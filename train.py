"""
--------------------------------- train.py ---------------------------
Trains the good vs bad shell classifier, tracking model loss and
accuracy per epoch for both train and validation.
After training, evaluates on the validation set and plots a confusion
matrix.

Usage: python train.py
-----------------------------------------------------------------------
"""

import csv
import os
import random
import time
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

from model import ShellClassifier
from dataset import make_loader

global ax_metrics

SEED = 42


def set_seed(seed=SEED):
    """
    Fixes random weight initialization and data shuffling order, so
    runs are reproducible and can be fairly compared - without this,
    every run gets a different random starting point, and a genuinely
    unlucky one can cause a 'dead network' (e.g. every ReLU unit stuck
    outputting zero from the first epoch, permanently killing its own
    gradient) that looks like a training failure but is actually just
    bad luck on initialization, indistinguishable from a real problem
    without a fixed seed to isolate what actually changed between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    """
    H:MM:SS for anything over an hour, otherwise M:SS.
    """
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"
    return f"{int(minutes)}:{secs:05.2f}"


def run_epoch(model, loader, device, criterion, optimizer=None):
    """
    optimizer=None -> eval mode, no gradient step. Returns (loss, accuracy).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def compute_class_weights(train_dir, classes):
    """
    Inverse-frequency class weights, so CrossEntropyLoss can't take
    the shortcut of leaning toward whichever class has more images.
    Necessary as soon as classes stop being roughly equal in count -
    e.g. if you keep photographing new good shells (real diversity gain)
    while bad stays capped by reshuffling the same limited physical
    stock (no real diversity gain, just more images of it), good will
    naturally pull ahead in raw count. That's fine and worth doing
    anyway - this weighting is what stops the count difference itself
    from biasing the model, so you don't have to hold back on
    photographing new good shells just to keep the numbers looking even.
    """
    counts = []
    for cls in classes:
        folder = os.path.join(train_dir, cls)
        n = len([f for f in os.listdir(folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        counts.append(n)

    counts = torch.tensor(counts, dtype=torch.float32)
    weights = counts.sum() / (len(counts) * counts)  # inverse frequency, normalised
    return weights

def compute_binary_metrics(cm, classes):
    """
    Returns None if classes isn't exactly length 2 - TP/TN/FP/FN
    doesn't have one clean meaning past binary classification. Extracted
    as its own function so both the confusion matrix AND logger can use
    the same numbers without computing them twice.
    """
    if len(classes) != 2:
        return None
    tn, fp, fn, tp = cm.ravel()
    negative_class, positive_class = classes[0], classes[1]
    return {
        "positive_class": positive_class,
        "negative_class": negative_class,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def log_training_run(log_path="outputs/training_log.csv", **kwargs):
    """
    Appends one row summarising this run to a persistent .csv, so
    results across training sessions accumulate in one place you
    can open in Excel. Creates the file with a header on the first-ever run;
    every run afterwards appends a new row.
    """
    metrics = kwargs.get("metrics", None)
    class_weights = kwargs.get("class_weights", None)
    classes = kwargs.get("classes", None)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(kwargs["device"]),
        "classes": "|".join(classes),
        "n_train": kwargs["n_train"],
        "n_val": kwargs["n_val"],
        "epochs_requested": kwargs["epochs_requested"],
        "epochs_run": kwargs["epochs_run"],
        "total_duration": format_duration(kwargs["total_seconds"]),
        "avg_seconds_per_epoch": round(kwargs["avg_epoch_seconds"], 2),
        "lr": kwargs["lr"],
        "weight_decay": kwargs["weight_decay"],
        "patience": kwargs["patience"],
        "class_weights": "|".join(
            f"{c}={w:.3f}" for c, w in zip(classes, class_weights.tolist())),
        "best_val_loss": round(kwargs["best_val_loss"], 5),
        # kept present (blank if unavailable) on every row, rather than
        # only appearing sometimes, so the CSV's columns never shift
        # between runs regardless of how many classes a given run had
        "accuracy": round(metrics["accuracy"], 4) if metrics else "",
        "recall": round(metrics["recall"], 4) if metrics else "",
        "specificity": round(metrics["specificity"], 4) if metrics else "",
        "precision": round(metrics["precision"], 4) if metrics else "",
        "npv": round(metrics["npv"], 4) if metrics else "",
        "positive_class": metrics["positive_class"] if metrics else "",
    }

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended this run to {log_path}")


def main(train_dir="dataset_images/train", val_dir="dataset_images/validation",
         epochs=30, lr=1e-3, weight_decay=1e-4, patience=8):
    training_start = time.perf_counter()
    set_seed()
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, n_train, classes = make_loader(train_dir, augment=True)
    val_loader, n_val, val_classes = make_loader(val_dir, augment=False, shuffle=False)
    assert classes == val_classes, (
        f"Train classes {classes} don't match validation classes {val_classes} - "
        f"check both dataset_images/train/ and dataset_images/validation/ have the same subfolders."
    )
    print(f"Classes: {classes}")
    print(f"{n_train} training images, {n_val} validation images")

    class_weights = compute_class_weights(train_dir, classes)
    print(f"Class weights (inverse frequency): "
          f"{dict(zip(classes, [round(w, 3) for w in class_weights.tolist()]))}")

    model = ShellClassifier(img_size=128, num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_seconds": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, optimizer=None)

        epoch_seconds = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_seconds"].append(epoch_seconds)

        print(f"Epoch {epoch + 1}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2%}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}  "
              f"({epoch_seconds:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "outputs/shell_classifier.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    # reload best checkpoint (mirrors Keras EarlyStopping's restore_best_weights)
    model.load_state_dict(torch.load("outputs/shell_classifier.pt"))

    with open("outputs/classes.txt", "w") as f:
        f.write("\n".join(classes))

    total_seconds = time.perf_counter() - training_start
    epochs_run = len(history["epoch_seconds"])
    avg_epoch_seconds = sum(history["epoch_seconds"]) / epochs_run

    print(f"\nTotal training time: {format_duration(total_seconds)}  "
          f"({epochs_run} epochs, avg {avg_epoch_seconds:.1f}s/epoch)")

    with open("outputs/training_duration.txt", "w") as f:
        f.write(f"device: {device}\n")
        f.write(f"epochs_run: {epochs_run}\n")
        f.write(f"total_seconds: {total_seconds:.2f}\n")
        f.write(f"total_duration: {format_duration(total_seconds)}\n")
        f.write(f"avg_seconds_per_epoch: {avg_epoch_seconds:.2f}\n")
        f.write(f"n_train_images: {n_train}\n")
        f.write(f"n_val_images: {n_val}\n")
    print("Saved outputs/training_duration.txt")

    plot_history(history)
    metrics = plot_confusion_matrix(model, val_loader, device, classes)
    log_training_run(
        device=device, classes=classes, n_train=n_train, n_val=n_val,
        epochs_requested=epochs, epochs_run=epochs_run,
        total_seconds=total_seconds, avg_epoch_seconds=avg_epoch_seconds,
        lr=lr, weight_decay=weight_decay, patience=patience,
        class_weights=class_weights, best_val_loss=best_val_loss,
        metrics=metrics,
    )
    return model, history


def plot_history(history):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11))

    ax1.plot(history["train_loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Training Accuracy")
    ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True)

    ax3.plot(history["epoch_seconds"], label="Seconds per Epoch", color="purple")
    ax3.set_title("Epoch Duration (useful for spotting slowdowns across a session)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Seconds")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=120)
    print("Saved outputs/training_history.png")


def plot_confusion_matrix(model, val_loader, device, classes):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    metrics = compute_binary_metrics(cm, classes)

    # Metrics panel only makes unambiguous sense for exactly 2 classes -
    # TP/TN/FP/FN don't have one clean meaning once a 3rd+ class exists
    # (that needs one-vs-rest per class instead). Skip it automatically
    # if this project grows past good/bad, rather than showing something
    # misleading.
    show_metrics = metrics is not None

    if show_metrics:
        fig, (ax_cm, ax_metrics) = plt.subplots(
            1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 2]})
    else:
        fig, ax_cm = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues", ax=ax_cm)
    ax_cm.set_title("Confusion Matrix (Validation Set)")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")

    if show_metrics:
        positive_class, negative_class = metrics["positive_class"], metrics["negative_class"]
        metrics_text = (
            f"Positive class: '{positive_class}'\n"
            f"Negative class: '{negative_class}'\n"
            f"\n"
            f"Accuracy:     {metrics['accuracy']:.1%}\n"
            f"\n"
            f"Recall (Sens.): {metrics['recall']:.1%}\n"
            f"  real '{positive_class}' caught\n"
            f"\n"
            f"Specificity:  {metrics['specificity']:.1%}\n"
            f"  real '{negative_class}' caught\n"
            f"\n"
            f"Precision:    {metrics['precision']:.1%}\n"
            f"  predicted '{positive_class}' correct\n"
            f"\n"
            f"NPV:          {metrics['npv']:.1%}\n"
            f"  predicted '{negative_class}' correct"
        )
        ax_metrics.axis("off")
        ax_metrics.text(0.02, 0.98, metrics_text, transform=ax_metrics.transAxes,
                        fontsize=10, verticalalignment="top", family="monospace",
                        bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=120)
    print("Saved outputs/confusion_matrix.png")
    return metrics


if __name__ == "__main__":
    main()