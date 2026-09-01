"""
--------------------------------- train.py ---------------------------
Trains the good vs missing_open_eyelid classifier, tracking loss AND
accuracy per epoch for both train and validation.
After training, evaluates on the validation set and plots a confusion
matrix.
-----------------------------------------------------------------------
"""

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

from model import ShellClassifier
from dataset import make_loader


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


def main(train_dir="dataset_images/train", val_dir="dataset_images/validation",
         epochs=30, lr=1e-3, weight_decay=1e-4, patience=8):

    # Model Training parameters:
    # 1) Epochs = 30
    # 2) Learning rate = 1e-3
    # 3) weight decay = 1e-4
    # 5) patience = 8
    # Remember to only adjust one value at a time across retrains to be able to track how they affect the overall training

    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #checks to see if the gpu is available for training otherwise run on the cpu.
    print(f"Using device: {device}")

    train_loader, n_train, classes = make_loader(train_dir, augment=True)
    val_loader, n_val, val_classes = make_loader(val_dir, augment=False, shuffle=False)
    assert classes == val_classes, (
        f"Train classes {classes} don't match validation classes {val_classes} - "
        f"check both dataset_images/train/ and dataset_images/validation/ have the same subfolders."
    )
    print(f"Classes: {classes}")
    print(f"{n_train} training images, {n_val} validation images")

    model = ShellClassifier(img_size=128, num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2%}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "outputs/shell_classifier.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)") #implemented early stopping routine to end training when no improvement is seen
                break

    # reload best checkpoint (mirrors Keras EarlyStopping's restore_best_weights)
    model.load_state_dict(torch.load("outputs/shell_classifier.pt"))

    with open("outputs/classes.txt", "w") as f:
        f.write("\n".join(classes))

    plot_history(history)
    plot_confusion_matrix(model, val_loader, device, classes)
    return model, history


def plot_history(history):
    """
    We need to be able to plot the training history so that we can see how the model
    is progressing across the epochs. This will also help us with adjusting the various
    parameters of the model as it trains (e.g. number of epochs, learning rate etc.)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

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

    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=120)
    print("Saved outputs/training_history.png")


def plot_confusion_matrix(model, val_loader, device, classes):
    """
    We need to be able to evaluate the trained model before we can begin testing
    the inference, so an easy way is to generate a confusion matrix of the model
    after it has completed training, this will allow us to view how the model is
    handling the different classes.
    """
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
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues")
    plt.title("Confusion Matrix (Validation Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=120)
    print("Saved outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()