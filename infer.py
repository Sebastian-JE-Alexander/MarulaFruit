"""
----------------------------------------- infer.py ------------------------------------------------

Loads the trained classifier and predicts good vs bad
(or whichever classes were trained) for a new image, with per-image
inference timing - only the model forward pass is timed, not image load/preprocess, and the first
call is reported separately since it includes first time warmup cycle.

Usage: python infer.py path/to/image.png [path/to/another.png ...]
---------------------------------------------------------------------------------------------------
"""

import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

from model import ShellClassifier
from dataset import build_transform
import config


def load_model(weights_path=config.MODEL_WEIGHTS_PATH,
               classes_path=config.CLASSES_PATH):
    with open(classes_path) as f:
        classes = f.read().strip().split("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShellClassifier(img_size=config.IMG_SIZE, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device, classes


def predict(model, device, image_path, classes):
    """
    Returns (predicted_class: str, confidence: float, inference_time_ms: float,
    all_probs: dict).
    """
    img = Image.open(image_path)
    transform = build_transform(augment=False)
    x = transform(img).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_ms = (time.perf_counter() - start) * 1000

    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    all_probs = {cls: float(p) for cls, p in zip(classes, probs)}

    return classes[pred_idx], confidence, inference_time_ms, all_probs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py path/to/image.png [path/to/another.png ...]")
        sys.exit(1)

    model, device, classes = load_model()
    print(f"Loaded model, classes = {classes}\n")

    times = []
    for path in sys.argv[1:]:
        pred_class, confidence, inference_time_ms, all_probs = predict(model, device, path, classes)
        times.append(inference_time_ms)
        probs_str = ", ".join(f"{c}={p:.2%}" for c, p in all_probs.items())
        print(f"{path}: {pred_class} ({confidence:.2%})  [{probs_str}]  "
              f"({inference_time_ms:.2f} ms)")

    if len(times) > 1:
        print(f"\nFirst call: {times[0]:.2f} ms  |  "
              f"Remaining calls avg: {sum(times[1:]) / len(times[1:]):.2f} ms")
