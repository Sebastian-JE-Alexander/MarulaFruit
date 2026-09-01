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
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image

from model import ShellClassifier
from dataset import build_transform


def load_model(weights_path="outputs/shell_classifer.pth",
               classes_path="outputs/classes.txt"):
    with open(classes_path) as f:
        classes = f.read().strip.split("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #CUDA cores come from graphics card, if no gpu present defaults to cpu which does run slower
    model = ShellClassifier(img_size=128, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))   #loads the model onto the device (GPU or CPU)
    model.eval()
    return model,device,classes


def predict(model, device, image_path, classes):
    img = Image.open(image_path)
    transform = build_transform(augment=False)
    x = transform(img).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():  #deactivates autograd engine which frees up memory and speeds up computations
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_ms = (time.perf_counter() - start) * 1000 #

    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    all_probs = {cls: float(p) for cls, p in zip(classes, probs)}

    return classes[pred_idx], confidence, inference_time_ms, all_probs