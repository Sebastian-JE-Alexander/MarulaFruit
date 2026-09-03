"""
---------------------- detect_and_classify.py -------------------------------------------

Finds every shell in an image, classifies each one
independently, and draws the result back onto the image as a labelled
bounding box. The classifier itself only ever sees one cropped
shell at a time; this script is the layer that finds however many
shells are in a frame and hands each one to the classifier separately.

NOTE: When specifying colours for classes in the dict, remember that OpenCv uses
      BGR order not the standard RGB.


Usage: python detect_and_classify.py path/to/image.png
------------------------------------------------------------------------------------------
"""

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import ShellClassifier
from dataset import build_transform
from segment_grid_photos import find_blobs, crop_shell

COLOURS = {
    # BGR - extend this if you add more classes later; unlisted classes
    # fall back to purple so nothing fails to draw
    "good": (0, 200, 0),
    "bad": (0, 0, 255),
}
DEFAULT_COLOUR = (128, 0, 128)


def load_model(weights_path="outputs/shell_classifier.pt",
                classes_path="outputs/classes.txt"):
    with open(classes_path) as f:
        classes = f.read().strip().split("\n")
    missing = [c for c in classes if c not in COLOURS]
    if missing:
        print(f"Warning: no Colour entry for class(es): {missing} - "
              f"the classes will be draw using default colour {DEFAULT_COLOUR}."
              f"Add them to the COLOURS dict for visual clarity of classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShellClassifier(img_size=128, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device, classes


def classify_crop(model, device, crop_gray, classes):
    """
    crop_grey: (H,W) uint8 numpy array. Returns (class_name, confidence, ms).
    """
    img = Image.fromarray(crop_gray)
    x = build_transform(augment=False)(img).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000

    idx = int(probs.argmax())
    return classes[idx], float(probs[idx]), ms


def detect_and_classify(image_path, model, device, classes, output_path=None,
                         min_area=5000):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = find_blobs(gray, min_area=min_area)

    annotated = img.copy()
    results = []
    for (x, y, w, h) in boxes:
        crop = crop_shell(gray, (x, y, w, h))
        pred_class, confidence, ms = classify_crop(model, device, crop, classes)
        results.append({"bbox": (x, y, w, h), "class": pred_class,
                         "confidence": confidence, "inference_ms": ms})

        colour = COLOURS.get(pred_class, DEFAULT_COLOUR)
        print("LOOKUP:", repr(pred_class), "in COLOURS:", pred_class in COLOURS, "->",
              COLOURS.get(pred_class, DEFAULT_COLOUR))

        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 4)
        label = f"{pred_class} {confidence:.0%}"
        # label background for readability against a busy image
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(annotated, (x, max(y - th - 12, 0)), (x + tw + 8, y), colour, -1)
        cv2.putText(annotated, label, (x + 4, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    output_path = output_path or (image_path.rsplit(".", 1)[0] + "_annotated.png")
    cv2.imwrite(output_path, annotated)

    return results, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model, device, classes = load_model()
    print(f"Loaded model, classes = {classes}\n")

    results, output_path = detect_and_classify(args.image_path, model, device, classes,
                                                 output_path=args.output)

    print(f"Found {len(results)} shell(s):")
    for i, r in enumerate(results):
        print(f"  Shell {i+1}: {r['class']} ({r['confidence']:.2%})  "
              f"bbox={r['bbox']}  ({r['inference_ms']:.2f} ms)")
    print(f"\nSaved annotated image -> {output_path}")