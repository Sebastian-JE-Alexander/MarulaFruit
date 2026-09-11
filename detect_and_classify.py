"""
---------------------- detect_and_classify.py -------------------------------------------

Finds every shell in an image, classifies each one
independently, and draws the result back onto the image as a labelled
bounding box. The classifier itself only ever sees one cropped
shell at a time; this script is the layer that finds however many
shells are in a frame and hands each one to the classifier separately.


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
import onnxruntime as ort

from model import ShellClassifier
from dataset import build_transform
from segment_grid_photos import find_blobs, crop_shell
import config


def load_model(weights_path=config.MODEL_WEIGHTS_PATH,
               classes_path=config.CLASSES_PATH):
    """
    Single entry point for both backends - callers (process_frame,
    camera_gui.py, etc.) never need to know or care which one is active,
    they just get back (model, device, classes) and pass model/device
    straight into classify_crop() below, which branches internally on
    the same config.USE_ONNX flag.
    """
    if config.USE_ONNX:
        return _load_onnx_model()
    return _load_pytorch_model(weights_path, classes_path)


def _load_pytorch_model(weights_path, classes_path):
    """
    Loads the normal pytorch model that is created with train.py for inference.
    """
    with open(classes_path) as f:
        classes = f.read().strip().split("\n")

    _warn_if_missing_colours(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShellClassifier(img_size=config.IMG_SIZE, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device, classes


def _load_onnx_model():
    """
    Loads the generated ONNX model stored within the project outputs folder.
    An ONNX model must first be generated using model_export.py
    """

    with open(config.ONNX_CLASSES_PATH) as f:
        classes = f.read().strip().split("\n")

    _warn_if_missing_colours(classes)
    session = ort.InferenceSession(config.ONNX_MODEL_PATH, providers=config.ONNX_PROVIDERS)
    input_name = session.get_inputs()[0].name

    active_providers = session.get_providers()
    print(f"ONNX Runtime active provider(s): {active_providers}")
    if "CUDAExecutionProvider" in config.ONNX_PROVIDERS and \
            "CUDAExecutionProvider" not in active_providers:
        print("WARNING: CUDAExecutionProvider was requested in config.ONNX_PROVIDERS "
              "but is NOT active - silently running on CPU instead. Check that "
              "onnxruntime-gpu (not plain onnxruntime) is installed and that your "
              "CUDA/cuDNN versions match what this onnxruntime-gpu build expects.")

    return (session, input_name), None, classes  # device kept as None - ONNX Runtime
    # manages its own execution device


def _warn_if_missing_colours(classes):
    """
    Checks if a new class for detection was declared but not assigned
    a colour in the dict found in config.py
    """
    missing = [c for c in classes if c not in config.CLASS_COLOURS]
    if missing:
        print(f"WARNING: no CLASS_COLOURS entry in config.py for class(es) {missing} - "
              f"they'll draw as the default colour {config.DEFAULT_COLOUR}. "
              f"Add them to the CLASS_COLOURS dict in config.py if "
              f"you want them visually distinct.")


def classify_crop(model, device, crop_gray, classes):
    """
    crop_gray: (H,W) uint8 numpy array. Returns (class_name, confidence, ms).
    Branches on config.USE_ONNX to match whichever backend load_model()
    actually loaded - model/device are whatever load_model() returned,
    unchanged, so this only needs to know how to use each shape.
    """
    img = Image.fromarray(crop_gray)
    x = build_transform(augment=False)(img).unsqueeze(0)

    if config.USE_ONNX:
        session, input_name = model
        start = time.perf_counter()
        logits = session.run(None, {input_name: x.numpy()})[0]
        probs = torch.softmax(torch.tensor(logits), dim=1)[0]
        ms = (time.perf_counter() - start) * 1000
    else:
        x = x.to(device)
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


def process_frame(gray, model, device, classes, min_area=config.MIN_BLOB_AREA):
    """
    Core pipeline on an in-memory (H,W) grayscale numpy array - no disk
    I/O. This is what camera_gui.py calls directly on a live-grabbed
    frame; detect_and_classify() below wraps this for the file-based CLI
    usage, so both share exactly one implementation rather than drifting
    apart over time.

    Returns (annotated_bgr, results, total_ms) - total_ms covers the
    whole segmentation+classification pass over the frame, which is the
    number worth showing on a live GUI (not just one crop's inference
    time) since it's what determines how quickly a result appears after
    a trigger.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    frame_start = time.perf_counter()
    boxes = find_blobs(gray, min_area=min_area)

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    results = []
    for (x, y, w, h) in boxes:
        crop = crop_shell(gray, (x, y, w, h))
        pred_class, confidence, ms = classify_crop(model, device, crop, classes)
        results.append({"bbox": (x, y, w, h), "class": pred_class,
                        "confidence": confidence, "inference_ms": ms})

        colour = config.CLASS_COLOURS.get(pred_class, config.DEFAULT_COLOUR)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 4)
        label = f"{pred_class} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(annotated, (x, max(y - th - 12, 0)), (x + tw + 8, y), colour, -1)
        cv2.putText(annotated, label, (x + 4, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    total_ms = (time.perf_counter() - frame_start) * 1000
    return annotated, results, total_ms


def detect_and_classify(image_path, model, device, classes, output_path=None,
                        min_area=config.MIN_BLOB_AREA):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    annotated, results, total_ms = process_frame(gray, model, device, classes, min_area)

    output_path = output_path or (image_path.rsplit(".", 1)[0] + "_annotated.png")
    cv2.imwrite(output_path, annotated)

    return results, output_path


def fuse_pass_fail(camera_results):
    """
    Combines single-shell results from multiple camera angles into
    one PASS/FAIL verdict, for the "one shell at a time, multiple
    camera angles" demo setup.

    camera_results: dict of camera_name -> list of shell-result dicts
    (each with 'class'/'confidence'), i.e. process_frame()'s `results`
    output per camera, for a frame expected to contain exactly one shell.

    Rule: FAIL if ANY camera calls it 'bad'. PASS only if EVERY camera
    that saw the shell called it 'good'. Deliberately NOT a confidence
    average across cameras - the customer explicitly said missing/open
    eyelid shells are "particularly important... to remove", meaning a
    missed defect (false negative) costs more than a wrongly-rejected
    good shell (false positive). The whole point of a second camera
    angle is catching a defect that's only visible from one side -
    averaging a clear detection from one camera against a "can't see
    anything wrong from here" read from the other would dilute exactly
    the signal the second camera exists to provide. A plain OR toward
    'bad' preserves it instead.

    Returns (verdict, explanation, per_camera_summary):
      verdict: "PASS", "FAIL", or "ERROR" (wrong shell count in some camera)
      explanation: one-line human-readable reason, good for display
      per_camera_summary: {camera_name: single result dict}, cameras
        with a valid single-shell read only
    """
    per_camera_summary = {}
    problems = []

    for name, results in camera_results.items():
        if len(results) != 1:
            problems.append(f"{name}: {len(results)} shell(s) found (expected 1)")
            continue
        per_camera_summary[name] = results[0]

    if problems:
        return "ERROR", "; ".join(problems), per_camera_summary

    if not per_camera_summary:
        return "ERROR", "No camera results to fuse", per_camera_summary

    bad_cams = [name for name, r in per_camera_summary.items() if r["class"] == "bad"]

    if bad_cams:
        detail = ", ".join(f"{name} ({per_camera_summary[name]['confidence']:.0%})"
                           for name in bad_cams)
        return "FAIL", f"bad detected by: {detail}", per_camera_summary

    detail = ", ".join(f"{name} ({r['confidence']:.0%})" for name, r in per_camera_summary.items())
    return "PASS", f"all camera(s) agree good: {detail}", per_camera_summary


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
        print(f"  Shell {i + 1}: {r['class']} ({r['confidence']:.2%})  "
              f"bbox={r['bbox']}  ({r['inference_ms']:.2f} ms)")
    print(f"\nSaved annotated image -> {output_path}")