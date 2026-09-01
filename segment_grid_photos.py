"""
--------------------------------- segment_grid_photos.py -----------------------

Splits 3x3 grid photos of ONE class into individual shell crops, using
Otsu-based segmentation.
Saves into dataset/train/<class_name>/ and dataset/validation/<class_name>/,
matching the folder-per-class structure dataset.py's ImageFolder
expects.

Run this once per class.
Nothing else in the pipeline needs to change when
you add a class this way, dataset.py picks up new folders automatically.

Splits at the PHOTO level (not the individual crop level) into
train/validation - same physical-shell-leakage reasoning as before: if
any photos are reshuffled/rephotographed shots of the same physical
shells, keeping whole photos on one side of the split guarantees no
crossover between validation and training images.
"""

import os
import random
import glob
import argparse

import cv2
import numpy as np


def find_blobs(gray, min_area=5000, max_aspect=2.5, use_adaptive=True):
    """
    Otsu's method finds ONE global foreground/background split for
    the whole image - this works well when all objects in frame have
    similar contrast against the background.

    Fix: also run adaptive thresholding (a local threshold computed per
    neighbourhood, not one global value) and merge its detections with
    Otsu's. Adaptive thresholding is more sensitive to local contrast
    regardless of an object's absolute brightness, so it catches
    lower-contrast objects a global method can miss; Otsu is kept
    because it's typically cleaner/less noisy on the higher-contrast
    cases. Overlapping detections from both methods are de-duplicated.
    """

    masks = []

    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_otsu_inv = cv2.bitwise_not(mask_otsu)
    otsu_fg = mask_otsu if (mask_otsu > 0).sum() < (mask_otsu_inv > 0).sum() else mask_otsu_inv
    masks.append(otsu_fg)

    if use_adaptive:
        # Run adaptive thresholding on a downscaled copy - at full camera
        # resolution (~5472x3648) with the large block size needed above,
        # this step far too slow to process a real batch of images.
        # Shell locations don't need pixel-perfect
        # precision (crop_shell adds padding anyway), so detecting at a
        # smaller scale and mapping boxes back up is a large speedup.
        max_dim = 900
        scale = min(1.0, max_dim / max(gray.shape[:2]))
        small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray

        block_size = _odd(max(int(min(small.shape[:2]) * 0.35), 151))
        adaptive_small = cv2.adaptiveThreshold(
            small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block_size, 5)
        adaptive = cv2.resize(adaptive_small, (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        masks.append(adaptive)

    k = max(int(min(gray.shape[:2]) * 0.01), 3)
    kernel = np.ones((k, k), np.uint8)

    all_boxes = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > max_aspect:
                continue
            all_boxes.append((x, y, w, h))

    boxes = _dedupe_boxes(all_boxes)
    return _reject_size_outliers(boxes)


def _reject_size_outliers(boxes, min_fraction_of_median=0.35):
    """
    Rejects blobs much smaller than the median blob size in this
    image - real shells in one photo should all be roughly consistent
    size, so a blob at a fraction of that size is far more likely to be
    a small artifact (a shadow, a lighting speck near the frame edge)
    than a genuine shell. More robust than a fixed min_area, which has
    to be re-tuned any time image resolution or the distance from the
    camera changes; this adapts automatically to whatever scale the
    marula shells happen to appear at in a given image.
    """

    if len(boxes) < 2:
        return boxes
    areas = sorted(b[2] * b[3] for b in boxes)
    median_area = areas[len(areas) // 2]
    return [b for b in boxes if (b[2] * b[3]) >= median_area * min_fraction_of_median]


def _odd(n):
    return n if n % 2 == 1 else n + 1


def _dedupe_boxes(boxes, iou_threshold=0.5):
    """
    Removes duplicate detections of the same shell found by both
    Otsu and adaptive thresholding, keeping the larger (usually
    tighter/more complete) box of any overlapping pair.
    """

    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        if not any(_iou(box, k) > iou_threshold for k in kept):
            kept.append(box)
    return kept


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def crop_shell(gray, box, pad_factor=1.25, out_size=160):
    x, y, w, h = box
    cx, cy = x + w // 2, y + h // 2
    side = int(max(w, h) * pad_factor)
    H, W = gray.shape[:2]
    x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x1, y1 = min(cx + side // 2, W), min(cy + side // 2, H)
    crop = gray[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size))


def process_class(input_dir, class_name, train_root="dataset/train",
                   val_root="dataset/validation", val_fraction=0.2, seed=42,
                   min_area=5000, expected_per_photo=9):
    """
    Random photo-level split - ONLY safe when every photo in input_dir
    shows genuinely independent physical shells never repeated in any
    other photo (no reshuffling/rephotographing the same batch). If
    you're reshuffling the same shells for extra pose variety, use
    process_two_folders() instead.
    """

    train_out = os.path.join(train_root, class_name)
    val_out = os.path.join(val_root, class_name)
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"[{class_name}] Found {len(paths)} source photos in {input_dir}")
    if not paths:
        print(f"[{class_name}] No .png files found - check input_dir path.")
        return

    random.seed(seed)
    shuffled = paths[:]
    random.shuffle(shuffled)
    n_val_photos = max(int(len(shuffled) * val_fraction), 1)
    val_photos = set(shuffled[:n_val_photos])

    _segment_photos(paths, val_photos, train_out, val_out, class_name,
                     min_area, expected_per_photo)


def process_two_folders(train_input_dir, val_input_dir, class_name,
                         train_root="dataset/train", val_root="dataset/validation",
                         min_area=5000, expected_per_photo=9):
    """
    Use this when you've reshuffled/rephotographed the same physical
    shells for extra pose variety.

    The fix is to decide train vs validation at the PHYSICAL
    level, before any reshuffling: physically set aside a validation
    portion of your shells first, then reshuffle and rephotograph each
    group as many times as you like WITHIN the group, keeping their photos
    in two separate folders from the start. This function then just
    processes each folder independently - no random splitting, because
    the separation already happened physically, at capture time.
    """

    train_out = os.path.join(train_root, class_name)
    val_out = os.path.join(val_root, class_name)
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    train_paths = sorted(glob.glob(os.path.join(train_input_dir, "*.png")))
    val_paths = sorted(glob.glob(os.path.join(val_input_dir, "*.png")))
    print(f"[{class_name}] Found {len(train_paths)} train photos in {train_input_dir}")
    print(f"[{class_name}] Found {len(val_paths)} validation photos in {val_input_dir}")
    if not train_paths or not val_paths:
        print(f"[{class_name}] Missing photos in one of the two folders - check both paths.")
        return

    all_paths = train_paths + val_paths
    val_set = set(val_paths)
    _segment_photos(all_paths, val_set, train_out, val_out, class_name,
                     min_area, expected_per_photo)


def _segment_photos(paths, val_photos, train_out, val_out, class_name,
                     min_area, expected_per_photo):
    counts_per_photo = []
    total_train, total_val = 0, 0

    for path in paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        boxes = find_blobs(gray, min_area=min_area)
        counts_per_photo.append((os.path.basename(path), len(boxes)))

        out_dir = val_out if path in val_photos else train_out
        stem = os.path.splitext(os.path.basename(path))[0]
        for i, box in enumerate(boxes):
            crop = crop_shell(gray, box)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{i:02d}.png"), crop)

        if path in val_photos:
            total_val += len(boxes)
        else:
            total_train += len(boxes)

    n_val_photos = len(val_photos)
    print(f"[{class_name}] Train: {total_train} crops from {len(paths) - n_val_photos} photos -> {train_out}")
    print(f"[{class_name}] Val:   {total_val} crops from {n_val_photos} photos -> {val_out}")

    off_count = [c for c in counts_per_photo if c[1] != expected_per_photo]
    if off_count:
        print(f"[{class_name}] {len(off_count)} photo(s) did NOT yield exactly "
              f"{expected_per_photo} blobs (worth a manual check):")
        for name, n in off_count:
            print(f"  {name}: {n} blobs found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                         help="Folder of raw 3x3 grid .png photos for ONE class - "
                              "random photo-level split. Only safe if photos are NOT "
                              "reshuffled shots of the same physical shells.")
    parser.add_argument("--val_input_dir",
                         help="If your shells were physically separated into train/validation "
                              "groups before photographing (recommended if you reshuffled for "
                              "extra photos), pass --input_dir as the TRAIN photo folder and "
                              "this as the VALIDATION photo folder. No random splitting is done "
                              "in this mode - each folder is processed as-is.")
    parser.add_argument("--class_name", required=True,
                         help="Class name - becomes the subfolder name under dataset/train and dataset/validation")
    args = parser.parse_args()

    if args.val_input_dir:
        process_two_folders(args.input_dir, args.val_input_dir, args.class_name)
    else:
        process_class(args.input_dir, args.class_name)