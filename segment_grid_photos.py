"""
--------------------------------- segment_grid_photos.py --------------------------------

Splits 3x3 grid photos of ONE class into individual shell crops, using
Otsu-based segmentation.
Saves into dataset_images/train/<class_name>/ and dataset_images/validation/<class_name>/,
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


Usage:
python segment_grid_photos.py --input_dir raw_imgs/train/good --val_input_dir raw_imgs/validation/good --class_name good
python segment_grid_photos.py --input_dir raw_imgs/train/bad --val_input_dir raw_imgs/validation/bad --class_name bad
--------------------------------------------------------------------------------------------
"""

import os
import random
import glob
import argparse

import cv2
import numpy as np

import config


def debug_find_blobs(gray, min_area=config.MIN_BLOB_AREA, max_aspect=config.MAX_BLOB_ASPECT):
    """
    Same detection as find_blobs(), but returns which method (otsu or
    adaptive) produced each surviving box, and the boxes BEFORE
    deduplication too - for diagnosing exactly why a real frame produces
    more boxes than physical shells actually present. Not used by the
    normal pipeline - purely a diagnostic entry point.

    Mirrors find_blobs()'s downscaled-throughout approach (see that
    function's docstring) so this diagnostic reflects what production
    actually does, not an older/slower path.

    Returns dict with keys: 'otsu_raw', 'adaptive_raw' (boxes before any
    merging/dedup, tagged by source, in FULL-RESOLUTION coordinates) and
    'final' (what find_blobs() would actually return).
    """
    max_dim = 900
    scale = min(1.0, max_dim / max(gray.shape[:2]))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray

    masks = {}
    _, mask_otsu = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_otsu_inv = cv2.bitwise_not(mask_otsu)
    masks["otsu"] = mask_otsu if (mask_otsu > 0).sum() < (mask_otsu_inv > 0).sum() else mask_otsu_inv

    block_size = _odd(max(int(min(small.shape[:2]) * 0.35), 151))
    masks["adaptive"] = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 5)

    k = max(int(min(small.shape[:2]) * 0.01), 3)
    kernel = np.ones((k, k), np.uint8)
    min_area_scaled = min_area * scale * scale

    per_method_boxes = {}
    all_boxes = []
    for source, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes_this_source = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_scaled:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > max_aspect:
                continue
            if scale < 1.0:
                x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
            boxes_this_source.append((x, y, w, h))
            all_boxes.append((x, y, w, h))
        per_method_boxes[source] = boxes_this_source

    final = _reject_size_outliers(_dedupe_boxes(all_boxes))

    return {"otsu_raw": per_method_boxes["otsu"], "adaptive_raw": per_method_boxes["adaptive"],
            "final": final}


def find_blobs(gray, min_area=config.MIN_BLOB_AREA, max_aspect=config.MAX_BLOB_ASPECT,
               use_adaptive=True):
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

    PERFORMANCE NOTE: everything (Otsu, adaptive threshold, morphology,
    contour-finding) runs on a DOWNSCALED copy of the frame - only the
    final box coordinates get scaled back up to full resolution. An
    earlier version only downscaled the adaptive threshold's own
    computation, then resized the result back UP to full resolution
    before morphology/contours - meaning the expensive per-pixel work
    (morphology + contour tracing, done twice: once for Otsu's native
    full-res mask, once for the upscaled adaptive mask) was still
    happening on ~20 megapixel masks regardless. Measured on a real
    live-camera setup: this was costing 120-200ms per frame, scaling
    directly with camera resolution. Doing the expensive steps at
    the small scale and only scaling coordinates at the very end removes that cost
    almost entirely, without changing the underlying detection logic.
    """
    max_dim = 900
    scale = min(1.0, max_dim / max(gray.shape[:2]))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray

    masks = []

    _, mask_otsu = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_otsu_inv = cv2.bitwise_not(mask_otsu)
    otsu_fg = mask_otsu if (mask_otsu > 0).sum() < (mask_otsu_inv > 0).sum() else mask_otsu_inv
    masks.append(otsu_fg)

    if use_adaptive:
        block_size = _odd(max(int(min(small.shape[:2]) * 0.35), 151))
        adaptive = cv2.adaptiveThreshold(
            small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block_size, 5)
        masks.append(adaptive)

    k = max(int(min(small.shape[:2]) * 0.01), 3)
    kernel = np.ones((k, k), np.uint8)

    # min_area is specified in FULL-RESOLUTION pixel units -
    # since contours are now measured
    # in the downscaled image, the threshold needs the same scale-down
    # (area scales with scale^2, not scale) or it would reject
    # everything at this smaller pixel count.
    min_area_scaled = min_area * scale * scale

    all_boxes = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_scaled:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)  # scale-invariant, fine to check before scaling up
            if aspect > max_aspect:
                continue
            # Scale coordinates back to full resolution - cheap (a few
            # numbers), unlike scaling the mask itself would have been.
            if scale < 1.0:
                x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
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
    to be re-tuned any time image resolution or shell distance-from-
    camera changes; this adapts automatically to whatever scale the
    real shells happen to appear at in a given photo.
    """
    if len(boxes) < 2:
        return boxes
    areas = sorted(b[2] * b[3] for b in boxes)
    median_area = areas[len(areas) // 2]
    return [b for b in boxes if (b[2] * b[3]) >= median_area * min_fraction_of_median]


def _odd(n):
    return n if n % 2 == 1 else n + 1


def _dedupe_boxes(boxes, iou_threshold=0.5, containment_threshold=0.7):
    """
    Removes duplicate detections of the same shell found by both
    Otsu and adaptive thresholding, keeping the larger (usually
    tighter/more complete) box of any overlapping pair.

    Two separate checks, not just IOU (intersection over union): real frames
    showed a smaller box sitting almost entirely INSIDE a larger one
    (adaptive catching the whole shell, Otsu catching a sub-region of
    it, or vice versa) with IOU of only 0.19 and 0.38 - both well under
    the 0.5 threshold, so pure IOU-based deduplication never caught either one.
    This isn't a threshold-tuning problem: a small box fully contained
    in a big one structurally has low IOU regardless of threshold,
    because the union stays dominated by the big box's area. Containment
    is a genuinely different relationship from overlap and needs its
    own check - if most of a smaller box's area sits inside a larger
    box, treat them as the same detection regardless of what IOU says.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        is_duplicate = any(
            _iou(box, k) > iou_threshold or _containment_fraction(box, k) > containment_threshold
            for k in kept
        )
        if not is_duplicate:
            kept.append(box)
    return kept


def _containment_fraction(small, big):
    """
    What fraction of small area overlaps with big - 1.0 means
    small sits entirely inside big. Order-independent in effect since
    _dedupe_boxes always calls this with the box being considered as
    small (boxes are processed largest-first, so anything already in
    kept is >= box in area) - this checks how much of the new,
    smaller-or-equal box is lost by what's already kept.
    """
    sx0, sy0, sw, sh = small
    bx0, by0, bw, bh = big
    sx1, sy1 = sx0 + sw, sy0 + sh
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(sx0, bx0), max(sy0, by0)
    ix1, iy1 = min(sx1, bx1), min(sy1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    small_area = sw * sh
    return intersection / small_area if small_area else 0.0


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


def process_class(input_dir, class_name, train_root=config.TRAIN_DIR,
                  val_root=config.VAL_DIR, val_fraction=config.VAL_FRACTION,
                  seed=config.SEED, min_area=config.MIN_BLOB_AREA,
                  expected_per_photo=config.EXPECTED_SHELLS_PER_GRID_PHOTO):
    """
    Random photo-level split - ONLY safe when every photo in input_dir
    shows genuinely independent physical shells never repeated in any
    other photo (no reshuffling/rephotographing the same batch). If
    you're reshuffling the same shells for extra pose variety, use
    process_two_folders() instead - see its docstring for why.

    This function has been depreciated but kept here as an example.
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
                        train_root=config.TRAIN_DIR, val_root=config.VAL_DIR,
                        min_area=config.MIN_BLOB_AREA,
                        expected_per_photo=config.EXPECTED_SHELLS_PER_GRID_PHOTO):
    """
    Use this when you've reshuffled/rephotographed the same physical
    shells for extra pose variety. Random per-photo splitting (see
    process_class) can't safely handle that: if the same 9 physical
    shells are reshuffled and rephotographed 3 times, a random split
    could put 2 of those photos in training and 1 in validation, meaning
    the same physical shells appear on both sides of the split - just in
    different poses. That defeats the point of a held-out validation set.

    The fix is to decide train vs validation at the PHYSICAL SHELL
    level, before any reshuffling: physically set aside a validation
    portion of your shells first, then reshuffle and rephotograph each
    group as many times as you like WITHIN itself, keeping their photos
    in two separate folders from the start. This function then just
    processes each folder independently - no random splitting, because
    the separation already happened physically, at capture time.

    Set up your raw images folder into train/valid subfolders with their
    own class folders within each containing the relevant images.
    This means the function just has to be pointed at the top folder path for
    it to process through it (see usage note in the docstring at the top of this
    script.)
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