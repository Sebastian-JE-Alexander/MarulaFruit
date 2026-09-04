"""
----------------------------- dataset_health_check.py -----------------------------------------
Run this script before training to catch problems with our image dataset before training starts.

1. Per-class, per-split image counts, train/validation ratio, and class balance (train.py class
   weighting compensates for imbalance automatically, but very large ratios can show that the smaller
   class is lacking enough real images to learn from).

2. File integrity - anything that fails to open, inconsistent image sizes, and
   unexpected colour content (current camera is mono8 grayscale, so any image
   with RGB content is worth noting)

3. Duplicate files across train/validation - catches any copy-paste of the same
   file in both sides of the split.

Useage: python dataset_health_check.py

"""

import hashlib
import os
from collections import Counter

import cv2
import numpy as np

TRAIN_DIR = "dataset_images/train"
TEST_DIR = "dataset_images/validation"
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(VALID_EXTENSIONS)]


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def check_class_counts(train_dir, val_dir):
    print("=" * 64)
    print("Class Counts")
    print("=" * 64)

    classes = set()
    for base in (train_dir, val_dir):
        if os.path.isdir(base):
            classes.update(c for c in os.listdir(base) if os.path.isdir(os.path.join(base, c)))

    classes = sorted(classes)

    if not classes:
        print(" No class folders found under either directory")
        return {}

    counts = {}
    for cls in classes:
        n_train = len(list_images(os.path.join(train_dir, cls)))
        n_val = len(list_images(os.path.join(val_dir, cls)))
        total = n_train + n_val
        val_pct = (n_val / total * 100) if total else 0
        counts[cls] = (n_train, n_val)

        flag = ""
        if total == 0:
            flag = " <-- Empty, no images found "
        elif n_train == 0:
            flag = " <-- no training images found "
        elif n_val == 0:
            flag = " <-- no validation images found "
        elif not (10 <= val_pct <= 30):
            flag = " <-- training validation split is {val_pct:.0f}% (target ~15-25%) "

        print(f" {cls:25s} train={n_train:4d}  val={n_val:4d} "
              f"(val {val_pct:4.0f}%){flag}")

    train_counts = [c[0] for c in counts.values() if c[0] > 0]
    if len(train_counts) >= 2:
        ratio = max(train_counts) / min(train_counts)
        print(f"\n Class balance (train, max/min: {ratio:.2f}x", end="")
        if ratio > 5:
            print(" <-- large imbalance. train.py's class weighting "
                  "compensates in the loss function, but the smaller"
                  "class might not have enough images to learn"
                  "regardless of class weighting.")
        elif ratio > 2:
            print(" <-- moderate imbalance, class weighting should handle this. ")
        else:
            print(" <-- small imbalance. ")
    return counts


def check_file_integrity(train_dir, val_dir):
    print("\n" + "=" * 64)
    print ("File Integrity")
    print("=" * 64)

    all_paths = []
    for base in (train_dir, val_dir):
        if not os.path.isdir(base):
            continue
        for cls in os.listdir(base):
            cls_dir = os.path.join(base, cls)
            if os.path.isdir(cls_dir):
                all_paths.extend(list_images(cls_dir))

    if not all_paths:
        print(" No images found to check.")
        return

    unreadable, sizes, color_flags = [], [], []
    for path in all_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            unreadable.append(path)
            continue
        h, w = img.shape[:2]
        sizes.append((w, h))
        if img.ndim == 3 and img.shape[2] >= 3:
            b, g, r = img[..., 0], img[..., 1], img[..., 2]
            if not (np.array_equal(b, g) and np.array_equal(g, r)):
                color_flags.append(path)

    if unreadable:
        print(f" {len(unreadable)} files failed to open:")
        for p in unreadable:
            print(f"   {p}")
        else:
            print(f" All {len(all_paths)} images opened successfully.")
    if sizes:
        size_counts = Counter(sizes)
        if len(size_counts) > 1:
            (common_size, common_n) = size_counts.most_common(1)[0]
            print(f"\n Image sizes are NOT all consistent, most common is " 
                  f"{common_size[0]}x{common_size[1]}  ({common_n} /{len(sizes)})."
                  f"Other sizes found:")
            for size, n in size_counts.most_common():
                if size != common_size:
                    print(f"  {size[0]}x{size[1]}: {n} images")

        else:
            (w, h) = next(iter(size_counts))
            print(f"\n All images are a consistent {w}x{h}.")

    if color_flags:
        print(f"\n  {len(color_flags)} image(s) have real colour content "
              f"(R/G/B channels differ) - unexpected for a Mono8 camera "
              f"source, worth checking these weren't saved from somewhere "
              f"else by accident:")
        for p in color_flags[:10]:
            print(f"    {p}")
        if len(color_flags) > 10:
            print(f"    ... and {len(color_flags) - 10} more")
    else:
        print(f"\n  All images are genuinely grayscale (R=G=B) - "
              f"consistent with a Mono8 camera source.")

def check_duplicates_across_split(train_dir, val_dir):
    print("\n" + "=" * 64)
    print("EXACT DUPLICATE FILES ACROSS TRAIN/VALIDATION")
    print("=" * 64)
    print("  (Catches an accidental copy-paste of the same file into both\n"
          "  sides. Does NOT catch reshuffled photos of the same physical\n"
          "  shell - those differ pixel-for-pixel - that risk still needs\n"
          "  the physically-separate-before-reshuffling discipline.)\n")

    train_hashes = []
    for cls in (os.listdir(train_dir) if os.path.isdir(train_dir) else []):
        cls_dir = os.path.join(train_dir, cls)
        if os.path.isdir(cls_dir):
            for os.path.isdir(cls_dir):
                for path in list_images(cls_dir):
                    train_hashes[file_hash(path)] = path

    duplicates = []
    for cls in (os.listdir(val_dir) if os.path.isdir(val_dir) else []):
        cls_dir = os.path.join(val_dir, cls)
        if os.path.isdir(cls_dir):
            for path in list_images(cls_dir):
                h = file_hash(path)
                if h in train_hashes:
                    duplicates.append((train_hashes[h], path))

    if duplicates:
        print(f" Found {len(duplicates)} exact duplicates.")
        for train_path, val_path in duplicates:
            print(f"   {train_path}\n    == {val_path}")
        else:
            print(f" No exact duplicate files found between train and validation.")

def main():
    print("Dataset Health Check")







