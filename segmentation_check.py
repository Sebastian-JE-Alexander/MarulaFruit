"""
--------------------------------------- segementation_check.py -------------------------------------------
Visual + numeric check of how segmentation behaves on a batch of camera test frames. This tool is purely for
diagnostics. It doesn't assume the 9x9 grid we used for segment_grid_photos.py and doesn't write anything
into dataset_images.

For each image in the input folder:
  - Draws every detected box on a copy of the frame (green = normal,
    orange = flagged as a size outlier within that same frame) and
    saves it for a quick visual scan
  - Reports blob count and size spread per frame

Usage: python segmentation_check.py path/to/test_frames_folder

"""

import argparse
import glob
import os

import cv2
import numpy as np

from segment_grid_photos import find_blobs
import config


def check_frame(path, output_dir, min_area=config.MIN_BLOB_AREA):
    img = cv2.imread(path)
    if img is None:
        print(f"  COULD NOT READ: {path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    boxes = find_blobs(gray, min_area=min_area)
    areas = [w * h for (_, _, w, h) in boxes]
    median_area = sorted(areas)[len(areas) // 2] if areas else 0

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_outliers = 0
    for i, (x, y, w, h) in enumerate(boxes):
        area = w * h
        # Same outlier idea as the real pipeline's own rejection filter,
        # but shown here rather than silently dropped, so you can see
        # what's borderline, not just what already passed. Threshold
        # tuned down from an initial 2.0x/0.5x after testing found a
        # genuine two-shell merge only reached 1.9x median area (some
        # overlap is lost in the merge, so it doesn't cleanly double) -
        # 1.6x/0.6x catches that case without over-flagging normal
        # single-shell size variation in testing.
        is_outlier = median_area and (area < median_area * 0.6 or area > median_area * 1.6)
        n_outliers += int(is_outlier)
        colour = (0, 165, 255) if is_outlier else (0, 200, 0)  # orange flagged / green normal
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 4)
        cv2.putText(annotated, str(i + 1), (x, max(y - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

    fname = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{fname}_annotated.png")
    cv2.imwrite(out_path, annotated)

    return {"path": path, "n_blobs": len(boxes), "areas": areas,
            "n_outliers": n_outliers, "out_path": out_path}


def main(input_dir, output_dir=config.SEGMENTATION_CHECK_DIR, min_area=config.MIN_BLOB_AREA):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for p in glob.glob(os.path.join(input_dir, ext))
    )
    if not paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Checking {len(paths)} frame(s) from {input_dir}\n")

    results = []
    for path in paths:
        r = check_frame(path, output_dir, min_area=min_area)
        if r is None:
            continue
        results.append(r)

        size_note = ""
        if len(r["areas"]) >= 2:
            ratio = max(r["areas"]) / max(min(r["areas"]), 1)
            if ratio > 3:
                size_note = f"  <-- wide size spread ({ratio:.1f}x), {r['n_outliers']} flagged"

        print(f"  {os.path.basename(path)}: {r['n_blobs']} blob(s) found{size_note}")

    if results:
        counts = [r["n_blobs"] for r in results]
        flagged_frames = sum(1 for r in results if r["n_outliers"] > 0)
        print(f"\nSummary: {len(results)} frame(s) checked, "
              f"{min(counts)}-{max(counts)} blobs per frame "
              f"(average {sum(counts) / len(counts):.1f})")
        print(f"{flagged_frames} frame(s) had at least one size-outlier box.")
        print(f"\nAnnotated frames saved to {output_dir}/ - "
              f"orange boxes were flagged as size outliers within their own "
              f"frame (worth a manual look), green boxes look normal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Folder of test camera frames to check")
    parser.add_argument("--output_dir", default=config.SEGMENTATION_CHECK_DIR)
    parser.add_argument("--min_area", type=int, default=config.MIN_BLOB_AREA,
                        help="Minimum blob area in pixels - scale this to your "
                             "camera's actual resolution/distance from shells")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.min_area)



