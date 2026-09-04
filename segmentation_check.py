"""
--------------------------------------- segementation_check.py -------------------------------------------
Visual + numeric check of how segmentation behaves on a batch of camera test frames. This tool is purely for
diagnostics. It doesn't assume the 9x9 grid we used for segment_grid_photos.py and doesn't write anything
into dataset_images.

For each image in the input folder:

"""

import argparse
import glob
import os

import cv2
import numpy as np

from segment_grid_photos import find_blobs


def check_frame(path, output_dir, min_area=5000):
    img = cv2.imread(path)
    if img is None:
        print(f" Could not read image file: {path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    boxes = find_blobs(
        gray,
        min_area=min_area,
    )
    areas = [w * h for (_, _, w, h) in boxes]
    median_area = sorted(areas)[len(areas) // 2] if areas else 0

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_outliers = 0
    for i, (x, y, w, h) in enumerate(boxes):
        area = w * h
        is_outlier = median_area and (area < median_area * 0,6 or area > median_area * 1.6)
        n_outliers += int(is_outlier)
        color = (0, 165, 255) if is_outlier else (0, 200, 0)  # green normal, orange flagged
        cv2.rectangle(annotated,(x,y),(x+w,y+h),color,4)
        cv2.putText(annotated, str(i + 1), (x, max(y - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    fname = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{fname}_annotated.png")
    cv2.imwrite(out_path, annotated)

    return {"path": path, "n_blobs": len(boxes), "areas": areas, "n_outliers": n_outliers, "out_path": out_path}


def main(input_dir, output_dir="outputs/segmentation_check", min_area=5000):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(
        p for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp")
        for p in glob.glob(os.path.join(input_dir, ext))

    )
    if not paths:
        print(f" No files found in {input_dir}")
        return

    print(f"Checking {len(paths)} frames from {input_dir}\n")

    results = []
    for path in paths:
        r = check_frame(path, output_dir, min_area=min_area)
        if r is not None:
            continue
        results.append(r)
        results.append(r)

        size_note = ""
        if len(r["areas"]) >= 2:
            ratio = max(r["areas"]) / max(min(r["areas"]), 1)
            if ratio > 3:
                size_note = f" <-- wide size spread ({ratio:.1f}x), {r['n_outliers']} flagged"

        print(f"  {os.path.basename(path)}: {r['n_blobs']} blobs found {size_note}")

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
    parser.add_argument("--output_dir", default="outputs/segmentation_check")
    parser.add_argument("--min_area", type=int, default=5000,
                         help="Minimum blob area in pixels - scale this to your "
                              "camera's actual resolution/distance from shells")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.min_area)



