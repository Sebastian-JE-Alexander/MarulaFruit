"""
----------------------------- diagnose_double_detection.py -------------------------------
Run this against a captured image where a single physical shell got detected as two
bounding boxes. Draws every candidate box BEFORE deduplication is applied, then colour
codes them by which method found it (Otsu vs Adaptive), plus the final boxes that actually
returned. This helps identify which method is responsible and what it's picking up.

Usage: python diagnose_double_detection.py path/to/image.png
------------------------------------------------------------------------------------------
"""

import argparse

import cv2

from segment_grid_photos import debug_find_blobs

COLOURS = {"otsu": (0, 165, 255), "adaptive": (255, 0, 255), "final": (0, 255, 0)}

def diagnose(image_path, output_path=None, min_area=None):
    img = cv2.imread(image_path)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    kwargs = {"min_area": min_area} if min_area else {}
    result = debug_find_blobs(gray, **kwargs)

    print(f"Otsu found:    {len(result['otsu_raw'])} boxes")
    for b in result['otsu_raw']:
        print(f"   {b} area={b[2]*b[3]}")
    print(f"Adaptive found:    {len(result['adaptive_raw'])} boxes")
    for b in result['adaptive_raw']:
        print(f"   {b} area={b[2]*b[3]}")
    print(f" Final (after deduping + outlier rejection):  {len(result['final'])} boxes")
    for b in result['final']:
        print(f"   {b} area={b[2]*b[3]}")

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for b in result['otsu_raw']:
        x, y, w, h = b
        cv2.rectangle(annotated, (x, y), (x + w, y +h), COLOURS["otsu"], 3)
    for b in result['adaptive_raw']:
        x, y, w, h = b
        cv2.rectangle(annotated, (x - 6, y - 6), (x + w + 6, y + h + 6), COLOURS["adaptive"], 3)
    for b in result['final']:
        x, y, w, h = b
        cv2.rectangle(annotated, (x - 6, y - 6), (x + w + 6, y + h + 6), COLOURS["final"], 2)

    legend_y = 30
    for label, colour in [("Otsu (orange)", COLOURS["otsu"]),
                          ("Adaptive (magenta)", COLOURS["adaptive"]),
                          ("Final (green)", COLOURS["final"])]:
        cv2.putText(annotated, label, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX,0.8, colour, 2)
        legend_y += 30

    output_path = output_path or (image_path.rsplit(".", 1)[0] + "_diagnosed.png")
    cv2.imwrite(output_path, annotated)
    print(f"\nSaved -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--min_area", default=None)
    args = parser.parse_args()
    diagnose(args.image_path, args.output, args.min_area)


