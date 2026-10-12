"""
03_label_generator.py
---------------------
Automatically labels each 256x256 chip as 'green' or 'non_green'
using the mean NDVI value of the tile.

Labelling rule:
  mean NDVI > 0.3  ->  green      (vegetation, parks, trees, farmland)
  mean NDVI <= 0.3 ->  non_green  (urban, roads, bare soil, water)

NDVI band is index 10 (last band) in the 11-band stack saved by
02_tile_generator.py.

Outputs:
  - Updates the 'label' column in data/chip_index.csv  (in-place)
  - Prints a class distribution summary
  - Saves data/chips/previews with a coloured border overlay:
      green border  -> green label
      red border    -> non_green label

Usage:
  python 03_label_generator.py
"""

import os
import csv
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INDEX_CSV    = os.path.join("data", "chip_index.csv")
OUT_PREVIEW  = os.path.join("data", "chips", "previews")

NDVI_BAND    = 10       # 0-indexed band position of NDVI in the .npy stack
NDVI_THRESH  = 0.3      # mean NDVI threshold for green vs non_green
BORDER_PX    = 6        # border thickness in pixels for labelled previews

GREEN_COLOR  = (34,  139, 34)   # forest green
RED_COLOR    = (220,  20, 60)   # crimson


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_mean_ndvi(npy_path):
    """Load chip and return mean NDVI, ignoring NaN values."""
    chip = np.load(npy_path)          # shape: (11, 256, 256)
    ndvi = chip[NDVI_BAND].astype(np.float32)
    valid = ndvi[np.isfinite(ndvi)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def add_border(png_path, color, border_px):
    """
    Add a solid coloured border to an existing PNG preview to visually
    indicate the assigned label. Overwrites the original file.
    """
    img  = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(border_px):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    img.save(png_path)


# ---------------------------------------------------------------------------
# Main labelling loop
# ---------------------------------------------------------------------------

def label_tiles():
    # Read existing CSV
    with open(INDEX_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    if "label" not in fieldnames:
        fieldnames.append("label")

    total      = len(rows)
    n_green    = 0
    n_nongreen = 0

    print(f"[INFO] Labelling {total} tiles  (NDVI threshold = {NDVI_THRESH})\n")

    for i, row in enumerate(rows):
        mean_ndvi = compute_mean_ndvi(row["npy_path"])
        label     = "green" if mean_ndvi > NDVI_THRESH else "non_green"

        row["label"]      = label
        row["mean_ndvi"]  = round(mean_ndvi, 4)

        # Add coloured border to preview PNG
        png_path = row["png_path"]
        if os.path.exists(png_path):
            color = GREEN_COLOR if label == "green" else RED_COLOR
            add_border(png_path, color, BORDER_PX)

        if label == "green":
            n_green += 1
        else:
            n_nongreen += 1

        if (i + 1) % 50 == 0:
            print(f"  ... processed {i + 1}/{total}")

    # Ensure mean_ndvi column is in fieldnames
    if "mean_ndvi" not in fieldnames:
        fieldnames.append("mean_ndvi")

    # Write updated CSV back
    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"\n{'='*45}")
    print(f"  Labelling complete")
    print(f"{'='*45}")
    print(f"  Total tiles   : {total}")
    print(f"  green         : {n_green}  ({n_green/total*100:.1f}%)")
    print(f"  non_green     : {n_nongreen}  ({n_nongreen/total*100:.1f}%)")
    print(f"  Index CSV     : {INDEX_CSV}")
    print(f"  Previews      : {OUT_PREVIEW}  (green border = green, red border = non_green)")
    print(f"{'='*45}")

    # Warn if heavily imbalanced
    ratio = max(n_green, n_nongreen) / max(min(n_green, n_nongreen), 1)
    if ratio > 3:
        print(f"\n[WARN] Class imbalance detected (ratio {ratio:.1f}:1).")
        print("       Consider using weighted loss or oversampling the minority class during training.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    label_tiles()
