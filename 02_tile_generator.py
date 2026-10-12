"""
02_tile_generator.py
--------------------
Slices the Bengaluru Sentinel-2 GeoTIFF into 256x256 pixel chips
ready for labelling and model training.

Output layout
-------------
data/
  chips/
    images/    <- .npy files  (11 bands, float32, shape 11x256x256)
    previews/  <- RGB .png files for quick visual inspection
  chip_index.csv  <- tile id, file path, bounding box, valid-pixel %

Usage
-----
  python 02_tile_generator.py

Tunable parameters (see CONFIG section below):
  TILE_SIZE      - chip size in pixels   (default 256)
  STRIDE         - step between tiles    (default 256 = no overlap)
  MIN_VALID_PCT  - skip chips with fewer valid pixels than this threshold
"""

import os
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_TIF    = os.path.join("data", "bengaluru_s2_ndvi_composite_feb_apr_2024.tif")
OUT_IMAGES   = os.path.join("data", "chips", "images")
OUT_PREVIEW  = os.path.join("data", "chips", "previews")
INDEX_CSV    = os.path.join("data", "chip_index.csv")

TILE_SIZE     = 256    # pixels
STRIDE        = 256    # pixels (== TILE_SIZE means no overlap)
MIN_VALID_PCT = 70.0   # skip chips where valid (non-NaN) pixels < this %

# Sentinel-2 band order in the stack (0-indexed)
# B2=0 B3=1 B4=2 B5=3 B6=4 B7=5 B8=6 B8A=7 B11=8 B12=9 NDVI=10
RGB_BANDS = (2, 1, 0)  # B4, B3, B2 -> Red, Green, Blue for RGB preview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_stretch(arr, lo=2, hi=98):
    """
    Stretch a single-band float array to [0, 255] uint8 using percentile clipping.
    NaN values are mapped to 0 after stretch.
    """
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    if p_hi == p_lo:
        return np.zeros_like(arr, dtype=np.uint8)
    stretched = np.clip((arr - p_lo) / (p_hi - p_lo) * 255.0, 0, 255)
    stretched = np.nan_to_num(stretched, nan=0.0)
    return stretched.astype(np.uint8)


def make_rgb_preview(chip, rgb_bands):
    """
    Create a PIL RGB image from a chip array (bands, H, W) for visual preview.
    """
    r = percentile_stretch(chip[rgb_bands[0]])
    g = percentile_stretch(chip[rgb_bands[1]])
    b = percentile_stretch(chip[rgb_bands[2]])
    rgb = np.stack([r, g, b], axis=-1)  # H x W x 3
    return Image.fromarray(rgb, mode="RGB")


def valid_pixel_pct(chip):
    """
    Return the percentage of finite (non-NaN, non-Inf) pixels across all bands.
    """
    total = chip.size
    valid = np.sum(np.isfinite(chip))
    return (valid / total) * 100.0


# ---------------------------------------------------------------------------
# Main tiling loop
# ---------------------------------------------------------------------------

def generate_tiles():
    os.makedirs(OUT_IMAGES,  exist_ok=True)
    os.makedirs(OUT_PREVIEW, exist_ok=True)

    with rasterio.open(INPUT_TIF) as src:
        width     = src.width
        height    = src.height
        n_bands   = src.count
        transform = src.transform

        print(f"[INFO] Source  : {INPUT_TIF}")
        print(f"[INFO] Size    : {width} x {height} px,  {n_bands} bands")
        print(f"[INFO] Tile    : {TILE_SIZE} x {TILE_SIZE} px,  stride {STRIDE} px")

        cols = list(range(0, width  - TILE_SIZE + 1, STRIDE))
        rows = list(range(0, height - TILE_SIZE + 1, STRIDE))
        total_possible = len(cols) * len(rows)
        print(f"[INFO] Max tiles: {total_possible}  (will skip chips < {MIN_VALID_PCT}% valid)\n")

        saved   = 0
        skipped = 0
        csv_rows = []

        for row_off in rows:
            for col_off in cols:
                window = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)

                # Read all bands for this window -> shape (bands, H, W)
                chip = src.read(window=window).astype(np.float32)

                # Replace nodata values with NaN
                nodata = src.nodata
                if nodata is not None:
                    chip[chip == nodata] = np.nan

                # Skip chips with too many NaN pixels (edges, cloud gaps)
                vpct = valid_pixel_pct(chip)
                if vpct < MIN_VALID_PCT:
                    skipped += 1
                    continue

                # Geographic bounds of this chip
                win_transform = rasterio.windows.transform(window, transform)
                left  = win_transform.c
                top   = win_transform.f
                right = left + TILE_SIZE * transform.a
                bot   = top  + TILE_SIZE * transform.e  # negative (north-up)

                tile_id  = f"tile_{row_off:05d}_{col_off:05d}"
                npy_path = os.path.join(OUT_IMAGES,  f"{tile_id}.npy")
                png_path = os.path.join(OUT_PREVIEW, f"{tile_id}.png")

                # Save chip array as numpy file
                np.save(npy_path, chip)

                # Save RGB preview image
                preview = make_rgb_preview(chip, RGB_BANDS)
                preview.save(png_path)

                csv_rows.append({
                    "tile_id"   : tile_id,
                    "npy_path"  : npy_path,
                    "png_path"  : png_path,
                    "col_off"   : col_off,
                    "row_off"   : row_off,
                    "left"      : round(left,  8),
                    "right"     : round(right, 8),
                    "top"       : round(top,   8),
                    "bottom"    : round(bot,   8),
                    "valid_pct" : round(vpct,  2),
                    "label"     : "",  # filled in Stage 3
                })

                saved += 1
                if saved % 50 == 0:
                    print(f"  ... saved {saved} chips  (skipped {skipped} so far)")

    # Write index CSV
    fieldnames = ["tile_id", "npy_path", "png_path", "col_off", "row_off",
                  "left", "right", "top", "bottom", "valid_pct", "label"]
    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[DONE] Tiles saved   : {saved}")
    print(f"       Tiles skipped  : {skipped}  (< {MIN_VALID_PCT}% valid pixels)")
    print(f"       Index CSV      : {INDEX_CSV}")
    print(f"       Images folder  : {OUT_IMAGES}")
    print(f"       Previews folder: {OUT_PREVIEW}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_tiles()
