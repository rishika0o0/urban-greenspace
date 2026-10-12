"""
01_gee_extractor.py
-------------------
Google Earth Engine data extractor for Bengaluru Urban Green Space Classifier.

Workflow:
  1. Authenticate & initialise the Earth Engine API.
  2. Define the Bengaluru bounding-box AOI.
  3. Load Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED) for Feb–Apr 2024.
  4. Filter by cloud cover (<10 %).
  5. Apply a cloud-mask using the SCL (Scene Classification Layer).
  6. Compute a median composite.
  7. Derive an NDVI band from B8 (NIR) and B4 (Red).
  8. Stack the spectral bands together with NDVI.
  9. Export the image to Google Drive (or optionally as a local GeoTIFF via
     getDownloadURL for small test chips).

Usage:
  python 01_gee_extractor.py

Requirements:
  pip install earthengine-api
  Run `earthengine authenticate` once before the first execution.
"""

import ee
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# 1. Authenticate and Initialise
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Your Google Cloud Project linked to Earth Engine
# (visible in the Task Manager header: "Cloud Project: my-project-511412")
# ---------------------------------------------------------------------------
GEE_PROJECT = "my-project-511412"


def authenticate_and_init() -> None:
    """
    Authenticate with the Earth Engine API and initialise the session.

    Strategy:
      - Always call ee.Authenticate() on first run to cache credentials.
      - Use ee.Initialize(project=...) every time — required by EE API v0.1.370+.
      - If credentials already exist, ee.Authenticate() is skipped via the
        quiet=True + auth_mode='notebook' path; otherwise a browser opens.
    """
    credentials_path = os.path.expanduser("~/.config/earthengine/credentials")

    if not os.path.exists(credentials_path):
        print("[INFO] No cached credentials found – starting authentication…")
        print("[INFO] A browser window will open. Log in and paste the token here.")
        ee.Authenticate()
        print("[INFO] Authentication complete.")
    else:
        print("[INFO] Cached credentials found.")

    ee.Initialize(project=GEE_PROJECT)
    print(f"[INFO] Earth Engine initialised (project: {GEE_PROJECT}).")


# ---------------------------------------------------------------------------
# 2. Area of Interest – Bengaluru bounding box
# ---------------------------------------------------------------------------

# (min_lon, min_lat, max_lon, max_lat)
BENGALURU_BBOX = (77.45, 12.83, 77.75, 13.14)

def get_aoi(bbox: tuple) -> ee.Geometry.Rectangle:
    """
    Convert a (min_lon, min_lat, max_lon, max_lat) tuple to an
    ee.Geometry.Rectangle.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    print(f"[INFO] AOI defined: lon [{min_lon}, {max_lon}], lat [{min_lat}, {max_lat}]")
    return aoi


# ---------------------------------------------------------------------------
# 3–5. Load, filter, and cloud-mask Sentinel-2 SR
# ---------------------------------------------------------------------------

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Cloud-mask a Sentinel-2 SR image using the SCL band.

    SCL classes kept (clear-sky pixels):
      4  – Vegetation
      5  – Not-vegetated
      6  – Water
      7  – Unclassified
      11 – Snow / Ice

    Classes removed (clouds, shadows, cirrus, saturated):
      1, 2, 3, 8, 9, 10
    """
    scl = image.select("SCL")
    clear_mask = (
        scl.eq(4)
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
        .Or(scl.eq(11))
    )
    # Preserve original image scale; mask clouds
    return image.updateMask(clear_mask)


def load_sentinel2(aoi: ee.Geometry, start_date: str, end_date: str, max_cloud_pct: float = 10.0) -> ee.ImageCollection:
    """
    Load Sentinel-2 Surface Reflectance image collection.

    Parameters
    ----------
    aoi           : ee.Geometry  – Area of interest
    start_date    : str          – 'YYYY-MM-DD'
    end_date      : str          – 'YYYY-MM-DD'
    max_cloud_pct : float        – Maximum CLOUDY_PIXEL_PERCENTAGE (0–100)

    Returns
    -------
    ee.ImageCollection filtered, cloud-masked Sentinel-2 SR images.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(mask_s2_clouds)
    )

    count = collection.size().getInfo()
    print(f"[INFO] Sentinel-2 SR collection: {count} scenes found "
          f"({start_date} → {end_date}, cloud < {max_cloud_pct}%)")
    return collection


# ---------------------------------------------------------------------------
# 6. Median composite
# ---------------------------------------------------------------------------

def make_composite(collection: ee.ImageCollection) -> ee.Image:
    """Return a per-pixel median composite of the collection."""
    composite = collection.median()
    print("[INFO] Median composite computed.")
    return composite


# ---------------------------------------------------------------------------
# 7. Calculate NDVI
# ---------------------------------------------------------------------------

def add_ndvi(image: ee.Image) -> ee.Image:
    """
    Compute NDVI = (NIR – Red) / (NIR + Red) and add it as a band named 'NDVI'.

    Sentinel-2 bands used:
      B8  – NIR  (~842 nm)
      B4  – Red  (~665 nm)

    Reflectance values in S2 SR are scaled by 10 000; NDVI is scale-invariant
    so no additional scaling is needed.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


# ---------------------------------------------------------------------------
# 8. Band selection and stacking
# ---------------------------------------------------------------------------

# Core spectral bands to retain for the classifier
SPECTRAL_BANDS = [
    "B2",   # Blue      ~490 nm
    "B3",   # Green     ~560 nm
    "B4",   # Red       ~665 nm
    "B5",   # Red-Edge1 ~705 nm
    "B6",   # Red-Edge2 ~740 nm
    "B7",   # Red-Edge3 ~783 nm
    "B8",   # NIR       ~842 nm
    "B8A",  # Narrow NIR ~865 nm
    "B11",  # SWIR1     ~1610 nm
    "B12",  # SWIR2     ~2190 nm
    "NDVI", # Derived vegetation index
]

def build_stack(image: ee.Image, aoi: ee.Geometry) -> ee.Image:
    """
    Select the desired bands, cast all to Float32 (so NDVI Float64 and
    S2 reflectance Float32 share the same type), and clip to the AOI.

    GEE requires all exported bands to have the same data type.
    Float32 is sufficient precision for both reflectance and NDVI values.
    """
    stacked = image.select(SPECTRAL_BANDS).toFloat().clip(aoi)
    print(f"[INFO] Final band stack (Float32): {SPECTRAL_BANDS}")
    return stacked


# ---------------------------------------------------------------------------
# 9. Export to Google Drive
# ---------------------------------------------------------------------------

def export_to_drive(
    image: ee.Image,
    aoi: ee.Geometry,
    description: str = "Bengaluru_S2_GreenSpace_Composite",
    folder: str = "GEE_Exports",
    filename: str = "bengaluru_s2_ndvi_composite",
    scale_m: int = 10,
    crs: str = "EPSG:4326",
) -> ee.batch.Task:
    """
    Submit an Earth Engine Export.image.toDrive task.

    Parameters
    ----------
    image       : ee.Image   – Stacked composite to export
    aoi         : ee.Geometry– Export region
    description : str        – Task description (visible in GEE Task Manager)
    folder      : str        – Google Drive folder name
    filename    : str        – Output GeoTIFF base name
    scale_m     : int        – Pixel resolution in metres (10 m = S2 native)
    crs         : str        – Output coordinate reference system

    Returns
    -------
    ee.batch.Task – the started export task.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=filename,
        region=aoi,
        scale=scale_m,
        crs=crs,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"[INFO] Export task submitted: '{description}'")
    print(f"[INFO] Output → Google Drive / {folder} / {filename}.tif")
    print("[INFO] Monitor progress at: https://code.earthengine.google.com/tasks")
    return task


# ---------------------------------------------------------------------------
# Optional: Download a small thumbnail/chip for quick local inspection
# ---------------------------------------------------------------------------

def get_download_url(
    image: ee.Image,
    aoi: ee.Geometry,
    scale_m: int = 30,
    bands: list | None = None,
) -> str:
    """
    Generate a signed download URL for a small local test chip.

    NOTE: Use only for small AOIs / coarse resolution to stay within the
    getDownloadURL size limit (~32 MB). For the full Bengaluru tile use
    export_to_drive() instead.
    """
    if bands is None:
        bands = ["B4", "B3", "B2", "NDVI"]  # True-colour + NDVI

    url = image.select(bands).getDownloadURL(
        {
            "region": aoi,
            "scale": scale_m,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        }
    )
    print(f"[INFO] Download URL (scale={scale_m}m, bands={bands}):\n  {url}")
    return url


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print(" Bengaluru Urban Green Space – GEE Data Extractor")
    print("=" * 60)

    # -- 1. Auth & init -------------------------------------------------------
    authenticate_and_init()

    # -- 2. AOI ---------------------------------------------------------------
    aoi = get_aoi(BENGALURU_BBOX)

    # -- 3-5. Collection ------------------------------------------------------
    START_DATE = "2024-02-01"
    END_DATE   = "2024-04-30"
    collection = load_sentinel2(aoi, START_DATE, END_DATE, max_cloud_pct=10.0)

    # -- 6. Composite ---------------------------------------------------------
    composite = make_composite(collection)

    # -- 7. NDVI --------------------------------------------------------------
    composite_ndvi = add_ndvi(composite)

    # -- 8. Stack & clip ------------------------------------------------------
    final_image = build_stack(composite_ndvi, aoi)

    # -- 9. Export ------------------------------------------------------------
    # Primary: full-resolution export to Google Drive (10 m pixels)
    task = export_to_drive(
        image=final_image,
        aoi=aoi,
        description="Bengaluru_S2_GreenSpace_Composite_FebApr2024",
        folder="GEE_Exports",
        filename="bengaluru_s2_ndvi_composite_feb_apr_2024",
        scale_m=10,
        crs="EPSG:4326",
    )

    # Optional: quick local chip at 30 m for sanity-check
    print("\n[INFO] Generating quick-inspect download URL (30 m resolution)…")
    try:
        _ = get_download_url(final_image, aoi, scale_m=30, bands=["B4", "B3", "B2", "NDVI"])
    except Exception as exc:
        print(f"[WARN] Could not generate download URL (AOI may be too large at 30m): {exc}")

    print("\n[DONE] Pipeline complete. Check GEE Task Manager for export status.")
    print("=" * 60)


if __name__ == "__main__":
    main()
