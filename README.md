# 🌿 Bengaluru Urban Green Space Classifier

A deep learning pipeline that detects and maps urban green spaces across Bengaluru using multispectral Sentinel-2 satellite imagery. The project covers the full ML lifecycle — from satellite data acquisition through model training to an interactive production-grade web dashboard.

---

## 📸 Dashboard Preview

> **Live app:** `streamlit run app/app.py`

The dashboard features an interactive Folium map with prediction overlays, an NDVI vegetation heatmap, dark-theme KPI cards, tile explorer, and model evaluation reports.

---

## 🗂️ Project Structure

```
green-space-classifier/
├── 01_gee_extractor.py       # Stage 1 — Google Earth Engine data export
├── 02_tile_generator.py      # Stage 2 — 256×256 chip generation
├── 03_label_generator.py     # Stage 3 — NDVI-threshold auto-labelling
├── 04_train.py               # Stage 4 — ResNet-18 model training
├── 05_evaluate.py            # Stage 5 — Evaluation & metrics
├── app/
│   └── app.py                # Stage 6 — Streamlit dashboard
├── data/
│   ├── bengaluru_s2_ndvi_composite_feb_apr_2024.tif   # (not tracked — large)
│   ├── chip_index.csv
│   └── chips/
│       ├── images/           # .npy chip arrays  (not tracked — large)
│       └── previews/         # RGB preview PNGs  (not tracked — large)
├── models/
│   ├── best_model.pth        # Best checkpoint   (not tracked — large)
│   ├── norm_stats.npy        # Normalisation stats (not tracked)
│   ├── train_log.csv         # Per-epoch metrics
│   ├── predictions.csv       # All tile predictions
│   ├── evaluation_report.png # Confusion matrix & F1 charts
│   └── training_curves.png   # Loss & accuracy curves
└── .gitignore
```

---

## 🛰️ Data Source

| Property | Value |
|---|---|
| Satellite | Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED) |
| Type | Surface Reflectance (Level-2A) |
| Period | February – April 2024 |
| Cloud filter | < 10% CLOUDY_PIXEL_PERCENTAGE |
| Bands | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 + derived NDVI |
| Resolution | 10 m / pixel |
| AOI | Bengaluru — lon [77.45, 77.75], lat [12.83, 13.14] |

---

## 🔁 Pipeline

### Stage 1 — GEE Data Extraction (`01_gee_extractor.py`)
- Authenticates with Google Earth Engine
- Loads Sentinel-2 SR collection, filters by cloud cover, applies SCL cloud mask
- Computes a per-pixel median composite
- Calculates NDVI = (B8 − B4) / (B8 + B4)
- Exports an 11-band Float32 GeoTIFF to Google Drive

### Stage 2 — Tile Generation (`02_tile_generator.py`)
- Slices the GeoTIFF into 256 × 256 px chips with no overlap
- Skips chips with < 70% valid pixels
- Saves each chip as a `.npy` array (shape `11 × 256 × 256`) and an RGB preview PNG
- Writes `data/chip_index.csv` with tile ID, geo-bounds, and valid pixel %

### Stage 3 — Auto-Labelling (`03_label_generator.py`)
- Labels each chip by mean NDVI value
  - NDVI > 0.30 → `green`
  - NDVI ≤ 0.30 → `non_green`
- Adds a coloured border to preview PNGs (green = green, red = non_green)
- Updates the `label` and `mean_ndvi` columns in `chip_index.csv`

### Stage 4 — Model Training (`04_train.py`)
- Architecture: **ResNet-18** pretrained on ImageNet, adapted for 11-band satellite input
- First conv layer expanded from 3 → 11 channels (extra channels warm-started from RGB mean)
- Loss: **Weighted cross-entropy** (handles 6:1 class imbalance)
- Optimiser: AdamW + Cosine Annealing LR schedule
- 80/20 train/val split with `WeightedRandomSampler` for balanced mini-batches
- Saves best checkpoint, normalisation stats, training curves, and per-epoch log

### Stage 5 — Evaluation (`05_evaluate.py`)
- Loads best model checkpoint
- Computes accuracy, precision, recall, F1 per class
- Generates confusion matrix, per-class bar chart, confidence distribution
- Saves `evaluation_report.png`, `predictions.csv`, and `prediction_grid.png`

### Stage 6 — Streamlit Dashboard (`app/app.py`)
- Production-grade dark-theme UI with custom CSS (Inter font, dark sidebar)
- Interactive Folium map — prediction overlay + NDVI heatmap layer
- Styled HTML popups on tile click (classification, confidence, NDVI, validation badge)
- Sidebar: layer controls, NDVI heatmap legend, confidence threshold, KPI cards, export
- Tabs: Map · Tile Explorer · Model Report · About

---

## 📊 Results

| Metric | Value |
|---|---|
| Best validation accuracy | **97.92%** (epoch 9) |
| Overall evaluation accuracy | **97.63%** |
| non_green — Precision / Recall / F1 | 100% / 97.24% / 98.60% |
| green — Precision / Recall / F1 | 85.71% / 100% / 92.31% |
| Total tiles | 169 |
| Green tiles | 24 (14.2%) |
| Non-green tiles | 145 (85.8%) |

The green class achieves **100% recall** — every actual green space is detected. The 85.71% precision reflects a small number of borderline tiles near the NDVI threshold.

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/rishika0o0/urban-greenspace.git
cd urban-greenspace
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install torch torchvision rasterio geopandas earthengine-api \
            streamlit folium streamlit-folium matplotlib pillow
```

### 4. Run the full pipeline

> **Note:** Steps 1–3 require the raw GeoTIFF from Google Drive. If you already have `data/bengaluru_s2_ndvi_composite_feb_apr_2024.tif`, start from Step 2.

```bash
# Step 1 — Authenticate GEE and export Sentinel-2 composite
earthengine authenticate
python 01_gee_extractor.py
# Download the exported GeoTIFF from Google Drive into data/

# Step 2 — Slice into chips
python 02_tile_generator.py

# Step 3 — Auto-label by NDVI
python 03_label_generator.py

# Step 4 — Train the classifier
python 04_train.py

# Step 5 — Evaluate
python 05_evaluate.py

# Step 6 — Launch the dashboard
streamlit run app/app.py
```

---

## 🧰 Tech Stack

| Layer | Library |
|---|---|
| Satellite data | Google Earth Engine (`earthengine-api`) |
| Raster I/O | `rasterio` |
| Geospatial | `geopandas` |
| Deep learning | `PyTorch`, `torchvision` |
| Visualisation | `matplotlib`, `Pillow` |
| Dashboard | `Streamlit`, `Folium`, `streamlit-folium` |

---

## 📁 What is not tracked by git

The following large files are excluded via `.gitignore` and must be regenerated locally:

- `data/*.tif` — GeoTIFF satellite composite (~200 MB)
- `data/chips/images/*.npy` — chip arrays
- `data/chips/previews/*.png` — preview images
- `models/*.pth` — model weights
- `models/norm_stats.npy` — normalisation statistics
- `venv/` — Python virtual environment

---

## 📄 License

This project is for educational and research purposes.
Sentinel-2 data © European Space Agency (ESA) — Copernicus Programme.
