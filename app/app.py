"""
app/app.py
----------
Bengaluru Urban Green Space Classifier — Production-grade Streamlit dashboard.

Usage (from project root):
  streamlit run app/app.py
"""

import os
import csv
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV   = os.path.join(ROOT, "models", "predictions.csv")
BEST_MODEL = os.path.join(ROOT, "models", "best_model.pth")
NORM_STATS = os.path.join(ROOT, "models", "norm_stats.npy")
REPORT_PNG = os.path.join(ROOT, "models", "evaluation_report.png")
CURVE_PNG  = os.path.join(ROOT, "models", "training_curves.png")

NUM_BANDS  = 11
CLASSES    = ["non_green", "green"]
LABEL_MAP  = {c: i for i, c in enumerate(CLASSES)}
MAP_CENTER = [12.9716, 77.5946]
MAP_ZOOM   = 11


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], .main, .main .block-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0a171d !important;
    color: #94a3b8 !important;
}

/* Hide default streamlit header padding & chrome */
#MainMenu        { visibility: hidden; display: none; }
header           { visibility: hidden; height: 0 !important; }
footer           { visibility: hidden; display: none; }
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* ---- Top banner (Matte Slate Card) ---- */
.top-banner {
    background: #0d1e26;
    border-radius: 8px;
    padding: 24px 30px;
    margin-bottom: 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}
.banner-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #8fae9b;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    padding: 4px 12px;
    border-radius: 4px;
    margin-bottom: 12px;
    text-transform: uppercase;
}
.banner-title {
    font-size: 26px;
    font-weight: 600;
    color: #f8fafc;
    margin: 4px 0 8px 0;
    line-height: 1.25;
    letter-spacing: -0.4px;
}
.banner-subtitle {
    font-size: 13.5px;
    color: #94a3b8;
    font-weight: 400;
    max-width: 680px;
    line-height: 1.6;
}
.banner-meta {
    margin-top: 16px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.banner-pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
    color: #cbd5e1;
    font-weight: 450;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 12px;
    font-weight: 600;
    color: #8fae9b;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 6px;
    margin-bottom: 14px;
    margin-top: 8px;
}

/* ---- Map container ---- */
.map-container {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* ---- Sidebar refinement ---- */
[data-testid="stSidebar"] {
    background-color: #071217 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
.sidebar-section {
    background: #0d1b22;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 12px;
}
.sidebar-section-title {
    font-size: 10.5px;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #8ca3af !important;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 500;
    color: #cbd5e1;
    margin-bottom: 14px;
}
.status-dot {
    width: 6px; height: 6px;
    background: #8fae9b;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ---- Sidebar KPI cards (Matte Vertical Stack) ---- */
.sidebar-kpi-card {
    background: #0d1b22;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 8px;
    box-shadow: none;
    transition: border-color 0.2s ease;
}
.sidebar-kpi-card:hover {
    border-color: rgba(255, 255, 255, 0.16);
}
.sidebar-kpi-label {
    font-size: 9.5px;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.sidebar-kpi-value {
    font-size: 21px;
    font-weight: 600;
    line-height: 1.1;
    margin-bottom: 2px;
}
.sidebar-kpi-sub {
    font-size: 10.5px;
    color: #8ca3af;
    margin-bottom: 5px;
}
.sidebar-kpi-badge {
    display: inline-block;
    font-size: 9.5px;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 3px;
    letter-spacing: 0.3px;
}

/* ---- Tile grid cards ---- */
.tile-card {
    background: #0d1b22;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    padding: 8px;
    text-align: center;
    box-shadow: none;
}
.tile-label-green    { color: #8fae9b; font-size: 11px; font-weight: 500; }
.tile-label-nongreen { color: #64748b; font-size: 11px; font-weight: 500; }
.tile-conf           { color: #64748b; font-size: 10px; }

/* ---- Tab styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #071217;
    padding: 4px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 4px;
    font-size: 13px;
    font-weight: 450;
    padding: 6px 18px;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: #0d1e26 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: none !important;
}

/* ---- About cards ---- */
.about-card {
    background: #0d1b22;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    padding: 18px 22px;
    margin-bottom: 12px;
}
.about-card h4 {
    font-size: 13px;
    font-weight: 600;
    color: #f8fafc;
    margin: 0 0 8px 0;
    letter-spacing: 0.2px;
}
.about-card p, .about-card li {
    font-size: 12.5px;
    color: #94a3b8;
    line-height: 1.6;
}

/* ---- Header wrap & Top Bar ---- */
.ci-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 14px;
    padding: 14px 0 2px 0;
}
.ci-header-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.ci-accent-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #8fae9b;
    box-shadow: none;
    flex-shrink: 0;
}
.ci-header-title {
    font-size: 24px;
    font-weight: 600;
    color: #f8fafc;
    letter-spacing: -0.3px;
    margin: 0;
}
.ci-header-subtitle {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 400;
    margin: 0 0 0 20px;
    line-height: 1.5;
}

/* ---- Back to Landing Page Button ---- */
.back-landing-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #0d1e26;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #cbd5e1 !important;
    text-decoration: none !important;
    cursor: pointer;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    letter-spacing: 0.2px;
    width: 100%;
    justify-content: center;
    margin-bottom: 14px;
    box-sizing: border-box;
}
.back-landing-btn:hover {
    background: #142a35;
    border-color: rgba(255, 255, 255, 0.24);
    color: #f8fafc !important;
    text-decoration: none !important;
}
.back-landing-btn svg {
    stroke: #8fae9b;
    transition: transform 0.15s ease;
    flex-shrink: 0;
}
.back-landing-btn:hover svg {
    transform: translateX(-3px);
    stroke: #b4cfc0;
}
.back-landing-btn-header {
    width: auto;
    margin-bottom: 0;
    padding: 7px 16px;
    flex-shrink: 0;
    margin-top: 2px;
}

/* ---- Heatmap legend (sidebar) ---- */
.heatmap-legend {
    background: #0d1b22;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    padding: 12px 14px;
    margin: 8px 0 14px 0;
}
.heatmap-legend-title {
    font-size: 10px;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.heatmap-legend-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 8px;
}
.heatmap-legend-row:last-child { margin-bottom: 0; }
.heatmap-swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    flex-shrink: 0;
    margin-top: 2px;
}
.heatmap-legend-label {
    font-size: 11.5px;
    font-weight: 500;
    color: #e2e8f0;
    line-height: 1.3;
}
.heatmap-legend-desc {
    font-size: 10px;
    color: #64748b;
    line-height: 1.4;
}

/* Streamlit Buttons & Dataframes */
.stButton > button, .stDownloadButton > button {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    color: #f8fafc !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
}
.stDataFrame {
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    overflow: hidden;
}
</style>
"""


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    def build(num_bands=NUM_BANDS, num_classes=2):
        m        = models.resnet18(weights=None)
        old_conv = m.conv1
        new_conv = nn.Conv2d(
            num_bands, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        m.conv1 = new_conv
        m.fc    = nn.Linear(m.fc.in_features, num_classes)
        return m

    if os.path.exists(BEST_MODEL) and os.path.exists(NORM_STATS):
        try:
            ckpt  = torch.load(BEST_MODEL, map_location="cpu")
            model = build()
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            stats = np.load(NORM_STATS, allow_pickle=True).item()
            return model, stats
        except Exception:
            return None, None
    return None, None


@st.cache_data
def load_predictions():
    with open(PRED_CSV, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_preview(png_path, size=(180, 180)):
    try:
        norm_path = png_path.replace("\\", os.sep)
        candidates = [
            norm_path,
            os.path.join(ROOT, norm_path),
            os.path.join(ROOT, "data", "chips", "previews", os.path.basename(norm_path)),
        ]
        for p in candidates:
            if os.path.exists(p):
                return Image.open(p).convert("RGB").resize(size, Image.LANCZOS)
        return Image.open(png_path).convert("RGB").resize(size, Image.LANCZOS)
    except Exception:
        return Image.new("RGB", size, (40, 40, 40))


def ndvi_color(val):
    v = max(-1.0, min(1.0, float(val) if val else 0.0))
    t = (v + 1.0) / 2.0
    r = int(140 * (1 - t) + 40)
    g = int(120 * t + 40)
    b = 55
    return f"#{r:02x}{g:02x}{b:02x}"


def records_to_csv_bytes(records):
    buf = io.StringIO()
    if not records:
        return buf.getvalue().encode()
    writer = csv.DictWriter(buf, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode()


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_map(records, show_ndvi=False, show_preds=True, min_conf=0.0):
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite Imagery",
        overlay=False,
    ).add_to(m)

    pred_layer = folium.FeatureGroup(name="Green Space Predictions", show=show_preds)
    ndvi_layer = folium.FeatureGroup(name="NDVI Vegetation Index",   show=show_ndvi)

    for rec in records:
        try:
            left   = float(rec["left"])
            right  = float(rec["right"])
            top    = float(rec["top"])
            bottom = float(rec["bottom"])
            conf   = float(rec["confidence"])
            label  = rec["pred_label"]
            ndvi   = float(rec.get("mean_ndvi", 0) or 0)
        except (ValueError, KeyError):
            continue

        if conf < min_conf:
            continue

        bounds = [[bottom, left], [top, right]]
        is_green = label == "green"

        # Styled popup (Matte Dark)
        correct_badge = (
            '<span style="background:rgba(74,124,89,0.25);color:#8fae9b;padding:2px 7px;'
            'border-radius:3px;font-size:11px;font-weight:500;">Correct</span>'
            if rec.get("correct") == "True" else
            '<span style="background:rgba(194,125,125,0.2);color:#c27d7d;padding:2px 7px;'
            'border-radius:3px;font-size:11px;font-weight:500;">Mismatch</span>'
        )
        label_color = "#8fae9b" if is_green else "#8ca3af"
        popup_html = f"""
        <div style="font-family:Inter,sans-serif;min-width:200px;padding:6px;background:#0d1b22;color:#e2e8f0;border-radius:6px;border:1px solid rgba(255,255,255,0.08)">
          <div style="font-size:10px;color:#64748b;margin-bottom:3px;letter-spacing:0.8px;text-transform:uppercase">
            TILE ID
          </div>
          <div style="font-size:13px;font-weight:600;color:#f8fafc;margin-bottom:8px">
            {rec['tile_id']}
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <tr>
              <td style="color:#94a3b8;padding:3px 0">Classification</td>
              <td style="font-weight:500;color:{label_color};text-align:right">
                {label.replace('_',' ').title()}
              </td>
            </tr>
            <tr>
              <td style="color:#94a3b8;padding:3px 0">Green Probability</td>
              <td style="font-weight:500;color:#f8fafc;text-align:right">{conf*100:.1f}%</td>
            </tr>
            <tr>
              <td style="color:#94a3b8;padding:3px 0">Mean NDVI</td>
              <td style="font-weight:500;color:#f8fafc;text-align:right">{ndvi:.3f}</td>
            </tr>
            <tr>
              <td style="color:#94a3b8;padding:3px 0">Validation</td>
              <td style="text-align:right">{correct_badge}</td>
            </tr>
          </table>
        </div>
        """

        folium.Rectangle(
            bounds=bounds,
            color="#3c6350" if is_green else "#334155",
            weight=1,
            fill=True,
            fill_color="#4a7c59" if is_green else "#475569",
            fill_opacity=0.45,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=folium.Tooltip(
                f"<b>{'Green Space' if is_green else 'Urban / Non-Green'}</b>"
                f"<br>Confidence: {conf*100:.0f}%",
                style="font-family:Inter,sans-serif;font-size:12px;",
            ),
        ).add_to(pred_layer)

        folium.Rectangle(
            bounds=bounds,
            color=ndvi_color(ndvi),
            weight=0,
            fill=True,
            fill_color=ndvi_color(ndvi),
            fill_opacity=0.55,
            tooltip=folium.Tooltip(
                f"NDVI: <b>{ndvi:.3f}</b>",
                style="font-family:Inter,sans-serif;font-size:12px;",
            ),
        ).add_to(ndvi_layer)

    pred_layer.add_to(m)
    ndvi_layer.add_to(m)

    folium.LayerControl(
        position="topright",
        collapsed=True,
    ).add_to(m)

    # Refined Matte legend
    legend_html = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:1000;
                background:rgba(10,23,29,0.92);padding:14px 18px;
                border-radius:6px;box-shadow:0 4px 16px rgba(0,0,0,0.3);
                font-family:Inter,sans-serif;border:1px solid rgba(255,255,255,0.08);">
      <div style="font-size:10px;font-weight:600;color:#8ca3af;letter-spacing:0.8px;
                  text-transform:uppercase;margin-bottom:8px;">
        Classification Legend
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <div style="width:13px;height:13px;background:#4a7c59;opacity:0.9;
                    border-radius:2px;border:1px solid #3c6350;"></div>
        <span style="font-size:12px;color:#e2e8f0;font-weight:450;">Green Canopy / Vegetation</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:13px;height:13px;background:#475569;opacity:0.9;
                    border-radius:2px;border:1px solid #334155;"></div>
        <span style="font-size:12px;color:#e2e8f0;font-weight:450;">Impervious / Urban Surface</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ---------------------------------------------------------------------------
# KPI ribbon
# ---------------------------------------------------------------------------

def render_kpi_ribbon(n_total, n_green, n_nongreen, coverage_pct, green_area_km2, acc):
    """Renders five KPI cards vertically inside the sidebar."""
    kpis = [
        ("SAMPLED TILES",      str(n_total),              "256 x 256 px chips",          "Sentinel-2 10m",    "#cbd5e1", "rgba(255,255,255,0.05)"),
        ("GREEN CANOPIES",     str(n_green),               "Detected vegetation tiles",    "NDVI > 0.30",       "#8fae9b", "rgba(74,124,89,0.2)"),
        ("IMPERVIOUS / URBAN", str(n_nongreen),            "Built-up & bare soil tiles",   "Non-vegetated",     "#94a3b8", "rgba(255,255,255,0.05)"),
        ("GREEN COVER RATIO",  f"{coverage_pct:.1f}%",    "of sampled AOI area",          f"~{green_area_km2:.0f} km2", "#b8977e", "rgba(184,151,126,0.15)"),
        ("MODEL ACCURACY",     f"{acc:.1f}%",              "ResNet-18 best checkpoint",    "Val epoch 9",       "#a3b18a", "rgba(255,255,255,0.05)"),
    ]
    for label, value, sub, badge, accent, badge_bg in kpis:
        st.markdown(f"""
        <div class="sidebar-kpi-card" style="border-left: 2px solid {accent};">
          <div class="sidebar-kpi-label">{label}</div>
          <div class="sidebar-kpi-value" style="color:{accent};">{value}</div>
          <div class="sidebar-kpi-sub">{sub}</div>
          <span class="sidebar-kpi-badge" style="background:{badge_bg};color:{accent};">{badge}</span>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(records):
    with st.sidebar:
        # Back to landing page button
        st.markdown("""
        <a href="http://localhost:3000" target="_self" class="back-landing-btn" id="sidebar-back-landing" title="Return to Landing Page">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="19" y1="12" x2="5" y2="12"></line>
            <polyline points="12 19 5 12 12 5"></polyline>
          </svg>
          <span>&larr; Back to Landing Page</span>
        </a>
        """, unsafe_allow_html=True)

        # Status pill
        st.markdown("""
        <div class="status-pill">
          <div class="status-dot"></div>
          Model Active &nbsp;&bull;&nbsp; ResNet-18 Ensemble
        </div>
        """, unsafe_allow_html=True)

        # Layer controls
        st.markdown("""
        <div class="sidebar-section-title">Layer Controls</div>
        """, unsafe_allow_html=True)
        with st.container():
            show_preds = st.toggle("Prediction overlay", value=True)
            show_ndvi  = st.toggle("NDVI vegetation heatmap", value=False)

        # Heatmap legend — shown right below the NDVI toggle (Matte swatches)
        st.markdown("""
        <div class="heatmap-legend">
          <div class="heatmap-legend-title">NDVI Heatmap Legend</div>
          <div class="heatmap-legend-row">
            <span class="heatmap-swatch" style="background:#4a7c59;"></span>
            <div>
              <div class="heatmap-legend-label">Dense Canopy</div>
              <div class="heatmap-legend-desc">Healthy mature trees, dense parks, thick forest cover</div>
            </div>
          </div>
          <div class="heatmap-legend-row">
            <span class="heatmap-swatch" style="background:#8fae9b;"></span>
            <div>
              <div class="heatmap-legend-label">Sparse Vegetation</div>
              <div class="heatmap-legend-desc">Lawns, shrubland, agricultural patches</div>
            </div>
          </div>
          <div class="heatmap-legend-row">
            <span class="heatmap-swatch" style="background:#64748b;"></span>
            <div>
              <div class="heatmap-legend-label">Urban / Impervious</div>
              <div class="heatmap-legend-desc">Roads, concrete rooftops, bare soil</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence filter
        st.markdown("""
        <div class="sidebar-section-title">Filter by Confidence</div>
        """, unsafe_allow_html=True)
        min_conf = st.slider(
            "Minimum prediction confidence",
            min_value=0.5,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            label_visibility="collapsed",
        )
        st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-section-title">Dataset KPIs</div>
        """, unsafe_allow_html=True)

    return show_preds, show_ndvi, min_conf


# ---------------------------------------------------------------------------
# Tab: Map
# ---------------------------------------------------------------------------

def render_map_tab(records, show_preds, show_ndvi, min_conf):
    st.markdown('<div class="section-header">Interactive Prediction Map</div>',
                unsafe_allow_html=True)
    st.caption(
        "Coloured rectangles show model predictions over Bengaluru. "
        "Click any tile for detailed classification info. "
        "Toggle layers via the control in the top-right of the map."
    )
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    fmap = build_map(records, show_ndvi=show_ndvi, show_preds=show_preds, min_conf=min_conf)
    st_folium(fmap, width=None, height=640, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab: Tile Explorer
# ---------------------------------------------------------------------------

def render_tiles_tab(records):
    st.markdown('<div class="section-header">Tile Explorer</div>',
                unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([2, 2, 4])
    with col_a:
        class_filter = st.selectbox("Class filter", ["All", "green", "non_green"], label_visibility="collapsed")
    with col_b:
        sort_by = st.selectbox("Sort by", ["Default", "Confidence (high)", "Confidence (low)", "NDVI (high)"], label_visibility="collapsed")
    with col_c:
        st.caption("Filter and sort the chip previews. Green border = green space, grey = urban.")

    filtered = records if class_filter == "All" else [r for r in records if r["pred_label"] == class_filter]

    if sort_by == "Confidence (high)":
        filtered = sorted(filtered, key=lambda r: float(r["confidence"]), reverse=True)
    elif sort_by == "Confidence (low)":
        filtered = sorted(filtered, key=lambda r: float(r["confidence"]))
    elif sort_by == "NDVI (high)":
        filtered = sorted(filtered, key=lambda r: float(r.get("mean_ndvi") or 0), reverse=True)

    st.markdown(
        f"<p style='font-size:12px;color:#94a3b8;margin-bottom:12px'>"
        f"Showing <b>{len(filtered)}</b> of <b>{len(records)}</b> tiles</p>",
        unsafe_allow_html=True,
    )

    cols_per_row = 7
    for row_start in range(0, len(filtered), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, rec in enumerate(filtered[row_start:row_start + cols_per_row]):
            with cols[i]:
                img = load_preview(rec["png_path"], size=(160, 160))
                st.image(img, use_container_width=True)
                is_green = rec["pred_label"] == "green"
                label_cls = "tile-label-green" if is_green else "tile-label-nongreen"
                ndvi_val = float(rec.get("mean_ndvi") or 0)
                st.markdown(
                    f"<div class='tile-card'>"
                    f"<div class='{label_cls}'>"
                    f"{'Green' if is_green else 'Urban'}"
                    f"</div>"
                    f"<div class='tile-conf'>{float(rec['confidence'])*100:.0f}% &bull; NDVI {ndvi_val:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Tab: Model Report
# ---------------------------------------------------------------------------

def render_eval_tab(records):
    st.markdown('<div class="section-header">Model Performance</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(REPORT_PNG):
            st.image(REPORT_PNG, caption="Evaluation Report — Confusion Matrix & Metrics",
                     use_container_width=True)
        else:
            st.info("Run `05_evaluate.py` to generate the evaluation report.")
    with col2:
        if os.path.exists(CURVE_PNG):
            st.image(CURVE_PNG, caption="Training & Validation Curves",
                     use_container_width=True)
        else:
            st.info("Run `04_train.py` to generate training curves.")

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Predictions Table</div>',
                unsafe_allow_html=True)

    table_data = [{
        "Tile ID":      r["tile_id"],
        "True Label":   r.get("true_label", ""),
        "Predicted":    r["pred_label"].replace("_", " ").title(),
        "Confidence":   f"{float(r['confidence'])*100:.1f}%",
        "Mean NDVI":    f"{float(r.get('mean_ndvi') or 0):.3f}",
        "Correct":      "Yes" if r.get("correct") == "True" else "No",
    } for r in records]

    st.dataframe(table_data, use_container_width=True, height=380)


# ---------------------------------------------------------------------------
# Tab: About
# ---------------------------------------------------------------------------

def render_about_tab():
    st.markdown('<div class="section-header">About This Platform</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="about-card">
          <h4>Data Source</h4>
          <ul>
            <li>Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED)</li>
            <li>Temporal composite: February - April 2024</li>
            <li>Cloud filter: &lt; 10% CLOUDY_PIXEL_PERCENTAGE</li>
            <li>Bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 + NDVI</li>
            <li>Native resolution: 10 m/pixel</li>
          </ul>
        </div>
        <div class="about-card">
          <h4>Processing Pipeline</h4>
          <ol>
            <li><code>01_gee_extractor.py</code> - GEE AOI export</li>
            <li><code>02_tile_generator.py</code> - 256x256 px chip generation</li>
            <li><code>03_label_generator.py</code> - NDVI threshold auto-labelling</li>
            <li><code>04_train.py</code> - ResNet-18 fine-tuning</li>
            <li><code>05_evaluate.py</code> - Metrics and confusion matrix</li>
            <li><code>app/app.py</code> - This dashboard</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
          <h4>Model Architecture</h4>
          <ul>
            <li>Base: ResNet-18 (ImageNet pretrained)</li>
            <li>Input adaptation: 3-channel to 11-channel conv layer</li>
            <li>Loss: Weighted cross-entropy (6:1 imbalance correction)</li>
            <li>Optimiser: AdamW + Cosine Annealing LR</li>
            <li>Best validation accuracy: <strong>97.92%</strong> (epoch 9)</li>
            <li>Overall evaluation accuracy: <strong>97.63%</strong></li>
          </ul>
        </div>
        <div class="about-card">
          <h4>Area of Interest</h4>
          <ul>
            <li>City: Bengaluru, Karnataka, India</li>
            <li>Bounding box: lon [77.45, 77.75], lat [12.83, 13.14]</li>
            <li>Total chips: 169 (all 100% valid pixels)</li>
            <li>Green space chips: 24 (14.2%)</li>
            <li>Urban/non-green chips: 145 (85.8%)</li>
          </ul>
        </div>
        <div class="about-card">
          <h4>Navigation</h4>
          <p style="margin-bottom: 10px; font-size: 12.5px; color: #94a3b8;">
            Return to the high-level project presentation landing page and interactive system overview.
          </p>
          <a href="http://localhost:3000" target="_self" class="back-landing-btn" style="width: auto; padding: 7px 16px; margin-bottom: 0;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="19" y1="12" x2="5" y2="12"></line>
              <polyline points="12 19 5 12 12 5"></polyline>
            </svg>
            <span>&larr; Back to Project Landing Page</span>
          </a>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Bengaluru Urban Green Space Analytics",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Load data
    with st.spinner("Initialising platform..."):
        _model, _stats = load_model()
        records        = load_predictions()

    # Compute KPIs
    n_total       = len(records)
    n_green       = sum(1 for r in records if r["pred_label"] == "green")
    n_nongreen    = n_total - n_green
    n_correct     = sum(1 for r in records if r.get("correct") == "True")
    acc           = n_correct / max(n_total, 1) * 100
    tile_area_km2 = (256 * 0.01) ** 2
    green_area    = n_green * tile_area_km2
    total_area    = n_total * tile_area_km2
    coverage_pct  = green_area / max(total_area, 1) * 100

    # Custom header with Back to Landing Page button
    st.markdown("""
    <div class="ci-header-bar">
      <div>
        <div class="ci-header-wrap">
          <div class="ci-accent-dot"></div>
          <h1 class="ci-header-title">Bengaluru Canopy Intelligence</h1>
        </div>
        <p class="ci-header-subtitle">
          High-resolution semantic segmentation benchmarked across
          Sentinel-2 Level-2A surface reflectance &mdash; Greater Bengaluru AOI.
        </p>
      </div>
      <a href="http://localhost:3000" target="_self" class="back-landing-btn back-landing-btn-header" id="header-back-landing" title="Return to Project Landing Page">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="19" y1="12" x2="5" y2="12"></line>
          <polyline points="12 19 5 12 12 5"></polyline>
        </svg>
        <span>&larr; Back to Landing Page</span>
      </a>
    </div>
    """, unsafe_allow_html=True)

    # Executive banner
    st.markdown("""
    <div class="top-banner">
      <div class="banner-badge">Urban Analytics Platform &bull; Bengaluru</div>
      <div class="banner-title">Urban Green Space Canopy &amp; Vegetation Classifier</div>
      <div class="banner-subtitle">
        High-resolution semantic segmentation benchmarked across
        Sentinel-2 Level-2A surface reflectance. Deep learning pipeline
        trained on multispectral satellite imagery covering Greater Bengaluru.
      </div>
      <div class="banner-meta">
        <span class="banner-pill">Sentinel-2 SR &bull; Feb-Apr 2024</span>
        <span class="banner-pill">10 m Resolution</span>
        <span class="banner-pill">11 Spectral Bands + NDVI</span>
        <span class="banner-pill">ResNet-18 Classifier</span>
        <span class="banner-pill">97.92% Val Accuracy</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar (returns controls) + KPIs rendered inside it
    show_preds, show_ndvi, min_conf = render_sidebar(records)
    with st.sidebar:
        render_kpi_ribbon(n_total, n_green, n_nongreen, coverage_pct, green_area, acc)

        st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

        # Model Info — below KPIs
        st.markdown("""
        <div class="sidebar-section-title">Model Info</div>
        """, unsafe_allow_html=True)
        st.caption("Architecture: ResNet-18 (11-band)")
        st.caption("Input: Sentinel-2 SR + NDVI")
        st.caption("AOI: Bengaluru [77.45-77.75, 12.83-13.14]")
        st.caption("Composite: Feb-Apr 2024 (cloud < 10%)")

        st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

        # Export Options — at the bottom
        st.markdown("""
        <div class="sidebar-section-title">Export Options</div>
        """, unsafe_allow_html=True)
        csv_bytes = records_to_csv_bytes(records)
        st.download_button(
            label="Download predictions CSV",
            data=csv_bytes,
            file_name="bengaluru_green_space_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if os.path.exists(REPORT_PNG):
            with open(REPORT_PNG, "rb") as f:
                st.download_button(
                    label="Download evaluation report (PNG)",
                    data=f.read(),
                    file_name="evaluation_report.png",
                    mime="image/png",
                    use_container_width=True,
                )

    # Tabs
    tab_map, tab_tiles, tab_eval, tab_about = st.tabs([
        "Map", "Tile Explorer", "Model Report", "About",
    ])

    with tab_map:
        render_map_tab(records, show_preds, show_ndvi, min_conf)

    with tab_tiles:
        render_tiles_tab(records)

    with tab_eval:
        render_eval_tab(records)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
