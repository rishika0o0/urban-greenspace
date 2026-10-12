"""
05_evaluate.py
--------------
Evaluates the best trained model on the full labelled dataset and produces:
  - Classification report  (precision, recall, F1 per class)
  - Confusion matrix plot
  - Per-tile prediction grid (green=correct, red=wrong)
  - Saves models/evaluation_report.png and models/predictions.csv

Usage:
  python 05_evaluate.py
"""

import os
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INDEX_CSV    = os.path.join("data", "chip_index.csv")
BEST_MODEL   = os.path.join("models", "best_model.pth")
NORM_STATS   = os.path.join("models", "norm_stats.npy")
REPORT_PNG   = os.path.join("models", "evaluation_report.png")
PRED_CSV     = os.path.join("models", "predictions.csv")
PRED_GRID    = os.path.join("models", "prediction_grid.png")

NUM_BANDS    = 11
NUM_CLASSES  = 2
CLASSES      = ["non_green", "green"]
LABEL_MAP    = {c: i for i, c in enumerate(CLASSES)}
BORDER_PX    = 8

CORRECT_COLOR = (34, 197, 94)    # green
WRONG_COLOR   = (239, 68,  68)   # red


# ---------------------------------------------------------------------------
# Model (same architecture as training)
# ---------------------------------------------------------------------------

def build_model(num_bands=NUM_BANDS, num_classes=NUM_CLASSES):
    model    = models.resnet18(weights=None)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        num_bands, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    model.conv1 = new_conv
    model.fc    = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_chip(npy_path, mean, std):
    chip = np.load(npy_path).astype(np.float32)
    chip = np.nan_to_num(chip, nan=0.0)
    chip = (chip - mean[:, None, None]) / std[:, None, None]
    return torch.tensor(chip, dtype=torch.float32).unsqueeze(0)   # 1 x 11 x 256 x 256


def add_border(img, color, px):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(px):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    return img


def plot_confusion_matrix(ax, cm, class_names):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")


def compute_metrics(y_true, y_pred, classes):
    rows = []
    for i, cls in enumerate(classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == i and p == i)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != i and p == i)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == i and p != i)
        prec  = tp / max(tp + fp, 1)
        rec   = tp / max(tp + fn, 1)
        f1    = 2 * prec * rec / max(prec + rec, 1e-9)
        sup   = sum(1 for t in y_true if t == i)
        rows.append((cls, prec, rec, f1, sup))
    return rows


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------

def build_prediction_grid(records, thumb_size=96):
    """
    Tile all preview PNGs into a single image.
    Green border = correct prediction, Red border = wrong.
    """
    n        = len(records)
    cols_cnt = min(16, n)
    rows_cnt = math.ceil(n / cols_cnt)
    grid_w   = cols_cnt * thumb_size
    grid_h   = rows_cnt * thumb_size
    grid     = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))

    for idx, rec in enumerate(records):
        row = idx // cols_cnt
        col = idx % cols_cnt
        try:
            img = Image.open(rec["png_path"]).convert("RGB").resize((thumb_size, thumb_size))
        except Exception:
            img = Image.new("RGB", (thumb_size, thumb_size), (50, 50, 50))

        correct = rec["true_label"] == rec["pred_label"]
        color   = CORRECT_COLOR if correct else WRONG_COLOR
        img     = add_border(img, color, 4)
        grid.paste(img, (col * thumb_size, row * thumb_size))

    return grid


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # -- Load model ----------------------------------------------------------
    checkpoint = torch.load(BEST_MODEL, map_location=device)
    model      = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded best model (saved at epoch {checkpoint.get('epoch','?')}, "
          f"val_acc={checkpoint.get('val_acc', 0)*100:.2f}%)")

    # -- Load normalisation stats --------------------------------------------
    stats = np.load(NORM_STATS, allow_pickle=True).item()
    mean  = stats["mean"]
    std   = stats["std"]

    # -- Load all labelled records -------------------------------------------
    with open(INDEX_CSV, newline="") as f:
        all_records = [r for r in csv.DictReader(f) if r["label"] in LABEL_MAP]

    print(f"[INFO] Evaluating {len(all_records)} chips...")

    y_true, y_pred, confidences = [], [], []
    pred_records = []

    with torch.no_grad():
        for rec in all_records:
            x        = load_chip(rec["npy_path"], mean, std).to(device)
            logits   = model(x)                               # 1 x 2
            probs    = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            conf     = probs[pred_idx].item()

            true_idx = LABEL_MAP[rec["label"]]
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            confidences.append(conf)

            pred_records.append({
                **rec,
                "true_label" : rec["label"],
                "pred_label" : CLASSES[pred_idx],
                "confidence" : round(conf, 4),
                "correct"    : str(true_idx == pred_idx),
            })

    # -- Overall accuracy ----------------------------------------------------
    overall_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"\n[RESULT] Overall accuracy : {overall_acc*100:.2f}%")

    # -- Confusion matrix ----------------------------------------------------
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # -- Per-class metrics ---------------------------------------------------
    metrics = compute_metrics(y_true, y_pred, CLASSES)
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("-" * 50)
    for cls, prec, rec, f1, sup in metrics:
        print(f"{cls:<12} {prec*100:>9.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}% {sup:>9}")

    # -- Save predictions CSV ------------------------------------------------
    fieldnames = list(pred_records[0].keys())
    with open(PRED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_records)
    print(f"\n[INFO] Predictions saved : {PRED_CSV}")

    # -- Prediction grid image -----------------------------------------------
    grid_img = build_prediction_grid(pred_records)
    grid_img.save(PRED_GRID)
    print(f"[INFO] Prediction grid   : {PRED_GRID}")

    # -- Evaluation report figure --------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Model Evaluation  |  Bengaluru Green Space Classifier",
                 fontsize=14, fontweight="bold", y=0.98)

    # Confusion matrix
    ax1 = fig.add_subplot(2, 2, 1)
    plot_confusion_matrix(ax1, cm, CLASSES)

    # Per-class bar chart
    ax2 = fig.add_subplot(2, 2, 2)
    cls_names  = [m[0] for m in metrics]
    precisions = [m[1] * 100 for m in metrics]
    recalls    = [m[2] * 100 for m in metrics]
    f1s        = [m[3] * 100 for m in metrics]
    x          = np.arange(len(cls_names))
    width      = 0.25
    ax2.bar(x - width, precisions, width, label="Precision", color="#3b82f6")
    ax2.bar(x,         recalls,    width, label="Recall",    color="#10b981")
    ax2.bar(x + width, f1s,        width, label="F1",        color="#f59e0b")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(cls_names)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Per-class Metrics", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in ax2.patches:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1,
                     f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    # Confidence distribution
    ax3 = fig.add_subplot(2, 2, 3)
    correct_conf   = [c for c, t, p in zip(confidences, y_true, y_pred) if t == p]
    incorrect_conf = [c for c, t, p in zip(confidences, y_true, y_pred) if t != p]
    ax3.hist(correct_conf,   bins=20, alpha=0.7, label="Correct",   color="#10b981")
    ax3.hist(incorrect_conf, bins=20, alpha=0.7, label="Incorrect", color="#ef4444")
    ax3.set_xlabel("Confidence")
    ax3.set_ylabel("Count")
    ax3.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Summary text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    n_correct   = sum(t == p for t, p in zip(y_true, y_pred))
    n_incorrect = len(y_true) - n_correct
    summary = (
        f"Overall Accuracy   :  {overall_acc*100:.2f}%\n\n"
        f"Total Chips        :  {len(y_true)}\n"
        f"Correct            :  {n_correct}\n"
        f"Incorrect          :  {n_incorrect}\n\n"
        f"Mean Confidence    :  {np.mean(confidences)*100:.2f}%\n\n"
        f"Class Distribution :\n"
        f"  non_green  ->  {sum(1 for t in y_true if t == 0)}\n"
        f"  green      ->  {sum(1 for t in y_true if t == 1)}\n\n"
        f"Model checkpoint   :\n  {os.path.basename(BEST_MODEL)}"
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0fdf4", alpha=0.8))
    ax4.set_title("Summary", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(REPORT_PNG, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Evaluation report : {REPORT_PNG}")
    print(f"\n{'='*50}")
    print(f"  Evaluation complete")
    print(f"  Overall accuracy : {overall_acc*100:.2f}%")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()
