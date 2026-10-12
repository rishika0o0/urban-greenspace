"""
04_train.py
-----------
Trains a ResNet-18 binary classifier on the 256x256 Sentinel-2 chips.

Key design decisions:
  - First conv layer adapted from 3 -> 11 input channels (all S2 bands + NDVI).
  - Weighted cross-entropy loss to handle the green/non_green class imbalance.
  - AdamW optimiser with cosine-annealing LR schedule.
  - Best checkpoint saved to models/best_model.pth (by val accuracy).
  - Training curves saved to models/training_curves.png.

Usage:
  python 04_train.py

Outputs:
  models/best_model.pth        <- best weights by validation accuracy
  models/last_model.pth        <- weights at final epoch
  models/training_curves.png   <- loss and accuracy plots
  models/train_log.csv         <- per-epoch metrics
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INDEX_CSV    = os.path.join("data", "chip_index.csv")
MODEL_DIR    = "models"
BEST_MODEL   = os.path.join(MODEL_DIR, "best_model.pth")
LAST_MODEL   = os.path.join(MODEL_DIR, "last_model.pth")
CURVE_PNG    = os.path.join(MODEL_DIR, "training_curves.png")
LOG_CSV      = os.path.join(MODEL_DIR, "train_log.csv")

NUM_BANDS    = 11       # all S2 bands + NDVI
NUM_CLASSES  = 2        # green, non_green
TILE_SIZE    = 256

BATCH_SIZE   = 16
NUM_EPOCHS   = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.2
SEED         = 42

CLASSES      = ["non_green", "green"]   # index 0 / 1
LABEL_MAP    = {c: i for i, c in enumerate(CLASSES)}

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChipDataset(Dataset):
    """
    Loads pre-tiled .npy chips (shape: 11 x 256 x 256, float32).

    Normalisation: per-band standardisation using dataset-level mean/std
    computed on the training split and stored for inference reuse.
    """

    def __init__(self, records, stats=None):
        """
        records : list of dicts from chip_index.csv
        stats   : dict with 'mean' and 'std' arrays (shape 11,)
                  If None, computed from this split (use for train set).
        """
        self.records = records
        self.stats   = stats

        if self.stats is None:
            self.stats = self._compute_stats()

    def _compute_stats(self):
        """Compute per-band mean and std across the entire split."""
        print("[INFO] Computing per-band normalisation statistics...")
        all_means = []
        all_stds  = []
        for rec in self.records:
            chip = np.load(rec["npy_path"]).astype(np.float32)  # 11 x H x W
            chip = np.nan_to_num(chip, nan=0.0)
            all_means.append(chip.reshape(NUM_BANDS, -1).mean(axis=1))
            all_stds.append(chip.reshape(NUM_BANDS, -1).std(axis=1))
        mean = np.stack(all_means).mean(axis=0)
        std  = np.stack(all_stds).mean(axis=0)
        std  = np.where(std < 1e-6, 1.0, std)   # avoid division by zero
        print(f"[INFO] Stats computed over {len(self.records)} chips.")
        return {"mean": mean, "std": std}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        chip  = np.load(rec["npy_path"]).astype(np.float32)   # 11 x 256 x 256
        chip  = np.nan_to_num(chip, nan=0.0)

        # Per-band standardisation
        mean  = self.stats["mean"][:, None, None]   # 11 x 1 x 1
        std   = self.stats["std"][:, None, None]
        chip  = (chip - mean) / std

        label = LABEL_MAP[rec["label"]]

        return torch.tensor(chip, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model: ResNet-18 with 11-channel input
# ---------------------------------------------------------------------------

def build_model(num_bands=NUM_BANDS, num_classes=NUM_CLASSES):
    """
    Load pretrained ResNet-18 and adapt it for 11-band satellite input.

    The original first conv (3 -> 64) is replaced by (11 -> 64).
    Weights for the first 3 channels are copied from the pretrained layer;
    the remaining 8 channels are initialised with the mean of those 3.
    This 'warm-start' strategy preserves low-level feature detectors.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    old_conv  = model.conv1
    new_conv  = nn.Conv2d(
        num_bands,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        # Copy RGB weights into first 3 channels
        new_conv.weight[:, :3, :, :] = old_conv.weight
        # Initialise extra channels with mean of RGB weights
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)   # 64 x 1 x 7 x 7
        for c in range(3, num_bands):
            new_conv.weight[:, c:c+1, :, :] = mean_weight

    model.conv1 = new_conv
    model.fc    = nn.Linear(model.fc.in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_class_weights(records):
    """Inverse-frequency weights for weighted cross-entropy loss."""
    counts  = [0] * NUM_CLASSES
    for rec in records:
        counts[LABEL_MAP[rec["label"]]] += 1
    total   = sum(counts)
    weights = [total / (NUM_CLASSES * max(c, 1)) for c in counts]
    print(f"[INFO] Class counts : {dict(zip(CLASSES, counts))}")
    print(f"[INFO] Loss weights : {dict(zip(CLASSES, [round(w,3) for w in weights]))}")
    return torch.tensor(weights, dtype=torch.float32)


def make_sampler(records):
    """
    WeightedRandomSampler so each mini-batch sees a balanced mix of classes,
    complementing the weighted loss.
    """
    counts  = [0] * NUM_CLASSES
    for rec in records:
        counts[LABEL_MAP[rec["label"]]] += 1

    class_weight = [1.0 / max(c, 1) for c in counts]
    sample_weights = [class_weight[LABEL_MAP[rec["label"]]] for rec in records]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")

    # -- Load index ----------------------------------------------------------
    with open(INDEX_CSV, newline="") as f:
        all_records = [r for r in csv.DictReader(f) if r["label"] in LABEL_MAP]

    random.shuffle(all_records)
    split = int(len(all_records) * (1 - VAL_SPLIT))
    train_records = all_records[:split]
    val_records   = all_records[split:]
    print(f"[INFO] Train: {len(train_records)} chips  |  Val: {len(val_records)} chips")

    # -- Datasets & loaders --------------------------------------------------
    train_ds = ChipDataset(train_records, stats=None)          # computes stats
    val_ds   = ChipDataset(val_records,   stats=train_ds.stats) # reuse train stats

    # Save normalisation stats alongside the model for inference
    np.save(os.path.join(MODEL_DIR, "norm_stats.npy"), train_ds.stats)

    sampler    = make_sampler(train_records)
    train_dl   = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
    val_dl     = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=0)

    # -- Model, loss, optimiser ----------------------------------------------
    model       = build_model().to(device)
    class_weights = compute_class_weights(train_records).to(device)
    criterion   = nn.CrossEntropyLoss(weight=class_weights)
    optimiser   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

    # -- Epoch loop ----------------------------------------------------------
    best_val_acc = 0.0
    log_rows     = []

    train_losses, val_losses   = [], []
    train_accs,   val_accs     = [], []

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>9}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):

        # -- Train -----------------------------------------------------------
        model.train()
        t_loss, t_acc = 0.0, 0.0
        for chips, labels in train_dl:
            chips, labels = chips.to(device), labels.to(device)
            optimiser.zero_grad()
            out  = model(chips)
            loss = criterion(out, labels)
            loss.backward()
            optimiser.step()
            t_loss += loss.item()
            t_acc  += accuracy(out, labels)

        t_loss /= len(train_dl)
        t_acc  /= len(train_dl)

        # -- Validate --------------------------------------------------------
        model.eval()
        v_loss, v_acc = 0.0, 0.0
        with torch.no_grad():
            for chips, labels in val_dl:
                chips, labels = chips.to(device), labels.to(device)
                out  = model(chips)
                loss = criterion(out, labels)
                v_loss += loss.item()
                v_acc  += accuracy(out, labels)

        v_loss /= len(val_dl)
        v_acc  /= len(val_dl)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:>6}  {t_loss:>11.4f}  {t_acc*100:>9.2f}%  {v_loss:>9.4f}  {v_acc*100:>7.2f}%  {current_lr:>9.6f}")

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        log_rows.append({
            "epoch": epoch, "train_loss": round(t_loss, 6),
            "train_acc": round(t_acc, 6), "val_loss": round(v_loss, 6),
            "val_acc": round(v_acc, 6), "lr": round(current_lr, 8),
        })

        # Save best checkpoint
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": v_acc,
                "classes": CLASSES,
                "num_bands": NUM_BANDS,
            }, BEST_MODEL)
            print(f"         ^ best model saved (val_acc={v_acc*100:.2f}%)")

    # Save final weights
    torch.save({"epoch": NUM_EPOCHS, "model_state_dict": model.state_dict()}, LAST_MODEL)

    # -- Training log --------------------------------------------------------
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc","lr"])
        writer.writeheader()
        writer.writerows(log_rows)

    # -- Training curves plot ------------------------------------------------
    epochs = list(range(1, NUM_EPOCHS + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses,   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train")
    ax2.plot(epochs, [a * 100 for a in val_accs],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle("ResNet-18  |  Bengaluru Green Space Classifier", fontsize=13)
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150)
    plt.close()

    print(f"\n{'='*55}")
    print(f"  Training complete")
    print(f"{'='*55}")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Best model        : {BEST_MODEL}")
    print(f"  Training curves   : {CURVE_PNG}")
    print(f"  Log CSV           : {LOG_CSV}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
