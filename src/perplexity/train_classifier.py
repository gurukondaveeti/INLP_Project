"""
================================================================================
MGT Detection - Transformer Encoder Classifier Trainer
================================================================================

Loads pre-extracted .npy feature files from the `features/` directory and
trains a small Transformer Encoder for binary classification (human=0 / machine=1).

Architecture (FeatureClassifier):
  Input  : aggregated feature vector  [batch, feature_dim=12]
  Layer 1: Linear projection  →  hidden_dim (256)
  Layer 2: nn.TransformerEncoder  (2 layers, 4 heads, ff_dim=512)
  Layer 3: Classification head  →  2 logits

Training:
  - AdamW,  lr=1e-4,  weight_decay=0.01
  - CrossEntropyLoss
  - Best checkpoint saved by dev accuracy

Usage:
  python train_classifier.py
  python train_classifier.py --epochs 20 --batch-size 64 --hidden-dim 256
================================================================================
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Defaults (can be overridden via CLI)
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_DIR   = Path("features")
CHECKPOINT_DIR = Path("checkpoints")
PLOT_DIR       = Path("plots")
RESULTS_DIR    = Path("results")
FEATURE_DIM   = 12   # 6 features × 2 models
HIDDEN_DIM    = 256
N_HEADS       = 4
N_LAYERS      = 2
FF_DIM        = 512
DROPOUT       = 0.1
BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 1e-4
WEIGHT_DECAY  = 0.01
EARLY_STOPPING_PATIENCE = 3


# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────
class FeatureClassifier(nn.Module):
    """
    Small Transformer Encoder classifier on top of aggregated LM features.

    The input is a single fixed-size vector per text (not a sequence), so we
    treat it as a sequence of length 1 for the Transformer — this lets us reuse
    nn.TransformerEncoder without modification and easily scale to longer
    context windows in the future.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        n_heads:     int = N_HEADS,
        n_layers:    int = N_LAYERS,
        ff_dim:      int = FF_DIM,
        dropout:     float = DROPOUT,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # 1. Linear projection: feature_dim → hidden_dim
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # 2. Transformer Encoder (lightweight: 2 layers, 4 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # input shape: (batch, seq_len, hidden_dim)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )

        # 3. Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim) float tensor
        Returns:
            logits: (batch, num_classes)
        """
        # Project to hidden dim: (batch, hidden_dim)
        x = self.input_proj(x)

        # Add sequence dimension: (batch, 1, hidden_dim)
        x = x.unsqueeze(1)

        # Transformer Encoder
        x = self.transformer(x)  # (batch, 1, hidden_dim)

        # Remove sequence dim: (batch, hidden_dim)
        x = x.squeeze(1)

        # Classification head
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_split(split: str, feature_dir: Path, prefix: str = "") -> TensorDataset:
    feat_path  = feature_dir / f"{prefix}{split}_features.npy"
    label_path = feature_dir / f"{prefix}{split}_labels.npy"

    # Fallback: if prefixed labels don't exist, try unprefixed labels
    if not label_path.exists() and prefix:
        label_path = feature_dir / f"{split}_labels.npy"

    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feat_path}\n"
            "Run `python extract_features.py` (or `python fuse_features.py` for fused) first."
        )

    features = torch.tensor(np.load(feat_path), dtype=torch.float32)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) # Scrub FP16 Underflows
    labels   = torch.tensor(np.load(label_path), dtype=torch.long)

    log.info(f"[{split}] Loaded features {tuple(features.shape)}, "
             f"labels {tuple(labels.shape)} (prefix='{prefix}')")
    return TensorDataset(features, labels)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """One training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in tqdm(loader, desc="  train", leave=False):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "eval",
) -> tuple[float, float]:
    """Evaluation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for features, labels in tqdm(loader, desc=f"  {desc}", leave=False):
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: list[dict], plot_dir: Path) -> None:
    """Generate and save training & validation loss/accuracy curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.2)
    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss curves ──
    ax1.plot(epochs, [h["train_loss"] for h in history],
             "o-", color="#e74c3c", linewidth=2, markersize=5, label="Train Loss")
    ax1.plot(epochs, [h["dev_loss"] for h in history],
             "s-", color="#3498db", linewidth=2, markersize=5, label="Dev Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # ── Accuracy curves ──
    ax2.plot(epochs, [h["train_acc"] * 100 for h in history],
             "o-", color="#e74c3c", linewidth=2, markersize=5, label="Train Acc")
    ax2.plot(epochs, [h["dev_acc"] * 100 for h in history],
             "s-", color="#3498db", linewidth=2, markersize=5, label="Dev Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=max(0, min(h["dev_acc"] * 100 for h in history) - 5))

    plt.tight_layout()
    path = plot_dir / "training_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved training curves → {path}")


def plot_lr_schedule(lr_history: list[float], plot_dir: Path) -> None:
    """Plot the learning rate schedule over epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(lr_history) + 1), lr_history,
            "-", color="#9b59b6", linewidth=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Cosine Annealing Learning Rate Schedule")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plot_dir / "lr_schedule.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved LR schedule → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train MGT Transformer classifier")
    parser.add_argument("--feature-dir",  default=str(FEATURE_DIR))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--feature-dim",  type=int, default=FEATURE_DIM)
    parser.add_argument("--hidden-dim",   type=int, default=HIDDEN_DIM)
    parser.add_argument("--n-heads",      type=int, default=N_HEADS)
    parser.add_argument("--n-layers",     type=int, default=N_LAYERS)
    parser.add_argument("--ff-dim",       type=int, default=FF_DIM)
    parser.add_argument("--dropout",      type=float, default=DROPOUT)
    parser.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs",       type=int, default=EPOCHS)
    parser.add_argument("--patience",     type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--feature-prefix", default="",
                        help="Prefix for feature files, e.g. 'fused_' to load fused_train_features.npy")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_dir    = Path(args.feature_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    plot_dir       = PLOT_DIR
    results_dir    = RESULTS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    prefix = args.feature_prefix
    train_ds = load_split("train", feature_dir, prefix=prefix)
    dev_ds   = load_split("dev",   feature_dir, prefix=prefix)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Derive actual feature dim from the loaded data
    actual_feature_dim = train_ds[0][0].shape[0]
    if actual_feature_dim != args.feature_dim:
        log.warning(
            f"Feature dim mismatch: expected {args.feature_dim}, "
            f"got {actual_feature_dim}. Using {actual_feature_dim}."
        )
        args.feature_dim = actual_feature_dim

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FeatureClassifier(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Classifier parameters: {n_params:,}")

    # ── Optimiser & loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine LR schedule — warms up for 10% of training, decays to 1e-6
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training ──────────────────────────────────────────────────────────────
    best_dev_loss = float('inf')
    best_dev_acc  = 0.0
    best_ckpt     = checkpoint_dir / "best_classifier.pt"
    history       = []
    lr_history    = []
    patience_counter = 0

    log.info(f"Starting training for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        # Record LR at the start of each epoch
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()
        dev_loss, dev_acc = evaluate(
            model, dev_loader, criterion, device, desc="dev"
        )

        marker = ""
        # We track best model by dev_acc (user's original logic), 
        # but apply early stopping based on dev_loss as requested.
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dev_loss": dev_loss,
                    "dev_acc": dev_acc,
                    "args": vars(args),
                },
                best_ckpt,
            )
            marker = " ← best acc"

        # Early stopping logic (monitor validation loss)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            if "best acc" not in marker:
                marker = " ← best loss"
        else:
            patience_counter += 1

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "dev_loss": dev_loss, "dev_acc": dev_acc, "lr": current_lr}
        )
        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"dev loss {dev_loss:.4f}  acc {dev_acc:.4f}{marker}"
        )

        if patience_counter >= args.patience:
            log.info(f"Early stopping triggered! Validation loss hasn't improved for {args.patience} epochs.")
            break

    log.info(f"Training complete. Best dev accuracy: {best_dev_acc:.4f}")
    log.info(f"Best checkpoint saved to: {best_ckpt}")

    # ── Save training history ─────────────────────────────────────────────────
    hist_path = checkpoint_dir / "training_history.json"
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)
    log.info(f"Training history saved to: {hist_path}")

    # ── Generate training plots ───────────────────────────────────────────────
    log.info("Generating training plots …")
    plot_training_curves(history, plot_dir)
    plot_lr_schedule(lr_history, plot_dir)

    # ── Save training summary ─────────────────────────────────────────────────
    summary = {
        "best_dev_accuracy": best_dev_acc,
        "best_epoch": max(history, key=lambda h: h["dev_acc"])["epoch"],
        "total_epochs": args.epochs,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_acc": history[-1]["train_acc"],
        "final_dev_loss": history[-1]["dev_loss"],
        "final_dev_acc": history[-1]["dev_acc"],
        "classifier_params": n_params,
        "hyperparameters": {
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "seed": args.seed,
        },
        "lm_models": ["gpt2-medium", "gpt2-large"],
    }
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
