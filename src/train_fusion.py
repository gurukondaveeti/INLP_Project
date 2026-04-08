"""
train_fusion.py
───────────────
End-to-end training pipeline for the Anti-Overshadowing Multimodal Fusion MLP.

Usage:
    python train_fusion.py                         # train with stylo + perplexity only
    python train_fusion.py --semantic_dir ./sem/    # train with all 3 branches

Author: Fusion Team — SemEval-2024 Task 8A
"""

import argparse
import json
import os
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from fusion_dataset import FusionDataset, fit_scalers, load_and_align
from fusion_model import FusionLoss, FusionMLP

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Train Fusion MLP for MGT Detection")

    # ── Data Paths ─────────────────────────────────────────────
    p.add_argument("--stylo_pkl", type=str, default="../data/features/extracted_features_cache.pkl",
                   help="Path to Afzal's stylometric cache pkl")
    p.add_argument("--train_perp", type=str, default="../data/features/train_perplexity_features.pkl",
                   help="Path to Vinay's train perplexity pkl")
    p.add_argument("--val_perp", type=str, default="../data/features/dev_perplexity_features.pkl",
                   help="Path to Vinay's val/dev perplexity pkl")
    p.add_argument("--test_perp", type=str, default="../data/features/test_perplexity_features.pkl",
                   help="Path to Vinay's test perplexity pkl")
    p.add_argument("--semantic_dir", type=str, default="../data/features",
                   help="Directory containing semantic_{train,val,test}.pkl files")

    # ── Model Architecture ─────────────────────────────────────
    p.add_argument("--stylo_dim", type=int, default=11)
    p.add_argument("--perp_dim", type=int, default=12)
    p.add_argument("--sem_dim", type=int, default=64)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout_rates", type=float, nargs="+", default=[0.4, 0.3, 0.2])
    p.add_argument("--branch_proj_dim", type=int, default=48,
                   help="Projection dim for each branch (equalizes dimensions)")
    p.add_argument("--no_branch_proj", action="store_true",
                   help="Disable branch projection (use raw dims)")

    # ── Anti-Overshadowing ─────────────────────────────────────
    p.add_argument("--perp_dropout", type=float, default=0.40,
                   help="Modality dropout probability for perplexity branch")
    p.add_argument("--perp_grad_scale", type=float, default=0.10,
                   help="Gradient scale for perplexity branch")
    p.add_argument("--perp_gate_init", type=float, default=0.12,
                   help="Initial gate value for perplexity (sigmoid output)")
    p.add_argument("--primary_loss_weight", type=float, default=0.70,
                   help="Weight of primary loss vs auxiliary loss")
    p.add_argument("--label_smoothing", type=float, default=0.05)

    # ── Training Hyperparameters ───────────────────────────────
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7,
                   help="Early stopping patience (epochs)")
    p.add_argument("--scheduler", type=str, default="onecycle",
                   choices=["onecycle", "cosine", "none"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # ── Output ─────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="../experiments/fusion_output",
                   help="Directory for saving model, plots, predictions")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════
def train_one_epoch(
    model: FusionMLP,
    loader: DataLoader,
    criterion: FusionLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    total_primary = 0.0
    total_aux = 0.0
    all_preds = []
    all_labels = []

    for stylo, perp, sem, labels in loader:
        stylo = stylo.to(device)
        perp = perp.to(device)
        sem = sem.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        primary_logits, aux_logits = model(stylo, perp, sem)
        loss, loss_p, loss_a = criterion(primary_logits, aux_logits, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        total_primary += loss_p.item() * labels.size(0)
        total_aux += loss_a.item() * labels.size(0)

        preds = (torch.sigmoid(primary_logits) > 0.5).long().squeeze(-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": total_loss / n,
        "loss_primary": total_primary / n,
        "loss_aux": total_aux / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(
    model: FusionMLP,
    loader: DataLoader,
    criterion: FusionLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_primary = 0.0
    total_aux = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    for stylo, perp, sem, labels in loader:
        stylo = stylo.to(device)
        perp = perp.to(device)
        sem = sem.to(device)
        labels = labels.to(device)

        primary_logits, aux_logits = model(stylo, perp, sem)
        loss, loss_p, loss_a = criterion(primary_logits, aux_logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_primary += loss_p.item() * labels.size(0)
        total_aux += loss_a.item() * labels.size(0)

        probs = torch.sigmoid(primary_logits).squeeze(-1)
        preds = (probs > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return {
        "loss": total_loss / n,
        "loss_primary": total_primary / n,
        "loss_aux": total_aux / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
    }


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════
def plot_training_history(history: dict, output_dir: str):
    """Generate and save training plots."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── 1. Loss Curves ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0].set_title("Total Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_loss_primary"], "b--", label="Train Primary")
    axes[1].plot(epochs, history["val_loss_primary"], "r--", label="Val Primary")
    axes[1].plot(epochs, history["train_loss_aux"], "b:", label="Train Aux")
    axes[1].plot(epochs, history["val_loss_aux"], "r:", label="Val Aux")
    axes[1].set_title("Primary vs Auxiliary Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_f1"], "b-", label="Train F1", linewidth=2)
    axes[2].plot(epochs, history["val_f1"], "r-", label="Val F1", linewidth=2)
    axes[2].plot(epochs, history["train_acc"], "b--", label="Train Acc", alpha=0.6)
    axes[2].plot(epochs, history["val_acc"], "r--", label="Val Acc", alpha=0.6)
    axes[2].set_title("F1 & Accuracy", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved training_curves.png")

    # ── 2. Gate Values Over Time ───────────────────────────────
    if "gate_stylo" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["gate_stylo"], "g-", label="Stylo Gate", linewidth=2)
        ax.plot(epochs, history["gate_perp"], "r-", label="Perplexity Gate", linewidth=2)
        ax.plot(epochs, history["gate_sem"], "b-", label="Semantic Gate", linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Fair share")
        ax.set_title("Learnable Gate Values (Anti-Overshadowing)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gate Activation (0–1)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gate_values.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Saved gate_values.png")

    # ── 3. Learning Rate Schedule ──────────────────────────────
    if "lr" in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history["lr"], "k-", linewidth=1)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lr_schedule.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Saved lr_schedule.png")


def plot_confusion_matrix(labels, preds, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Human", "Machine"],
        yticklabels=["Human", "Machine"],
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    # Write counts in cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved confusion_matrix.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # ── Reproducibility ────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MULTIMODAL FUSION MLP — MGT DETECTION")
    print(f"  Device: {device}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resolve semantic paths ─────────────────────────────────
    sem_train_path = sem_val_path = sem_test_path = None
    if args.semantic_dir is not None:
        sem_dir = Path(args.semantic_dir)
        sem_train_path = str(sem_dir / "semantic_train.pkl")
        # Handle both "val" and "validation" naming conventions
        if (sem_dir / "semantic_val.pkl").exists():
            sem_val_path = str(sem_dir / "semantic_val.pkl")
        else:
            sem_val_path = str(sem_dir / "semantic_validation.pkl")
        sem_test_path = str(sem_dir / "semantic_test.pkl")
        if not Path(sem_train_path).exists():
            print("WARNING: Semantic train file not found: " + sem_train_path)
            print("    Falling back to 2-branch mode (stylo + perplexity)\n")
            sem_train_path = sem_val_path = sem_test_path = None

    # ══════════════════════════════════════════════════════════
    # PHASE 1: DATA LOADING
    # ══════════════════════════════════════════════════════════
    print("📂 Phase 1: Loading and aligning data...")

    # Load train (unscaled)
    _, raw_stylo_train, raw_perp_train, raw_sem_train = load_and_align(
        "train", args.stylo_pkl, args.train_perp, sem_train_path
    )

    # Fit scalers on training data
    print("\n  🔧 Fitting per-branch StandardScalers on training data...")
    stylo_scaler, perp_scaler, sem_scaler = fit_scalers(
        raw_stylo_train, raw_perp_train, raw_sem_train
    )

    # Reload all splits with fitted scalers
    train_ds, _, _, _ = load_and_align(
        "train", args.stylo_pkl, args.train_perp, sem_train_path,
        stylo_scaler, perp_scaler, sem_scaler,
    )
    val_ds, _, _, _ = load_and_align(
        "val", args.stylo_pkl, args.val_perp, sem_val_path,
        stylo_scaler, perp_scaler, sem_scaler,
    )
    test_ds, _, _, _ = load_and_align(
        "test", args.stylo_pkl, args.test_perp, sem_test_path,
        stylo_scaler, perp_scaler, sem_scaler,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"\n  ✅ Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  ✅ Dims — Stylo: {train_ds.stylo_dim}, Perp: {train_ds.perp_dim}, "
          f"Sem: {train_ds.sem_dim}")

    # ── Class balance ──────────────────────────────────────────
    train_labels = train_ds.labels
    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"  ✅ Class balance — Neg(human): {n_neg:,}  Pos(machine): {n_pos:,}")
    print(f"     pos_weight = {pos_weight.item():.4f}")

    # Detect effective semantic dim
    effective_sem_dim = train_ds.sem_dim

    # ══════════════════════════════════════════════════════════
    # PHASE 2: MODEL CONSTRUCTION
    # ══════════════════════════════════════════════════════════
    print(f"\n🏗️  Phase 2: Building model...")

    model = FusionMLP(
        stylo_dim=train_ds.stylo_dim,
        perp_dim=train_ds.perp_dim,
        sem_dim=effective_sem_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout_rates=tuple(args.dropout_rates),
        perp_modality_dropout=args.perp_dropout,
        perp_grad_scale=args.perp_grad_scale,
        perp_gate_init=args.perp_gate_init,
        use_branch_projection=not args.no_branch_proj,
        branch_proj_dim=args.branch_proj_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Gate values:      {model.get_gate_values()}")

    criterion = FusionLoss(
        primary_weight=args.primary_loss_weight,
        label_smoothing=args.label_smoothing,
        pos_weight=pos_weight,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    if args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy="cos",
        )
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    else:
        scheduler = None

    # ══════════════════════════════════════════════════════════
    # PHASE 3: TRAINING
    # ══════════════════════════════════════════════════════════
    print(f"\n🚀 Phase 3: Training for up to {args.epochs} epochs...")
    print(f"   Batch size: {args.batch_size}  |  LR: {args.lr}  |  WD: {args.weight_decay}")
    print(f"   Scheduler: {args.scheduler}  |  Patience: {args.patience}")
    print(f"   Anti-overshadowing: perp_dropout={args.perp_dropout}, "
          f"perp_grad_scale={args.perp_grad_scale}, perp_gate_init={args.perp_gate_init}")
    print(f"   Loss weights: primary={args.primary_loss_weight}, "
          f"aux={1-args.primary_loss_weight}")
    print()

    history = {
        "train_loss": [], "val_loss": [],
        "train_loss_primary": [], "val_loss_primary": [],
        "train_loss_aux": [], "val_loss_aux": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "gate_stylo": [], "gate_perp": [], "gate_sem": [],
        "lr": [],
    }

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler if args.scheduler == "onecycle" else None,
            device,
        )
        if args.scheduler == "cosine":
            scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Gate values
        gates = model.get_gate_values()

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_loss_primary"].append(train_metrics["loss_primary"])
        history["val_loss_primary"].append(val_metrics["loss_primary"])
        history["train_loss_aux"].append(train_metrics["loss_aux"])
        history["val_loss_aux"].append(val_metrics["loss_aux"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["gate_stylo"].append(gates["gate_stylo"])
        history["gate_perp"].append(gates["gate_perp"])
        history["gate_sem"].append(gates["gate_sem"])
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start

        # Print progress
        print(
            f"  Epoch {epoch:02d}/{args.epochs} │ "
            f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} │ "
            f"F1: {train_metrics['f1']:.4f}/{val_metrics['f1']:.4f} │ "
            f"Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} │ "
            f"Gates: S={gates['gate_stylo']:.3f} P={gates['gate_perp']:.3f} "
            f"E={gates['gate_sem']:.3f} │ "
            f"LR: {current_lr:.2e} │ "
            f"{elapsed:.1f}s"
        )

        # Early stopping check
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_acc": val_metrics["accuracy"],
                "gates": gates,
                "args": vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "best_fusion_model.pt"))
            print(f"    ✨ New best! Val F1 = {best_val_f1:.4f}  (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    print(f"\n{'─'*60}")
    print(f"  Best epoch: {best_epoch}  |  Best Val F1: {best_val_f1:.4f}")
    print(f"{'─'*60}")

    # ══════════════════════════════════════════════════════════
    # PHASE 4: EVALUATION
    # ══════════════════════════════════════════════════════════
    print(f"\n📊 Phase 4: Final evaluation on test set...")

    # Load best model
    ckpt = torch.load(os.path.join(args.output_dir, "best_fusion_model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║    FINAL TEST RESULTS                ║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  Accuracy:  {test_metrics['accuracy']:.4f}                  ║")
    print(f"  ║  F1 Score:  {test_metrics['f1']:.4f}                  ║")
    print(f"  ║  Precision: {test_metrics['precision']:.4f}                  ║")
    print(f"  ║  Recall:    {test_metrics['recall']:.4f}                  ║")
    print(f"  ╚══════════════════════════════════════╝")

    # Classification report
    report = classification_report(
        test_metrics["labels"], test_metrics["predictions"],
        target_names=["Human", "Machine"], digits=4,
    )
    print(f"\n{report}")

    # Save classification report
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(f"SemEval-2024 Task 8A — MGT Detection — Fusion Model\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best epoch: {best_epoch}  |  Best Val F1: {best_val_f1:.4f}\n")
        f.write(f"{'='*60}\n\n")
        f.write(report)
        f.write(f"\nGate values at best epoch: {ckpt['gates']}\n")

    # ══════════════════════════════════════════════════════════
    # PHASE 5: SAVE OUTPUTS
    # ══════════════════════════════════════════════════════════
    print(f"\n💾 Phase 5: Saving outputs to {args.output_dir}/...")

    # Plots
    plot_training_history(history, args.output_dir)
    plot_confusion_matrix(test_metrics["labels"], test_metrics["predictions"], args.output_dir)

    # Training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  💾 Saved training_history.json")

    # Test predictions
    preds_df = {
        "id": list(range(len(test_metrics["predictions"]))),
        "prediction": test_metrics["predictions"].tolist(),
        "probability": [round(p, 5) for p in test_metrics["probabilities"].tolist()],
        "label": test_metrics["labels"].tolist(),
    }
    with open(os.path.join(args.output_dir, "test_predictions.json"), "w") as f:
        json.dump(preds_df, f, indent=2)
    print(f"  💾 Saved test_predictions.json")

    # Save scalers for inference
    scalers = {
        "stylo_scaler": stylo_scaler,
        "perp_scaler": perp_scaler,
        "sem_scaler": sem_scaler,
    }
    with open(os.path.join(args.output_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    print(f"  💾 Saved scalers.pkl")

    # Config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  💾 Saved config.json")

    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"  📁 All outputs saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
