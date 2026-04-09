"""
train_cross_attn.py
────────────────────
Dedicated training script for the Cross-Branch Attention Fusion model.
Uses ONLY Stylometric + Perplexity branches (Semantic disabled — per ablation study findings).

All outputs are saved to:   experiments/architecture_experiments/cross_attention_v1/
Config file:                configs/cross_attention_v1.json

Author: Fusion Team — SemEval-2024 Task 8A
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Make src/ importable ───────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent.parent  # src/
sys.path.insert(0, str(SRC_DIR))

from fusion_dataset import load_and_align, fit_scalers
from architectures.cross_attention_fusion import CrossAttentionFusionMLP


# ══════════════════════════════════════════════════════════════════
# ARG PARSING + CONFIG LOADING
# ══════════════════════════════════════════════════════════════════
def build_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--stylo_train", default="data/features/extracted_features_cache.pkl")
    p.add_argument("--perp_train",  default="data/features/train_perplexity_features.pkl")
    p.add_argument("--stylo_val",   default="data/features/extracted_features_cache.pkl")
    p.add_argument("--perp_val",    default="data/features/dev_perplexity_features.pkl")
    p.add_argument("--stylo_test",  default="data/features/extracted_features_cache.pkl")
    p.add_argument("--perp_test",   default="data/features/test_perplexity_features.pkl")

    # Model hyperparameters
    p.add_argument("--d_model",     type=int,   default=32)
    p.add_argument("--num_heads",   type=int,   default=4)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.25)

    # Training hyperparameters
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-3)
    p.add_argument("--epochs",          type=int,   default=60)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--patience",        type=int,   default=12)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--scheduler",       type=str,   default="onecycle",
                   choices=["onecycle", "cosine", "none"])
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num_workers",     type=int,   default=0)

    # Output
    p.add_argument("--output_dir", default="experiments/architecture_experiments/cross_attention_v1")
    p.add_argument("--config",     default=None)

    args, _ = p.parse_known_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        p.set_defaults(**config)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
# TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []
    for stylo, perp, _, labels in loader:
        stylo, perp, labels = stylo.to(device), perp.to(device), labels.to(device)
        logits = model(stylo, perp)
        loss = criterion(logits.squeeze(-1), labels.float())
        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "preds":     all_preds,
        "probs":     np.array(all_probs),
        "labels":    all_labels,
    }


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, label_smoothing=0.0):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for stylo, perp, _, labels in loader:
        stylo, perp, labels = stylo.to(device), perp.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(stylo, perp)
        # Apply label smoothing manually
        smooth_labels = labels.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        loss = criterion(logits.squeeze(-1), smooth_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits).squeeze(-1) > 0.5).long()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average="macro"),
    }


# ══════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════
def save_training_curves(history, out_dir):
    epochs = range(1, len(history["train_f1"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#3498db")
    axes[0].set_title("Loss over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.4)

    axes[1].plot(epochs, history["train_f1"], label="Train F1", color="#e74c3c")
    axes[1].plot(epochs, history["val_f1"],   label="Val F1",   color="#3498db")
    axes[1].set_title("F1 Score over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close()


def save_confusion_matrix(preds, labels, out_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "Machine"]); ax.set_yticklabels(["Human", "Machine"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Cross-Attention Fusion", fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def save_lr_schedule(lr_history, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lr_history, color="#9b59b6")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "lr_schedule.png", dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print("  CROSS-BRANCH ATTENTION FUSION — MGT DETECTION")
    print(f"  Device: {device}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ── Phase 1: Load Data ─────────────────────────────────────
    print("\n📂 Phase 1: Loading Data...")
    train_ds, stylo_train_np, perp_train_np, _ = load_and_align(
        split="train",
        stylo_pkl_path=args.stylo_train,
        perp_pkl_path=args.perp_train,
        sem_pkl_path=None,
    )

    print("\n  🔥 Fitting scalers on training data...")
    stylo_scaler, perp_scaler, _ = fit_scalers(stylo_train_np, perp_train_np)

    # Reload all splits with fitted scalers
    train_ds, _, _, _ = load_and_align("train", args.stylo_train, args.perp_train, None,
                                       stylo_scaler, perp_scaler)
    val_ds, _, _, _   = load_and_align("val",   args.stylo_val,   args.perp_val,   None,
                                       stylo_scaler, perp_scaler)
    test_ds, _, _, _  = load_and_align("test",  args.stylo_test,  args.perp_test,  None,
                                       stylo_scaler, perp_scaler)

    print(f"\n  ✅ Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  ✅ Dims — Stylo: {train_ds.stylo_dim}, Perp: {train_ds.perp_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

    # ── Phase 2: Build Model ───────────────────────────────────
    print("\n🏗️  Phase 2: Building Cross-Attention Fusion Model...")
    model = CrossAttentionFusionMLP(
        stylo_dim=train_ds.stylo_dim,
        perp_dim=train_ds.perp_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Class-balanced loss
    pos_count = sum(1 for l in train_ds.labels if l == 1)
    neg_count = sum(1 for l in train_ds.labels if l == 0)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    label_smoothing = args.label_smoothing  # applied manually inside train loop

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_loader), epochs=args.epochs,
            pct_start=0.3, anneal_strategy="cos",
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # ── Phase 3: Training ──────────────────────────────────────
    print(f"\n⏳ Phase 3: Training for up to {args.epochs} epochs...")
    print(f"   Batch size: {args.batch_size}  |  LR: {args.lr}  |  WD: {args.weight_decay}")

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    lr_history = []
    best_val_f1 = 0.0
    best_epoch  = 0
    stale_count = 0

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler if args.scheduler == "onecycle" else None,
            device, label_smoothing=label_smoothing
        )
        if args.scheduler == "cosine":
            scheduler.step()

        val_m = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_f1"].append(train_m["f1"])
        history["val_f1"].append(val_m["f1"])
        lr_history.append(optimizer.param_groups[0]["lr"])

        print(f"  Epoch {epoch:02d}/{args.epochs} │ "
              f"Loss: {train_m['loss']:.4f}/{val_m['loss']:.4f} │ "
              f"F1: {train_m['f1']:.4f}/{val_m['f1']:.4f} │ "
              f"Acc: {train_m['accuracy']:.4f}/{val_m['accuracy']:.4f}")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch  = epoch
            stale_count = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1, "args": vars(args),
            }, out_dir / "best_model.pt")
            print(f"    ✨ New best! Val F1 = {best_val_f1:.4f}  (saved)")
        else:
            stale_count += 1
            if stale_count >= args.patience:
                print(f"\n  ⏱️  Early stopping at epoch {epoch}")
                break

    print(f"\n{'─'*60}")
    print(f"  Best epoch: {best_epoch}  |  Best Val F1: {best_val_f1:.4f}")
    print(f"{'─'*60}")

    # ── Phase 4: Test Evaluation ───────────────────────────────
    print("\n📊 Phase 4: Final evaluation on test set...")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_m = evaluate(model, test_loader, criterion, device)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║    FINAL TEST RESULTS                ║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  Accuracy:  {test_m['accuracy']:.4f}                  ║")
    print(f"  ║  F1 Score:  {test_m['f1']:.4f}                  ║")
    print(f"  ║  Precision: {test_m['precision']:.4f}                  ║")
    print(f"  ║  Recall:    {test_m['recall']:.4f}                  ║")
    print(f"  ╚══════════════════════════════════════╝")

    report = classification_report(
        test_m["labels"], test_m["preds"],
        target_names=["Human", "Machine"], digits=4
    )
    print(f"\n{report}")

    # ── Phase 5: Save Outputs ──────────────────────────────────
    print(f"\n💾 Phase 5: Saving outputs to {out_dir}/...")
    save_training_curves(history, out_dir)
    print("  📈 Saved training_curves.png")
    save_confusion_matrix(test_m["preds"], test_m["labels"], out_dir)
    print("  📈 Saved confusion_matrix.png")
    save_lr_schedule(lr_history, out_dir)
    print("  📈 Saved lr_schedule.png")

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Cross-Attention Fusion — SemEval-2024 Task 8A — MGT Detection\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best epoch: {best_epoch}  |  Best Val F1: {best_val_f1:.4f}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy:  {test_m['accuracy']:.4f}\n")
        f.write(f"F1 Score:  {test_m['f1']:.4f}\n")
        f.write(f"Precision: {test_m['precision']:.4f}\n")
        f.write(f"Recall:    {test_m['recall']:.4f}\n\n")
        f.write(report)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Save scalers
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump({"stylo_scaler": stylo_scaler, "perp_scaler": perp_scaler}, f)

    print("\n" + "="*60)
    print("  ✅ TRAINING COMPLETE!")
    print(f"  📁 All outputs saved to: {out_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
