"""
================================================================================
MGT Detection - Inference / Prediction + Report Plots & Analysis
================================================================================

Loads the saved feature files and the best trained classifier checkpoint, then
outputs predictions, accuracy, and generates publication-quality plots for the
project report.

Usage:
  python predict.py
  python predict.py --split test --checkpoint checkpoints/best_classifier.pt
  python predict.py --output predictions.json
================================================================================
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Re-use the model definition from train_classifier
from train_classifier import FeatureClassifier, load_split, evaluate

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PLOT_DIR    = Path("plots")
RESULTS_DIR = Path("results")

FEATURE_NAMES = [
    "mean_log_prob (med)", "std_log_prob (med)",
    "mean_entropy (med)",  "std_entropy (med)",
    "mean_top1_lp (med)",  "std_top1_lp (med)",
    "mean_log_prob (lg)",  "std_log_prob (lg)",
    "mean_entropy (lg)",   "std_entropy (lg)",
    "mean_top1_lp (lg)",   "std_top1_lp (lg)",
]


# ──────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────────────
def _setup_plot_style():
    """Set up consistent publication-quality plot style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2)
    return plt, sns


def plot_confusion_matrix(all_labels, all_preds, plot_dir):
    """Confusion matrix heatmap with counts and percentages."""
    plt, sns = _setup_plot_style()
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    # Create annotation labels with count + percentage
    annot = np.array([
        [f"{cm[i][j]:,}\n({cm_pct[i][j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=["Human", "Machine"],
                yticklabels=["Human", "Machine"],
                linewidths=1.5, linecolor="white",
                cbar_kws={"label": "Count"})
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = plot_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confusion matrix → {path}")


def plot_classification_report(all_labels, all_preds, plot_dir):
    """Render sklearn classification report as a styled table image."""
    plt, sns = _setup_plot_style()
    from sklearn.metrics import classification_report

    report = classification_report(all_labels, all_preds,
                                   target_names=["Human", "Machine"],
                                   output_dict=True)

    # Build a table from the dict
    rows = ["Human", "Machine", "macro avg", "weighted avg"]
    cols = ["precision", "recall", "f1-score", "support"]
    data = []
    for r in rows:
        data.append([report[r][c] for c in cols])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{v:.4f}" if c != "support" else f"{int(v):,}"
                   for c, v in zip(cols, row)]
                  for row in data],
        rowLabels=rows,
        colLabels=["Precision", "Recall", "F1-Score", "Support"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#3498db")
            cell.set_text_props(color="white", fontweight="bold")
        elif j == -1:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#fafafa" if i % 2 == 0 else "white")

    ax.set_title("Classification Report", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path = plot_dir / "classification_report.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved classification report → {path}")

    # Also save text version
    report_text = classification_report(all_labels, all_preds,
                                        target_names=["Human", "Machine"])
    text_path = RESULTS_DIR / "classification_report.txt"
    with open(text_path, "w") as f:
        f.write(report_text)
    log.info(f"Saved classification report text → {text_path}")


def plot_roc_curve(all_labels, all_probs, plot_dir):
    """ROC curve with AUC score."""
    plt, sns = _setup_plot_style()
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e74c3c", linewidth=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#95a5a6", linewidth=1.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plot_dir / "roc_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved ROC curve (AUC={roc_auc:.4f}) → {path}")
    return roc_auc


def plot_precision_recall_curve(all_labels, all_probs, plot_dir):
    """Precision-Recall curve with Average Precision score."""
    plt, sns = _setup_plot_style()
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, 1])
    ap = average_precision_score(all_labels, all_probs[:, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#2ecc71", linewidth=2.5,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.15, color="#2ecc71")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plot_dir / "precision_recall_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved PR curve (AP={ap:.4f}) → {path}")
    return ap


def plot_confidence_distribution(all_labels, all_preds, all_probs, plot_dir):
    """Histogram of prediction confidence for correct vs incorrect predictions."""
    plt, sns = _setup_plot_style()

    max_probs = all_probs.max(axis=1)
    correct_mask = all_preds == all_labels

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(max_probs[correct_mask], bins=50, alpha=0.7, color="#2ecc71",
            label=f"Correct ({correct_mask.sum():,})", density=True, edgecolor="white")
    ax.hist(max_probs[~correct_mask], bins=50, alpha=0.7, color="#e74c3c",
            label=f"Incorrect ({(~correct_mask).sum():,})", density=True, edgecolor="white")
    ax.set_xlabel("Prediction Confidence (max probability)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = plot_dir / "confidence_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confidence distribution → {path}")


def plot_feature_distributions(features, labels, plot_dir):
    """Violin/box plots of all 12 features split by human vs machine."""
    plt, sns = _setup_plot_style()
    import pandas as pd

    n_features = features.shape[1]
    names = FEATURE_NAMES[:n_features] if n_features <= len(FEATURE_NAMES) else \
            [f"feat_{i}" for i in range(n_features)]

    # Build a long-form DataFrame for seaborn
    rows = []
    for i in range(n_features):
        for j in range(len(features)):
            rows.append({
                "Feature": names[i],
                "Value": float(features[j, i]),
                "Class": "Human" if labels[j] == 0 else "Machine"
            })
    df = pd.DataFrame(rows)

    # Plot in a 3x4 grid for 12 features
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_features == 1 else axes.flatten()

    palette = {"Human": "#3498db", "Machine": "#e74c3c"}

    for i, name in enumerate(names):
        ax = axes[i]
        subset = df[df["Feature"] == name]
        sns.violinplot(data=subset, x="Class", y="Value", ax=ax,
                       palette=palette, inner="quartile", cut=0)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Feature Distributions: Human vs Machine", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = plot_dir / "feature_distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved feature distributions → {path}")


def plot_feature_correlation(features, plot_dir):
    """Heatmap of feature correlations."""
    plt, sns = _setup_plot_style()

    n_features = features.shape[1]
    names = FEATURE_NAMES[:n_features] if n_features <= len(FEATURE_NAMES) else \
            [f"feat_{i}" for i in range(n_features)]

    corr = np.corrcoef(features.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1,
                xticklabels=names, yticklabels=names,
                ax=ax, linewidths=0.5,
                cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    path = plot_dir / "feature_correlation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved feature correlation → {path}")


def plot_tsne(features, labels, plot_dir, n_samples=5000):
    """2D t-SNE of the feature space, colored by class."""
    plt, sns = _setup_plot_style()
    from sklearn.manifold import TSNE

    # Subsample for speed if needed
    if len(features) > n_samples:
        idx = np.random.choice(len(features), n_samples, replace=False)
        feats_sub = features[idx]
        labels_sub = labels[idx]
    else:
        feats_sub = features
        labels_sub = labels

    log.info(f"Running t-SNE on {len(feats_sub)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embedding = tsne.fit_transform(feats_sub)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = np.array(["#3498db" if l == 0 else "#e74c3c" for l in labels_sub])
    ax.scatter(embedding[labels_sub == 0, 0], embedding[labels_sub == 0, 1],
               c="#3498db", s=8, alpha=0.5, label="Human", rasterized=True)
    ax.scatter(embedding[labels_sub == 1, 0], embedding[labels_sub == 1, 1],
               c="#e74c3c", s=8, alpha=0.5, label="Machine", rasterized=True)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE Visualization of LM Features", fontsize=14, fontweight="bold")
    ax.legend(markerscale=4, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = plot_dir / "tsne_features.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved t-SNE plot → {path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MGT classifier — predict")
    parser.add_argument("--feature-dir",  default="features")
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_classifier.pt",
        help="Path to a .pt checkpoint saved by train_classifier.py"
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "dev", "test"],
        help="Which split to run predictions on"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Inference batch size"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write JSON predictions (id → predicted_label)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (useful for quick inference only)"
    )
    parser.add_argument(
        "--feature-prefix", default="",
        help="Prefix for feature files, e.g. 'fused_' to load fused_test_features.npy"
    )
    args = parser.parse_args()

    feature_dir  = Path(args.feature_dir)
    ckpt_path    = Path(args.checkpoint)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run `python train_classifier.py` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    log.info(f"Loading checkpoint from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    saved_args    = ckpt.get("args", {})
    feature_dim   = saved_args.get("feature_dim", 12)
    hidden_dim    = saved_args.get("hidden_dim",  256)
    n_heads       = saved_args.get("n_heads",     4)
    n_layers      = saved_args.get("n_layers",    2)
    ff_dim        = saved_args.get("ff_dim",      512)
    dropout       = saved_args.get("dropout",     0.1)

    # ── Load data ─────────────────────────────────────────────────────────────
    prefix = args.feature_prefix
    dataset = load_split(args.split, feature_dir, prefix=prefix)

    # Verify / fix feature dim from actual data
    actual_feature_dim = dataset[0][0].shape[0]
    if actual_feature_dim != feature_dim:
        log.warning(
            f"Feature dim mismatch: checkpoint expects {feature_dim}, "
            f"data has {actual_feature_dim}. Using {actual_feature_dim}."
        )
        feature_dim = actual_feature_dim

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = FeatureClassifier(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Checkpoint from epoch {ckpt.get('epoch', '?')} "
             f"(dev acc at save: {ckpt.get('dev_acc', 'N/A'):.4f})")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    all_probs  = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, labels in tqdm(loader, desc=f"Predicting [{args.split}]"):
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy = (all_preds == all_labels).mean()
    log.info(f"[{args.split}] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class breakdown
    per_class = {}
    for cls in [0, 1]:
        cls_name = "human"   if cls == 0 else "machine"
        mask     = all_labels == cls
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == all_labels[mask]).mean()
            per_class[cls_name] = {"count": int(mask.sum()), "accuracy": float(cls_acc)}
            log.info(f"  Class {cls} ({cls_name}): {mask.sum()} samples, "
                     f"accuracy {cls_acc:.4f}")

    # Confusion-matrix style counts
    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    tn = int(((all_preds == 0) & (all_labels == 0)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    log.info(
        f"  TP={tp}  TN={tn}  FP={fp}  FN={fn} | "
        f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}"
    )

    # ── Generate plots & analysis ─────────────────────────────────────────────
    if not args.no_plots:
        log.info("=" * 60)
        log.info("Generating report plots and analysis …")
        log.info("=" * 60)

        # Load raw features for feature-level analysis
        raw_features = np.load(Path(args.feature_dir) / f"{prefix}{args.split}_features.npy")
        raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Confusion matrix
        plot_confusion_matrix(all_labels, all_preds, PLOT_DIR)

        # 2. Classification report (image + text)
        plot_classification_report(all_labels, all_preds, PLOT_DIR)

        # 3. ROC curve
        roc_auc = plot_roc_curve(all_labels, all_probs, PLOT_DIR)

        # 4. Precision-Recall curve
        avg_precision = plot_precision_recall_curve(all_labels, all_probs, PLOT_DIR)

        # 5. Confidence distribution
        plot_confidence_distribution(all_labels, all_preds, all_probs, PLOT_DIR)

        # 6. Feature distributions (violin plots)
        plot_feature_distributions(raw_features, all_labels, PLOT_DIR)

        # 7. Feature correlation heatmap
        plot_feature_correlation(raw_features, PLOT_DIR)

        # 8. t-SNE visualization
        plot_tsne(raw_features, all_labels, PLOT_DIR)

        # ── Save comprehensive metrics JSON ───────────────────────────────────
        metrics = {
            "split": args.split,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "average_precision": float(avg_precision),
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "per_class": per_class,
            "total_samples": len(all_labels),
            "checkpoint_epoch": ckpt.get("epoch", "unknown"),
        }
        metrics_path = RESULTS_DIR / "evaluation_metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        log.info(f"Saved evaluation metrics → {metrics_path}")

        log.info("=" * 60)
        log.info(f"All plots saved to:   {PLOT_DIR}/")
        log.info(f"All results saved to: {RESULTS_DIR}/")
        log.info("=" * 60)

    # ── Optional output file ──────────────────────────────────────────────────
    if args.output:
        output_data = {
            "split":    args.split,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall":   float(recall),
            "f1":       float(f1),
            "predictions": [
                {
                    "index":           int(i),
                    "true_label":      int(all_labels[i]),
                    "predicted_label": int(all_preds[i]),
                    "prob_human":      float(all_probs[i][0]),
                    "prob_machine":    float(all_probs[i][1]),
                }
                for i in range(len(all_preds))
            ],
        }
        out_path = Path(args.output)
        with open(out_path, "w") as fh:
            json.dump(output_data, fh, indent=2)
        log.info(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
