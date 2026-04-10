"""
================================================================================
MGT Detection - Export Pickle Files for Fusion
================================================================================

Exports two types of pickle files per split (train/dev/test):

1. RAW FEATURE PICKLES (12-D probability fingerprints):
   {split}_perplexity_features.pkl → Dict with:
     - "features": np.ndarray (N, 12)  — the 12-D fingerprint per text
     - "labels":   np.ndarray (N,)     — 0=human, 1=machine
     - "ids":      list[int] (N,)      — original JSONL line index
     - "feature_names": list[str]      — human-readable feature names
     - "metadata": dict                — model info, extraction details

2. CLASSIFIER EMBEDDINGS (256-D from trained model):
   {split}_perplexity_embeddings.pkl → Dict with:
     - "embeddings": np.ndarray (N, 256)  — 256-D hidden representations
     - "labels":     np.ndarray (N,)      — 0=human, 1=machine
     - "ids":        list[int] (N,)       — original JSONL line index
     - "predictions": np.ndarray (N,)     — model's predicted labels
     - "probabilities": np.ndarray (N, 2) — [P(human), P(machine)]
     - "metadata": dict

Usage:
  python export_pickles.py
  python export_pickles.py --splits train dev test
  python export_pickles.py --output-dir pickle_exports
================================================================================
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "mean_log_prob_gpt2medium",
    "std_log_prob_gpt2medium",
    "mean_entropy_gpt2medium",
    "std_entropy_gpt2medium",
    "mean_top1_log_prob_gpt2medium",
    "std_top1_log_prob_gpt2medium",
    "mean_log_prob_gpt2large",
    "std_log_prob_gpt2large",
    "mean_entropy_gpt2large",
    "std_entropy_gpt2large",
    "mean_top1_log_prob_gpt2large",
    "std_top1_log_prob_gpt2large",
]


# ──────────────────────────────────────────────────────────────────────────────
# Model definition (must match train_classifier.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
class FeatureClassifier(nn.Module):
    def __init__(
        self,
        feature_dim=12,
        hidden_dim=256,
        n_heads=4,
        n_layers=2,
        ff_dim=512,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)

    def get_embeddings(self, x):
        """Extract the 256-D hidden representation BEFORE the classification head."""
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return x  # (batch, hidden_dim=256)


def main():
    parser = argparse.ArgumentParser(
        description="Export perplexity features and embeddings as pickle files"
    )
    parser.add_argument(
        "--feature-dir", default="features",
        help="Directory with extracted .npy feature files"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_classifier.pt",
        help="Path to trained classifier checkpoint"
    )
    parser.add_argument(
        "--output-dir", default="pickle_exports",
        help="Directory to save pickle files"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "dev", "test"],
        help="Which splits to export"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for embedding extraction"
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Load trained model (for embeddings) ───────────────────────────────
    ckpt_path = Path(args.checkpoint)
    model = None
    if ckpt_path.exists():
        log.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})

        model = FeatureClassifier(
            feature_dim=saved_args.get("feature_dim", 12),
            hidden_dim=saved_args.get("hidden_dim", 256),
            n_heads=saved_args.get("n_heads", 4),
            n_layers=saved_args.get("n_layers", 2),
            ff_dim=saved_args.get("ff_dim", 512),
            dropout=saved_args.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        log.info(
            f"Model loaded (epoch {ckpt.get('epoch', '?')}, "
            f"dev_acc={ckpt.get('dev_acc', 'N/A'):.4f})"
        )
    else:
        log.warning(
            f"Checkpoint not found: {ckpt_path}. "
            "Will export raw features only (no embeddings)."
        )

    # ── Process each split ────────────────────────────────────────────────
    for split in args.splits:
        feat_path = feature_dir / f"{split}_features.npy"
        label_path = feature_dir / f"{split}_labels.npy"

        if not feat_path.exists():
            log.warning(f"[{split}] Feature file not found: {feat_path} — skipping")
            continue

        log.info(f"{'='*60}")
        log.info(f"Processing split: {split}")
        log.info(f"{'='*60}")

        # Load features and labels
        features = np.load(feat_path).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if label_path.exists():
            labels = np.load(label_path).astype(np.int64)
        else:
            labels = np.zeros(len(features), dtype=np.int64)
            log.warning(f"[{split}] Labels not found, using zeros")

        ids = list(range(len(features)))

        log.info(f"[{split}] Features: {features.shape}, Labels: {labels.shape}")

        # ── 1. Export raw features pickle ─────────────────────────────────
        raw_pkl = {}
        for i, idx in enumerate(ids):
            raw_pkl[idx] = {
                "vector": features[i],
                "label": int(labels[i])
            }

        raw_path = out_dir / f"{split}_perplexity_features.pkl"
        with open(raw_path, "wb") as f:
            pickle.dump(raw_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = raw_path.stat().st_size / (1024 * 1024)
        log.info(f"[{split}] ✓ Raw features saved: {raw_path} ({size_mb:.1f} MB)")

        # ── 2. Export embeddings pickle (if model available) ──────────────
        if model is not None:
            feat_tensor = torch.tensor(features, dtype=torch.float32)
            label_tensor = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(feat_tensor, label_tensor)
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
            )

            all_embeddings = []
            all_preds = []
            all_probs = []

            with torch.no_grad():
                for batch_feats, batch_labels in tqdm(
                    loader, desc=f"  Extracting embeddings [{split}]"
                ):
                    batch_feats = batch_feats.to(device)

                    # Get embeddings (256-D)
                    emb = model.get_embeddings(batch_feats)
                    all_embeddings.append(emb.cpu().numpy())

                    # Also get predictions and probabilities
                    logits = model(batch_feats)
                    probs = torch.softmax(logits, dim=-1)
                    preds = logits.argmax(dim=-1)

                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())

            all_embeddings = np.concatenate(all_embeddings, axis=0)
            all_preds = np.concatenate(all_preds, axis=0)
            all_probs = np.concatenate(all_probs, axis=0)

            emb_pkl = {}
            for i, idx in enumerate(ids):
                emb_pkl[idx] = {
                    "vector": all_embeddings[i],
                    "label": int(labels[i])
                }

            emb_path = out_dir / f"{split}_perplexity_embeddings.pkl"
            with open(emb_path, "wb") as f:
                pickle.dump(emb_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)
            size_mb = emb_path.stat().st_size / (1024 * 1024)
            log.info(
                f"[{split}] ✓ Embeddings saved: {emb_path} ({size_mb:.1f} MB) "
                f"shape={all_embeddings.shape}"
            )

            # Print accuracy for verification
            acc = (all_preds == labels).mean()
            log.info(f"[{split}] Classifier accuracy: {acc:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("EXPORT COMPLETE — Files for your friend:")
    log.info("=" * 60)
    for f in sorted(out_dir.glob("*.pkl")):
        size_mb = f.stat().st_size / (1024 * 1024)
        log.info(f"  {f.name:45s} ({size_mb:.1f} MB)")
    log.info("")
    log.info("Your friend can load any pickle file with:")
    log.info("  import pickle")
    log.info("  with open('train_perplexity_features.pkl', 'rb') as f:")
    log.info("      data = pickle.load(f)")
    log.info("  # Format: { id: {'vector': array(12-D), 'label': 0 or 1 } }")
    log.info("  sample_id = list(data.keys())[0]")
    log.info("  print(data[sample_id]['vector'])")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
