"""
Semantic Fingerprint Extraction — MGT Late Fusion Pipeline
Single DeBERTa pass → saves TWO .pkl files per split:

  embeddings_<split>.pkl  →  {id: {vector: 768-D CLS, label}}   (raw sentence embeddings)
  semantic_<split>.pkl    →  {id: {vector:  64-D compressed, label}}  (bottleneck features)

Kaggle dual-GPU optimised:
  - Explicit CUDA_VISIBLE_DEVICES + DataParallel(device_ids=[0,1]) — both GPUs always used
  - Mixed precision (torch.autocast) for ~2x throughput
  - Length-bucketed batching to minimize padding waste
  - Both vectors extracted in ONE forward pass — DeBERTa runs only once
  - Safe dictionary-keyed .pkl output per split for Late Fusion alignment

Output (6 files total):
  /kaggle/working/semantic_features/embeddings_train.pkl
  /kaggle/working/semantic_features/embeddings_validation.pkl
  /kaggle/working/semantic_features/embeddings_test.pkl
  /kaggle/working/semantic_features/semantic_train.pkl
  /kaggle/working/semantic_features/semantic_validation.pkl
  /kaggle/working/semantic_features/semantic_test.pkl
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CFG = {
    "model_name":     "microsoft/deberta-v3-base",
    "output_dir":     "/kaggle/working/semantic_features",
    "batch_size":     64,       # per-GPU; DataParallel doubles throughput with 2 GPUs
    "max_length":     512,
    "num_workers":    4,
    "bottleneck_in":  768,
    "bottleneck_out": 64,
    "seed":           42,
    "fp16":           True,     # mixed precision — safe on T4/P100 on Kaggle

    # ── Actual filenames for SemEval 2024 Task 8 SubtaskA monolingual
    "splits": {
        "train":      "subtaskA_train_monolingual.jsonl",
        "validation": "subtaskA_dev_monolingual.jsonl",
        "test":       "subtaskA_test_monolingual.jsonl",
    }
}


# ─────────────────────────────────────────────
# 0. FILE DISCOVERY
# ─────────────────────────────────────────────
def find_file(name: str) -> Path:
    """Auto-discover a file anywhere under /kaggle/input/ regardless of subfolder depth."""
    matches = list(Path("/kaggle/input").rglob(name))
    if not matches:
        raise FileNotFoundError(
            f"'{name}' not found under /kaggle/input/. "
            "Make sure the dataset is attached in the Kaggle sidebar."
        )
    if len(matches) > 1:
        print(f"  ⚠  Multiple matches for '{name}', using: {matches[0]}")
    return matches[0]


# ─────────────────────────────────────────────
# 1. MODEL: DeBERTa + Bottleneck
# ─────────────────────────────────────────────
class SemanticEncoder(nn.Module):
    """
    Wraps DeBERTa and returns BOTH vectors in a single forward pass:
      - cls_vec    : raw [CLS] token → shape [B, 768]  (sentence embedding)
      - compressed : bottleneck output → shape [B, 64]  (semantic fingerprint)

    The bottleneck is intentionally untrained (random init) as specified.
    Everything is frozen — pure inference, no gradients needed.
    """

    def __init__(self, model_name: str, in_dim: int = 768, out_dim: int = 64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Frozen random bottleneck — fixed across the entire run
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

        # Freeze everything: no gradients needed, pure inference
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        cls_vec    = outputs.last_hidden_state[:, 0, :]   # [B, 768]
        compressed = self.bottleneck(cls_vec)              # [B, 64]
        return cls_vec, compressed


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
class MGTDataset(Dataset):
    """
    Reads a .jsonl file and sorts by text length (length bucketing)
    to minimise padding waste within batches.
    """

    def __init__(self, filepath: str, max_length: int):
        self.max_length = max_length
        self.records: List[Dict] = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.records.append({
                    "id":    str(obj["id"]),
                    "text":  obj["text"],
                    "label": int(obj.get("label", -1)),
                })

        self.records.sort(key=lambda r: len(r["text"]))
        print(f"  Loaded {len(self.records):,} examples from {Path(filepath).name}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


def make_collate_fn(tokenizer, max_length: int):
    def collate_fn(batch: List[Dict]) -> Dict:
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ids":            [item["id"]    for item in batch],
            "labels":         [item["label"] for item in batch],
        }
    return collate_fn


# ─────────────────────────────────────────────
# 3. HARDWARE SETUP
# ─────────────────────────────────────────────
def setup_device() -> Tuple[torch.device, int]:
    """
    Explicitly sets CUDA_VISIBLE_DEVICES to ensure both Kaggle GPUs are used.
    Prints VRAM info for each GPU.
    """
    if not torch.cuda.is_available():
        print("⚠  No GPU detected — running on CPU (will be slow)")
        return torch.device("cpu"), 0

    n_gpus = torch.cuda.device_count()

    # Explicitly expose all GPUs to PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_gpus))

    print(f"  Found {n_gpus} GPU(s):")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / (1024 ** 3)
        print(f"    GPU {i}: {props.name} — {vram:.1f} GB VRAM")

    return torch.device("cuda:0"), n_gpus


# ─────────────────────────────────────────────
# 4. EXTRACTION LOOP
# ─────────────────────────────────────────────
@torch.no_grad()
def extract_split(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    use_fp16:   bool,
    split_name: str,
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Single DeBERTa pass over one split.
    Returns:
      embedding_dict : {id: {vector: np.ndarray[768], label: int}}
      feature_dict   : {id: {vector: np.ndarray[64],  label: int}}
    """
    embedding_dict: Dict[str, Dict] = {}
    feature_dict:   Dict[str, Dict] = {}
    total_batches = len(dataloader)
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        doc_ids        = batch["ids"]
        labels         = batch["labels"]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            cls_vec, compressed = model(input_ids=input_ids, attention_mask=attention_mask)
            # DataParallel scatters output across GPUs — gather both back to cuda:0
            if cls_vec.device != device:
                cls_vec    = cls_vec.to(device)
                compressed = compressed.to(device)

        # Cast to float32 on GPU BEFORE CPU transfer — avoids fp16 precision loss
        cls_np  = cls_vec.float().cpu().numpy()     # [B, 768]
        comp_np = compressed.float().cpu().numpy()  # [B, 64]

        for i, doc_id in enumerate(doc_ids):
            embedding_dict[doc_id] = {"vector": cls_np[i],  "label": labels[i]}
            feature_dict[doc_id]   = {"vector": comp_np[i], "label": labels[i]}

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            elapsed  = time.time() - t0
            progress = (batch_idx + 1) / total_batches
            eta      = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            print(
                f"  [{split_name}] Batch {batch_idx+1:>5}/{total_batches}"
                f"  |  {progress*100:.1f}%"
                f"  |  Elapsed: {elapsed:.0f}s"
                f"  |  ETA: {eta:.0f}s"
            )

    return embedding_dict, feature_dict


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Semantic Fingerprint Extraction")
    print("  Model :", CFG["model_name"])
    print("  Output:", output_dir)
    print("="*60)

    # ── Device setup
    device, n_gpus = setup_device()
    use_fp16 = CFG["fp16"] and (n_gpus > 0)

    # ── Tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    # ── Model — move to GPU BEFORE DataParallel wrapping
    print("[2/3] Building SemanticEncoder...")
    model = SemanticEncoder(
        model_name = CFG["model_name"],
        in_dim     = CFG["bottleneck_in"],
        out_dim    = CFG["bottleneck_out"],
    )
    model.eval()
    model = model.to(device)   # ← must be on GPU before DataParallel

    if n_gpus >= 2:
        # Explicitly pass device_ids so PyTorch uses BOTH GPUs
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
        effective_batch = CFG["batch_size"] * n_gpus
        print(f"  ✓ DataParallel across GPUs: {list(range(n_gpus))}")
        print(f"  ✓ Effective batch size: {effective_batch}  ({CFG['batch_size']} × {n_gpus})")
    elif n_gpus == 1:
        effective_batch = CFG["batch_size"]
        print(f"  ✓ Single GPU (cuda:0) — batch size: {effective_batch}")
    else:
        effective_batch = max(CFG["batch_size"] // 4, 8)
        print(f"  ⚠  CPU mode — reduced batch size: {effective_batch}")

    # ── Process each split
    print("\n[3/3] Extracting features...\n")
    for split_name, filename in CFG["splits"].items():

        try:
            jsonl_path = find_file(filename)
        except FileNotFoundError as e:
            print(f"  ⚠  Skipping '{split_name}': {e}\n")
            continue

        size_mb = jsonl_path.stat().st_size / (1024 ** 2)
        print(f"── Split: {split_name.upper()}  ({size_mb:.0f} MB) ──")
        print(f"   File: {jsonl_path}")

        dataset = MGTDataset(str(jsonl_path), max_length=CFG["max_length"])

        dataloader = DataLoader(
            dataset,
            batch_size         = effective_batch,
            shuffle            = False,              # CRITICAL — preserves alignment
            num_workers        = CFG["num_workers"],
            pin_memory         = (n_gpus > 0),       # faster CPU→GPU transfer
            collate_fn         = make_collate_fn(tokenizer, CFG["max_length"]),
            prefetch_factor    = 2 if CFG["num_workers"] > 0 else None,
            persistent_workers = (CFG["num_workers"] > 0),
        )

        t_start = time.time()
        embedding_dict, feature_dict = extract_split(
            model=model, dataloader=dataloader,
            device=device, use_fp16=use_fp16, split_name=split_name,
        )
        elapsed = time.time() - t_start

        # ── Sanity checks
        n_docs     = len(feature_dict)
        sample_key = next(iter(feature_dict))
        s_emb      = embedding_dict[sample_key]["vector"]
        s_feat     = feature_dict[sample_key]["vector"]

        assert s_emb.shape  == (CFG["bottleneck_in"],),  f"Bad embedding shape: {s_emb.shape}"
        assert s_feat.shape == (CFG["bottleneck_out"],), f"Bad feature shape: {s_feat.shape}"
        assert s_emb.dtype  == np.float32
        assert s_feat.dtype == np.float32

        print(f"\n  ✓ {n_docs:,} docs in {elapsed:.1f}s  ({n_docs/elapsed:.0f} docs/sec)")
        print(f"  ✓ Embedding : {s_emb.shape}   dtype={s_emb.dtype}")
        print(f"  ✓ Feature   : {s_feat.shape}  dtype={s_feat.dtype}")
        print(f"  ✓ Labels    : {sorted(set(v['label'] for v in feature_dict.values()))}")

        def save_pkl(data: Dict, path: Path):
            with open(path, "wb") as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  ✓ Saved → {path.name}  ({path.stat().st_size/1024**2:.1f} MB)")

        save_pkl(embedding_dict, output_dir / f"embeddings_{split_name}.pkl")  # 768-D
        save_pkl(feature_dict,   output_dir / f"semantic_{split_name}.pkl")    # 64-D
        print()

    print("="*60)
    print("  All done! Files in output dir:")
    for f in sorted(output_dir.glob("*.pkl")):
        print(f"    {f.name}  ({f.stat().st_size/1024**2:.1f} MB)")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# 6. VERIFICATION UTILITY
# ─────────────────────────────────────────────
def verify_pkl(pkl_path: str, n_samples: int = 3):
    """
    Sanity-check any .pkl file produced by this script.

    Usage:
        verify_pkl("/kaggle/working/semantic_features/semantic_train.pkl")
        verify_pkl("/kaggle/working/semantic_features/embeddings_train.pkl")
    """
    print(f"\nVerifying: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"  Total documents : {len(data):,}")
    for k in list(data.keys())[:n_samples]:
        v = data[k]
        print(f"  id={k!r:20s}  label={v['label']}  "
              f"shape={v['vector'].shape}  dtype={v['vector'].dtype}  "
              f"norm={np.linalg.norm(v['vector']):.4f}")


if __name__ == "__main__":
    main()

    # ── Verify all 6 output files
    output_dir = Path(CFG["output_dir"])
    print("\n" + "="*60)
    print("  Post-extraction verification (all 6 files)")
    print("="*60)
    for split_name in CFG["splits"]:
        for prefix in ["embeddings", "semantic"]:
            pkl = output_dir / f"{prefix}_{split_name}.pkl"
            if pkl.exists():
                verify_pkl(str(pkl))
