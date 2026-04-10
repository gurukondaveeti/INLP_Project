"""
================================================================================
MGT Detection - Feature Extraction (Kaggle Dual-GPU / 30GB VRAM)
================================================================================

HOW TO RUN THE FULL PIPELINE ON KAGGLE:
  1. Upload project files or zip to your Kaggle Notebook.
  2. Run: python extract_features.py
  3. Run: python train_classifier.py
  4. Run: python predict.py

This script:
  - Uses BOTH T4 GPUs on Kaggle via DataParallel (full 30GB VRAM)
  - Loads the entire dataset into memory (~30GB RAM available on Kaggle)
  - Uses "Length-Bucketing": Sorting texts by length to minimize padding waste
  - Automatically restores original file order before saving .npy
================================================================================
"""

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODELS = ["gpt2-medium", "gpt2-large"]
MAX_LENGTH = 512
BATCH_SIZE = 32         
FEATURES_PER_MODEL = 6
FEATURE_DIM = FEATURES_PER_MODEL * len(MODELS)

DATA_FILES = {
    "train": "subtaskA_train_monolingual.jsonl",
    "dev":   "subtaskA_dev_monolingual.jsonl",
    "test":  "subtaskA_test_monolingual.jsonl",
}
OUTPUT_DIR = Path("features")

# we can use this function to log the vram usage
def log_vram(tag: str = "") -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            used  = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            log.info(f"[VRAM] {tag} GPU-{i}: {used:.2f}/{total:.2f} GB used")


def masked_mean_std(tensor: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = mask.sum(dim=1).clamp(min=1)
    mean = (tensor * mask).sum(dim=1) / lengths
    diff = tensor - mean.unsqueeze(1)
    var = ((diff ** 2) * mask).sum(dim=1) / lengths
    std = torch.sqrt(var.clamp(min=1e-8))
    std = torch.where(lengths > 1, std, torch.zeros_like(std))
    return mean, std


@torch.no_grad()
def extract_batch_features(texts: list, model, tokenizer: AutoTokenizer, device: torch.device) -> np.ndarray:
    """Extract features from a batch of texts. `model` can be raw or DataParallel-wrapped."""
    try:
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

        if input_ids.shape[1] < 2:
            return np.zeros((len(texts), FEATURES_PER_MODEL), dtype=np.float32)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            # DataParallel returns CausalLMOutputWithCrossAttentions, which works fine
            outputs = model(input_ids)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Compute log_probs and immediately delete logits to save VRAM
        log_probs = F.log_softmax(shift_logits, dim=-1)
        del shift_logits
        del outputs
        del logits
        
        log_probs = torch.clamp(log_probs, min=-10000.0)  # Prevent fp16 -inf -> NaN
        
        # Extract the features we need directly
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        top1_log_probs = log_probs.max(dim=-1).values
        
        # Calculate entropy and free all vocab-sized tensors
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        del probs
        del log_probs

        mean_lp, std_lp = masked_mean_std(token_log_probs, shift_mask)
        mean_ent, std_ent = masked_mean_std(entropy, shift_mask)
        mean_top1, std_top1 = masked_mean_std(top1_log_probs, shift_mask)

        feats = torch.stack([mean_lp, std_lp, mean_ent, std_ent, mean_top1, std_top1], dim=1)
        return feats.float().cpu().numpy()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
            if len(texts) > 1:
                mid = len(texts) // 2
                log.warning(f"OOM on batch {len(texts)}: subdividing into {mid} + {len(texts)-mid}...")
                feats1 = extract_batch_features(texts[:mid], model, tokenizer, device)
                feats2 = extract_batch_features(texts[mid:], model, tokenizer, device)
                return np.concatenate([feats1, feats2], axis=0)
            else:
                log.error("OOM on single text. Returning zero-features.")
                return np.zeros((len(texts), FEATURES_PER_MODEL), dtype=np.float32)
        else:
            raise e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log.info(f"Using device: {device} | GPUs detected: {n_gpus}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Preload and Length-Bucket the data ────────────────────────────
    split_records = {}

    for split in args.splits:
        jsonl_path = DATA_FILES.get(split)
        if not jsonl_path or not os.path.exists(jsonl_path):
            continue

        log.info(f"[{split}] Loading full dataset into RAM for Length-Bucketing...")
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    records.append((i, obj["text"], int(obj.get("label", 0))))

        if args.max_samples:
            import random
            random.seed(42)
            c0 = [r for r in records if r[2] == 0]
            c1 = [r for r in records if r[2] == 1]
            limit = args.max_samples // 2
            records = random.sample(c0, min(len(c0), limit)) + random.sample(c1, min(len(c1), limit))

        records.sort(key=lambda r: len(r[1]))
        split_records[split] = records
        log.info(f"[{split}] Length-Bucketed {len(records)} records.")

    per_model_features = {}

    for model_name in MODELS:
        log.info(f"Loading tokenizer & model: {model_name} …")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        # Load model onto GPU 0 first
        raw_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).eval().to(device)

        # Wrap with DataParallel if multiple GPUs are available
        if n_gpus > 1:
            model = nn.DataParallel(raw_model)
            log.info(f"  DataParallel enabled across {n_gpus} GPUs!")
        else:
            model = raw_model
            log.info(f"  Single GPU mode.")

        log_vram(f"after loading {model_name}")
        per_model_features[model_name] = {}

        for split in args.splits:
            if split not in split_records:
                continue
            records = split_records[split]

            log.info(f"  [{split}] Extracting features...")
            pbar = tqdm(total=len(records), desc=f"{model_name}/{split}", unit="text", dynamic_ncols=True)

            extracted_blocks = []
            cbatch = []

            for _, text, _ in records:
                cbatch.append(text)
                if len(cbatch) == args.batch_size:
                    feats = extract_batch_features(cbatch, model, tokenizer, device)
                    extracted_blocks.append(feats)
                    pbar.update(len(cbatch))
                    cbatch = []

            if cbatch:
                extracted_blocks.append(extract_batch_features(cbatch, model, tokenizer, device))
                pbar.update(len(cbatch))

            pbar.close()

            combined_feats_sorted = np.concatenate(extracted_blocks, axis=0)
            original_indices = [r[0] for r in records]
            restore_order = np.argsort(original_indices)
            per_model_features[model_name][split] = combined_feats_sorted[restore_order]

        # Free GPU memory before loading the next model
        del model
        del raw_model
        for i in range(n_gpus if n_gpus > 0 else 1):
            torch.cuda.empty_cache()
        log.info(f"Unloaded {model_name}.")

    # ── Final Save ────────────────────────────────────────────────────
    for split in args.splits:
        if split not in split_records:
            continue
        records = split_records[split]

        original_indices = [r[0] for r in records]
        restore_order = np.argsort(original_indices)

        sorted_labels = np.array([r[2] for r in records], dtype=np.int64)
        labels_arr = sorted_labels[restore_order]

        label_path = OUTPUT_DIR / f"{split}_labels.npy"
        np.save(label_path, labels_arr)
        log.info(f"[{split}] Saved labels {labels_arr.shape} → {label_path}")

        arrays = [per_model_features[m][split] for m in MODELS if split in per_model_features.get(m, {})]
        if not arrays:
            continue

        combined = np.concatenate(arrays, axis=1)
        feat_path = OUTPUT_DIR / f"{split}_features.npy"
        np.save(feat_path, combined)
        log.info(f"[{split}] Saved features {combined.shape} → {feat_path}")

    log.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
