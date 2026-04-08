"""
fusion_dataset.py
─────────────────
PyTorch Dataset that aligns and loads:
  • Afzal's   11-D stylometric features   (DataFrame rows)
  • Vinay's   12-D perplexity features    (dict keyed by int id)
  • Teammate3 64-D semantic features      (dict keyed by int id)

Each __getitem__ returns:
    stylo_feat  : (11,) float32 tensor
    perp_feat   : (12,) float32 tensor
    sem_feat    : (64,) float32 tensor   [zeros if file not available]
    label       : scalar long tensor

Author: Fusion Team — SemEval-2024 Task 8A
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Optional, List, Tuple, Dict


# ── Configuration ──────────────────────────────────────────────
# Set to a list of column names to use only specific stylometric features.
# Set to "ALL" to use all 11 features.
# Example:  STYLO_FEATURES = ["dependency_tree_depth", "function_word_adjacency",
#                              "sentence_length_deviation", "verb_gap_deviation"]
STYLO_FEATURES = "ALL"
# ───────────────────────────────────────────────────────────────


def _dict_pkl_to_arrays(
    pkl_data: Dict, n_samples: int, vec_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert  {id: {"vector": np.array, "label": int}}  →  (features, labels).
    Handles both int keys (e.g., 0, 1, 2) and str keys (e.g., '0', '117646').
    The output arrays are ordered by id  0 … n_samples-1  so that they align
    with Afzal's row-indexed DataFrames.
    """
    features = np.zeros((n_samples, vec_dim), dtype=np.float32)
    labels = np.full(n_samples, -1, dtype=np.int64)

    # Detect key type and build a normalised lookup
    sample_key = next(iter(pkl_data))
    if isinstance(sample_key, str):
        # Keys are strings — could be str(int).  Build int→entry mapping.
        int_keyed = {int(k): v for k, v in pkl_data.items()}
    else:
        int_keyed = pkl_data

    for idx in range(n_samples):
        if idx in int_keyed:
            entry = int_keyed[idx]
            features[idx] = entry["vector"]
            labels[idx] = int(entry["label"])
        else:
            # Missing id — leave as zeros (will be caught by NaN safety net)
            pass

    found = np.sum(labels >= 0)
    if found < n_samples:
        print(f"    WARNING: {n_samples - found} ids missing from pkl "  
              f"(found {found}/{n_samples})")

    return features, labels


class FusionDataset(Dataset):
    """A multimodal dataset that fuses stylometric + perplexity + semantic."""

    def __init__(
        self,
        stylo_features: np.ndarray,      # (N, 11) float64
        stylo_labels: np.ndarray,         # (N,)   int64
        perp_features: np.ndarray,        # (N, 12) float32
        sem_features: Optional[np.ndarray] = None,  # (N, 64) float32 or None
        stylo_scaler: Optional[StandardScaler] = None,
        perp_scaler: Optional[StandardScaler] = None,
        sem_scaler: Optional[StandardScaler] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        stylo_features : ndarray (N, D_stylo)
        stylo_labels   : ndarray (N,)
        perp_features  : ndarray (N, 12)
        sem_features   : ndarray (N, D_sem) or None
        stylo_scaler   : fitted StandardScaler (if None, raw values used)
        perp_scaler    : fitted StandardScaler (if None, raw values used)
        sem_scaler     : fitted StandardScaler (if None, raw values used)
        feature_columns: column names (for logging only)
        """
        super().__init__()

        # Apply scalers ─────────────────────────────────────────
        self.stylo = (
            stylo_scaler.transform(stylo_features).astype(np.float32)
            if stylo_scaler is not None
            else stylo_features.astype(np.float32)
        )
        self.perp = (
            perp_scaler.transform(perp_features).astype(np.float32)
            if perp_scaler is not None
            else perp_features.astype(np.float32)
        )

        self.has_semantic = sem_features is not None
        if self.has_semantic:
            self.sem = (
                sem_scaler.transform(sem_features).astype(np.float32)
                if sem_scaler is not None
                else sem_features.astype(np.float32)
            )
            self.sem_dim = self.sem.shape[1]
        else:
            self.sem_dim = 64  # default placeholder dim
            self.sem = np.zeros(
                (len(stylo_features), self.sem_dim), dtype=np.float32
            )

        self.labels = stylo_labels.astype(np.int64)
        self.feature_columns = feature_columns

        # Replace any NaN / Inf with 0 (safety net) ────────────
        for arr in [self.stylo, self.perp, self.sem]:
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.stylo[idx]),
            torch.from_numpy(self.perp[idx]),
            torch.from_numpy(self.sem[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    @property
    def stylo_dim(self) -> int:
        return self.stylo.shape[1]

    @property
    def perp_dim(self) -> int:
        return self.perp.shape[1]


def load_and_align(
    split: str,
    stylo_pkl_path: str,
    perp_pkl_path: str,
    sem_pkl_path: Optional[str] = None,
    stylo_scaler: Optional[StandardScaler] = None,
    perp_scaler: Optional[StandardScaler] = None,
    sem_scaler: Optional[StandardScaler] = None,
) -> Tuple[FusionDataset, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load all pkl files for a given split and return an aligned FusionDataset.

    Parameters
    ----------
    split : 'train', 'val', or 'test'
    stylo_pkl_path : path to extracted_features_cache.pkl
    perp_pkl_path  : path to {train,dev,test}_perplexity_features.pkl
    sem_pkl_path   : path to semantic_{train,val,test}.pkl (optional)
    *_scaler       : fitted StandardScaler instances (pass None for train→fit later)

    Returns
    -------
    dataset : FusionDataset
    raw_stylo, raw_perp, raw_sem : unscaled numpy arrays (for fitting scalers)
    """
    # ── 1. Load Afzal's stylometric data ───────────────────────
    with open(stylo_pkl_path, "rb") as f:
        stylo_cache = pickle.load(f)

    key_map = {"train": "train", "val": "val", "test": "test"}
    s = key_map[split]
    stylo_df = stylo_cache[f"{s}_features"]
    stylo_labels = stylo_cache[f"y_{s}"]

    # Feature selection
    if STYLO_FEATURES != "ALL":
        stylo_df = stylo_df[STYLO_FEATURES]

    feature_columns = list(stylo_df.columns)
    stylo_np = stylo_df.values.astype(np.float64)
    n_samples = stylo_np.shape[0]

    print(f"  [{split}] Stylometric: {stylo_np.shape}  columns={feature_columns}")

    # ── 2. Load Vinay's perplexity data ────────────────────────
    with open(perp_pkl_path, "rb") as f:
        perp_dict = pickle.load(f)

    perp_np, perp_labels = _dict_pkl_to_arrays(perp_dict, n_samples, vec_dim=12)
    print(f"  [{split}] Perplexity:  {perp_np.shape}")

    # Verify label consistency
    label_match = np.all(stylo_labels == perp_labels)
    if not label_match:
        n_mismatch = np.sum(stylo_labels != perp_labels)
        print(f"  ⚠️  Label mismatch between stylo and perplexity: {n_mismatch} / {n_samples}")
        print(f"      Using stylometric labels as ground truth.")

    # ── 3. Load Semantic data (optional) ───────────────────────
    sem_np = None
    if sem_pkl_path is not None:
        # Try the exact path first, then common naming variants
        candidates = [
            sem_pkl_path,
            sem_pkl_path.replace("_val.", "_validation."),
            sem_pkl_path.replace("_validation.", "_val."),
        ]
        actual_path = None
        for c in candidates:
            if Path(c).exists():
                actual_path = c
                break

        if actual_path is not None:
            with open(actual_path, "rb") as f:
                sem_dict = pickle.load(f)
            first_entry = sem_dict[list(sem_dict.keys())[0]]
            sem_dim = len(first_entry["vector"])
            sem_np, sem_labels = _dict_pkl_to_arrays(sem_dict, n_samples, vec_dim=sem_dim)
            print(f"  [{split}] Semantic:    {sem_np.shape}  (from {Path(actual_path).name})")

            # Verify label consistency (ignore missing entries with label=-1)
            valid_mask = sem_labels >= 0
            if valid_mask.sum() > 0:
                sem_match = np.all(stylo_labels[valid_mask] == sem_labels[valid_mask])
                if not sem_match:
                    n_mismatch = np.sum(stylo_labels[valid_mask] != sem_labels[valid_mask])
                    print(f"  WARNING: Label mismatch with semantic: {n_mismatch} / {valid_mask.sum()}")
        else:
            print(f"  [{split}] Semantic:    NOT FOUND at {sem_pkl_path}")
    else:
        print(f"  [{split}] Semantic:    NOT AVAILABLE (will use zeros)")

    # ── 4. Build dataset ───────────────────────────────────────
    dataset = FusionDataset(
        stylo_features=stylo_np,
        stylo_labels=stylo_labels,
        perp_features=perp_np,
        sem_features=sem_np,
        stylo_scaler=stylo_scaler,
        perp_scaler=perp_scaler,
        sem_scaler=sem_scaler,
        feature_columns=feature_columns,
    )

    return dataset, stylo_np, perp_np, sem_np


def fit_scalers(
    stylo_train: np.ndarray,
    perp_train: np.ndarray,
    sem_train: Optional[np.ndarray] = None,
) -> Tuple[StandardScaler, StandardScaler, Optional[StandardScaler]]:
    """Fit StandardScalers on training data only."""
    stylo_scaler = StandardScaler().fit(stylo_train)
    perp_scaler = StandardScaler().fit(perp_train)
    sem_scaler = None
    if sem_train is not None:
        sem_scaler = StandardScaler().fit(sem_train)
    return stylo_scaler, perp_scaler, sem_scaler
