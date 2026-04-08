"""
Inspect ALL available pkl files to confirm structures before building fusion.
"""
import pickle
import numpy as np
import pandas as pd

# ============================================================
# 1. AFZAL — Stylometric Features
# ============================================================
print("=" * 60)
print("AFZAL: extracted_features_cache.pkl")
print("=" * 60)
with open("extracted_features_cache.pkl", "rb") as f:
    stylo = pickle.load(f)

for k, v in stylo.items():
    if isinstance(v, pd.DataFrame):
        print(f"  {k}: DataFrame, shape={v.shape}")
        print(f"    columns: {list(v.columns)}")
        print(f"    dtypes: {v.dtypes.unique()}")
        print(f"    sample row 0: {v.iloc[0].values[:5]}...")
        print(f"    any NaN: {v.isnull().any().any()}")
    elif isinstance(v, np.ndarray):
        print(f"  {k}: ndarray, shape={v.shape}, dtype={v.dtype}")
        print(f"    unique: {np.unique(v)}")
    else:
        print(f"  {k}: {type(v).__name__}")

# ============================================================
# 2. VINAY — Perplexity Features (train, dev, test)
# ============================================================
for fname, label in [
    ("train_perplexity_features.pkl", "TRAIN"),
    ("dev_perplexity_features.pkl", "DEV/VAL"),
    ("test_perplexity_features.pkl", "TEST"),
]:
    print()
    print("=" * 60)
    print(f"VINAY Perplexity: {fname} ({label})")
    print("=" * 60)
    with open(fname, "rb") as f:
        data = pickle.load(f)
    
    print(f"  type: {type(data).__name__}, num_entries: {len(data)}")
    
    # Check key types
    keys = list(data.keys())
    print(f"  key type: {type(keys[0]).__name__}")
    print(f"  key range: min={min(keys)}, max={max(keys)}")
    
    # Inspect first entry
    first = data[keys[0]]
    print(f"  entry type: {type(first).__name__}")
    
    if isinstance(first, dict):
        for ek, ev in first.items():
            if hasattr(ev, "shape"):
                print(f"    '{ek}': shape={ev.shape}, dtype={ev.dtype}, sample={ev[:4]}")
            else:
                print(f"    '{ek}': value={ev}, type={type(ev).__name__}")
    
    # Check a few more entries for consistency
    missing_vector = 0
    missing_label = 0
    dims = set()
    for i, (kid, entry) in enumerate(data.items()):
        if isinstance(entry, dict):
            if "vector" in entry:
                dims.add(len(entry["vector"]))
            else:
                missing_vector += 1
            if "label" not in entry:
                missing_label += 1
        if i >= 100:
            break
    print(f"  vector dims seen (first 100): {dims}")
    print(f"  missing 'vector' (first 100): {missing_vector}")
    print(f"  missing 'label' (first 100): {missing_label}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Stylo train: {stylo['train_features'].shape[0]} samples")
print(f"Stylo val:   {stylo['val_features'].shape[0]} samples")
print(f"Stylo test:  {stylo['test_features'].shape[0]} samples")

with open("train_perplexity_features.pkl", "rb") as f:
    pt = pickle.load(f)
with open("dev_perplexity_features.pkl", "rb") as f:
    pv = pickle.load(f)
with open("test_perplexity_features.pkl", "rb") as f:
    pte = pickle.load(f)
print(f"Perplexity train: {len(pt)} samples")
print(f"Perplexity val:   {len(pv)} samples")
print(f"Perplexity test:  {len(pte)} samples")

# Verify label consistency
y_train_stylo = stylo["y_train"]
perp_labels_sample = np.array([pt[i]["label"] for i in range(min(100, len(pt)))])
stylo_labels_sample = y_train_stylo[:100]
match = np.all(perp_labels_sample == stylo_labels_sample)
print(f"\nLabel consistency (first 100 train): {'MATCH' if match else 'MISMATCH!'}")
if not match:
    mismatches = np.where(perp_labels_sample != stylo_labels_sample)[0]
    print(f"  Mismatches at indices: {mismatches[:10]}")
