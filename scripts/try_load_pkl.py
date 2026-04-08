"""
Quick script to try loading Afzal's pkl and diagnose version issues.
"""
import pickle
import sys
import pandas as pd
import numpy as np

print(f"Pandas version: {pd.__version__}")
print(f"Python version: {sys.version}")
print()

pkl_path = "extracted_features_cache.pkl"

# Attempt 1: Direct load
print("=" * 60)
print("Attempt 1: Direct pickle.load()")
print("=" * 60)
try:
    with open(pkl_path, 'rb') as f:
        cache = pickle.load(f)
    print("SUCCESS! Loaded without errors.")
    for key, val in cache.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: type={type(val).__name__}, shape={val.shape}, dtype={getattr(val, 'dtype', 'N/A')}")
        else:
            print(f"  {key}: type={type(val).__name__}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

    # Attempt 2: Try with pickle5 or different protocol
    print()
    print("=" * 60)
    print("Attempt 2: Trying pd.read_pickle()")
    print("=" * 60)
    try:
        # Sometimes pd.read_pickle handles version compat better
        data = pd.read_pickle(pkl_path)
        print(f"SUCCESS with pd.read_pickle! Type: {type(data)}")
    except Exception as e2:
        print(f"FAILED: {type(e2).__name__}: {e2}")

    # Attempt 3: Try loading with encoding fixes
    print()
    print("=" * 60)
    print("Attempt 3: Trying with encoding='latin1'")
    print("=" * 60)
    try:
        with open(pkl_path, 'rb') as f:
            cache = pickle.load(f, encoding='latin1')
        print("SUCCESS with latin1 encoding!")
    except Exception as e3:
        print(f"FAILED: {type(e3).__name__}: {e3}")

    # Show what Pandas version likely created this
    print()
    print("=" * 60)
    print("Diagnosis")
    print("=" * 60)
    print("The pkl contains Pandas DataFrames that were pickled with a")
    print("different Pandas version. The StringDtype error suggests a")
    print("major version mismatch (likely pickled with Pandas 2.x and")  
    print("you are running 3.x, or vice versa).")
    print()
    print("Solutions:")
    print("  1. pip install pandas==2.2.3  (try older 2.x)")
    print("  2. pip install pandas==2.1.4  (try older 2.x)")
    print("  3. Ask Afzal to re-save as numpy arrays")
