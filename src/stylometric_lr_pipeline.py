# %% [markdown]
# # 🔬 Stylometric AI-Text Detector — Simple Logistic Regression Pipeline
# ## Full Pipeline: Feature Caching · Global Scaling · CSV-Driven Training · Model Pickling · Inference
# 
# ---
# 
# ### Pipeline Overview
# 
# | Step | Cell | Description |
# |------|------|-------------|
# | **1** | Install & Imports | All dependencies |
# | **2** | Configuration | Paths, constants, feature names |
# | **3** | Feature Extraction + Cache | Extract 11 features ONCE, save to `extracted_features_cache.pkl` |
# | **4** | Global Scaler | Fit `StandardScaler` on train, transform all splits, save scaler |
# | **5** | CSV-Driven LR Training | Read `feature_combinations.csv`, train & pickle one model per combo |
# | **6** | Results Export | Save all metrics to `simple_lr_results.csv` |
# | **7** | Inference Function | Load saved model + scaler, predict on new texts |
# 
# ### 11 Stylometric Features
# | # | Feature | AI Signature |
# |---|---------|-------------|
# | 1 | `burstiness` | Low — uniform rhythm |
# | 2 | `sentence_length_deviation` | Low — consistent length |
# | 3 | `dependency_tree_depth` | Lower — simpler syntax |
# | 4 | `hapax_legomena_ratio` | Low — repetitive vocab |
# | 5 | `clause_density` | Higher — over-structured |
# | 6 | `verb_gap_deviation` | Low — regular verb spacing |
# | 7 | `lexical_density` | Higher — dense phrasing |
# | 8 | `stopword_gradient` | Near-zero — uniform |
# | 9 | `content_unique_ratio` | Lower — less variety |
# | 10 | `subject_verb_distance` | Lower — tight syntax |
# | 11 | `function_word_adjacency` | Lower — spaced grammar |

# %% [markdown]
# ---
# ## Cell 1 — Installation
# Run once in a fresh environment.

# %%
!pip install scikit-learn pandas numpy tqdm spacy --quiet
!python -m spacy download en_core_web_sm

# %% [markdown]
# ---
# ## Cell 2 — Library Imports

# %%
# ── Standard library ──────────────────────────────────────────────────────────
import hashlib
import itertools
import json
import math
import pickle
import re
import time
import warnings
from collections import Counter
from pathlib import Path

# ── Numerics & data ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ── NLP ───────────────────────────────────────────────────────────────────────
import spacy

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
print('✅  Core imports successful.')

# %% [markdown]
# ---
# ## Cell 3 — Configuration
# **Single source of truth.** All paths and constants live here.

# %%
# ── File paths ───────────────────────────────────────────────────────────────
TRAIN_PATH   = Path('train.jsonl')
VAL_PATH     = Path('validation.jsonl')
TEST_PATH    = Path('test.jsonl')

# ── Cache / output paths ─────────────────────────────────────────────────────
CACHE_PATH         = Path('extracted_features_cache.pkl')   # raw feature cache
SCALER_PATH        = Path('global_scaler.pkl')              # fitted StandardScaler
MODELS_DIR         = Path('lr_models')                      # directory for per-combo pickles
COMBOS_CSV         = Path('feature_combinations.csv')       # input: combos to train
RESULTS_CSV        = Path('simple_lr_results.csv')          # output: all metrics

MODELS_DIR.mkdir(exist_ok=True)

# ── Feature names — must match keys returned by extract_stylometric_features ─
FEATURE_NAMES = [
    'burstiness',
    'sentence_length_deviation',
    'dependency_tree_depth',
    'hapax_legomena_ratio',
    'clause_density',
    'verb_gap_deviation',
    'lexical_density',
    'stopword_gradient',
    'content_unique_ratio',
    'subject_verb_distance',
    'function_word_adjacency',
]

# ── Stopword gradient binning ─────────────────────────────────────────────────
BIN_SIZE = 50

print('✅  Configuration ready.')
print(f'   Feature names  : {len(FEATURE_NAMES)}')
print(f'   Cache path     : {CACHE_PATH}')
print(f'   Models dir     : {MODELS_DIR}')
print(f'   Input combos   : {COMBOS_CSV}')
print(f'   Results output : {RESULTS_CSV}')

# %% [markdown]
# ---
# ## Cell 4 — spaCy Setup & Stylometric Feature Extractor
# spaCy is loaded **once**. The 11 feature functions are identical to the companion notebook.

# %%
# ── Load spaCy model (once) ───────────────────────────────────────────────────
spacy.prefer_gpu()           # no-op when no GPU — always safe
nlp             = spacy.load('en_core_web_sm')
SPACY_STOPWORDS = nlp.Defaults.stop_words

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}
CLAUSE_DEPS = {'relcl', 'advcl', 'ccomp', 'xcomp', 'acl', 'csubj', 'csubjpass'}

print(f'✅  spaCy {spacy.__version__} loaded.')

# %%
# ── Helper: bin word list ─────────────────────────────────────────────────────
def _bin_words(words: list, bin_size: int = BIN_SIZE) -> list:
    if len(words) <= bin_size:
        return [words]
    return [
        words[i : i + bin_size]
        for i in range(0, len(words), bin_size)
        if words[i : i + bin_size]
    ]


# ── Helper: IQR outlier filter ────────────────────────────────────────────────
def _iqr_filter(values: list) -> list:
    if len(values) < 4:
        return values
    arr    = np.array(values, dtype=float)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr    = q3 - q1
    mask   = (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
    return arr[mask].tolist()


# ── Helper: recursive dependency tree depth ───────────────────────────────────
def _tree_depth(token) -> int:
    children = list(token.children)
    return 0 if not children else 1 + max(_tree_depth(c) for c in children)


# ── Main: extract all 11 stylometric features ─────────────────────────────────
def extract_stylometric_features(text: str) -> dict:
    """
    Compute 11 stylometric features for one text string via spaCy.
    Returns a dict keyed by FEATURE_NAMES; values default to 0.0.
    """
    text  = str(text)
    doc   = nlp(text)
    sents = list(doc.sents)
    words   = [t.text.lower() for t in doc if t.is_alpha]
    n_words = len(words)
    n_sents = len(sents)

    # 1. Burstiness
    sent_lens = [len([t for t in s if t.is_alpha]) for s in sents]
    if len(sent_lens) > 4:
        chunks     = np.array_split(sent_lens, 3)
        burstiness = float(np.std([np.std(c) for c in chunks if len(c) > 0]))
    elif len(sent_lens) > 1:
        mean_len   = max(np.mean(sent_lens), 1e-9)
        burstiness = float(np.std(sent_lens) / mean_len)
    else:
        burstiness = 0.0

    # 2. Sentence Length Deviation
    sentence_length_deviation = float(np.std(sent_lens)) if sent_lens else 0.0

    # 3. Dependency Tree Depth
    depths = []
    for sent in sents:
        roots = [t for t in sent if t.dep_ == 'ROOT']
        if roots:
            depths.append(_tree_depth(roots[0]))
    dependency_tree_depth = float(np.mean(depths)) if depths else 0.0

    # 4. Hapax Legomena Ratio
    freq                 = Counter(words)
    hapax_legomena_ratio = (
        sum(1 for c in freq.values() if c == 1) / n_words
        if n_words > 0 else 0.0
    )

    # 5. Clause Density
    n_clauses      = sum(1 for t in doc if t.dep_ in CLAUSE_DEPS) + n_sents
    clause_density = n_clauses / n_sents if n_sents > 0 else 0.0

    # 6. Verb Gap Deviation
    verb_positions = [t.i for t in doc if t.pos_ in ('VERB', 'AUX')]
    if len(verb_positions) >= 2:
        gaps       = [verb_positions[i+1] - verb_positions[i]
                      for i in range(len(verb_positions) - 1)]
        gaps_clean = _iqr_filter(gaps)
        verb_gap_deviation = (
            float(np.mean(np.abs(np.array(gaps_clean) - np.mean(gaps_clean))))
            if gaps_clean else 0.0
        )
    else:
        verb_gap_deviation = 0.0

    # 7. Lexical Density
    lexical_density = (
        sum(1 for t in doc if t.pos_ in CONTENT_POS) / n_words
        if n_words > 0 else 0.0
    )

    # 8. Stopword Gradient
    bins      = _bin_words(words, BIN_SIZE)
    sw_ratios = [
        sum(1 for w in b if w in SPACY_STOPWORDS) / len(b)
        for b in bins if len(b) > 0
    ]
    stopword_gradient = (
        float(np.polyfit(np.arange(len(sw_ratios)), sw_ratios, 1)[0])
        if len(sw_ratios) >= 2 else 0.0
    )

    # 9. Content Unique Ratio
    content_words = [
        t.text.lower() for t in doc
        if t.pos_ in CONTENT_POS and t.is_alpha
    ]
    content_unique_ratio = (
        len(set(content_words)) / len(content_words)
        if content_words else 0.0
    )

    # 10. Subject–Verb Distance
    sv_distances = [
        abs(t.i - child.i)
        for t in doc if t.pos_ in ('VERB', 'AUX')
        for child in t.children if child.dep_ in ('nsubj', 'csubj')
    ]
    subject_verb_distance = float(np.mean(sv_distances)) if sv_distances else 0.0

    # 11. Function Word Adjacency
    if n_words >= 2:
        is_fw  = [1 if w in SPACY_STOPWORDS else 0 for w in words]
        n_adj  = n_words - 1
        n_both = sum(1 for i in range(n_adj) if is_fw[i] and is_fw[i+1])
        function_word_adjacency = n_both / n_adj
    else:
        function_word_adjacency = 0.0

    return {
        'burstiness'               : burstiness,
        'sentence_length_deviation': sentence_length_deviation,
        'dependency_tree_depth'    : dependency_tree_depth,
        'hapax_legomena_ratio'     : hapax_legomena_ratio,
        'clause_density'           : clause_density,
        'verb_gap_deviation'       : verb_gap_deviation,
        'lexical_density'          : lexical_density,
        'stopword_gradient'        : stopword_gradient,
        'content_unique_ratio'     : content_unique_ratio,
        'subject_verb_distance'    : subject_verb_distance,
        'function_word_adjacency'  : function_word_adjacency,
    }


print('✅  Feature extractor defined (11 features).')

# %% [markdown]
# ---
# ## Cell 5 — Feature Extraction with Disk Cache
# 
# **Critical constraint:** Feature extraction runs **ONCE**. On subsequent runs the cache
# (`extracted_features_cache.pkl`) is loaded directly — no reprocessing.
# 
# Cache contents:
# ```
# {
#   'train_features' : DataFrame (N_train × 11),
#   'val_features'   : DataFrame (N_val   × 11),
#   'test_features'  : DataFrame (N_test  × 11),
#   'y_train'        : ndarray,
#   'y_val'          : ndarray,
#   'y_test'         : ndarray,
# }
# ```

# %%
def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a .jsonl file into a DataFrame. Skips blank lines silently."""
    records = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def build_feature_matrix(df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    """Apply extract_stylometric_features row-wise. Returns filled DataFrame."""
    feats = []
    for text in tqdm(df['text'],
                     desc=f'  Extracting [{split_label:>10}]',
                     leave=True):
        feats.append(extract_stylometric_features(text))
    return pd.DataFrame(feats, columns=FEATURE_NAMES).fillna(0.0)


# ── Cache-or-compute logic ────────────────────────────────────────────────────
if CACHE_PATH.exists():
    # ✅  CACHE HIT — load from disk, skip expensive extraction
    print('=' * 60)
    print(f'  📦  Cache found at "{CACHE_PATH}"')
    print('  Loading pre-extracted features (skipping recomputation) ...')
    print('=' * 60)
    t0 = time.perf_counter()
    with open(CACHE_PATH, 'rb') as f:
        _cache = pickle.load(f)
    train_features = _cache['train_features']
    val_features   = _cache['val_features']
    test_features  = _cache['test_features']
    y_train        = _cache['y_train']
    y_val          = _cache['y_val']
    y_test         = _cache['y_test']
    elapsed = time.perf_counter() - t0
    print(f'\n✅  Cache loaded in {elapsed:.2f}s')

else:
    # 🔄  CACHE MISS — extract features and save to disk
    print('=' * 60)
    print('  📂  No cache found — extracting features from raw JSONL files')
    print('  ⚠️   This is slow (may take hours for large datasets).')
    print('  ✅  It will NEVER run again once the cache is saved.')
    print('=' * 60)

    # Load raw splits
    print('\n  Loading datasets ...')
    df_train = load_jsonl(TRAIN_PATH)
    df_val   = load_jsonl(VAL_PATH)
    df_test  = load_jsonl(TEST_PATH)

    for name, df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        print(
            f'  [{name:>5}]  {len(df):>7,} texts  |'
            f'  Human: {(df["label"]==0).sum():>6,}'
            f'  AI: {(df["label"]==1).sum():>6,}'
        )

    # Extract features
    print('\n  Extracting stylometric features ...')
    t0             = time.perf_counter()
    train_features = build_feature_matrix(df_train, 'train')
    val_features   = build_feature_matrix(df_val,   'validation')
    test_features  = build_feature_matrix(df_test,  'test')
    elapsed        = time.perf_counter() - t0

    # Label arrays
    y_train = df_train['label'].values.astype(int)
    y_val   = df_val['label'].values.astype(int)
    y_test  = df_test['label'].values.astype(int)

    # Save cache to disk
    _cache = {
        'train_features': train_features,
        'val_features'  : val_features,
        'test_features' : test_features,
        'y_train'       : y_train,
        'y_val'         : y_val,
        'y_test'        : y_test,
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'\n✅  Feature extraction complete in {elapsed:.1f}s')
    print(f'   💾  Cache saved to "{CACHE_PATH}" — future runs will load this.')


# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n   train_features : {train_features.shape}')
print(f'   val_features   : {val_features.shape}')
print(f'   test_features  : {test_features.shape}')
print(f'   y_train        : {y_train.shape}  (0=Human, 1=AI)')
train_features.head(3)

# %% [markdown]
# ---
# ## Cell 6 — Global StandardScaler
# 
# **Critical:** `StandardScaler` is fitted **exclusively on the training set**.
# Fitting on val/test would leak their distribution and inflate metrics.
# The scaler is also saved to `global_scaler.pkl` so the inference function
# can reload it without retraining.

# %%
print('=' * 60)
print('  Scaling  (StandardScaler fit on TRAIN only)')
print('  No PolynomialFeatures · No PCA')
print('=' * 60)

# Raw numpy arrays (needed for column-indexing in training loop)
X_train_raw = train_features.values
X_val_raw   = val_features.values
X_test_raw  = test_features.values

# Fit ONCE on training data, transform all three splits
global_scaler = StandardScaler()
X_train_sc    = global_scaler.fit_transform(X_train_raw)   # fit + transform
X_val_sc      = global_scaler.transform(X_val_raw)          # transform only
X_test_sc     = global_scaler.transform(X_test_raw)         # transform only

# Persist the fitted scaler so inference can reload it
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(global_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f'\n  X_train_sc : {X_train_sc.shape}')
print(f'  X_val_sc   : {X_val_sc.shape}')
print(f'  X_test_sc  : {X_test_sc.shape}')
print(f'\n✅  Scaling complete — scaler saved to "{SCALER_PATH}"')

# %% [markdown]
# ---
# ## Cell 7 — Utility: Model Filename Generator
# 
# Each feature-combination string is converted to a deterministic, safe filename
# using two strategies:
# - **Clean slug** (human-readable): abbreviates each feature name, joins with `_`
# - **MD5 hash suffix** (collision-proof): appended to the slug
# 
# Example: `burstiness | lexical_density | clause_density`  
# → `lr_bur_lex_cla_a3f9d2b1.pkl`

# %%
# Short abbreviations for each of the 11 features (stable, human-readable)
_ABBREV = {
    'burstiness'               : 'bur',
    'sentence_length_deviation': 'sld',
    'dependency_tree_depth'    : 'dtd',
    'hapax_legomena_ratio'     : 'hlr',
    'clause_density'           : 'cld',
    'verb_gap_deviation'       : 'vgd',
    'lexical_density'          : 'lex',
    'stopword_gradient'        : 'swg',
    'content_unique_ratio'     : 'cur',
    'subject_verb_distance'    : 'svd',
    'function_word_adjacency'  : 'fwa',
}


def combo_to_filename(feature_str: str) -> str:
    """
    Convert a feature combination string like
      'burstiness | lexical_density | clause_density'
    to a deterministic, filesystem-safe pickle filename like
      'lr_bur_lex_cld_a3f9d2b1.pkl'

    The MD5 hash ensures uniqueness even if abbreviations collide.
    """
    features = [f.strip() for f in feature_str.split(' | ')]
    # Sort canonically so that order variations produce the same filename
    # NOTE: We sort here only for filename stability. The actual feature
    # order used during training is preserved separately via combo_idx.
    sorted_features = sorted(features)
    slug  = '_'.join(_ABBREV.get(f, f[:3]) for f in sorted_features)
    hash8 = hashlib.md5(feature_str.encode()).hexdigest()[:8]
    return f'lr_{slug}_{hash8}.pkl'


# ── Quick sanity check ────────────────────────────────────────────────────────
_example = 'burstiness | lexical_density | clause_density'
print(f'Example combo  : "{_example}"')
print(f'Generated name : "{combo_to_filename(_example)}"')
print('\n✅  Filename generator ready.')

# %% [markdown]
# ---
# ## Cell 8 — CSV-Driven Simple Logistic Regression Training
# 
# Reads `feature_combinations.csv` (column `Features_Used`).  
# For every combination row:
# 1. Subset the globally-scaled data to those features.
# 2. Train `LogisticRegression(max_iter=1000)` — **no class_weight, no modifications**.
# 3. Evaluate on validation and test sets.
# 4. Save the trained model to `lr_models/<smart_filename>.pkl`.
# 5. Record all metrics.
# 
# Already-pickled models are **skipped** (resume-safe on interruption).

# %%
# ── Load the feature combinations CSV ────────────────────────────────────────
assert COMBOS_CSV.exists(), (
    f'❌  "{COMBOS_CSV}" not found.\n'
    f'    Create a CSV with a column named "Features_Used" where each row\n'
    f'    contains feature names separated by " | "\n'
    f'    e.g.: burstiness | dependency_tree_depth | lexical_density'
)

combos_df = pd.read_csv(COMBOS_CSV)
assert 'Features_Used' in combos_df.columns, (
    f'❌  Column "Features_Used" not found in {COMBOS_CSV}.\n'
    f'    Found columns: {list(combos_df.columns)}'
)

combos_df = combos_df.dropna(subset=['Features_Used']).reset_index(drop=True)
print(f'✅  Loaded {len(combos_df):,} feature combinations from "{COMBOS_CSV}"')
print(f'   Preview of first 3 rows:')
display(combos_df[['Features_Used']].head(3))

# %%
# ── Feature name → column index lookup (global, built once) ──────────────────
FEAT_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


def parse_combo(feature_str: str):
    """
    Parse a " | "-separated feature string into:
      - features   : list of feature name strings
      - combo_idx  : tuple of column indices into the 11-feature matrix
    Raises ValueError if any feature name is unrecognised.
    """
    features = [f.strip() for f in feature_str.split(' | ')]
    unknown  = [f for f in features if f not in FEAT_INDEX]
    if unknown:
        raise ValueError(f'Unknown feature(s): {unknown}.  Valid: {FEATURE_NAMES}')
    combo_idx = tuple(FEAT_INDEX[f] for f in features)
    return features, combo_idx


# ── Main training loop ────────────────────────────────────────────────────────
results   = []
skipped   = 0
trained   = 0
errors    = 0

print('=' * 65)
print('  🚀  Simple Logistic Regression — CSV-Driven Training')
print(f'  Model type    : LogisticRegression(max_iter=1000)  [NO class_weight]')
print(f'  Total combos  : {len(combos_df):,}')
print(f'  Models dir    : {MODELS_DIR}')
print(f'  Resume-safe   : already-pickled models are skipped ✓')
print('=' * 65)

with tqdm(total=len(combos_df), desc='Training combos') as pbar:
    for _, row in combos_df.iterrows():
        feature_str = str(row['Features_Used']).strip()

        # ── Parse feature string ──────────────────────────────────────────────
        try:
            features, combo_idx = parse_combo(feature_str)
        except ValueError as e:
            tqdm.write(f'  ⚠️  Skipping invalid combo "{feature_str}": {e}')
            errors += 1
            pbar.update(1)
            continue

        # ── Determine pickle path ─────────────────────────────────────────────
        pkl_name = combo_to_filename(feature_str)
        pkl_path = MODELS_DIR / pkl_name

        # ── Resume: if model already pickled, load metrics from existing pkl ─
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                saved = pickle.load(f)
            results.append(saved['metrics'])
            skipped += 1
            pbar.update(1)
            continue

        # ── 1. Subset the globally scaled matrices ────────────────────────────
        X_tr_sub = X_train_sc[:, combo_idx]
        X_va_sub = X_val_sc[:,   combo_idx]
        X_te_sub = X_test_sc[:,  combo_idx]

        # ── 2. Train Simple Logistic Regression ──────────────────────────────
        #       CRITICAL: Default settings. NO class_weight. NO modifications.
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr_sub, y_train)

        # ── 3. Evaluate on Validation ─────────────────────────────────────────
        y_va_pred = model.predict(X_va_sub)
        val_acc   = accuracy_score(y_val,  y_va_pred)
        val_f1    = f1_score(y_val,        y_va_pred, zero_division=0)
        val_prec  = precision_score(y_val, y_va_pred, zero_division=0)
        val_rec   = recall_score(y_val,    y_va_pred, zero_division=0)

        # ── 4. Evaluate on Test ───────────────────────────────────────────────
        y_te_pred  = model.predict(X_te_sub)
        test_acc   = accuracy_score(y_test,  y_te_pred)
        test_f1    = f1_score(y_test,        y_te_pred, zero_division=0)
        test_prec  = precision_score(y_test, y_te_pred, zero_division=0)
        test_rec   = recall_score(y_test,    y_te_pred, zero_division=0)

        # ── 5. Build metrics dict ─────────────────────────────────────────────
        metrics = {
            'Model_Type'    : 'Simple_LR_No_Weights',
            'Num_Features'  : len(features),
            'Features_Used' : feature_str,
            'Pickle_File'   : str(pkl_path),
            'Val_Accuracy'  : round(val_acc,  6),
            'Val_F1'        : round(val_f1,   6),
            'Val_Precision' : round(val_prec, 6),
            'Val_Recall'    : round(val_rec,  6),
            'Test_Accuracy' : round(test_acc,  6),
            'Test_F1'       : round(test_f1,   6),
            'Test_Precision': round(test_prec, 6),
            'Test_Recall'   : round(test_rec,  6),
        }

        # ── 6. Pickle model + metadata ────────────────────────────────────────
        payload = {
            'model'          : model,
            'feature_names'  : features,
            'feature_indices': combo_idx,
            'feature_str'    : feature_str,
            'scaler_path'    : str(SCALER_PATH),
            'metrics'        : metrics,
        }
        with open(pkl_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        results.append(metrics)
        trained += 1
        pbar.update(1)


# ── Build & save results DataFrame ───────────────────────────────────────────
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(
    by=['Val_F1', 'Val_Accuracy'], ascending=[False, False]
).reset_index(drop=True)
df_results.to_csv(RESULTS_CSV, index=False)

print(f'\n✅  Training complete.')
print(f'   Newly trained  : {trained:,}')
print(f'   Loaded (cache) : {skipped:,}')
print(f'   Skipped errors : {errors}')
print(f'   Total results  : {len(df_results):,}')
print(f'   💾  Results saved to "{RESULTS_CSV}"')

# %% [markdown]
# ---
# ## Cell 9 — Preview Results

# %%
print('\n🏆 TOP 10 SIMPLE LR MODELS (Sorted by Val F1 → Val Accuracy)')
print('=' * 90)
display(
    df_results[[
        'Num_Features', 'Val_F1', 'Val_Accuracy',
        'Test_F1', 'Test_Accuracy', 'Features_Used'
    ]].head(10)
)

print('\n📊 Summary Statistics')
print('=' * 50)
print(df_results[['Val_F1', 'Val_Accuracy', 'Test_F1', 'Test_Accuracy']].describe().round(4))

# %% [markdown]
# ---
# ## Cell 10 — Inference Function
# 
# **No retraining.** Given a feature combination string and one or more raw text strings,
# this function:
# 1. Derives the expected pickle filename from the feature string.
# 2. Loads the corresponding `.pkl` model from `lr_models/`.
# 3. Loads the global scaler from `global_scaler.pkl`.
# 4. Extracts the 11 stylometric features from the input texts.
# 5. Scales with the global scaler, subsets to the combo's features.
# 6. Returns predictions and probabilities.
# 
# ```python
# # Usage
# results = predict_texts(
#     texts=['Some text here...', 'Another text...'],
#     feature_str='burstiness | lexical_density | clause_density'
# )
# ```

# %%
def predict_texts(
    texts: list,
    feature_str: str,
    models_dir: Path = MODELS_DIR,
    scaler_path: Path = SCALER_PATH,
) -> pd.DataFrame:
    """
    Run inference on a list of raw text strings using the pre-trained
    Logistic Regression model for a given feature combination.

    Parameters
    ----------
    texts        : list of str — raw text inputs to classify
    feature_str  : str — feature combination string used during training,
                   e.g. 'burstiness | lexical_density | clause_density'
    models_dir   : Path — directory containing .pkl model files (default: MODELS_DIR)
    scaler_path  : Path — path to the global scaler pickle (default: SCALER_PATH)

    Returns
    -------
    pd.DataFrame with columns:
        text          : original input text (truncated to 80 chars for display)
        prediction    : 0 (Human) or 1 (AI)
        label         : 'Human' or 'AI-Generated'
        prob_human    : probability of class 0
        prob_ai       : probability of class 1
        + all 11 raw stylometric features

    Raises
    ------
    FileNotFoundError  : if the model or scaler pickle is missing
    ValueError         : if the feature string contains unknown feature names
    """
    feature_str = feature_str.strip()

    # ── 1. Resolve pickle path ────────────────────────────────────────────────
    pkl_name = combo_to_filename(feature_str)
    pkl_path = Path(models_dir) / pkl_name

    if not pkl_path.exists():
        raise FileNotFoundError(
            f'No trained model found for this combination.\n'
            f'  Feature string : "{feature_str}"\n'
            f'  Expected file  : "{pkl_path}"\n'
            f'  Make sure you have trained and pickled this combination first '
            f'(Cell 8).'
        )

    # ── 2. Load model payload ─────────────────────────────────────────────────
    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)

    model           = payload['model']
    feature_indices = payload['feature_indices']   # tuple of int
    feature_names   = payload['feature_names']     # list of str

    # ── 3. Load global scaler ─────────────────────────────────────────────────
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(
            f'Global scaler not found at "{scaler_path}".\n'
            f'  Run Cell 6 to fit and save the scaler.'
        )

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # ── 4. Extract 11 stylometric features from raw texts ────────────────────
    print(f'  Extracting features for {len(texts)} text(s) ...')
    raw_feats = []
    for text in tqdm(texts, desc='  Extracting', leave=False):
        raw_feats.append(extract_stylometric_features(str(text)))

    feat_df  = pd.DataFrame(raw_feats, columns=FEATURE_NAMES).fillna(0.0)
    X_raw    = feat_df.values  # shape: (N, 11)

    # ── 5. Apply global scaler then subset to combo features ──────────────────
    X_scaled = scaler.transform(X_raw)             # shape: (N, 11)
    X_subset = X_scaled[:, feature_indices]        # shape: (N, len(combo))

    # ── 6. Predict ────────────────────────────────────────────────────────────
    preds  = model.predict(X_subset)               # shape: (N,)
    probas = model.predict_proba(X_subset)          # shape: (N, 2)

    # ── 7. Build results DataFrame ────────────────────────────────────────────
    out_df = pd.DataFrame({
        'text'      : [str(t)[:80] + ('...' if len(str(t)) > 80 else '') for t in texts],
        'prediction': preds,
        'label'     : ['AI-Generated' if p == 1 else 'Human' for p in preds],
        'prob_human': probas[:, 0].round(4),
        'prob_ai'   : probas[:, 1].round(4),
    })

    # Append raw feature values for traceability
    for feat in FEATURE_NAMES:
        out_df[feat] = feat_df[feat].values.round(6)

    print(f'\n✅  Inference complete using model "{pkl_name}"')
    print(f'   Features used  : {feature_names}')
    print(f'   Texts processed: {len(texts)}')
    return out_df


print('✅  Inference function `predict_texts()` defined.')
print()
print('Usage:')
print('  results = predict_texts(')
print('      texts=["Your text here...", "Another text..."],')
print('      feature_str="burstiness | lexical_density | clause_density"')
print('  )')

# %% [markdown]
# ---
# ## Cell 11 — Example Inference Demo
# 
# Demonstrates `predict_texts()` using the **best** feature combination
# from the results CSV (highest Val F1).

# %%
# ── Pick the best combo from results ─────────────────────────────────────────
best_combo_str = df_results.iloc[0]['Features_Used']
best_val_f1    = df_results.iloc[0]['Val_F1']

print('🏆  Best feature combination (highest Val F1):')
print(f'   Features : {best_combo_str}')
print(f'   Val F1   : {best_val_f1:.4f}')
print()

# ── Sample texts to classify ──────────────────────────────────────────────────
sample_texts = [
    # Likely human-written (informal, varied rhythm, personal voice)
    """I honestly didn't know what to expect going in. The museum was packed, 
    and the kids were already complaining about their feet. But then we turned 
    a corner and — wow. The dinosaur exhibit was incredible. My youngest just 
    stood there with her mouth open for a full minute.""",

    # Likely AI-generated (formal, structured, uniform rhythm)
    """The implementation of renewable energy sources represents a pivotal 
    advancement in sustainable development. Solar panels, wind turbines, and 
    hydroelectric systems collectively contribute to reducing carbon emissions. 
    Furthermore, these technologies demonstrate significant economic benefits 
    through reduced operational costs and long-term energy security.""",
]

# ── Run inference ─────────────────────────────────────────────────────────────
inference_results = predict_texts(
    texts       = sample_texts,
    feature_str = best_combo_str,
)

# ── Display results ───────────────────────────────────────────────────────────
print('\n📋  Inference Results:')
print('=' * 70)
display(
    inference_results[['text', 'label', 'prob_human', 'prob_ai']]
)

print('\n📐  Raw Feature Values (for traceability):')
print('=' * 70)
display(inference_results[FEATURE_NAMES])

# %% [markdown]
# ---
# ## Cell 12 — Standalone Inference Helper (Load-from-Disk Only)
# 
# This cell provides a **completely self-contained** inference function that can be
# copied to any other script. It depends **only** on:
# - The `lr_models/` directory with pickled models
# - `global_scaler.pkl`
# 
# No training data, no scaler fitting, no sklearn split — just load and predict.

# %%
def load_model_for_combo(
    feature_str: str,
    models_dir: str = 'lr_models',
    scaler_path: str = 'global_scaler.pkl',
) -> dict:
    """
    Load a pre-trained LR model and the global scaler for a given feature
    combination string.  Returns a dict with keys:
        'model'           : trained LogisticRegression
        'scaler'          : fitted StandardScaler
        'feature_names'   : list of feature name strings
        'feature_indices' : tuple of column indices
        'pkl_path'        : path to the loaded pickle file
    """
    pkl_name = combo_to_filename(feature_str.strip())
    pkl_path = Path(models_dir) / pkl_name

    if not pkl_path.exists():
        raise FileNotFoundError(
            f'Model pickle not found: "{pkl_path}".\n'
            f'Train this combination first (Cell 8).'
        )

    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return {
        'model'           : payload['model'],
        'scaler'          : scaler,
        'feature_names'   : payload['feature_names'],
        'feature_indices' : payload['feature_indices'],
        'pkl_path'        : str(pkl_path),
    }


def predict_from_features(
    feature_matrix: np.ndarray,
    feature_str: str,
    models_dir: str = 'lr_models',
    scaler_path: str = 'global_scaler.pkl',
) -> dict:
    """
    Run inference directly from a pre-extracted (N × 11) feature matrix.
    Useful when you have already run extract_stylometric_features() externally.

    Parameters
    ----------
    feature_matrix : np.ndarray of shape (N, 11) — raw (unscaled) features
                     in the order of FEATURE_NAMES
    feature_str    : str — feature combination string

    Returns
    -------
    dict with:
        'predictions'   : np.ndarray of int (0=Human, 1=AI)
        'probabilities' : np.ndarray of shape (N, 2)
        'labels'        : list of str ('Human' / 'AI-Generated')
    """
    loaded   = load_model_for_combo(feature_str, models_dir, scaler_path)
    model    = loaded['model']
    scaler   = loaded['scaler']
    idx      = loaded['feature_indices']

    X_scaled = scaler.transform(feature_matrix)    # (N, 11)
    X_subset = X_scaled[:, idx]                    # (N, len(combo))

    preds  = model.predict(X_subset)
    probas = model.predict_proba(X_subset)
    labels = ['AI-Generated' if p == 1 else 'Human' for p in preds]

    return {
        'predictions'  : preds,
        'probabilities': probas,
        'labels'       : labels,
    }


print('✅  Standalone inference helpers defined:')
print('   load_model_for_combo(feature_str)          → loads model + scaler')
print('   predict_from_features(X_raw_11, feature_str) → predict from raw feature matrix')

# %% [markdown]
# ---
# ## Summary — File Structure After Full Run
# 
# ```
# project/
# ├── train.jsonl                         ← raw data
# ├── validation.jsonl
# ├── test.jsonl
# ├── feature_combinations.csv            ← input: column 'Features_Used'
# │
# ├── extracted_features_cache.pkl        ← [GENERATED] raw features, all 3 splits + labels
# ├── global_scaler.pkl                   ← [GENERATED] fitted StandardScaler
# ├── simple_lr_results.csv               ← [GENERATED] all metrics, sorted by Val F1
# │
# └── lr_models/
#     ├── lr_bur_lex_cld_a3f9d2b1.pkl     ← [GENERATED] one pickle per combo
#     ├── lr_dtd_hlr_svd_fwa_b7e2c401.pkl
#     └── ...  (one .pkl per row in feature_combinations.csv)
# ```
# 
# ### Quick Re-run Behaviour
# | Condition | Cell 5 (features) | Cell 6 (scaler) | Cell 8 (training) |
# |-----------|-------------------|-----------------|-------------------|
# | First run | Extracts from JSONL (slow) | Fits scaler | Trains all combos |
# | Re-run | Loads cache (fast) | Loads saved scaler | Skips pickled models |
# | Add new combos | Loads cache (fast) | Loads saved scaler | Only trains new ones |


