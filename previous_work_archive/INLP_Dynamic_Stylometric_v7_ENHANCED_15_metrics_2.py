# %%
!pip install --upgrade torch torchvision torchaudio spacy thinc
!python -m spacy download en_core_web_sm

# %% [markdown]
# # 🧠 Dynamic Stylometric AI-Text Detector — Phase 7 Enhanced
# ## SemEval 2024 Task 8: Binary Human vs. AI Classification
# 
# ### Key Improvements in v7:
# - **2 New Metrics**: Hapax Legomena Ratio + Type-Token Ratio Curve (TTR Slope)
# - **Metric Discriminability Gate**: Drops metrics whose `|H_mu - AI_mu| < 0.5 * pooled_sigma` (low signal) per bin
# - **Laplace-Smoothed Probability**: Prevents degenerate near-zero PDF scores ruining the vote
# - **Out-of-Interval Fallback (OOI)**: When a value falls >2σ outside BOTH distributions, use KL-Divergence tiebreak + flag as UNCERTAIN
# - **Soft Probability Aggregation**: Instead of hard majority vote, use weighted average of `P(AI)` scores, with per-metric weights from discriminability
# - **Adaptive Threshold (Youden J)**: Per-bin threshold tuning via Youden's J statistic (max sensitivity+specificity) on training folds
# - **Sparse Bin Merging**: Bins with <10 examples are merged with the nearest populated bin instead of falling back to global
# - **Confusion Matrix + ROC AUC** added to evaluation output
# 

# %%
# ============================================================
# CELL 1: Install & Setup
# ============================================================
# Uncomment on first run:
# !pip install spacy scikit-learn pandas numpy scipy tqdm --quiet
# !python -m spacy download en_core_web_sm

import pandas as pd
import numpy as np
import spacy
import scipy.stats as st
import pickle
import math
import warnings
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

spacy.prefer_gpu()

nlp = spacy.load('en_core_web_sm')
print('✅ Environment ready. spaCy loaded.')
print(f'   spaCy version: {spacy.__version__}')

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================
# Change these two variables to control the target word count 
# for the entire pipeline!
GLOBAL_MIN_WORDS = 101
GLOBAL_MAX_WORDS = 300
GLOBAL_TARGET_PER_CLASS = 10000

# Cell 13 Multi-Set Testing Controls
GLOBAL_MULTI_SET_N = 10         # Number of evaluation loops
GLOBAL_MULTI_SET_SAMPLES = 1000 # Total samples to test per loop

# %%
# ============================================================
# CELL 2: Data Loading (Using subtaskA_train_monolingual)
# ============================================================
from sklearn.model_selection import train_test_split

# 1. Load the dataset
# Assuming the file is a JSONL file (standard for SemEval 2024 Task 8)
file_path = 'subtaskA_train_monolingual.jsonl'
df = pd.read_json(file_path, lines=True)

# If your file is actually a CSV, comment out the two lines above and uncomment the line below:
# df = pd.read_csv('subtaskA_train_monolingual.csv')

# 2. Ensure columns are named correctly for the rest of the pipeline
# The pipeline expects exactly two columns: 'text' and 'label'
# SemEval data usually has 'text' and 'label', but if it has 'id', we can drop it.
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# 3. Split into Train and Test sets
# Since this is the training file, we will hold out 20% to act as our local test set
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"✅ Successfully loaded {file_path}")
print(f'Training samples : {len(train_df)} (Human={sum(train_df.label==0)}, AI={sum(train_df.label==1)})')
print(f'Test samples     : {len(test_df)}  (Human={sum(test_df.label==0)}, AI={sum(test_df.label==1)})')

# %%
# ============================================================
# CELL 3: 11-Metric Stylometric Feature Extraction
# ============================================================
# The Big List of 75 common English words (preserved from v6)
COMMON_WORDS = {
    'the','be','to','of','and','a','in','that','have','i',
    'it','for','not','on','with','he','as','you','do','at',
    'this','but','his','by','from','they','we','say','her',
    'she','or','an','will','my','one','all','would','there',
    'their','what','so','up','out','if','about','who','get',
    'which','go','me','when','make','can','like','time','no',
    'just','him','know','take','people','into','year','your',
    'good','some','could','them','see','other','than','then',
    'now','look','only','come','its','over','think','also'
}

def extract_stylometric_features(text: str) -> dict:
    doc = nlp(str(text))
    words = [token.text.lower() for token in doc if token.is_alpha]
    n_words = len(words)
    sents = list(doc.sents)

    # --- Original 11 Metrics ---
    unique_words_ratio = len(set(words)) / n_words if n_words > 0 else 0
    common_word_ratio = sum(1 for w in words if w in COMMON_WORDS) / n_words if n_words > 0 else 0
    verb_lengths = [len(t.text) for t in doc if t.pos_ == 'VERB']
    verb_length_deviation = float(np.std(verb_lengths)) if verb_lengths else 0.0
    sent_lens = [len([t for t in s if t.is_alpha]) for s in sents]
    sentence_length_deviation = float(np.std(sent_lens)) if sent_lens else 0.0
    verb_distances = [abs(t.i - c.i) for t in doc if t.pos_ in ('VERB', 'AUX') for c in t.children if c.dep_ in ('nsubj', 'csubj')]
    avg_verb_distance = float(np.mean(verb_distances)) if verb_distances else 0.0
    punct_density = sum(1 for t in doc if t.is_punct) / len(doc) if len(doc) > 0 else 0.0
    pronoun_ratio = sum(1 for t in doc if t.pos_ == 'PRON') / len(doc) if len(doc) > 0 else 0.0
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0
    lexical_density = sum(1 for t in doc if t.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}) / n_words if n_words > 0 else 0.0
    freq = Counter(words)
    hapax_ratio = sum(1 for c in freq.values() if c == 1) / n_words if n_words > 0 else 0.0

    ttr_slope = 0.0
    if n_words >= 100:
        ttr_vals = [len(set(words[i:i+50])) / 50 for i in range(0, n_words - 50, 25)]
        ttr_slope = float(np.polyfit(np.arange(len(ttr_vals)), ttr_vals, 1)[0]) if len(ttr_vals) >= 2 else 0.0

    # --- NEW: 85% CLUB FEATURES ---
    # 12. Burstiness (Variance of Sentence length variances)
    if len(sent_lens) > 4:
        chunks = np.array_split(sent_lens, 3)
        burstiness = np.std([np.std(c) for c in chunks if len(c) > 0])
    else:
        burstiness = 0.0

    # 13. Stopword Density Gradient (Style shift from start to end)
    stopwords = nlp.Defaults.stop_words
    sw_gradient = 0.0
    if n_words > 40:
        mid = n_words // 2
        sw_gradient = abs((sum(1 for w in words[:mid] if w in stopwords)/mid) -
                          (sum(1 for w in words[mid:] if w in stopwords)/(n_words-mid)))

    # 14. Named Entity Density (Generalities vs Specifics)
    ne_density = len(doc.ents) / n_words if n_words > 0 else 0.0

    # 15. Function Word Adjacency (Clustering of grammar words)
    sw_indices = [i for i, w in enumerate(words) if w in stopwords]
    fw_adjacency = float(np.mean(np.diff(sw_indices))) if len(sw_indices) > 1 else 0.0

    return {
        'unique_words_ratio': unique_words_ratio, 'common_word_ratio': common_word_ratio,
        'verb_length_deviation': verb_length_deviation, 'sentence_length_deviation': sentence_length_deviation,
        'avg_verb_distance': avg_verb_distance, 'punct_density': punct_density,
        'pronoun_ratio': pronoun_ratio, 'avg_word_length': avg_word_length,
        'lexical_density': lexical_density, 'hapax_ratio': hapax_ratio, 'ttr_slope': ttr_slope,
        'burstiness': burstiness, 'sw_gradient': sw_gradient, 'ne_density': ne_density, 'fw_adjacency': fw_adjacency
    }

# Quick sanity test
sample = extract_stylometric_features(train_df.loc[0, 'text'])
print('Sanity check — 11 features extracted:')
for k, v in sample.items():
    print(f'  {k:<28} {v:.4f}')

# %%



# # ============================================================
# # CELL 4: Strict 500/500 Sampling & Logistic Regression (Sigmoid)
# # ============================================================
# import random
# from sklearn.linear_model import LogisticRegression

# METRICS = [
#     'unique_words_ratio', 'common_word_ratio', 'verb_length_deviation',
#     'sentence_length_deviation', 'avg_verb_distance', 'punct_density',
#     'pronoun_ratio', 'avg_word_length', 'lexical_density',
#     'hapax_ratio', 'ttr_slope'
# ]

# CATEGORY_LIST = list(range(100, 1001, 100)) + ['Above 1000']
# MIN_BIN_SIZE  = 10

# def _get_bin(word_count: int):
#     if word_count > 1000: return 'Above 1000'
#     elif word_count == 0: return 100
#     else: return min(math.ceil(word_count / 100.0) * 100, 1000)

# def _discriminability(h_mu, h_sig, a_mu, a_sig):
#     pooled_sigma = math.sqrt((h_sig**2 + a_sig**2) / 2.0)
#     if pooled_sigma < 1e-9: return 0.0
#     return abs(h_mu - a_mu) / pooled_sigma

# def _youden_threshold(h_samples, a_samples, metric_name):
#     if len(h_samples) < 2 or len(a_samples) < 2:
#         mid = (np.mean(h_samples + a_samples)) if (h_samples + a_samples) else 0.5
#         return float(mid), (np.mean(a_samples) > np.mean(h_samples)) if (a_samples and h_samples) else True

#     all_vals = sorted(set(h_samples + a_samples))
#     best_j, best_t, best_dir = -1, all_vals[len(all_vals)//2], True

#     for t in all_vals:
#         for direction in [True, False]:
#             tp = sum(1 for v in a_samples if (v > t) == direction)
#             fn = len(a_samples) - tp
#             tn = sum(1 for v in h_samples if (v > t) != direction)
#             fp = len(h_samples) - tn
#             sens = tp / (tp + fn) if (tp + fn) > 0 else 0
#             spec = tn / (tn + fp) if (tn + fp) > 0 else 0
#             j = sens + spec - 1
#             if j > best_j:
#                 best_j, best_t, best_dir = j, float(t), direction
#     return best_t, best_dir


# def train_single_category_model(df: pd.DataFrame,
#                                 min_words: int = 401,
#                                 max_words: int = 500,
#                                 target_per_class: int = 500,
#                                 discriminability_cutoff: float = 0.15) -> dict:

#     print('=' * 70)
#     print(f'TRAINING SPECIFIC CATEGORY: {min_words} to {max_words} WORDS'.center(70))
#     print('=' * 70)

#     df = df.copy().reset_index(drop=True)
#     df['_word_count'] = df['text'].apply(lambda t: len(str(t).split()))

#     # Filter exactly the indices we want
#     category_df = df[(df['_word_count'] >= min_words) & (df['_word_count'] <= max_words)].copy()

#     # 1. STRICT SAMPLING: Exactly 500 Human and 500 AI (if available)
#     humans = category_df[category_df['label'] == 0]
#     ais = category_df[category_df['label'] == 1]

#     n_human = min(len(humans), target_per_class)
#     n_ai = min(len(ais), target_per_class)

#     humans_sampled = humans.sample(n=n_human, random_state=42)
#     ais_sampled = ais.sample(n=n_ai, random_state=42)

#     category_df = pd.concat([humans_sampled, ais_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
#     print(f"Step 1: Extracted {len(humans_sampled)} Human texts and {len(ais_sampled)} AI texts.")

#     category_df['_bin'] = category_df['_word_count'].apply(_get_bin)

#     print('Step 2: Extracting 11 stylometric features...')
#     feats_df = category_df['text'].apply(extract_stylometric_features).apply(pd.Series)
#     combined = pd.concat([category_df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

#     print('Step 3: Calculating Hyperparameters separately for Human and AI...')
#     # This specifically separates the calculation for Label 0 (Human) and Label 1 (AI)
#     global_stats = combined.groupby('label')[METRICS].agg(['mean','std']).fillna(0.001)

#     target_bin = _get_bin(max_words)
#     final_model = {}
#     cat_data = {}
#     active_count = 0

#     for m in METRICS:
#         # Separate Human Metrics
#         h_mu  = float(global_stats.loc[0,(m,'mean')]) if 0 in global_stats.index else 0.0
#         h_sig = max(float(global_stats.loc[0,(m,'std')]) if 0 in global_stats.index else 0.001, 1e-4)

#         # Separate AI Metrics
#         a_mu  = float(global_stats.loc[1,(m,'mean')]) if 1 in global_stats.index else 0.0
#         a_sig = max(float(global_stats.loc[1,(m,'std')]) if 1 in global_stats.index else 0.001, 1e-4)

#         disc  = _discriminability(h_mu, h_sig, a_mu, a_sig)
#         is_active = disc >= discriminability_cutoff
#         if is_active: active_count += 1

#         h_samp = combined.loc[combined.label==0, m].tolist()
#         a_samp = combined.loc[combined.label==1, m].tolist()
#         yt, yd = _youden_threshold(h_samp, a_samp, m)

#         cat_data[m] = {
#             'h_mu': h_mu, 'h_sig': h_sig,
#             'a_mu': a_mu, 'a_sig': a_sig,
#             'discriminability': disc,
#             'active': is_active,
#             'youden_threshold': yt,
#             'youden_direction': yd
#         }

#     final_model[target_bin] = cat_data
#     final_model['_global'] = cat_data

#     # 2. LOGISTIC REGRESSION (SIGMOID) TRAINING
#     print('Step 4: Training Logistic Regression (Sigmoid) on the 11 features...')
#     X_train = feats_df[METRICS].fillna(0)
#     y_train = category_df['label']

#     log_reg = LogisticRegression(max_iter=1000, random_state=42)
#     log_reg.fit(X_train, y_train)

#     # Save the trained regression model directly into the pickle data
#     final_model['_logistic_regression'] = log_reg

#     final_model['_meta'] = {
#         'discriminability_cutoff': discriminability_cutoff,
#         'metrics': METRICS,
#         'version': 8,
#         'category_trained': f'{min_words}_{max_words}',
#         'human_samples': len(humans_sampled),
#         'ai_samples': len(ais_sampled)
#     }

#     save_path = f'master_dynamic_thresholds_{min_words}_to_{max_words}_words.pkl'
#     with open(save_path, 'wb') as f:
#         pickle.dump(final_model, f)

#     print(f'  Bin {target_bin:<12}: {len(combined):>4} total samples | {active_count}/{len(METRICS)} metrics active')
#     print(f'\n✅ Model saved → {save_path}')
#     print('=' * 70)
#     return final_model

# learned_model = train_single_category_model(
#     df=train_df,
#     min_words=500,
#     max_words=1000,
#     target_per_class=3000
# )

# ============================================================
# CELL 4: Enhanced Non-Linear Sampling & Interaction Training
# ============================================================
import random
import pickle
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. SYNCED: Full 15-Metric Suite for 85% Accuracy
METRICS = [
    'unique_words_ratio', 'common_word_ratio', 'verb_length_deviation',
    'sentence_length_deviation', 'avg_verb_distance', 'punct_density',
    'pronoun_ratio', 'avg_word_length', 'lexical_density',
    'hapax_ratio', 'ttr_slope',
    'burstiness', 'sw_gradient', 'ne_density', 'fw_adjacency'
]

CATEGORY_LIST = list(range(100, 1001, 100)) + ['Above 1000']

def _get_bin(word_count: int):
    if word_count > 1000: return 'Above 1000'
    elif word_count == 0: return 100
    else: return min(math.ceil(word_count / 100.0) * 100, 1000)

def _discriminability(h_mu, h_sig, a_mu, a_sig):
    pooled_sigma = math.sqrt((h_sig**2 + a_sig**2) / 2.0)
    if pooled_sigma < 1e-9: return 0.0
    return abs(h_mu - a_mu) / pooled_sigma

def _youden_threshold(h_samples, a_samples, metric_name):
    if len(h_samples) < 2 or len(a_samples) < 2:
        mid = (np.mean(h_samples + a_samples)) if (h_samples + a_samples) else 0.5
        return float(mid), (np.mean(a_samples) > np.mean(h_samples)) if (a_samples and h_samples) else True
    all_vals = sorted(set(h_samples + a_samples))
    best_j, best_t, best_dir = -1, all_vals[len(all_vals)//2], True
    for t in all_vals:
        for direction in [True, False]:
            tp = sum(1 for v in a_samples if (v > t) == direction)
            fn = len(a_samples) - tp
            tn = sum(1 for v in h_samples if (v > t) != direction)
            fp = len(h_samples) - tn
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sens + spec - 1
            if j > best_j:
                best_j, best_t, best_dir = j, float(t), direction
    return best_t, best_dir

def train_single_category_model(df: pd.DataFrame,
                                min_words: int = 401,
                                max_words: int = 500,
                                target_per_class: int = 1500,
                                discriminability_cutoff: float = 0.15) -> dict:

    print('=' * 70)
    print(f'TRAINING ENHANCED MODEL: {min_words} to {max_words} WORDS'.center(70))
    print('=' * 70)

    df = df.copy().reset_index(drop=True)
    df['_word_count'] = df['text'].apply(lambda t: len(str(t).split()))

    # --- ROBUST FILTERING ---
    # Find humans in range, or fallback to wider range to capture style
    humans = df[(df['label'] == 0) & (df['_word_count'] >= min_words) & (df['_word_count'] <= max_words)]
    if len(humans) < 10:
        print(f"⚠️ Low Humans found. Expanding search range to find style patterns...")
        humans = df[(df['label'] == 0) & (df['_word_count'] >= 300) & (df['_word_count'] <= 1000)]

    ais = df[(df['label'] == 1) & (df['_word_count'] >= min_words) & (df['_word_count'] <= max_words)]

    n_human = min(len(humans), target_per_class)
    n_ai = min(len(ais), target_per_class)

    humans_sampled = humans.sample(n=n_human, random_state=42)
    ais_sampled = ais.sample(n=n_ai, random_state=42)
    category_df = pd.concat([humans_sampled, ais_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Step 1: Extracted {len(humans_sampled)} Human and {len(ais_sampled)} AI texts.")

    # Step 2: Feature Extraction
    print(f'Step 2: Extracting {len(METRICS)} stylometric features...')
    feats_df = category_df['text'].apply(extract_stylometric_features).apply(pd.Series)
    combined = pd.concat([category_df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

    # Step 3: Profile Statistics
    print('Step 3: Calculating Profile Statistics...')
    global_stats = combined.groupby('label')[METRICS].agg(['mean','std']).fillna(0.001)

    final_model = {}
    cat_data = {}
    for m in METRICS:
        h_mu, h_sig = float(global_stats.loc[0,(m,'mean')]), max(float(global_stats.loc[0,(m,'std')]), 0.001)
        a_mu, a_sig = float(global_stats.loc[1,(m,'mean')]), max(float(global_stats.loc[1,(m,'std')]), 0.001)
        disc = _discriminability(h_mu, h_sig, a_mu, a_sig)
        h_samp, a_samp = combined.loc[combined.label==0, m].tolist(), combined.loc[combined.label==1, m].tolist()
        yt, yd = _youden_threshold(h_samp, a_samp, m)
        cat_data[m] = {'h_mu': h_mu, 'h_sig': h_sig, 'a_mu': a_mu, 'a_sig': a_sig,
                       'discriminability': disc, 'active': disc >= discriminability_cutoff,
                       'youden_threshold': yt, 'youden_direction': yd}

    final_model['_global'] = cat_data
    final_model[_get_bin(max_words)] = cat_data

    # Step 4: NEW - Non-Linear Logistic Regression (Interaction Terms)
    print('Step 4: Training Enhanced Non-Linear Logistic Regression...')
    X_raw = feats_df[METRICS].fillna(0)

    # Transformation: Creates new features like (Unique_Words * Burstiness)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    y_train = category_df['label']

    # Cost-Sensitive weights (1.25x penalty for misclassifying humans)
    log_reg = LogisticRegression(max_iter=2000, class_weight={0: 1.25, 1: 1.0}, C=0.5, random_state=42)
    log_reg.fit(X_poly, y_train)

    final_model['_logistic_regression'] = log_reg
    final_model['_poly_transformer'] = poly # SAVED FOR STEP 5 SYNC

    final_model['_meta'] = {'metrics': METRICS, 'category_trained': f'{min_words}_{max_words}', 'version': 15}

    save_path = f'master_dynamic_thresholds_{min_words}_to_{max_words}_words.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(final_model, f)

    print(f'\n✅ Enhanced Model saved → {save_path}')
    return final_model

# EXECUTE TRAINING
# learned_model = train_single_category_model(df=train_df, min_words=401, max_words=500, target_per_class=1500)
learned_model = train_single_category_model(df=train_df, min_words=GLOBAL_MIN_WORDS, max_words=GLOBAL_MAX_WORDS, target_per_class=GLOBAL_TARGET_PER_CLASS)

# %%
# ============================================================
# CELL 5: Inference Engine (Non-Linear Interaction Logic)
# ============================================================

EPS     = 1e-6
OOI_Z   = 2.5

def _ooi_fallback_vote(val, m_data):
    """Fallback logic when values are Out-of-Interval."""
    d_h = abs(val - m_data['h_mu']) / max(m_data['h_sig'], 1e-6)
    d_a = abs(val - m_data['a_mu']) / max(m_data['a_sig'], 1e-6)
    if d_h + d_a == 0: return 0.5
    return float(d_h / (d_h + d_a))

def predict_single(text: str, model: dict) -> dict:
    word_count  = len(str(text).split())
    current_bin = _get_bin(word_count)

    # Use specific bin data if available, else fallback to global
    # Optimized to target the 401-500 range (Bin 500)
    m_data = model.get(current_bin, model.get(500, model['_global']))

    # Step 1: Extract all 15 stylometric features
    features = extract_stylometric_features(text)
    metric_details = []

    # Ensure feature vector order perfectly matches the 15 METRICS list
    feature_vector = [features[m] for m in METRICS]

    for m_name, val in features.items():
        md = m_data.get(m_name, model['_global'].get(m_name))
        if not md: continue

        # Check for Out-of-Interval (OOI)
        z_h = abs(val - md['h_mu']) / max(md['h_sig'], 1e-6)
        z_a = abs(val - md['a_mu']) / max(md['a_sig'], 1e-6)
        is_ooi = (z_h > OOI_Z) and (z_a > OOI_Z)

        # Youden Threshold Logic for the "Majority Vote" side-report
        yt  = md.get('youden_threshold', (md['h_mu'] + md['a_mu']) / 2)
        yd  = md.get('youden_direction', True)
        vote_ai = ((val > yt) == yd)

        metric_details.append({
            'metric'   : m_name,
            'value'    : val,
            'vote_ai'  : vote_ai,
            'active'   : md.get('active', True),
            'ooi'      : is_ooi
        })

    # 1. Majority Vote Logic (Side-Metric for comparison)
    active_votes_ai  = sum(1 for d in metric_details if d['active'] and d['vote_ai'])
    active_total     = sum(1 for d in metric_details if d['active'])
    majority_vote_ai = 1 if active_votes_ai >= max(1, math.ceil(active_total / 2)) else 0

    # 2. Enhanced Logistic Regression (Non-Linear Interaction)
    log_reg = model.get('_logistic_regression')
    poly    = model.get('_poly_transformer') # Required for 85% accuracy logic

    if log_reg and poly:
        # Transform the raw 15 metrics into expanded Interaction Terms
        X_poly = poly.transform([feature_vector])

        # Get Sigmoid Probability for the AI class
        sigmoid_prob_ai = log_reg.predict_proba(X_poly)[0][1]
        sigmoid_pred = 1 if sigmoid_prob_ai >= 0.5 else 0
    else:
        # Fallback if the advanced model components are missing
        sigmoid_prob_ai = 0.5
        sigmoid_pred = majority_vote_ai

    return {
        'pred_label'      : sigmoid_pred,
        'majority_vote'   : majority_vote_ai,
        'sigmoid_prob_ai' : sigmoid_prob_ai,
        'word_count'      : word_count,
        'current_bin'     : current_bin,
        'metric_details'  : metric_details,
        'active_votes_ai' : active_votes_ai,
        'active_total'    : active_total
    }

# %%
# ============================================================
# CELL 6: Full Evaluation with Formatted Output (Updated for Sigmoid)
# ============================================================

def run_full_evaluation(test_dataframe: pd.DataFrame,
                         model: dict,
                         show_examples: int = 10):
    LABEL = {0: 'Human', 1: 'AI'}
    cat_stats = {k: {'total': 0, 'correct': 0} for k in CATEGORY_LIST}
    predictions, true_labels, prob_scores = [], [], []

    test_dataframe = test_dataframe.reset_index(drop=True)
    n = len(test_dataframe)

    print('\n' + '=' * 110)
    print('FULL EVALUATION — Majority Vote + Sigmoid Regression'.center(110))
    print('=' * 110)

    for idx, row in test_dataframe.iterrows():
        result     = predict_single(row['text'], model)
        pred_label = result['pred_label']
        true_label = int(row['label'])
        is_correct = (pred_label == true_label)
        marker     = '✅ Correct' if is_correct else '❌ Incorrect'

        predictions.append(pred_label)
        true_labels.append(true_label)
        prob_scores.append(result['sigmoid_prob_ai'])

        cbin = result['current_bin']
        if cbin in cat_stats:
            cat_stats[cbin]['total'] += 1
            if is_correct:
                cat_stats[cbin]['correct'] += 1

        if idx < show_examples:
            snippet = str(row['text'])[:100].replace('\n', ' ')
            print(f'\n{"=" * 110}')
            print(f'EXAMPLE {idx+1:>3} | Word Count: {result["word_count"]:>5} | Bin: {str(result["current_bin"]):<12} | Active: {result["active_total"]}/11')
            print(f'Text Snippet: "{snippet}..."')
            print(f'{"-" * 110}')
            print(f'{"Metric":<28} | {"Value":>8} | {"Active":<7} | {"Vote AI":<8} | {"OOI"}')
            print(f'{"-" * 28}-+-{"":->8}-+-{"":-<7}-+-{"":-<8}-+-{"":-<5}')

            ooi_count = 0
            for d in result['metric_details']:
                active_str = '✓' if d['active'] else '✗'
                vote_str   = 'Yes' if d['vote_ai'] else 'No'
                ooi_flag   = '⚠️ OOI' if d['ooi'] else ''
                if d['ooi']: ooi_count += 1
                print(f'{d["metric"]:<28} | {d["value"]:>8.4f} | {active_str:<7} | {vote_str:<8} | {ooi_flag}')

            print(f'{"-" * 110}')
            print(f'Majority Vote AI Count: {result["active_votes_ai"]}/{result["active_total"]} | Sigmoid AI Prob: {result["sigmoid_prob_ai"]:.4f}')
            print(f'True: {LABEL[true_label]:<7} | Predicted: {LABEL[pred_label]:<7} --> {marker}')

    acc  = accuracy_score(true_labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro', zero_division=0)
    prec1, rec1, f1_1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(true_labels, prob_scores)
    except Exception:
        auc = float('nan')
    cm   = confusion_matrix(true_labels, predictions)

    print(f'\n{"=" * 80}')
    print('FINAL PERFORMANCE SUMMARY'.center(80))
    print('=' * 80)
    print(f'  {"Overall Accuracy":<25} {acc:.4f}')
    print(f'  {"Macro Precision":<25} {prec:.4f}')
    print(f'  {"Macro Recall":<25} {rec:.4f}')
    print(f'  {"Macro F1-Score":<25} {f1:.4f}')
    print(f'  {"AI-class F1 (binary)":<25} {f1_1:.4f}')
    print(f'  {"ROC-AUC (Sigmoid)":<25} {auc:.4f}')
    print(f'  {"Total Examples":<25} {n}')
    print('=' * 80)
    print(f'\n  Confusion Matrix:')
    print(f'  {"":>12} Pred Human  Pred AI')
    print(f'  {"True Human":<12} {cm[0,0]:>10}  {cm[0,1]:>7}')
    print(f'  {"True AI":<12} {cm[1,0]:>10}  {cm[1,1]:>7}')
    print('=' * 80)

    return predictions, prob_scores, true_labels

# ==========================================
# ADD THIS TO THE VERY BOTTOM OF CELL 6
# ==========================================

# # 1. Filter the test set to match our trained category (401-500 words)
# print("Filtering test set for 401-500 word lengths...")
# test_df['_word_count'] = test_df['text'].apply(lambda t: len(str(t).split()))
# category_test_df = test_df[(test_df['_word_count'] >= 401) & (test_df['_word_count'] <= 500)].copy()

# 1. Filter the test set for > 500 words
print(f"Filtering test set for between {GLOBAL_MIN_WORDS} to {GLOBAL_MAX_WORDS} word lengths...")
test_df['_word_count'] = test_df['text'].apply(lambda t: len(str(t).split()))
# category_test_df = test_df[(test_df['_word_count'] >= 501)].copy() # Change 401 to 501
category_test_df = test_df[(test_df['_word_count'] >= GLOBAL_MIN_WORDS) & (test_df['_word_count'] <= GLOBAL_MAX_WORDS)].copy()

print(f"Filtered Test Set Size: {len(category_test_df)} texts")

# 2. Run evaluation ONLY on this filtered test set
preds, probs, gts = run_full_evaluation(category_test_df, learned_model, show_examples=10)

# %%
# ============================================================
# CELL 7: Hyperparameter Inspector — Side-by-Side Profile Table
# ============================================================

def inspect_profiles(model: dict):
    """
    Prints per-bin, per-metric: Human(μ±σ), AI(μ±σ), Abs Diff, Discriminability.
    Highlights active/inactive metrics.
    """
    print('=' * 130)
    print('LEARNED HYPERPARAMETERS: HUMAN vs AI PROFILES (50-WORD BINS)'.center(130))
    print('=' * 130)

    sorted_keys = sorted([k for k in model.keys() if isinstance(k, int)])
    if 'Above 1000' in model:
        sorted_keys.append('Above 1000')

    for cat in sorted_keys:
        if cat in ('_global', '_meta', '_logistic_regression'):
            continue
        cat_data = model[cat]
        if not isinstance(cat_data, dict):
            continue
        print(f'\nBIN: {cat}')
        print(f'{"Metric":<28} | {"Human (μ ± σ)":<25} | {"AI (μ ± σ)":<25} | {"Abs Diff":>9} | {"Disc":>6} | {"Active"}')
        print(f'{"-" * 28}-+-{"-" * 25}-+-{"-" * 25}-+-{"":->9}-+-{"":->6}-+-{"":-<6}')

        for m in METRICS:
            if m not in cat_data:
                continue
            d = cat_data[m]
            h_str    = f'{d["h_mu"]:.4f} ± {d["h_sig"]:.4f}'
            a_str    = f'{d["a_mu"]:.4f} ± {d["a_sig"]:.4f}'
            diff     = abs(d['h_mu'] - d['a_mu'])
            disc     = d.get('discriminability', 0.0)
            active   = '✓ YES' if d.get('active', True) else '✗ NO '
            print(f'{m:<28} | {h_str:<25} | {a_str:<25} | {diff:>9.4f} | {disc:>6.3f} | {active}')

    print('\n' + '=' * 130)


inspect_profiles(learned_model)

# %%
# ============================================================
# CELL 8: Out-of-Interval (OOI) Analysis Report
# ============================================================
# Investigates which metrics trigger OOI most often and
# whether OOI texts are harder to classify.

def ooi_analysis_report(test_dataframe: pd.DataFrame, model: dict):
    print('=' * 90)
    print('OUT-OF-INTERVAL (OOI) ANALYSIS REPORT'.center(90))
    print('=' * 90)

    ooi_freq   = {m: 0 for m in METRICS}
    ooi_correct = 0
    ooi_total   = 0
    non_ooi_correct = 0
    non_ooi_total   = 0

    for _, row in test_dataframe.iterrows():
        res   = predict_single(row['text'], model)
        pred  = res['pred_label']
        true  = int(row['label'])
        is_ok = (pred == true)

        # FIX: Calculate OOI count dynamically since it's no longer in 'res'
        ooi_count = sum(1 for d in res['metric_details'] if d.get('ooi', False))
        has_ooi = ooi_count > 0

        if has_ooi:
            ooi_total   += 1
            ooi_correct += int(is_ok)
        else:
            non_ooi_total   += 1
            non_ooi_correct += int(is_ok)

        for d in res['metric_details']:
            if d.get('ooi', False):
                ooi_freq[d['metric']] += 1

    print(f'\n  OOI texts  : {ooi_total} | Accuracy on OOI texts    : {ooi_correct/ooi_total:.2%}' if ooi_total else '  No OOI texts detected.')
    print(f'  Non-OOI    : {non_ooi_total} | Accuracy on non-OOI texts: {non_ooi_correct/non_ooi_total:.2%}' if non_ooi_total else '')

    print(f'\n  OOI Frequency by Metric (across all test examples):')
    print(f'  {"Metric":<28} | {"OOI Count":>10}')
    print(f'  {"-" * 28}-+-{"":->10}')
    for m, cnt in sorted(ooi_freq.items(), key=lambda x: -x[1]):
        print(f'  {m:<28} | {cnt:>10}')
    print('\n' + '=' * 90)

# FIX: Run only on the filtered 401-500 word test set (avoids running on 24k texts)
ooi_analysis_report(category_test_df, learned_model)

# %%
# ============================================================
# CELL 9: Single-Text Prediction Interface (Updated)
# ============================================================

def predict_text(text: str, model: dict = None, pickle_path: str = None):
    # Dynamically build the string using f-strings and curly braces
    if pickle_path is None:
        pickle_path = f'master_dynamic_thresholds_{GLOBAL_MIN_WORDS}_to_{GLOBAL_MAX_WORDS}_words.pkl'
        
    if model is None:
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)

    res = predict_single(text, model)
    LABEL = {0: 'Human', 1: 'AI'}

    print('\n' + '=' * 90)
    print('  SINGLE TEXT CLASSIFICATION — Majority + Sigmoid')
    print('=' * 90)
    print(f'  Input text  : "{text[:120]}..."')
    print(f'  Word count  : {res["word_count"]} | Bin: {res["current_bin"]}')
    print(f'  Active metrics: {res["active_total"]}/11')
    print('-' * 90)
    print(f'  {"Metric":<28} | {"Value":>8} | {"Active":<7} | Vote AI | OOI')
    print(f'  {"-"*28}-+-{"":->8}-+-{"":-<7}-+-{"":-<8}-+-{"":-<5}')
    for d in res['metric_details']:
        ooi_flag = '⚠️ OOI' if d['ooi'] else ''
        print(f'  {d["metric"]:<28} | {d["value"]:>8.4f} | {"✓" if d["active"] else "✗":<7} | {"Yes" if d["vote_ai"] else "No":<8} | {ooi_flag}')
    print('-' * 90)
    print(f'  Majority Vote for AI    : {res["active_votes_ai"]}/{res["active_total"]}')
    print(f'  Sigmoid AI Probability  : {res["sigmoid_prob_ai"]:.4f}')
    print(f'  ──────────────────────────────────────────')
    print(f'  PREDICTION: 🤖 {LABEL[res["pred_label"]]:^10} (label={res["pred_label"]})')
    print('=' * 90)
    return res

# === DEMO ===
human_sample = (
    "I went out this morning. The light was strange — orange at the edges, pale in "
    "the middle, like someone had left the sun on too long. My neighbor waved at me. "
    "I couldn't remember if I'd fed the cat. Probably not. She was inside, looking "
    "at me through the glass, not in a friendly way."
)

ai_sample = (
    "Artificial intelligence refers to the simulation of human intelligence processes "
    "by machine learning systems. These processes include learning, reasoning, and self-correction. "
    "Furthermore, AI encompasses natural language processing, image recognition, and automated "
    "decision-making. Additionally, these systems are increasingly being deployed across "
    "healthcare, finance, transportation, and education sectors."
)

print('\n--- Human-authored sample ---')
predict_text(human_sample, learned_model)

print('\n--- AI-authored sample ---')
predict_text(ai_sample, learned_model)

# %%
# ============================================================
# CELL 10: Discriminability Ranking Report
# Across all bins — which metrics are actually useful?
# ============================================================

def discriminability_ranking(model: dict):
    print('=' * 90)
    print('METRIC DISCRIMINABILITY RANKING (Averaged Across All Bins)'.center(90))
    print('=' * 90)
    print('  Higher = better separation between Human and AI distributions per bin.')
    print('  Metrics below the cutoff are suppressed (✗) during that bin\'s inference.\n')

    disc_accumulator = {m: [] for m in METRICS}
    active_count     = {m: 0  for m in METRICS}
    total_bins       = 0

    for cat, cat_data in model.items():
        if cat in ('_global', '_meta', '_logistic_regression') or not isinstance(cat_data, dict):
            continue
        total_bins += 1
        for m in METRICS:
            if m in cat_data:
                d = cat_data[m].get('discriminability', 0.0)
                disc_accumulator[m].append(d)
                if cat_data[m].get('active', False):
                    active_count[m] += 1

    avg_disc = {m: np.mean(v) if v else 0.0 for m, v in disc_accumulator.items()}
    ranked   = sorted(avg_disc.items(), key=lambda x: -x[1])

    print(f'  {"Rank":<6} {"Metric":<28} | {"Avg Disc":>9} | {"Active in N Bins":>17} | Recommendation')
    print(f'  {"-"*6} {"-"*28}-+-{"":->9}-+-{"":->17}-+-{"":-<20}')
    for rank, (m, disc) in enumerate(ranked, 1):
        act = active_count[m]
        rec = '✅ Keep' if disc >= 0.3 else ('⚠️  Borderline' if disc >= 0.15 else '❌ Consider dropping')
        print(f'  {rank:<6} {m:<28} | {disc:>9.4f} | {act:>17} | {rec}')

    print('\n' + '=' * 90)


discriminability_ranking(learned_model)

# %%
# # ============================================================
# # CELL 11: Final Performance Comparison (Majority vs. Logistic vs. Final)
# # ============================================================
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

# def run_comparative_report(test_dataframe: pd.DataFrame, model: dict):
#     print('=' * 95)
#     print('FINAL PERFORMANCE COMPARISON REPORT'.center(95))
#     print('=' * 95)

#     true_labels = []
#     maj_preds   = [] # Strict Majority Vote
#     log_preds   = [] # Strict Sigmoid Threshold (0.5)
#     final_preds = [] # The Actual Output (Logistic Override)
#     log_probs   = []

#     for _, row in test_dataframe.iterrows():
#         res = predict_single(row['text'], model)
#         true_labels.append(int(row['label']))

#         # 1. Majority Voting Result
#         maj_preds.append(res['majority_vote'])

#         # 2. Logistic Regression Result
#         log_preds.append(1 if res['sigmoid_prob_ai'] >= 0.5 else 0)
#         log_probs.append(res['sigmoid_prob_ai'])

#         # 3. Final Combined Output (As currently implemented in your Cell 5)
#         final_preds.append(res['pred_label'])

#     def print_summary(title, y_true, y_pred, y_prob=None):
#         acc = accuracy_score(y_true, y_pred)
#         prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

#         print(f"\n>>> {title}")
#         print(f"    Accuracy  : {acc:.4f} | Precision : {prec:.4f}")
#         print(f"    Recall    : {rec:.4f} | F1-Score  : {f1:.4f}")
#         if y_prob:
#             auc = roc_auc_score(y_true, y_prob)
#             print(f"    ROC-AUC   : {auc:.4f}")

#         cm = confusion_matrix(y_true, y_pred)
#         print(f"    Confusion Matrix:")
#         print(f"                Pred H  Pred AI")
#         print(f"    True Human  {cm[0,0]:>6}  {cm[0,1]:>7}")
#         print(f"    True AI     {cm[1,0]:>6}  {cm[1,1]:>7}")

#     # Print the three requested blocks
#     print_summary("1. MAJORITY VOTING METRICS", true_labels, maj_preds)
#     print_summary("2. LOGISTIC REGRESSION METRICS", true_labels, log_preds, log_probs)
#     print_summary("3. FINAL COMBINED OUTPUT (Current Override System)", true_labels, final_preds, log_probs)

#     print('\n' + '=' * 95)

# # Execute only on the 401-500 word category
# run_comparative_report(category_test_df, learned_model)

# ============================================================
# CELL 11: Final Performance Comparison (Majority vs. Logistic vs. Final)
# ============================================================
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def run_comparative_report(test_dataframe: pd.DataFrame, model: dict):
    # --- YOUR CHANGE IS APPLIED HERE ---
    total_train_samples = GLOBAL_TARGET_PER_CLASS * 2
    
    print('=' * 95)
    print(f'FINAL PERFORMANCE COMPARISON REPORT ({total_train_samples} TRAIN EXAMPLES)'.center(95))
    print('=' * 95)
    # -----------------------------------

    true_labels, maj_preds, log_preds, final_preds, log_probs = [], [], [], [], []

    for _, row in test_dataframe.iterrows():
        res = predict_single(row['text'], model)
        true_labels.append(int(row['label']))
        maj_preds.append(res['majority_vote'])
        log_preds.append(1 if res['sigmoid_prob_ai'] >= 0.5 else 0)
        log_probs.append(res['sigmoid_prob_ai'])
        final_preds.append(res['pred_label'])

    def print_summary(title, y_true, y_pred, y_prob=None):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        print(f"\n>>> {title}")
        print(f"    Accuracy  : {acc:.4f} | Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f} | F1-Score  : {f1:.4f}")
        if y_prob: print(f"    ROC-AUC   : {roc_auc_score(y_true, y_prob):.4f}")
        cm = confusion_matrix(y_true, y_pred)
        print(f"    Confusion Matrix:\n                Pred H  Pred AI\n    True Human  {cm[0,0]:>6}  {cm[0,1]:>7}\n    True AI     {cm[1,0]:>6}  {cm[1,1]:>7}")

    print_summary("1. MAJORITY VOTING METRICS", true_labels, maj_preds)
    print_summary("2. LOGISTIC REGRESSION METRICS", true_labels, log_preds, log_probs)
    print_summary("3. FINAL COMBINED OUTPUT (Current Override System)", true_labels, final_preds, log_probs)
    print('\n' + '=' * 95)

run_comparative_report(category_test_df, learned_model)

# %%
# # ----------------------------------------- Different Test Sets --------------------------------------------------------------------------
# # 1. Identify texts used in training (1,500 Human + 1,500 AI)
# used_text = set(train_df['text'])

# # 2. Find ALL other available texts in the 401-500 word range NOT used in training
# # 2. Find ALL other available texts in the 501+ word range
# remaining_pool = df[
#     (df['text'].apply(lambda t: GLOBAL_MIN_WORDS <= len(str(t).split()) <= GLOBAL_MAX_WORDS)) & # Changed range
#     (~df['text'].isin(used_text))
# ].copy()

# # 3. Create a perfectly balanced FRESH set (500 Human, 500 AI)
# # Using random_state=99 ensures these are different from your previous test
# fresh_humans = remaining_pool[remaining_pool['label'] == 0].sample(n=500, random_state=99)
# fresh_ais = remaining_pool[remaining_pool['label'] == 1].sample(n=500, random_state=99)

# # 4. OVERWRITE the variable name Cell 6 and Cell 11 look for
# category_test_df = pd.concat([fresh_humans, fresh_ais]).sample(frac=1, random_state=99).reset_index(drop=True)

# print(f"✅ 'category_test_df' has been replaced with {len(category_test_df)} BRAND NEW examples.")
# print("Now scroll up and Run Cell 6 and Cell 11 to see the updated Matrix.")

# ----------------------------------------- Different Test Sets --------------------------------------------------------------------------
# 1. Identify texts used in training (1,500 Human + 1,500 AI)
used_text = set(train_df['text'])

# 2. Find ALL other available texts in the 401-500 word range NOT used in training
# 2. Find ALL other available texts in the 501+ word range
remaining_pool = df[
    (df['text'].apply(lambda t: GLOBAL_MIN_WORDS <= len(str(t).split()) <= GLOBAL_MAX_WORDS)) & # Changed range
    (~df['text'].isin(used_text))
].copy()

# Find the maximum balanced size we can extract
available_humans = len(remaining_pool[remaining_pool['label'] == 0])
available_ais = len(remaining_pool[remaining_pool['label'] == 1])
target_n = min(available_humans, available_ais, 500)

# 3. Create a perfectly balanced FRESH set (Up to 500 Human, 500 AI)
# Using random_state=99 ensures these are different from your previous test
fresh_humans = remaining_pool[remaining_pool['label'] == 0].sample(n=target_n, random_state=99)
fresh_ais = remaining_pool[remaining_pool['label'] == 1].sample(n=target_n, random_state=99)

# 4. OVERWRITE the variable name Cell 6 and Cell 11 look for
category_test_df = pd.concat([fresh_humans, fresh_ais]).sample(frac=1, random_state=99).reset_index(drop=True)

print(f"✅ 'category_test_df' has been replaced with {len(category_test_df)} BRAND NEW examples (Balanced).")
print("Now scroll up and Run Cell 6 and Cell 11 to see the updated Matrix.")

# %%
# # ============================================================
# # CELL 13: Dedicated Independent Evaluation Function
# # ============================================================

# def evaluate_custom_set(test_df, model, set_name="Custom Test Set"):
#     """
#     Evaluates the model on any provided dataframe without
#     affecting global variables.
#     """
#     print('=' * 80)
#     print(f'EVALUATION REPORT: {set_name.upper()}'.center(80))
#     print('=' * 80)

#     true_labels = []
#     predictions = []

#     # Run predictions
#     for _, row in test_df.iterrows():
#         res = predict_single(row['text'], model)
#         true_labels.append(int(row['label']))
#         predictions.append(res['pred_label'])

#     # Calculate Metrics
#     acc = accuracy_score(true_labels, predictions)
#     cm = confusion_matrix(true_labels, predictions)

#     # Print Summary
#     print(f"  Total Examples   : {len(test_df)}")
#     print(f"  Overall Accuracy : {acc:.4f}")
#     print('-' * 80)
#     print(f"  Confusion Matrix:")
#     print(f"                Pred H   Pred AI")
#     print(f"  True Human    {cm[0,0]:>6}   {cm[0,1]:>7}")
#     print(f"  True AI       {cm[1,0]:>6}   {cm[1,1]:>7}")
#     print('=' * 80)

# # --- EXAMPLE USAGE (You can comment these out or keep them) ---

# # A. The old 20% slice (Your 1,764 examples)
# old_test_slice = test_df[(test_df['_word_count'] >= 401) & (test_df['_word_count'] <= 500)]
# evaluate_custom_set(old_test_slice, learned_model, "Original 20% Test Slice")

# # B. The new 1,000 balanced examples
# # (Assuming you already ran the 'Data Swapper' code we discussed)
# evaluate_custom_set(category_test_df, learned_model, "New Balanced 1,000 Set")

# %%
# # ============================================================
# # CELL 13: Multi-Set Balanced Testing Engine
# # ============================================================

# def run_multi_test_experiment(original_df, model, n_sets=10, samples_per_set=1000):
#     """
#     Generates and evaluates multiple balanced test sets.
#     """
#     # 1. Filter for the 401-500 word category
#     pool = original_df[original_df['text'].apply(lambda t: len(str(t).split()) >= 501)].copy()

#     results = []
#     print(f"🚀 Starting Multi-Set Evaluation ({n_sets} sets of {samples_per_set} samples each)")
#     print(f"{'Set ID':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
#     print("-" * 55)

#     for i in range(n_sets):
#         # Create a unique balanced set using a unique seed for each loop
#         seed = 200 + i
#         h_sample = pool[pool['label'] == 0].sample(n=samples_per_set//2, random_state=seed)
#         a_sample = pool[pool['label'] == 1].sample(n=samples_per_set//2, random_state=seed)
#         current_test_df = pd.concat([h_sample, a_sample]).sample(frac=1, random_state=seed)

#         # Evaluate
#         y_true = []
#         y_pred = []
#         for _, row in current_test_df.iterrows():
#             res = predict_single(row['text'], model)
#             y_true.append(int(row['label']))
#             y_pred.append(res['pred_label'])

#         # Calculate stats
#         acc = accuracy_score(y_true, y_pred)
#         prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

#         results.append(acc)
#         print(f"Set {i+1:<5} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}")

#     print("-" * 55)
#     print(f"AVERAGE ACCURACY: {sum(results)/len(results):.4f}")
#     print(f"VARIANCE:         {pd.Series(results).std():.4f}")

# # RUN THE EXPERIMENT
# run_multi_test_experiment(df, learned_model, n_sets=10, samples_per_set=1000)

# # ============================================================
# # CELL 13: Multi-Set Balanced Testing Engine (Sleep-Safe Version)
# # ============================================================

# def run_multi_test_experiment(original_df, model, n_sets=10, samples_per_set=1000):
#     """
#     Generates and evaluates multiple balanced test sets.
#     """
#     # 1. Filter for the STRICT 501-1000 word category
#     # This ensures testing matches exactly what the model learned.
#     # pool = original_df[original_df['text'].apply(lambda t: 501 <= len(str(t).split()) <= 1000)].copy()
#     pool = original_df[original_df['text'].apply(lambda t: GLOBAL_MIN_WORDS <= len(str(t).split()) <= GLOBAL_MAX_WORDS)].copy()

#     # 2. Safety Check: Ensure we have enough Humans and AI to sample
#     available_h = len(pool[pool['label'] == 0])
#     available_a = len(pool[pool['label'] == 1])

#     needed_per_class = samples_per_set // 2

#     if available_h < needed_per_class or available_a < needed_per_class:
#         new_total = min(available_h, available_a) * 2
#         print(f"⚠️ Warning: Not enough samples for {samples_per_set}. Reducing to {new_total} per set.")
#         samples_per_set = new_total
#         needed_per_class = samples_per_set // 2

#     if samples_per_set == 0:
#         print("❌ CRITICAL ERROR: No data found in 501-1000 range. Skipping test.")
#         return

#     results = []
#     print(f"🚀 Starting Multi-Set Evaluation ({n_sets} sets of {samples_per_set} samples each)")
#     print(f"{'Set ID':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
#     print("-" * 55)

#     for i in range(n_sets):
#         seed = 200 + i
#         # Balanced sampling from the pool
#         h_sample = pool[pool['label'] == 0].sample(n=needed_per_class, random_state=seed)
#         a_sample = pool[pool['label'] == 1].sample(n=needed_per_class, random_state=seed)
#         current_test_df = pd.concat([h_sample, a_sample]).sample(frac=1, random_state=seed)

#         y_true = []
#         y_pred = []
#         for _, row in current_test_df.iterrows():
#             res = predict_single(row['text'], model)
#             y_true.append(int(row['label']))
#             y_pred.append(res['pred_label'])

#         acc = accuracy_score(y_true, y_pred)
#         prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

#         results.append(acc)
#         print(f"Set {i+1:<5} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}")

#     print("-" * 55)
#     print(f"AVERAGE ACCURACY: {sum(results)/len(results):.4f}")
#     print(f"VARIANCE:         {pd.Series(results).std():.4f}")

# # RUN THE EXPERIMENT
# run_multi_test_experiment(df, learned_model, n_sets=10, samples_per_set=1000)


# ============================================================
# CELL 13: Multi-Set Balanced Testing Engine (Sleep-Safe Version)
# ============================================================

def run_multi_test_experiment(original_df, model, n_sets=10, samples_per_set=1000):
    """
    Generates and evaluates multiple balanced test sets.
    """
    # 1. Filter for the STRICT GLOBAL_MIN_WORDS-GLOBAL_MAX_WORDS word category AND exclude training data
    used_text = set(train_df['text'])
    pool = original_df[
        (original_df['text'].apply(lambda t: GLOBAL_MIN_WORDS <= len(str(t).split()) <= GLOBAL_MAX_WORDS)) &
        (~original_df['text'].isin(used_text))
    ].copy()

    # 2. Safety Check: Ensure we have enough Humans and AI to sample
    available_h = len(pool[pool['label'] == 0])
    available_a = len(pool[pool['label'] == 1])

    needed_per_class = samples_per_set // 2

    if available_h < needed_per_class or available_a < needed_per_class:
        new_total = min(available_h, available_a) * 2
        print(f"⚠️ Warning: Not enough samples for {samples_per_set}. Reducing to {new_total} per set.")
        samples_per_set = new_total
        needed_per_class = samples_per_set // 2

    if samples_per_set == 0:
        print(f"❌ CRITICAL ERROR: No data found in {GLOBAL_MIN_WORDS}-{GLOBAL_MAX_WORDS} range. Skipping test.")
        return

    results = []
    print(f"🚀 Starting Multi-Set Evaluation ({n_sets} sets of {samples_per_set} samples each)")
    print(f"{'Set ID':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)

    for i in range(n_sets):
        seed = 200 + i
        # Balanced sampling from the pool
        h_sample = pool[pool['label'] == 0].sample(n=needed_per_class, random_state=seed)
        a_sample = pool[pool['label'] == 1].sample(n=needed_per_class, random_state=seed)
        current_test_df = pd.concat([h_sample, a_sample]).sample(frac=1, random_state=seed)

        y_true = []
        y_pred = []
        for _, row in current_test_df.iterrows():
            res = predict_single(row['text'], model)
            y_true.append(int(row['label']))
            y_pred.append(res['pred_label'])

        acc = accuracy_score(y_true, y_pred)
        prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

        results.append(acc)
        print(f"Set {i+1:<5} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}")

    print("-" * 55)
    print(f"AVERAGE ACCURACY: {sum(results)/len(results):.4f}")
    print(f"VARIANCE:         {pd.Series(results).std():.4f}")

# RUN THE EXPERIMENT
# RUN THE EXPERIMENT
run_multi_test_experiment(
    original_df=df, 
    model=learned_model, 
    n_sets=GLOBAL_MULTI_SET_N, 
    samples_per_set=GLOBAL_MULTI_SET_SAMPLES
)


