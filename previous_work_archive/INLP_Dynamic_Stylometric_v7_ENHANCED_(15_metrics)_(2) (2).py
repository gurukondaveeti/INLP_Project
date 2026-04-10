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
# CELL 1: Install & Setup (Updated with GPU & GPT-2)
# ============================================================
# !pip install spacy scikit-learn pandas numpy scipy tqdm transformers torch --quiet
# !python -m spacy download en_core_web_sm

import pandas as pd
import numpy as np
import spacy
import scipy.stats as st
import pickle
import math
import warnings
import torch
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

warnings.filterwarnings('ignore')

# 1. Setup GPU Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Compute Device: {device.upper()}")

# 2. Load spaCy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])

# 3. Load GPT-2 Model & Tokenizer for Perplexity and BPE
print("Loading GPT-2 model into VRAM...")
lm_id = "gpt2"
lm_tokenizer = GPT2TokenizerFast.from_pretrained(lm_id)
lm_model = GPT2LMHeadModel.from_pretrained(lm_id).to(device)
lm_model.eval() # Freeze model for inference only

# Enable tqdm for pandas progress bars
tqdm.pandas()

print('✅ Environment ready. spaCy and GPT-2 loaded successfully.')

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
# CELL 3: 12-Metric Optimized Stylometric Extraction
# ============================================================
import numpy as np
import math
from collections import Counter

# The 12 optimal features based on Discriminability Ranking
METRICS = [
    'entropy', 'perplexity', 'unique_words_ratio', 'hapax_ratio',
    'pronoun_ratio', 'punct_density', 'lexical_density',
    'sentence_length_deviation', 'sw_gradient', 'avg_word_length',
    'burstiness', 'ttr_slope'
]

# Disable parser and ner for a massive speed boost
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'lemmatizer'])
stopwords = nlp.Defaults.stop_words

def extract_stylometric_features(text: str) -> dict:
    doc = nlp(str(text))
    words = [token.text.lower() for token in doc if token.is_alpha]
    n_words = len(words)
    
    # Fallback for sentences if parser is disabled
    sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]

    # --- 1. Information Theory & Lexical Richness ---
    freq = Counter(words)
    entropy = 0.0
    if n_words > 0:
        probs = [count / n_words for count in freq.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        
    perplexity = 2 ** entropy if entropy > 0 else 0.0
    unique_words_ratio = len(freq) / n_words if n_words > 0 else 0.0
    hapax_ratio = sum(1 for c in freq.values() if c == 1) / n_words if n_words > 0 else 0.0

    # --- 2. POS & Density Markers ---
    pronoun_ratio = sum(1 for t in doc if t.pos_ == 'PRON') / len(doc) if len(doc) > 0 else 0.0
    punct_density = sum(1 for t in doc if t.is_punct) / len(doc) if len(doc) > 0 else 0.0
    lexical_density = sum(1 for t in doc if t.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}) / n_words if n_words > 0 else 0.0
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

    # --- 3. Structural Predictability ---
    sent_lens = [len([t for t in s if t.is_alpha]) for s in sents]
    sentence_length_deviation = float(np.std(sent_lens)) if sent_lens else 0.0
    
    burstiness = 0.0
    if len(sent_lens) > 4:
        chunks = np.array_split(sent_lens, 3)
        burstiness = float(np.std([np.std(c) for c in chunks if len(c) > 0]))

    # --- 4. Trajectory Features ---
    ttr_slope = 0.0
    if n_words >= 100:
        ttr_vals = [len(set(words[i:i+50])) / 50 for i in range(0, n_words - 50, 25)]
        ttr_slope = float(np.polyfit(np.arange(len(ttr_vals)), ttr_vals, 1)[0]) if len(ttr_vals) >= 2 else 0.0

    sw_gradient = 0.0
    if n_words > 40:
        mid = n_words // 2
        sw_gradient = abs((sum(1 for w in words[:mid] if w in stopwords)/mid) -
                          (sum(1 for w in words[mid:] if w in stopwords)/(n_words-mid)))

    return {
        'entropy': entropy, 'perplexity': perplexity,
        'unique_words_ratio': unique_words_ratio, 'hapax_ratio': hapax_ratio,
        'pronoun_ratio': pronoun_ratio, 'punct_density': punct_density,
        'lexical_density': lexical_density, 'sentence_length_deviation': sentence_length_deviation,
        'sw_gradient': sw_gradient, 'avg_word_length': avg_word_length,
        'burstiness': burstiness, 'ttr_slope': ttr_slope
    }

print('✅ 12-Metric Fast Extraction loaded. (NER and Parser disabled for speed)')

# %%
# ============================================================
# CELL 4: Dynamic Target Controller & Enhanced Training
# ============================================================
import pickle
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# 🎛️ DYNAMIC CONTROLS: Change these to run different experiments!
TARGET_MIN_WORDS = 201
TARGET_MAX_WORDS = 300
TARGET_SAMPLES   = 1500

def train_dynamic_model(df: pd.DataFrame, min_words: int, max_words: int, max_samples: int):
    print('=' * 70)
    print(f'TRAINING DYNAMIC MODEL: {min_words} to {max_words} WORDS'.center(70))
    print('=' * 70)

    df = df.copy().reset_index(drop=True)
    df['_word_count'] = df['text'].apply(lambda t: len(str(t).split()))

    # Filter strictly to the desired range
    category_df = df[(df['_word_count'] >= min_words) & (df['_word_count'] <= max_words)].copy()

    humans = category_df[category_df['label'] == 0]
    ais = category_df[category_df['label'] == 1]

    # Auto-adjust if the bin doesn't have enough data
    final_n = min(len(humans), len(ais), max_samples)
    if final_n < 10:
        print(f"❌ ERROR: Only {final_n} balanced samples found. Try a wider word range.")
        return None

    humans_sampled = humans.sample(n=final_n, random_state=42)
    ais_sampled = ais.sample(n=final_n, random_state=42)
    train_pool = pd.concat([humans_sampled, ais_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Step 1: Extracted {final_n} Human and {final_n} AI texts.")
    print(f"Step 2: Extracting {len(METRICS)} optimized features...")
    
    feats_df = train_pool['text'].apply(extract_stylometric_features).apply(pd.Series)
    combined = pd.concat([train_pool, feats_df], axis=1)

    print('Step 3: Calculating Profile Statistics...')
    global_stats = combined.groupby('label')[METRICS].agg(['mean','std']).fillna(0.001)

    final_model = {}
    cat_data = {}
    
    for m in METRICS:
        h_mu = float(global_stats.loc[0,(m,'mean')])
        h_sig = max(float(global_stats.loc[0,(m,'std')]), 0.001)
        a_mu = float(global_stats.loc[1,(m,'mean')])
        a_sig = max(float(global_stats.loc[1,(m,'std')]), 0.001)
        
        pooled_sigma = math.sqrt((h_sig**2 + a_sig**2) / 2.0)
        disc = abs(h_mu - a_mu) / pooled_sigma if pooled_sigma > 1e-9 else 0.0
        
        h_samp = combined.loc[combined.label==0, m].tolist()
        a_samp = combined.loc[combined.label==1, m].tolist()
        
        # Simplified Youden threshold for dynamic speed
        mid = (np.mean(h_samp + a_samp)) if (h_samp + a_samp) else 0.5
        yd = (np.mean(a_samp) > np.mean(h_samp)) if (a_samp and h_samp) else True

        cat_data[m] = {'h_mu': h_mu, 'h_sig': h_sig, 'a_mu': a_mu, 'a_sig': a_sig,
                       'discriminability': disc, 'active': disc >= 0.15,
                       'youden_threshold': mid, 'youden_direction': yd}

    final_model['_global'] = cat_data
    final_model[min_words] = cat_data 

    print('Step 4: Training Enhanced Non-Linear Logistic Regression...')
    X_raw = feats_df[METRICS].fillna(0)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_raw)

    log_reg = LogisticRegression(max_iter=2000, class_weight={0: 1.25, 1: 1.0}, C=0.5, random_state=42)
    log_reg.fit(X_poly, train_pool['label'])

    final_model['_logistic_regression'] = log_reg
    final_model['_poly_transformer'] = poly 
    final_model['_meta'] = {'metrics': METRICS, 'category_trained': f'{min_words}_{max_words}'}

    save_path = f'master_dynamic_{min_words}_{max_words}.pkl'
    with open(save_path, 'wb') as f: pickle.dump(final_model, f)
    
    print(f'✅ Model saved → {save_path}')
    return final_model

learned_model = train_dynamic_model(train_df, TARGET_MIN_WORDS, TARGET_MAX_WORDS, TARGET_SAMPLES)

# %%
# ============================================================
# CELL 5: Dynamic Inference Engine (12 Metrics + Poly)
# ============================================================
import scipy.stats as st

EPS   = 1e-6
OOI_Z = 2.5

def _ooi_fallback_vote(val, m_data):
    d_h = abs(val - m_data['h_mu']) / max(m_data['h_sig'], 1e-6)
    d_a = abs(val - m_data['a_mu']) / max(m_data['a_sig'], 1e-6)
    if d_h + d_a == 0: return 0.5
    return float(d_h / (d_h + d_a))

def predict_single(text: str, model: dict) -> dict:
    word_count = len(str(text).split())
    
    # Since we train dynamically on one bin at a time, we just use the global data for that bin
    m_data   = model['_global']
    features = extract_stylometric_features(text)

    metric_details = []
    
    # Ensure feature vector order perfectly matches METRICS
    feature_vector = [features[m] for m in METRICS]

    for m_name in METRICS:
        val = features[m_name]
        md = m_data[m_name]

        z_h = abs(val - md['h_mu']) / max(md['h_sig'], 1e-6)
        z_a = abs(val - md['a_mu']) / max(md['a_sig'], 1e-6)
        is_ooi = (z_h > OOI_Z) and (z_a > OOI_Z)

        active = md.get('active', True)
        yt  = md.get('youden_threshold', (md['h_mu'] + md['a_mu']) / 2)
        yd  = md.get('youden_direction', True)
        vote_ai = ((val > yt) == yd)

        metric_details.append({
            'metric'   : m_name,
            'value'    : val,
            'vote_ai'  : vote_ai,
            'active'   : active,
            'ooi'      : is_ooi
        })

    # 1. Majority Vote Logic
    active_votes_ai  = sum(1 for d in metric_details if d['active'] and d['vote_ai'])
    active_total     = sum(1 for d in metric_details if d['active'])
    majority_vote_ai = 1 if active_votes_ai >= max(1, math.ceil(active_total / 2)) else 0

    # 2. Logistic Regression (Poly + Sigmoid) Logic
    log_reg = model.get('_logistic_regression')
    poly    = model.get('_poly_transformer')

    if log_reg and poly:
        poly_features = poly.transform([feature_vector])
        sigmoid_prob_ai = log_reg.predict_proba(poly_features)[0][1]
        sigmoid_pred = 1 if sigmoid_prob_ai >= 0.5 else 0
    else:
        sigmoid_prob_ai = 0.5
        sigmoid_pred = majority_vote_ai

    return {
        'pred_label'      : sigmoid_pred,
        'majority_vote'   : majority_vote_ai,
        'sigmoid_prob_ai' : sigmoid_prob_ai,
        'word_count'      : word_count,
        'metric_details'  : metric_details,
        'active_votes_ai' : active_votes_ai,
        'active_total'    : active_total
    }

# %%
# ============================================================
# CELL 6: Full Evaluation (Targeted Bin)
# ============================================================
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def run_full_evaluation(test_dataframe: pd.DataFrame, model: dict, show_examples: int = 5):
    LABEL = {0: 'Human', 1: 'AI'}
    predictions, true_labels, prob_scores = [], [], []

    test_dataframe = test_dataframe.reset_index(drop=True)
    n = len(test_dataframe)

    print('\n' + '=' * 110)
    print('FULL EVALUATION — Targeted Dynamic Range'.center(110))
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

        if idx < show_examples:
            snippet = str(row['text'])[:100].replace('\n', ' ')
            print(f'\n{"=" * 110}')
            print(f'EXAMPLE {idx+1:>3} | Word Count: {result["word_count"]:>5} | Active Metrics: {result["active_total"]}/{len(METRICS)}')
            print(f'Text Snippet: "{snippet}..."')
            print(f'{"-" * 110}')
            print(f'{"Metric":<28} | {"Value":>8} | {"Active":<7} | {"Vote AI":<8} | {"OOI"}')
            print(f'{"-" * 28}-+-{"":->8}-+-{"":-<7}-+-{"":-<8}-+-{"":-<5}')

            for d in result['metric_details']:
                active_str = '✓' if d['active'] else '✗'
                vote_str   = 'Yes' if d['vote_ai'] else 'No'
                ooi_flag   = '⚠️ OOI' if d['ooi'] else ''
                print(f'{d["metric"]:<28} | {d["value"]:>8.4f} | {active_str:<7} | {vote_str:<8} | {ooi_flag}')

            print(f'{"-" * 110}')
            print(f'Majority Vote AI Count: {result["active_votes_ai"]}/{result["active_total"]} | Sigmoid AI Prob: {result["sigmoid_prob_ai"]:.4f}')
            print(f'True: {LABEL[true_label]:<7} | Predicted: {LABEL[pred_label]:<7} --> {marker}')

    acc  = accuracy_score(true_labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(true_labels, prob_scores)
    except:
        auc = float('nan')
        
    cm = confusion_matrix(true_labels, predictions)

    print(f'\n{"=" * 80}')
    print('FINAL PERFORMANCE SUMMARY'.center(80))
    print('=' * 80)
    print(f'  {"Overall Accuracy":<25} {acc:.4f}')
    print(f'  {"Macro Precision":<25} {prec:.4f}')
    print(f'  {"Macro Recall":<25} {rec:.4f}')
    print(f'  {"Macro F1-Score":<25} {f1:.4f}')
    print(f'  {"ROC-AUC (Sigmoid)":<25} {auc:.4f}')
    print(f'  {"Total Examples Evaluated":<25} {n}')
    print('=' * 80)
    print(f'\n  Confusion Matrix:')
    print(f'  {"":>12} Pred Human  Pred AI')
    print(f'  {"True Human":<12} {cm[0,0]:>10}  {cm[0,1]:>7}')
    print(f'  {"True AI":<12} {cm[1,0]:>10}  {cm[1,1]:>7}')
    print('=' * 80)

    return predictions, prob_scores, true_labels

# ==========================================
# RUN EVALUATION ON DYNAMIC SUBSET
# ==========================================
print(f"Filtering test set for {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS} word lengths...")
test_df['_word_count'] = test_df['text'].apply(lambda t: len(str(t).split()))
category_test_df = test_df[(test_df['_word_count'] >= TARGET_MIN_WORDS) & 
                           (test_df['_word_count'] <= TARGET_MAX_WORDS)].copy()

print(f"Filtered Test Set Size: {len(category_test_df)} texts")
preds, probs, gts = run_full_evaluation(category_test_df, learned_model, show_examples=3)

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
# CELL 9: Single-Text Prediction Interface (Updated for 12 Features)
# ============================================================
import pickle

def predict_text(text: str, model: dict = None, pickle_path: str = 'master_dynamic_201_300.pkl'):
    if model is None:
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)

    res = predict_single(text, model)
    LABEL = {0: 'Human', 1: 'AI'}

    print('\n' + '=' * 90)
    print('  SINGLE TEXT CLASSIFICATION — Majority + Poly-Sigmoid')
    print('=' * 90)
    print(f'  Input text  : "{text[:120]}..."')
    print(f'  Word count  : {res["word_count"]}')
    
    # Removed current_bin and updated to 12 metrics
    print(f'  Active metrics: {res["active_total"]}/12') 
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
    print('=' * 95)
    print('FINAL PERFORMANCE COMPARISON REPORT (3000 TRAIN EXAMPLES)'.center(95))
    print('=' * 95)

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
# ----------------------------------------- Different Test Sets --------------------------------------------------------------------------
# 1. Identify texts used in training (1,500 Human + 1,500 AI)
used_text = set(train_df['text'])

# 2. Find ALL other available texts in the 401-500 word range NOT used in training
# 2. Find ALL other available texts in the 501+ word range
remaining_pool = df[
    (df['text'].apply(lambda t: 401 <= len(str(t).split()) <= 500)) & # Changed range
    (~df['text'].isin(used_text))
].copy()

# 3. Create a perfectly balanced FRESH set (500 Human, 500 AI)
# Using random_state=99 ensures these are different from your previous test
fresh_humans = remaining_pool[remaining_pool['label'] == 0].sample(n=500, random_state=99)
fresh_ais = remaining_pool[remaining_pool['label'] == 1].sample(n=500, random_state=99)

# 4. OVERWRITE the variable name Cell 6 and Cell 11 look for
category_test_df = pd.concat([fresh_humans, fresh_ais]).sample(frac=1, random_state=99).reset_index(drop=True)

print(f"✅ 'category_test_df' has been replaced with {len(category_test_df)} BRAND NEW examples.")
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

# ============================================================
# CELL 13: Multi-Set Balanced Testing Engine (Sleep-Safe Version)
# ============================================================

def run_multi_test_experiment(original_df, model, n_sets=10, samples_per_set=1000):
    """
    Generates and evaluates multiple balanced test sets.
    """
    # 1. Filter for the STRICT 501-1000 word category
    # This ensures testing matches exactly what the model learned.
    pool = original_df[original_df['text'].apply(lambda t: 401 <= len(str(t).split()) <= 500)].copy()

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
        print("❌ CRITICAL ERROR: No data found in 501-1000 range. Skipping test.")
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
run_multi_test_experiment(df, learned_model, n_sets=10, samples_per_set=1000)


