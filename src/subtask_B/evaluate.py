"""
SemEval 2024 Task 8 — Subtask B  |  Evaluation Script
=======================================================
Loads trained .pkl files for RoBERTa, DeBERTa-v3, FLAN-T5.
Produces:
  - Individual model metrics (F1, Accuracy, per-class breakdown)
  - Ensemble soft-voting metrics
  - Confusion matrix PNG plots
  - Timestamped JSON results file
  - Pushes everything to HuggingFace Hub

Usage (Kaggle Terminal):
  python evaluate.py                         # interactive
  python evaluate.py --mode individual       # evaluate each model separately
  python evaluate.py --mode ensemble         # ensemble only
  python evaluate.py --mode both             # individual + ensemble (recommended)
  python evaluate.py --mode both --split test
"""

import os, gc, sys, json, glob, argparse, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_score, recall_score, accuracy_score,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ================================================================
# PATHS
# ================================================================
KAGGLE = os.path.exists('/kaggle/working')

if KAGGLE:
    OUTPUT_DIR = '/kaggle/working'
    found = glob.glob('/kaggle/input/**/subtaskB_dev.jsonl', recursive=True)
    DATA_DIR = os.path.dirname(found[0]) if found else '/kaggle/input'
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ['HF_TOKEN'] = HF_TOKEN
        print("HuggingFace token loaded from Kaggle Secrets.")
    except Exception:
        HF_TOKEN = os.environ.get('HF_TOKEN', '')
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN not found. Hub push will be skipped.")
else:
    DATA_DIR   = '.'
    OUTPUT_DIR = '.'
    HF_TOKEN   = os.environ.get('HF_TOKEN', '')

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
PLOTS_DIR      = os.path.join(OUTPUT_DIR, 'plots')
LOGS_DIR       = os.path.join(OUTPUT_DIR, 'logs')
for d in [CHECKPOINT_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ================================================================
# CONFIG
# ================================================================
LABEL2NAME = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
NUM_CLASSES = 6
MAX_LENGTH  = 512
NUM_WORKERS = 0 if os.name == 'nt' else 2
TARGET_NAMES = [f"{i}-{LABEL2NAME[i]}" for i in range(NUM_CLASSES)]

# The models we support in the ensemble
ALL_MODELS = {
    'roberta': {
        'name':     'roberta-base',
        'model_id': 'roberta-base',
        'hf_repo':  'semeval2024-subtaskB-roberta',
    },
    'electra': {
        'name':     'electra-base',
        'model_id': 'google/electra-base-discriminator',
        'hf_repo':  'semeval2024-subtaskB-electra',
    },
    'deberta': {
        'name':     'deberta-v3-base',
        'model_id': 'microsoft/deberta-v3-base',
        'hf_repo':  'semeval2024-subtaskB-deberta',
    },
    'flant5': {
        'name':     'flan-t5-base',
        'model_id': 'google/flan-t5-base',
        'hf_repo':  'semeval2024-subtaskB-flant5',
    },
}

# Results HF repo
RESULTS_HF_REPO = 'semeval2024-subtaskB-results'

# ================================================================
# DATASET
# ================================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts;  self.labels = labels
        self.tokenizer = tokenizer;  self.max_length = max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]
        tids  = self.tokenizer.encode(text, add_special_tokens=False)
        ns    = self.tokenizer.num_special_tokens_to_add(pair=False)
        mc    = self.max_length - ns
        if len(tids) > mc:
            h    = mc // 2
            tids = tids[:h] + tids[-h:]
            text = self.tokenizer.decode(tids)
        enc = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_attention_mask=True, add_special_tokens=True,
        )
        return {
            'input_ids':      torch.tensor(enc['input_ids'],      dtype=torch.long).squeeze(),
            'attention_mask': torch.tensor(enc['attention_mask'], dtype=torch.long).squeeze(),
            'labels':         torch.tensor(label,                 dtype=torch.long),
        }

# ================================================================
# HELPERS
# ================================================================
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]


def find_available_checkpoints():
    """Return list of (key, config, ckpt_dir, pkl_path) for every model that has been trained."""
    available = []
    for key, cfg in ALL_MODELS.items():
        name     = cfg['name']
        ckpt_dir = os.path.join(CHECKPOINT_DIR, name)
        pkl_path = os.path.join(ckpt_dir, f'{name}_model.pkl')
        if os.path.exists(pkl_path):
            available.append((key, cfg, ckpt_dir, pkl_path))
            print(f"  [FOUND] {name:30s}  →  {pkl_path}")
        else:
            print(f"  [MISS]  {name:30s}  (run train.py first)")
    return available


def get_local_model_path(model_id):
    """Checks local paths for a directory containing pre-downloaded model files.
    Searches /kaggle/working/models/ first, then /kaggle/input/.
    """
    clean_id = model_id.split('/')[-1]

    # 0. Check /kaggle/working/models/ (pre-downloaded by notebook)
    working_path = f'/kaggle/working/models/{clean_id}'
    if os.path.exists(os.path.join(working_path, 'config.json')):
        return working_path

    # 1. Search for models/ directory inside any attached Kaggle dataset
    for candidate in glob.glob(f'/kaggle/input/**/models/{clean_id}/config.json', recursive=True):
        return os.path.dirname(candidate)

    # 2. Direct Kaggle dataset name patterns
    search_patterns = [
        f'/kaggle/input/{clean_id}',
        f'/kaggle/input/{clean_id.replace("-", "")}',
        f'/kaggle/input/{model_id.replace("/", "-")}',
    ]
    for pattern in search_patterns:
        if os.path.exists(os.path.join(pattern, 'config.json')):
            return pattern

    # 3. Broad recursive glob fallback
    found = glob.glob(f'/kaggle/input/**/{clean_id}/config.json', recursive=True)
    if found:
        return os.path.dirname(found[0])

    return model_id  # Fallback to internet string


def load_model_from_checkpoint(name, pkl_path, device):
    """Load tokenizer + model from the raw .pkl dictionary.
    
    Priority for loading base weights:
      1. The checkpoint directory itself (training saves tokenizer + model there)
      2. Pre-downloaded models in /kaggle/input/.../models/
      3. Internet download (last resort)
    """
    print(f"  Loading {name} from PKL...")
    checkpoint = torch.load(pkl_path, map_location=device)
    
    model_id = checkpoint['model_id']
    ckpt_dir = os.path.dirname(pkl_path)

    # Determine best source for tokenizer & model architecture
    # Priority: checkpoint dir > pre-downloaded local > internet
    if os.path.exists(os.path.join(ckpt_dir, 'config.json')):
        source = ckpt_dir
        print(f"  [LOCAL] Using checkpoint directory: {source}")
    else:
        source = get_local_model_path(model_id)
        if source != model_id:
            print(f"  [LOCAL] Using pre-downloaded model: {source}")
        else:
            print(f"  [WARN] No local model found, downloading {model_id} from internet...")

    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer


def get_probabilities(model, tokenizer, texts, labels, device, batch_size=8, desc="Inference"):
    """Run model inference, return (preds, probs, true_labels)."""
    ds = TextDataset(texts, labels, tokenizer, MAX_LENGTH)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"  {desc}", leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            with torch.cuda.amp.autocast():
                out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits.float()
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch['labels'].numpy())
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


# ================================================================
# METRICS + PLOTTING
# ================================================================
def compute_metrics(title, preds, true_labels):
    """Print and return full metrics dictionary."""
    macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
    accuracy  = accuracy_score(true_labels, preds)
    pc_f1     = f1_score(true_labels, preds, average=None, zero_division=0)
    pc_prec   = precision_score(true_labels, preds, average=None, zero_division=0)
    pc_rec    = recall_score(true_labels, preds, average=None, zero_division=0)

    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  Macro-F1 : {macro_f1:.4f}")
    print(f"  Accuracy : {accuracy:.4f}")
    print()
    print(classification_report(true_labels, preds,
                                 target_names=TARGET_NAMES, digits=4, zero_division=0))
    cm = confusion_matrix(true_labels, preds)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=TARGET_NAMES, columns=TARGET_NAMES).to_string())

    return {
        'macro_f1':            round(float(macro_f1), 5),
        'accuracy':            round(float(accuracy), 5),
        'per_class_f1':        [round(float(x), 5) for x in pc_f1],
        'per_class_precision': [round(float(x), 5) for x in pc_prec],
        'per_class_recall':    [round(float(x), 5) for x in pc_rec],
        'confusion_matrix':    confusion_matrix(true_labels, preds).tolist(),
    }


def save_confusion_matrix_plot(cm_data, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(np.array(cm_data), annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def save_comparison_plot(individual_results, split, save_path):
    """Bar chart comparing F1 scores across all models + ensemble."""
    models = list(individual_results.keys())
    f1s    = [individual_results[m].get(split, {}).get('macro_f1', 0) for m in models]
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, f1s, color=colors, edgecolor='black', linewidth=0.7)
    ax.set_ylabel('Macro-F1 Score', fontsize=13)
    ax.set_title(f'Model Comparison — {split.upper()} Set', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ================================================================
# HUGGINGFACE PUSH
# ================================================================
def push_results_to_hub(results_dir_files, individual_results, ensemble_results):
    if not HF_TOKEN:
        print("  [HUB] Skipping — no HF_TOKEN.")
        return
    try:
        from huggingface_hub import HfApi
        api  = HfApi(token=HF_TOKEN)
        user = api.whoami()['name']
        repo_id = f"{user}/{RESULTS_HF_REPO}"

        api.create_repo(repo_id=repo_id, exist_ok=True, token=HF_TOKEN, repo_type='dataset')
        print(f"\n  [HUB] Pushing evaluation results → {repo_id} ...")

        for fpath in results_dir_files:
            fname = os.path.basename(fpath)
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type='dataset',
                token=HF_TOKEN,
            )
            print(f"  [HUB] Uploaded: {fname}")

        print(f"  [HUB] Done → https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"  [HUB] Push failed: {e}")


# ================================================================
# EVALUATION RUNNERS
# ================================================================
def run_individual(available, splits_data):
    """Evaluate every trained model separately. Returns results dict."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    individual_results = {}
    generated_files    = []

    for key, cfg, ckpt_dir, pkl_path in available:
        name = cfg['name']
        print(f"\n\n{'='*65}")
        print(f"  Individual Evaluation: {name}")
        print(f"{'='*65}")

        model, tokenizer = load_model_from_checkpoint(name, pkl_path, device)
        model = model.to(device)

        individual_results[name] = {}

        for split_name, (texts, labels) in splits_data.items():
            preds, probs, true_labels = get_probabilities(
                model, tokenizer, texts, labels, device,
                batch_size=8, desc=f"{name} / {split_name}",
            )

            has_labels = any(l >= 0 for l in true_labels)
            if has_labels and len(set(true_labels)) > 1:
                metrics = compute_metrics(f"{name.upper()} — {split_name.upper()}", preds, true_labels)
            else:
                metrics = {'predictions': preds.tolist()}

            individual_results[name][split_name] = metrics

            if 'confusion_matrix' in metrics:
                cm_path = os.path.join(PLOTS_DIR, f"{name}_{split_name}_confusion_matrix.png")
                save_confusion_matrix_plot(
                    metrics['confusion_matrix'],
                    title=f"{name} — Confusion Matrix ({split_name})",
                    save_path=cm_path,
                )
                generated_files.append(cm_path)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return individual_results, generated_files


def run_ensemble(available, splits_data):
    """Soft-voting ensemble across all available models. Returns results dict."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ensemble_results = {}
    generated_files  = []
    model_names = [cfg['name'] for _, cfg, _, _ in available]

    print(f"\n\n{'='*65}")
    print(f"  Ensemble Evaluation (soft-voting)")
    print(f"  Models: {model_names}")
    print(f"{'='*65}")

    for split_name, (texts, labels) in splits_data.items():
        all_probs = []

        for key, cfg, ckpt_dir, pkl_path in available:
            name = cfg['name']
            model, tokenizer = load_model_from_checkpoint(name, pkl_path, device)
            model = model.to(device)

            _, probs, true_labels = get_probabilities(
                model, tokenizer, texts, labels, device,
                batch_size=8, desc=f"{name} / {split_name}",
            )
            all_probs.append(probs)

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        # Average probabilities across all models
        avg_probs  = np.mean(all_probs, axis=0)   # shape: (N, 6)
        ens_preds  = avg_probs.argmax(axis=1)

        has_labels = any(l >= 0 for l in true_labels)
        if has_labels and len(set(true_labels)) > 1:
            metrics = compute_metrics(
                f"ENSEMBLE ({len(available)} models) — {split_name.upper()}", ens_preds, true_labels
            )
        else:
            metrics = {'predictions': ens_preds.tolist()}

        ensemble_results[split_name] = metrics

        if 'confusion_matrix' in metrics:
            cm_path = os.path.join(PLOTS_DIR, f"ensemble_{split_name}_confusion_matrix.png")
            save_confusion_matrix_plot(
                metrics['confusion_matrix'],
                title=f"Ensemble — Confusion Matrix ({split_name})",
                save_path=cm_path,
            )
            generated_files.append(cm_path)

    return ensemble_results, generated_files


# ================================================================
# INTERACTIVE MENUS
# ================================================================
def select_mode():
    print("\n  Evaluation options:")
    print("    [1] individual  — Evaluate each model separately")
    print("    [2] ensemble    — Ensemble (soft-voting) only")
    print("    [3] both        — Individual + Ensemble  [RECOMMENDED]")
    choice = input("\n  Your choice (1/2/3 or name): ").strip()
    m = {'1': 'individual', '2': 'ensemble', '3': 'both',
         'individual': 'individual', 'ensemble': 'ensemble', 'both': 'both'}
    mode = m.get(choice)
    if not mode:
        print(f"  Invalid: '{choice}'")
        sys.exit(1)
    return mode


def select_split():
    print("\n  Which data split?")
    print("    [1] dev  — Validation set (has ground-truth labels)")
    print("    [2] test — Test set (may have labels)")
    print("    [3] both — Evaluate on dev AND test")
    choice = input("\n  Your choice (1/2/3): ").strip()
    m = {'1': ['dev'], '2': ['test'], '3': ['dev', 'test']}
    splits = m.get(choice.strip())
    if not splits:
        splits = ['dev']
    return splits


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   type=str, default=None,
                        help='individual | ensemble | both')
    parser.add_argument('--split',  type=str, default=None,
                        help='dev | test | both')
    parser.add_argument('--no-hub', action='store_true',
                        help='Skip HuggingFace push')
    args = parser.parse_args()

    if args.no_hub:
        global HF_TOKEN
        HF_TOKEN = ''

    print("\n" + "="*65)
    print("  SemEval 2024 Task 8 — Subtask B  |  Evaluation Script")
    print("="*65)

    # Discover trained checkpoints
    print("\n  Scanning for trained checkpoints...")
    available = find_available_checkpoints()
    if not available:
        print("\n  No trained checkpoints found. Run train.py first.")
        sys.exit(1)

    # Mode + split selection
    mode   = args.mode  or select_mode()
    splits = [s.strip() for s in args.split.split(',')] if args.split else select_split()

    # Load data
    print("\n  Loading data...")
    splits_data = {}
    for split in splits:
        fname = 'subtaskB_dev.jsonl' if split == 'dev' else 'subtaskB_test.jsonl'
        data  = load_jsonl(os.path.join(DATA_DIR, fname))
        texts  = [d['text']          for d in data]
        labels = [d.get('label', -1) for d in data]
        splits_data[split] = (texts, labels)
        print(f"  {split.upper():4s}: {len(texts):,} samples")

    # Run evaluation
    individual_results = {}
    ensemble_results   = {}
    all_plot_files     = []

    if mode in ('individual', 'both'):
        individual_results, plots = run_individual(available, splits_data)
        all_plot_files.extend(plots)

    if mode in ('ensemble', 'both'):
        ensemble_results, plots = run_ensemble(available, splits_data)
        all_plot_files.extend(plots)

    # Comparison bar chart (if we ran individual on dev or test)
    for split in splits:
        if individual_results and any(split in v for v in individual_results.values()):
            cpath = os.path.join(PLOTS_DIR, f"model_comparison_{split}.png")
            save_comparison_plot(individual_results, split, cpath)
            all_plot_files.append(cpath)

    # Save full results JSON
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = {
        'timestamp':     datetime.now().isoformat(),
        'mode':          mode,
        'splits':        splits,
        'models_used':   [cfg['name'] for _, cfg, _, _ in available],
        'individual':    individual_results,
        'ensemble':      ensemble_results,
    }
    json_path = os.path.join(LOGS_DIR, f'evaluation_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results JSON saved: {json_path}")

    # Also write a fixed-name summary for easy reference
    summary_path = os.path.join(LOGS_DIR, 'evaluation_results.json')
    with open(summary_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    # Push everything to HuggingFace Hub
    files_to_push = [json_path, summary_path] + all_plot_files
    push_results_to_hub(files_to_push, individual_results, ensemble_results)

    # Final summary printout
    print("\n" + "="*65)
    print("  EVALUATION SUMMARY")
    print("="*65)
    for mname, splits_res in individual_results.items():
        for split_name, metrics in splits_res.items():
            f1  = metrics.get('macro_f1', '—')
            acc = metrics.get('accuracy', '—')
            print(f"  {mname:30s}  [{split_name:4s}]  F1={f1}  Acc={acc}")
    for split_name, metrics in ensemble_results.items():
        f1  = metrics.get('macro_f1', '—')
        acc = metrics.get('accuracy', '—')
        print(f"  {'ENSEMBLE':30s}  [{split_name:4s}]  F1={f1}  Acc={acc}")

    print(f"\n  Plots : {PLOTS_DIR}")
    print(f"  Logs  : {LOGS_DIR}")
    print("="*65)


if __name__ == '__main__':
    main()
