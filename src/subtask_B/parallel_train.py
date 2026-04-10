"""
SemEval 2024 Task 8 — Subtask B
Parallel Training Script (Single GPU Worker)
===========================================
Enhanced with heartbeat logging and unbuffered output.
"""

import os, gc, sys, json, glob, random, argparse, warnings, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Force HuggingFace to show progress bars even in background processes
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
# Disable hf_transfer for stability in parallel runs (standard downloader handles locks better)
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
# Ensure TQDM uses a standard width for the PTY
os.environ['TQDM_MININTERVAL'] = '0.5' # Update every 0.5s to reduce log spam

# Speed up JSON loading
import json
try:
    import ujson as json
except ImportError:
    pass

# ============================================================
# PATHS
# ============================================================
OUTPUT_DIR = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
found = glob.glob('/kaggle/input/**/subtaskB_train.jsonl', recursive=True)
DATA_DIR = os.path.dirname(found[0]) if found else '.'

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOGS_DIR       = os.path.join(OUTPUT_DIR, 'logs')

def get_local_model_path(model_id):
    """Checks local paths for a directory containing model files.
    Searches /kaggle/working/models/ (notebook pre-download), then
    /kaggle/input/ (dataset zip), then falls back to internet.
    """
    clean_id = model_id.split('/')[-1]

    # 0. Check /kaggle/working/models/ (pre-downloaded by notebook)
    working_path = f'/kaggle/working/models/{clean_id}'
    if os.path.exists(os.path.join(working_path, 'config.json')):
        print(f"  [LOCAL] Found pre-downloaded model: {working_path}", flush=True)
        return working_path

    # 1. Search for models/ directory inside any attached Kaggle dataset
    for candidate in glob.glob(f'/kaggle/input/**/models/{clean_id}/config.json', recursive=True):
        path = os.path.dirname(candidate)
        print(f"  [LOCAL] Found model in dataset: {path}", flush=True)
        return path

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

    print(f"  [WARN] No local weights found for {model_id}, will download from internet.", flush=True)
    return model_id  # Fallback to internet string

# ============================================================
# CONFIG
# ============================================================
LABEL2NAME = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
NUM_CLASSES = 6
A_MODEL_TO_B_LABEL = {'human': 0, 'chatGPT': 1, 'cohere': 2, 'davinci': 3, 'dolly': 5}

MODEL_REGISTRY = {
    'roberta': {
        'name':     'roberta-base',
        'model_id': 'roberta-base',
        'batch_size': 8,
        'grad_accum': 4,
        'lr': 2e-5,
    },
    'electra': {
        'name':     'electra-base',
        'model_id': 'google/electra-base-discriminator',
        'batch_size': 8,
        'grad_accum': 4,
        'lr': 2e-5,
    }
}

MAX_LENGTH   = 512
EPOCHS       = 3
PATIENCE     = 2
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# ============================================================
# DATASET
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts; self.labels = labels
        self.tokenizer = tokenizer; self.max_length = max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]); label = self.labels[idx]
        tids = self.tokenizer.encode(text, add_special_tokens=False)
        ns   = self.tokenizer.num_special_tokens_to_add(pair=False)
        mc   = self.max_length - ns
        if len(tids) > mc:
            h = mc // 2; tids = tids[:h] + tids[-h:]
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

def load_jsonl(path, desc="data"):
    print(f"  [INIT] Loading {desc} from {os.path.basename(path)}...", flush=True)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
            if (i+1) % 50000 == 0:
                print(f"  [PROGRESS] Loaded {i+1} rows of {desc}...", flush=True)
    print(f"  [DONE] Finished loading {desc} ({len(data)} total rows).", flush=True)
    return data

# ============================================================
# WORKER
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--gpu',   type=int, required=True)
    args = parser.parse_args()

    # Pre-check GPU
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available!", flush=True)
        sys.exit(1)
    
    device = torch.device(f'cuda:{args.gpu}')
    cfg    = MODEL_REGISTRY[args.model]
    name, model_id = cfg['name'], cfg['model_id']
    bs, grad_accum, lr = cfg['batch_size'], cfg['grad_accum'], cfg['lr']
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    ckpt_dir = os.path.join(CHECKPOINT_DIR, name)

    print(f"[{name}] Starting Worker on GPU {args.gpu}", flush=True)

    # Data
    train_b = load_jsonl(os.path.join(DATA_DIR, 'subtaskB_train.jsonl'), "Train Set")
    dev_b   = load_jsonl(os.path.join(DATA_DIR, 'subtaskB_dev.jsonl'), "Dev Set")
    train_a = load_jsonl(os.path.join(DATA_DIR, 'subtaskA_train_monolingual.jsonl'), "Subtask A Augmentation")

    print(f"[{name}] Pre-processing data and applying augmentation...", flush=True)
    aug = [{'text': s['text'], 'label': A_MODEL_TO_B_LABEL[s.get('model','')]} 
           for s in train_a if s.get('model','') in A_MODEL_TO_B_LABEL]
    all_train = [{'text': d['text'], 'label': d['label']} for d in train_b] + aug
    random.shuffle(all_train)

    print(f"[{name}] Initializing Tokenizer (checking local cache first)...", flush=True)
    local_path = get_local_model_path(model_id)
    if local_path != model_id:
        print(f"  [FOUND] Using local weights from: {local_path}", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    train_ds  = TextDataset([d['text'] for d in all_train], [d['label'] for d in all_train], tokenizer)
    val_ds    = TextDataset([d['text'] for d in dev_b], [d['label'] for d in dev_b], tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs*2, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[{name}] Loading Model Weights to GPU {args.gpu}...", flush=True)
    cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=np.array([d['label'] for d in all_train]))
    for i in range(len(cw)):
        print(f"  Class {i} weights: {cw[i]:.3f}", flush=True)
    class_wts = torch.tensor(cw, dtype=torch.float32).to(device)

    print(f"[{name}] Loading weights (instant if local, otherwise downloading)...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(local_path, num_labels=NUM_CLASSES)
    
    print(f"[{name}] Moving model to GPU {args.gpu}...", flush=True)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = nn.CrossEntropyLoss(weight=class_wts)
    scaler       = torch.cuda.amp.GradScaler()

    vram = torch.cuda.memory_reserved(device) / 1e9
    print(f"[{name}] Setup Complete. Initial VRAM: {vram:.2f} GB. Starting Training Loop...", flush=True)

    best_f1 = 0.0
    history = {'train_loss': [], 'val_f1': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[{name}] Epoch {epoch}", leave=False)
        for step, batch in pbar:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            
            with torch.cuda.amp.autocast():
                out  = model(input_ids=ids, attention_mask=mask)
                loss = criterion(out.logits.float(), lbls)
                loss = loss / grad_accum
            
            scaler.scale(loss).backward()
            
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            running_loss += loss.item() * grad_accum
            
            pbar.set_postfix({'loss': f"{running_loss/(step+1):.4f}"})

        # Eval
        print(f"[{name}] Epoch {epoch} Complete. Running Validation...", flush=True)
        model.eval()
        p, l = [], []
        with torch.no_grad():
            for b in tqdm(val_loader, desc=f"[{name}] Validating", leave=False):
                out = model(b['input_ids'].to(device), b['attention_mask'].to(device))
                p.extend(out.logits.argmax(dim=-1).cpu().numpy())
                l.extend(b['labels'].numpy())
        f1 = f1_score(l, p, average='macro', zero_division=0)
        
        history['train_loss'].append(round(running_loss/len(train_loader), 5))
        history['val_f1'].append(round(f1, 5))
        print(f"[{name}] VAL RESULTS -> Epoch {epoch} | Macro-F1: {f1:.4f}", flush=True)

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_id':         model_id,
                'num_classes':      NUM_CLASSES,
                'label_map':        LABEL2NAME,
                'best_f1':          f1,
            }, os.path.join(ckpt_dir, f"{name}_model.pkl"))
            print(f"[{name}] [SAVED] New best F1 achieved ({f1:.4f}). Checkpoint updated.", flush=True)

    with open(os.path.join(LOGS_DIR, f"{name}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[{name}] FINISHED TRAINING. Best Val F1: {best_f1:.4f}. Script exiting.", flush=True)

if __name__ == '__main__':
    main()
