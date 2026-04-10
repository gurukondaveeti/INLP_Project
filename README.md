# 🤖 Multimodal Fusion for Machine-Generated Text Detection
### SemEval-2024 Task 8 — Subtasks A & B

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Subtask A F1](https://img.shields.io/badge/Subtask%20A%20F1-93.26%25-brightgreen)](experiments/final_optimized_model/)
[![Subtask B F1](https://img.shields.io/badge/Subtask%20B%20F1-73.00%25-blue)](experiments/)

> A heterogeneous late-fusion framework that achieves **93.26% macro F1** on Subtask A (binary MGT detection) and **73% macro F1** on Subtask B (6-class generator attribution) using a lightweight gated MLP — no fine-tuned LLM required for inference.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Ablation Study](#ablation-study)
- [Hyperparameter Search](#hyperparameter-search)
- [Key Findings](#key-findings)
- [Team](#team)

---

## Overview

This repository contains our system for **SemEval-2024 Task 8: Multigenerator, Multidomain, and Multilingual Black-Box Machine-Generated Text Detection**.

Instead of fine-tuning a large model end-to-end, we extract three independent **statistical feature families** from text and train a fast, interpretable **Gated Multi-Branch MLP** classifier:

| Modality | Dimensions | Method |
|---|---|---|
| **Stylometric** | 4-D | spaCy dependency trees + combinatorial feature search |
| **Perplexity** | 12-D | GPT-2 medium/large token log-probability fingerprints |
| **Semantic** | 64-D | DeBERTa-v3-base → frozen random bottleneck (discarded after ablation) |

**Key design principles:**
- **Anti-overshadowing** — Modality Dropout, Gradient Scaling, and biased Gate initialisation prevent perplexity from drowning out stylometry.
- **Learnable Independent Sigmoid Gates** — Each modality is weighted independently (not Softmax zero-sum), allowing synergistic fusion.
- **Auxiliary Loss Head** — A secondary MLP trained on Stylometric + Semantic only, forcing the stylometric branch to retain discriminative representations.

---

## Results

### Subtask A — Binary Classification (Human vs. Machine)

| System | Accuracy | Macro F1 | Precision | Recall |
|---|---|---|---|---|
| Semantic Only | 63.85% | 63.26% | — | — |
| Stylometric Only | 75.35% | 75.34% | — | — |
| Perplexity Only | 90.78% | 90.78% | — | — |
| Stylo + Semantic | 75.24% | 75.09% | — | — |
| Perp + Semantic | 90.45% | 90.44% | — | — |
| All Three Branches | 91.99% | 91.98% | — | — |
| **Stylo + Perp (Ours)** | **93.26%** | **93.26%** | **93.26%** | **93.37%** |

> **Key finding:** Semantic (DeBERTa) features *degrade* performance in **every** tested configuration. The 2-branch Stylo+Perp fusion is the winner.

### Subtask B — 6-Class Generator Attribution (RoBERTa + ELECTRA Ensemble)

| System | Macro F1 |
|---|---|
| RoBERTa (standalone) | ~68% |
| ELECTRA (standalone) | ~65% |
| **RoBERTa + ELECTRA Ensemble** | **73%** |

---

## Architecture

```
Input Text
    │
    ├── [Stylometric Pipeline] ────────────────────────────── 4-D
    │    spaCy → 4 features → StandardScaler → LayerNorm
    │    → BranchProjection(4→48-D) → Gate σ_S ──┐
    │                                              │
    ├── [Perplexity Pipeline] ─────────────────── 12-D       │
    │    GPT-2 log-probs → 12-D fingerprint                  │
    │    → StandardScaler → LayerNorm                        │
    │    → GradientScale(0.066×) → ModalityDropout(p=0.19)  │
    │    → BranchProjection(12→48-D) → Gate σ_P ────────────┤
    │                                              │          │
    └── [Semantic Pipeline] ─────────────────── 64-D       (DISCARDED after ablation)
         DeBERTa-v3-base [CLS] → Frozen Bottleneck(→64D)
         → StandardScaler → LayerNorm
         → BranchProjection(64→48-D) → Gate σ_E ──┘

    └──── Concatenation (144-D) ────► MLP[256→128] ────► Classification Head
                                                  ▲
                  Auxiliary MLP (Stylo+Sem only) ─┘  (training only)
```

---

## Project Structure

```
.
├── src/
│   ├── fusion_model.py            # Core FusionMLP architecture (Gates, AntI-Overshadowing)
│   ├── fusion_dataset.py          # Dataset loader — feature alignment & branch subsetting
│   ├── train_fusion.py            # Main training script (CLI + JSON config driven)
│   ├── run_ablation.py            # Automated 7-combination ablation orchestrator
│   ├── hyperparameter_search.py   # 15-trial random hyperparameter search
│   ├── permutation_importance.py  # Post-hoc permutation feature importance analysis
│   ├── extract_semantic_features.py # DeBERTa-v3 extraction pipeline (Kaggle GPU ready)
│   ├── extract_ablation_results.py  # CSV aggregation of ablation runs
│   └── architectures/
│       ├── cross_attention_fusion.py  # ❌ Failed architecture (Cross-Attention, 74.9% F1)
│       └── train_cross_attn.py
│
├── configs/
│   ├── optimized_stylo_perp.json  # ✅ Champion config — 2-branch, tuned hyperparams
│   ├── best_model.json            # Quick-reference best hyperparameters
│   └── cross_attention_v1.json
│
├── experiments/
│   ├── final_optimized_model/     # ✅ Best model weights, curves, confusion matrix
│   │   ├── best_fusion_model.pt   # Saved PyTorch model weights
│   │   ├── scalers.pkl            # Fitted StandardScalers (required for inference)
│   │   ├── config.json            # Exact hyperparameters used
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   └── gate_values.png        # Learned gate weights after training
│   ├── ablation_study/            # 7 sub-folders + results CSV + bar chart
│   └── optimizations/             # sweep_results.csv from random search
│
├── data/
│   └── features/                  # Extracted feature PKL files (not tracked in Git)
│       ├── extracted_features_cache.pkl   # Stylometric features
│       ├── train_perplexity_features.pkl
│       ├── dev_perplexity_features.pkl
│       └── test_perplexity_features.pkl
│
├── scripts/
│   ├── inspect_all_pkls.py        # Utility: inspect feature PKL shapes
│   └── try_load_pkl.py            # Utility: debug feature loading
│
├── report/
│   ├── final_merged_report.tex    # ✅ Full ACL-style paper (Subtasks A + B)
│   ├── viva_presentation.tex      # ✅ Beamer slides for viva/presentation
│   ├── references.bib
│   └── images/                    # All result plots and tables
│
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/gurukondaveeti/INLP_Project.git
cd INLP_Project

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers scikit-learn spacy pandas numpy matplotlib seaborn tqdm

# 4. Download spaCy model (for stylometric feature extraction)
python -m spacy download en_core_web_sm
```

---

## Data Setup

Place the SemEval-2024 Task 8 dataset files or pre-extracted feature `.pkl` files under `data/features/`:

```
data/
└── features/
    ├── extracted_features_cache.pkl      # Stylometric features (train+dev+test)
    ├── train_perplexity_features.pkl     # GPT-2 perplexity 12-D fingerprints
    ├── dev_perplexity_features.pkl
    └── test_perplexity_features.pkl
```

> **Note:** Raw `.pkl` feature files are not committed to Git due to size (\~GB). Re-extract using the steps below if needed.

### Re-extract Features

```bash
# Stylometric features (CPU, ~2-3 hours on full dataset)
# Implemented inside fusion_dataset.py — run train_fusion.py to trigger extraction

# Semantic features (GPU recommended — Kaggle/Colab)
python src/extract_semantic_features.py

# Perplexity features
# See src/fusion_dataset.py — GPT-2 extraction is cached automatically on first run
```

---

## Running the Pipeline

### Train the Champion 2-Branch Model

```bash
python src/train_fusion.py --config configs/optimized_stylo_perp.json
```

This will:
1. Load the Stylometric + Perplexity features
2. Train with Anti-Overshadowing defences for 75 epochs
3. Save the best model to `experiments/final_optimized_model/`
4. Generate `training_curves.png`, `confusion_matrix.png`, and `gate_values.png`

### Evaluate a Saved Model

```bash
python src/train_fusion.py --config configs/optimized_stylo_perp.json --eval_only
```

### Train with Custom Hyperparameters (CLI override)

```bash
python src/train_fusion.py \
    --config configs/optimized_stylo_perp.json \
    --lr 0.001 \
    --batch_size 512 \
    --epochs 50
```

---

## Ablation Study

Runs all 7 branch combinations automatically and generates a comparison bar chart:

```bash
python src/run_ablation.py
```

Results are saved to `experiments/ablation_study/`:
- `ablation_results.csv` — F1/Accuracy per combination
- `combination_performance.png` — Grouped bar chart

| Combination | F1 (%) |
|---|---|
| Semantic Only | 63.26 |
| Stylo Only | 75.34 |
| Perplexity Only | 90.78 |
| Stylo + Semantic | 75.09 |
| Perp + Semantic | 90.44 |
| **Stylo + Perp** | **92.60** |
| All Three | 91.98 |

---

## Hyperparameter Search

Launches a 15-trial random search across LR, weight decay, batch size, architecture, and anti-overshadowing parameters:

```bash
python src/hyperparameter_search.py
```

Results are saved to `experiments/optimizations/sweep_results.csv`. The winning Trial 15 configuration is already captured in `configs/optimized_stylo_perp.json`.

---

## Key Findings

### ✅ What Worked

1. **Perplexity is the dominant signal** — 12-D GPT-2 log-probability histogram alone achieves ~91% F1. Machine text concentrates probability mass in low-perplexity buckets; human text is "bursty."

2. **Stylometric features are complementary, not redundant** — Function Word Adjacency (2.79% F1 drop on permutation) and Dependency Tree Depth (2.56%) capture structural patterns *invisible* to GPT-2.

3. **4 features beat 11** — Exhaustive combinatorial search over ~1,700 subsets proved that `Sentence Length Deviation`, `Tree Depth`, `Verb Gap Deviation`, and `Function Word Adjacency` is the optimal core. More features = more noise.

4. **Independent Sigmoid Gates > Softmax Gating** — Sigmoid allows both branches to be "turned up" simultaneously (synergy), while Softmax forces zero-sum competition.

5. **Large batch sizes** (1024) stabilise training significantly on 34k samples.

### ❌ What Failed

1. **DeBERTa Semantic Features** — Degrade performance in every single 2-branch and 3-branch combination. Root cause: MGT classification depends on *how* text is generated, not *what* it means. The random frozen bottleneck adds high-dimensional noise.

2. **Cross-Attention Fusion** — Dropped performance from 92%+ to **74.9% F1**. Attention mechanisms require long token sequences to find patterns; applying them to 16 scalar features completely washes out the signal.

3. **Polynomial Feature Expansion** (degree-2, 136 features from 15) — Amplified noise without accuracy gain. Low-signal features squared become *louder* noise.

---

## 🔗 Quick Links & Hosted Models

* **GitHub Repository:** [gurukondaveeti/INLP_Project](https://github.com/gurukondaveeti/INLP_Project)
* **Perplexity Modality:** [🤗 HuggingFace - Perplexity Extraction](https://huggingface.co/VinayChaitu/MGT-Detection-SemEval2024)
* **Subtask B Models:** * [🤗 RoBERTa Base Weights](https://huggingface.co/Afzal143/semeval2024-task8-subtaskB-roberta-base) 
  * [🤗 ELECTRA Base Weights](https://huggingface.co/Afzal143/semeval2024-task8-subtaskB-electra-base)
  * [📊 Subtask B Final Results](https://huggingface.co/Afzal143/semeval2024-task8-subtaskB-results)

## Team

| Role | Contributor |
|---|---|
| Stylometric Feature Engineering | Afzal |
| Perplexity Pipeline + Subtask B | Vinay |
| Fusion Architecture + Ablation + Hyperparameter Search | Gurukondaveeti Sai Kiran |

---

## 📚 Theoretical Foundation

The stylometric features engineered for this project are heavily grounded in established linguistic and authorship attribution research. Our feature selection is directly validated by:

> **Stamatatos, E. (2009). *A Survey of Modern Authorship Attribution Methods*. JASIST, 60:538-556.**
> 
> Stamatatos provides a foundational framework for stylometry by categorizing writing style into lexical and syntactic features, supporting measures such as vocabulary richness and sentence-level statistics. His discussion of hapax legomena and function words validates our use of features like Hapax Legomena Ratio, Lexical Density, and Function Word Adjacency. Additionally, the emphasis on syntactic structures and sentence patterns aligns with our extraction of Dependency Tree Depth and Clause Density. This survey firmly justifies our approach of using structural and distributional features to capture stylistic variation in human vs. machine text.

## Citation

If you use this code, please cite SemEval-2024 Task 8:

```bibtex
@inproceedings{semeval2024task8,
  title     = {SemEval-2024 Task 8: Multigenerator, Multidomain, and Multilingual Machine-Generated Text Detection},
  booktitle = {Proceedings of the 18th International Workshop on Semantic Evaluation},
  year      = {2024}
}
```

@article{stamatatos2009survey,
  author  = {Stamatatos, Efstathios},
  title   = {A Survey of Modern Authorship Attribution Methods},
  journal = {JASIST},
  volume  = {60},
  pages   = {538--556},
  year    = {2009}
}


