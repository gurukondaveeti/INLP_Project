import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from fusion_dataset import load_and_align
from fusion_model import FusionMLP, FusionLoss
from train_fusion import evaluate

def main():
    print("\n🚀 Starting Permutation Feature Importance Analysis...")
    print("====================================================")
    
    device = torch.device("cpu")
    output_dir = Path("experiments/feature_importance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Reload Best Model Configuration ─────────────────────
    with open("configs/best_model.json", "r") as f:
        config = json.load(f)
        
    model_dir = Path(config["output_dir"])
    ckpt_path = model_dir / "best_fusion_model.pt"
    scalers_path = model_dir / "scalers.pkl"
    
    print(f"📁 Loading weights from: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing best model weights at {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
        
    stylo_scaler = scalers["stylo_scaler"]
    perp_scaler = scalers["perp_scaler"]
    sem_scaler = scalers["sem_scaler"]
    
    # ── 2. Load Testing Data ───────────────────────────────────
    print("📂 Loading 34k Document Test Set...")
    test_ds, _, _, _ = load_and_align(
        split="test",
        stylo_pkl_path="data/features/extracted_features_cache.pkl",
        perp_pkl_path="data/features/test_perplexity_features.pkl",
        sem_pkl_path="data/features/semantic_test.pkl",
        stylo_scaler=stylo_scaler,
        perp_scaler=perp_scaler,
        sem_scaler=sem_scaler
    )
    
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # ── 3. Initialize Model ────────────────────────────────────
    print("🏗️ Initializing FusionMLP network architecture...")
    model = FusionMLP(
        stylo_dim=test_ds.stylo_dim,
        perp_dim=test_ds.perp_dim,
        sem_dim=test_ds.sem_dim,
        hidden_dims=(256, 128, 64),
        dropout_rates=(0.4, 0.3, 0.2),
        perp_modality_dropout=config.get("perp_dropout", 0.4),
        perp_grad_scale=config.get("perp_grad_scale", 0.1),
        use_branch_projection=True,
        branch_proj_dim=48,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    
    criterion = FusionLoss(primary_weight=0.7)
    
    # ── 4. Baseline Evaluation ─────────────────────────────────
    print("\n🎯 Running Baseline Intelligence Reference...")
    baseline_metrics = evaluate(model, test_loader, criterion, device)
    baseline_f1 = baseline_metrics["f1"] * 100.0
    print(f"   Baseline F1-Score: {baseline_f1:.2f}%")
    
    # ── 5. Permutation Loop ────────────────────────────────────
    results = []
    
    stylo_feature_names = test_ds.feature_columns
    if not stylo_feature_names:
        stylo_feature_names = [f"Stylo_Feature_{i}" for i in range(test_ds.stylo_dim)]
        
    print("\n🧩 Commencing Sabotage (Feature Masking/Shuffling)...")
    
    # 5a. Stylometric Features
    for i, name in enumerate(stylo_feature_names):
        print(f"   Shuffling {name}...", end="", flush=True)
        # Store original
        original_col = test_ds.stylo[:, i].copy()
        
        # Scramble array across all 34k entries!
        test_ds.stylo[:, i] = np.random.permutation(original_col)
        
        # Re-evaluate F1
        shuffled_metrics = evaluate(model, test_loader, criterion, device)
        shuffled_f1 = shuffled_metrics["f1"] * 100.0
        
        importance = baseline_f1 - shuffled_f1
        results.append({"Feature": name, "Category": "Stylometric", "Importance_F1_Drop": importance})
        print(f" Drop: -{importance:.2f}% (F1 = {shuffled_f1:.2f}%)")
        
        # Restore!
        test_ds.stylo[:, i] = original_col


    # 5b. Perplexity Features
    for i in range(test_ds.perp_dim):
        name = f"GPT2_Perplexity_Bin_{i+1}"
        print(f"   Shuffling {name}...", end="", flush=True)
        # Store original
        original_col = test_ds.perp[:, i].copy()
        
        # Scramble
        test_ds.perp[:, i] = np.random.permutation(original_col)
        
        # Re-evaluate F1
        shuffled_metrics = evaluate(model, test_loader, criterion, device)
        shuffled_f1 = shuffled_metrics["f1"] * 100.0
        
        importance = baseline_f1 - shuffled_f1
        results.append({"Feature": name, "Category": "Perplexity", "Importance_F1_Drop": importance})
        print(f" Drop: -{importance:.2f}% (F1 = {shuffled_f1:.2f}%)")
        
        # Restore
        test_ds.perp[:, i] = original_col
        
    print("\n✅ Simulation Complete.")
    
    # ── 6. Save and Plot ───────────────────────────────────────
    df = pd.DataFrame(results)
    df = df.sort_values(by="Importance_F1_Drop", ascending=True) # Ascending for horizontal bar plot
    
    csv_path = output_dir / "permutation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved absolute results to {csv_path}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = df["Category"].map({"Stylometric": "#e74c3c", "Perplexity": "#3498db"}).tolist()
    
    bars = ax.barh(df["Feature"], df["Importance_F1_Drop"], color=colors)
    ax.set_xlabel("F1-Score Drop (Percentage %)\nLower is less important, Higher means Critical Dependency", fontweight="bold")
    ax.set_title("Permutation Feature Importance\nHow much accuracy is lost when destroying a single feature?", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    
    # Adding text values inside bars
    for bar in bars:
        val = bar.get_width()
        if val > 0:
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"-{val:.2f}%", va='center')
            
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Stylometric'),
                       Patch(facecolor='#3498db', label='Perplexity')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plot_path = output_dir / "feature_importance.png"
    plt.savefig(plot_path, dpi=200)
    print(f"📊 Saved horizontal bar chart graph to {plot_path}")

if __name__ == "__main__":
    main()
