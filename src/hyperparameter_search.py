import os
import sys
import json
import random
import subprocess
import pandas as pd
from pathlib import Path

def sample_hyperparameters():
    """Generates a random dictionary of hyperparameter ranges."""
    
    # ── Topology ──
    # Shallow vs Deep networks
    architectures = [
        {"hidden_dims": [128, 64], "dropout_rates": [0.4, 0.2]},
        {"hidden_dims": [256, 128, 64], "dropout_rates": [0.5, 0.3, 0.1]},
        {"hidden_dims": [64, 32], "dropout_rates": [0.3, 0.1]},
        {"hidden_dims": [256, 128], "dropout_rates": [0.4, 0.2]},
        {"hidden_dims": [128, 64, 32], "dropout_rates": [0.3, 0.2, 0.1]}
    ]
    arch = random.choice(architectures)
    
    # Log-uniform for continuous scaling parameters
    lr = 10 ** random.uniform(-4.0, -2.5) # between 0.0001 and ~0.003
    weight_decay = 10 ** random.uniform(-5.0, -2.5)
    
    # Discrete choices
    batch_size = random.choice([128, 256, 512, 1024])
    label_smoothing = random.choice([0.01, 0.05, 0.1, 0.15])
    
    # Anti-overshadowing features (very critical for 2-branch)
    perp_dropout = random.uniform(0.1, 0.5)
    perp_grad_scale = random.uniform(0.05, 0.4)
    perp_gate_init = random.uniform(0.05, 0.4)
    
    # Branch Dimension equalization
    branch_proj_dim = random.choice([16, 24, 32, 48, 64])

    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "label_smoothing": label_smoothing,
        "perp_dropout": perp_dropout,
        "perp_grad_scale": perp_grad_scale,
        "perp_gate_init": perp_gate_init,
        "branch_proj_dim": branch_proj_dim,
        "hidden_dims": arch["hidden_dims"],
        "dropout_rates": arch["dropout_rates"]
    }

def run_trial(trial_id, params, output_base_dir):
    """Executes train_fusion.py with the assigned param set."""
    trial_dir = output_base_dir / f"trial_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct base command (Disabling semantic!)
    cmd = [
        sys.executable, "src/train_fusion.py",
        "--disable_sem",
        "--epochs", "25", # Fast search epochs
        "--patience", "6", 
        "--output_dir", str(trial_dir),
        "--lr", str(params["lr"]),
        "--weight_decay", str(params["weight_decay"]),
        "--batch_size", str(params["batch_size"]),
        "--label_smoothing", str(params["label_smoothing"]),
        "--perp_dropout", str(params["perp_dropout"]),
        "--perp_grad_scale", str(params["perp_grad_scale"]),
        "--perp_gate_init", str(params["perp_gate_init"]),
        "--branch_proj_dim", str(params["branch_proj_dim"]),
        "--hidden_dims"
    ] + [str(d) for d in params["hidden_dims"]] + [
        "--dropout_rates"
    ] + [str(d) for d in params["dropout_rates"]]

    print(f"\n🚀 Running Trial {trial_id:03d}")
    print(f"Architecture: {params['hidden_dims']} | LR: {params['lr']:.5f} | Batch: {params['batch_size']}")
    
    try:
        # Stream the output so we see it live
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for line in process.stdout:
            # Optionally suppress high-volume epoch logs if we want a clean console, 
            # but printing it keeps the user informed
            print(line, end="")
        process.wait()
    except Exception as e:
        print(f"❌ Error in Trial {trial_id} execution: {e}")
        return None

    # Harvest score
    report_path = trial_dir / "classification_report.txt"
    f1_score = None
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("macro avg"):
                    f1_score = float(line.split()[-2]) * 100.0
                    break
    
    return f1_score


def main():
    NUM_TRIALS = 15
    output_base_dir = Path("experiments/optimizations")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "═"*60)
    print(f" 🧬 HYPERPARAMETER RANDOM SEARCH ({NUM_TRIALS} TRIALS)")
    print("═"*60)
    
    results = []
    
    for i in range(1, NUM_TRIALS + 1):
        params = sample_hyperparameters()
        f1 = run_trial(i, params, output_base_dir)
        
        if f1 is not None:
            # Flatten params for DataFrame
            flat_params = params.copy()
            flat_params["hidden_dims"] = str(flat_params["hidden_dims"])
            flat_params["dropout_rates"] = str(flat_params["dropout_rates"])
            flat_params["F1_Score"] = f1
            flat_params["Trial_ID"] = i
            
            results.append(flat_params)
            print(f"✅ Trial {i:03d} Completed — F1: {f1:.2f}%")
        else:
            print(f"⚠️ Trial {i:03d} Failed or Did Not Save Report")
            
    if not results:
        print("❌ All trials failed.")
        return
        
    df = pd.DataFrame(results)
    
    # Sort and save
    df = df.sort_values(by="F1_Score", ascending=False)
    df.to_csv(output_base_dir / "sweep_results.csv", index=False)
    
    best_row = df.iloc[0]
    best_trial = int(best_row["Trial_ID"])
    best_f1 = best_row["F1_Score"]
    
    print("\n" + "🌟"*20)
    print(f" WINNING CONFIGURATION (Trial {best_trial:03d}) — {best_f1:.2f}%")
    print("🌟"*20)
    print(best_row)
    
    # Extract the true params used for the best row
    # Re-reading to ensure exact dict format
    best_params_original = next(r for r in results if r["Trial_ID"] == best_trial)
    
    # Create final JSON config file
    final_config = {
        "lr": float(best_params_original["lr"]),
        "weight_decay": float(best_params_original["weight_decay"]),
        "batch_size": int(best_params_original["batch_size"]),
        "label_smoothing": float(best_params_original["label_smoothing"]),
        "perp_dropout": float(best_params_original["perp_dropout"]),
        "perp_grad_scale": float(best_params_original["perp_grad_scale"]),
        "perp_gate_init": float(best_params_original["perp_gate_init"]),
        "branch_proj_dim": int(best_params_original["branch_proj_dim"]),
        "hidden_dims": eval(best_params_original["hidden_dims"]),
        "dropout_rates": eval(best_params_original["dropout_rates"]),
        "epochs": 75,
        "patience": 15,
        "scheduler": "onecycle",
        "output_dir": "experiments/final_optimized_model"
    }
    
    config_path = Path("configs/optimized_stylo_perp.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=4)
        
    print(f"\n💾 Saved Champion configuration to {config_path}")
    print("Run this to launch the ultimate model:")
    print("python src/train_fusion.py --config configs/optimized_stylo_perp.json --disable_sem")

if __name__ == "__main__":
    main()
