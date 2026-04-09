import os
import sys
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Combinations to Test ────────────────────────────────────
combinations = [
    {"name": "Semantic_Only", "flags": ["--disable_stylo", "--disable_perp"]},
    {"name": "Stylo_Only", "flags": ["--disable_sem", "--disable_perp"]},
    {"name": "Perp_Only", "flags": ["--disable_stylo", "--disable_sem"]},
    {"name": "Stylo_Perp", "flags": ["--disable_sem"]},
    {"name": "Stylo_Semantic", "flags": ["--disable_perp"]},
    {"name": "Perp_Semantic", "flags": ["--disable_stylo"]},
    {"name": "All_Three", "flags": []},
]

BASE_ARGS = [
    "python", "src/train_fusion.py",
    "--config", "configs/best_model.json",
]

# Set environment
os.environ["PYTHONIOENCODING"] = "utf-8"
output_base_dir = Path("experiments/ablation_study")
output_base_dir.mkdir(parents=True, exist_ok=True)

results = []

print("🚀 Starting Multimodal Fusion Ablation Study...")
print("====================================================")

for combo in combinations:
    name = combo["name"]
    flags = combo["flags"]
    output_dir = output_base_dir / name
    
    print(f"\n🏃 Running [{name}]...")
    
    cmd = BASE_ARGS + flags + ["--output_dir", str(output_dir)]
    print(f"   Command: {' '.join(cmd)}")
    
    # Run the training
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"❌ Error running {name}: exited with {process.returncode}")
            continue
    except Exception as e:
        print(f"❌ Subprocess failed: {e}")
        continue
    
    # Extract results from classification_report.txt
    report_path = output_dir / "classification_report.txt"
    if not report_path.exists():
        print(f"⚠️ Warning: No report found for {name}")
        continue
        
    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    acc = None
    f1 = None
    # Parse our custom header
    # ║  Accuracy:  0.9199                  ║
    # ║  F1 Score:  0.9198                  ║
    for line in lines:
        if "Accuracy:" in line:
            acc = float(line.split(":")[1].strip().split()[0])
        elif "F1 Score:" in line:
            f1 = float(line.split(":")[1].strip().split()[0])
            
    if acc is not None and f1 is not None:
        results.append({"Combination": name, "Accuracy": acc * 100, "F1_Score": f1 * 100})
        print(f"   ✅ Done! Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
    else:
        print(f"   ⚠️ Could not parse results for {name}")

# ── Save Results and Plot ───────────────────────────────────
if not results:
    print("No results to save.")
    exit()

df = pd.DataFrame(results)
csv_path = output_base_dir / "ablation_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n💾 Saved results to {csv_path}")

# Sorting for nicer plot (Singles -> Doubles -> Triple)
df["Num_Branches"] = df["Combination"].apply(lambda x: 1 if "Only" in x else (3 if "All" in x else 2))
df = df.sort_values(by=["Num_Branches", "F1_Score"]).reset_index(drop=True)

# Plot F1 and Accuracy
fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(df))
width = 0.35

ax.bar([i - width/2 for i in x], df["Accuracy"], width, label='Accuracy', color="#4a90e2")
ax.bar([i + width/2 for i in x], df["F1_Score"], width, label='F1 Score', color="#50e3c2")

ax.set_ylabel("Percentage (%)", fontweight='bold')
ax.set_title("Multimodal Ablation Study - Performance Evaluation", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df["Combination"], rotation=30, ha="right")
ax.set_ylim([max(50, df["F1_Score"].min() - 5), 100]) # Start y-axis at a reasonable floor for visibility
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i in x:
    ax.text(i - width/2, df["Accuracy"][i] + 0.3, f"{df['Accuracy'][i]:.1f}", ha='center', fontsize=9)
    ax.text(i + width/2, df["F1_Score"][i] + 0.3, f"{df['F1_Score'][i]:.1f}", ha='center', fontsize=9)

plt.tight_layout()
plot_path = output_base_dir / "combination_performance.png"
plt.savefig(plot_path, dpi=200)
print(f"📊 Saved graph to {plot_path}")

print("\n🎉 Ablation Study Finished!")
