import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

combinations = [
    "Semantic_Only", "Stylo_Only", "Perp_Only", 
    "Stylo_Perp", "Stylo_Semantic", "Perp_Semantic", "All_Three"
]

output_base_dir = Path("experiments/ablation_study")
results = []

for name in combinations:
    report_path = output_base_dir / name / "classification_report.txt"
    if not report_path.exists():
        continue
        
    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    acc = None
    f1 = None
    
    for line in lines:
        if line.strip().startswith("accuracy"):
            # Parses: '    accuracy                         0.9199     34272'
            parts = line.split()
            acc = float(parts[-2])
        elif line.strip().startswith("macro avg"):
            # Parses: '   macro avg     0.9195    0.9201    0.9198     34272'
            parts = line.split()
            f1 = float(parts[-2])
            
    if acc is not None and f1 is not None:
        results.append({"Combination": name, "Accuracy": acc * 100, "F1_Score": f1 * 100})
        print(f"Extracted [{name}]: Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")

df = pd.DataFrame(results)
csv_path = output_base_dir / "ablation_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n💾 Saved results to {csv_path}")

df["Num_Branches"] = df["Combination"].apply(lambda x: 1 if "Only" in x else (3 if "All" in x else 2))
df = df.sort_values(by=["Num_Branches", "F1_Score"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(df))
width = 0.35

ax.bar([i - width/2 for i in x], df["Accuracy"], width, label='Accuracy', color="#4a90e2")
ax.bar([i + width/2 for i in x], df["F1_Score"], width, label='F1 Score', color="#50e3c2")

ax.set_ylabel("Percentage (%)", fontweight='bold')
ax.set_title("Multimodal Ablation Study - Performance Evaluation", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df["Combination"], rotation=30, ha="right")
ax.set_ylim([max(50, df["F1_Score"].min() - 5), 100]) # Start y-axis
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

for i in x:
    ax.text(i - width/2, df["Accuracy"][i] + 0.3, f"{df['Accuracy'][i]:.1f}", ha='center', fontsize=9)
    ax.text(i + width/2, df["F1_Score"][i] + 0.3, f"{df['F1_Score'][i]:.1f}", ha='center', fontsize=9)

plt.tight_layout()
plot_path = output_base_dir / "combination_performance.png"
plt.savefig(plot_path, dpi=200)
print(f"📊 Saved graph to {plot_path}")
