import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open("results/experiments/all_results.json") as f:
    rows = json.load(f)

df = pd.DataFrame(rows)

STRATEGY_COLORS = {"greedy": "#2196F3", "mcts": "#FF9800", "beam": "#4CAF50"}
CONFIGS = {"small": (2, 50), "medium": (5, 100), "large": (10, 200)}
CONFIG_LABELS = ["small\n(2 trials\n50 nbhds)", "medium\n(5 trials\n100 nbhds)", "large\n(10 trials\n200 nbhds)"]

sns.set(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("SPMiner on Wiki-Vote: Strategy & Hyperparameter Analysis", fontsize=15, fontweight="bold", y=1.02)

# ── Plot 1: Runtime vs Strategy per Config ──────────────────────────────────
ax1 = axes[0]
x = range(3)
width = 0.26
for i, strategy in enumerate(["greedy", "mcts", "beam"]):
    sub = df[df["strategy"] == strategy].sort_values("n_trials")
    ax1.bar([v + i * width for v in x], sub["runtime_s"].values,
            width=width, label=strategy.capitalize(),
            color=STRATEGY_COLORS[strategy], edgecolor="white", linewidth=0.8)
ax1.set_xticks([v + width for v in x])
ax1.set_xticklabels(CONFIG_LABELS)
ax1.set_title("Search Strategy vs. Runtime", fontweight="bold")
ax1.set_ylabel("Runtime (seconds)")
ax1.set_xlabel("Configuration")
ax1.legend(title="Strategy")

# ── Plot 2: Patterns Found vs Configuration ──────────────────────────────────
ax2 = axes[1]
for i, strategy in enumerate(["greedy", "mcts", "beam"]):
    sub = df[df["strategy"] == strategy].sort_values("n_trials")
    ax2.plot(CONFIG_LABELS, sub["n_patterns"].values,
             marker="o", linewidth=2, markersize=8,
             label=strategy.capitalize(), color=STRATEGY_COLORS[strategy])
    for j, (cfg, v) in enumerate(zip(CONFIG_LABELS, sub["n_patterns"].values)):
        ax2.annotate(str(v), (j, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
ax2.set_title("Configuration vs. Patterns Found", fontweight="bold")
ax2.set_ylabel("Number of Patterns Found")
ax2.set_xlabel("Configuration")
ax2.legend(title="Strategy")

# ── Plot 3: Metrics Table ─────────────────────────────────────────────────────
ax3 = axes[2]
ax3.axis("off")
table_data = []
best_rt = df["runtime_s"].min()
best_pat = df["n_patterns"].max()
for _, row in df.iterrows():
    rt = f"{row['runtime_s']}s {'★' if row['runtime_s'] == best_rt else ''}"
    pat = f"{row['n_patterns']} {'★' if row['n_patterns'] == best_pat else ''}"
    table_data.append([row["label"], row["strategy"].upper(), str(row["n_trials"]),
                       str(row["n_neighborhoods"]), rt, pat])
table = ax3.table(
    cellText=table_data,
    colLabels=["Config", "Strategy", "Trials", "Nbhds", "Runtime", "Patterns"],
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#424242")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("#E0E0E0")
ax3.set_title("Full Metrics Table", fontweight="bold")

plt.tight_layout()
plt.savefig("results/experiments/strategy_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: results/experiments/strategy_comparison.png")
