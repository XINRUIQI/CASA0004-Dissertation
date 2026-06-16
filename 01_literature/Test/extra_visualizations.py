"""
Additional visualizations for result analysis.
Run after run_all.py has completed.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

RESULTS_DIR = Path(__file__).resolve().parent / "results"
df = pd.read_csv(RESULTS_DIR / "all_results_combined.csv")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                      "savefig.bbox": "tight", "font.size": 9})

LAYERS = ["M1", "M2", "M3", "M4", "M5"]
LAYER_LABELS = {"M1": "M1\n(Financial)", "M2": "M2\n(Fin+RS)",
                "M3": "M3\n(Fin+Ship)", "M4": "M4\n(All)",
                "M5": "M5\n(All+GDELT)"}
COLORS = {"M1": "#2196F3", "M2": "#FF9800", "M3": "#4CAF50",
           "M4": "#9C27B0", "M5": "#F44336"}
MODEL_ORDER = ["Ridge", "LogisticRegression", "RandomForest", "SVM",
               "XGBoost", "LSTM", "TFT", "ST-GNN"]


# ── 1. Marginal improvement: M2/M3/M4 vs M1 per model ───────────
def plot_marginal_improvement():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    configs = [
        ("direction", "macro_f1", "Direction Macro-F1", True),
        ("price", "rmse", "Price RMSE", False),
        ("volatility", "rmse", "Volatility RMSE", False),
    ]
    for ax, (target, metric, title, higher_better) in zip(axes, configs):
        sub = df[df["target"] == target]
        if metric not in sub.columns or sub.empty:
            ax.set_visible(False)
            continue

        models = sorted(sub["model"].unique())
        x = np.arange(len(models))
        width = 0.22

        for i, layer in enumerate(["M2", "M3", "M4"]):
            improvements = []
            for model in models:
                m1_val = sub[(sub["model"] == model) & (sub["layer"] == "M1")][metric]
                mx_val = sub[(sub["model"] == model) & (sub["layer"] == layer)][metric]
                if m1_val.empty or mx_val.empty:
                    improvements.append(0)
                    continue
                m1_v, mx_v = m1_val.values[0], mx_val.values[0]
                if higher_better:
                    improvements.append((mx_v - m1_v) / max(abs(m1_v), 1e-9) * 100)
                else:
                    improvements.append((m1_v - mx_v) / max(abs(m1_v), 1e-9) * 100)

            color = COLORS[layer]
            ax.bar(x + (i - 1) * width, improvements, width,
                   label=LAYER_LABELS[layer].replace("\n", " "), color=color, alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("% Improvement over M1")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Marginal Improvement of Multimodal Data over Financial-Only Baseline (M1)",
                 fontsize=13, y=1.02)
    fig.savefig(RESULTS_DIR / "extra_01_marginal_improvement.png")
    plt.close(fig)
    print("[saved] extra_01_marginal_improvement.png")


# ── 2. Best model per layer (grouped bar) ────────────────────────
def plot_best_per_layer():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    configs = [
        ("direction", "macro_f1", "Direction: Best Macro-F1 per Layer", True),
        ("price", "rmse", "Price: Best RMSE per Layer", False),
        ("volatility", "rmse", "Volatility: Best RMSE per Layer", False),
    ]
    for ax, (target, metric, title, higher_better) in zip(axes, configs):
        sub = df[df["target"] == target]
        if metric not in sub.columns or sub.empty:
            ax.set_visible(False)
            continue

        best_models, best_vals = [], []
        for layer in LAYERS:
            lsub = sub[sub["layer"] == layer].dropna(subset=[metric])
            if lsub.empty:
                best_models.append("N/A")
                best_vals.append(0)
                continue
            if higher_better:
                idx = lsub[metric].idxmax()
            else:
                idx = lsub[metric].idxmin()
            best_models.append(lsub.loc[idx, "model"])
            best_vals.append(lsub.loc[idx, metric])

        bars = ax.bar(range(4), best_vals,
                      color=[COLORS[l] for l in LAYERS], alpha=0.85)
        ax.set_xticks(range(4))
        ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
        ax.set_ylabel(metric.upper())
        ax.set_title(title, fontsize=10)
        for bar, model in zip(bars, best_models):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    model, ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Best Model per Data Layer", fontsize=12, y=1.02)
    fig.savefig(RESULTS_DIR / "extra_02_best_per_layer.png")
    plt.close(fig)
    print("[saved] extra_02_best_per_layer.png")


# ── 3. Model ranking across all targets (radar-like dot plot) ────
def plot_model_ranking():
    rank_data = []
    for target, metric, higher_better in [
        ("direction", "macro_f1", True),
        ("price", "rmse", False),
        ("volatility", "rmse", False),
    ]:
        sub = df[(df["target"] == target) & (df["layer"] == "M4")]
        if metric not in sub.columns or sub.empty:
            continue
        sub = sub.dropna(subset=[metric]).copy()
        sub["rank"] = sub[metric].rank(ascending=not higher_better).astype(int)
        for _, row in sub.iterrows():
            rank_data.append({"model": row["model"], "target": target,
                              "rank": row["rank"], "value": row[metric]})

    if not rank_data:
        return
    rdf = pd.DataFrame(rank_data)
    pivot = rdf.pivot(index="model", columns="target", values="rank")

    fig, ax = plt.subplots(figsize=(10, 5))
    models = sorted(pivot.index)
    targets = sorted(pivot.columns)
    x = np.arange(len(models))
    for i, tgt in enumerate(targets):
        vals = [pivot.loc[m, tgt] if m in pivot.index and tgt in pivot.columns
                else np.nan for m in models]
        ax.scatter(x + i * 0.15, vals, s=100, label=tgt.title(), zorder=3)
        for xi, v in zip(x + i * 0.15, vals):
            if not np.isnan(v):
                ax.text(xi, v - 0.3, f"#{int(v)}", ha="center", fontsize=7)

    ax.set_xticks(x + 0.15)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Model Rankings on M4 (All Data) — Lower Rank = Better", fontsize=11)
    ax.legend()
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    fig.savefig(RESULTS_DIR / "extra_03_model_ranking_M4.png")
    plt.close(fig)
    print("[saved] extra_03_model_ranking_M4.png")


# ── 4. Layer progression line chart (M1→M2→M3→M4) ───────────────
def plot_layer_progression():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    configs = [
        ("direction", "macro_f1", "Direction Macro-F1"),
        ("price", "rmse", "Price RMSE"),
        ("volatility", "rmse", "Volatility RMSE"),
    ]
    cmap = plt.cm.tab10
    for ax, (target, metric, title) in zip(axes, configs):
        sub = df[df["target"] == target]
        if metric not in sub.columns or sub.empty:
            ax.set_visible(False)
            continue
        models = sorted(sub["model"].unique())
        for j, model in enumerate(models):
            msub = sub[sub["model"] == model]
            vals = []
            for layer in LAYERS:
                v = msub[msub["layer"] == layer][metric]
                vals.append(v.values[0] if len(v) > 0 else np.nan)
            ax.plot(range(4), vals, "o-", label=model, color=cmap(j),
                    linewidth=1.5, markersize=5, alpha=0.8)

        ax.set_xticks(range(4))
        ax.set_xticklabels([LAYER_LABELS[l] for l in LAYERS])
        ax.set_ylabel(metric.upper())
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, ncol=2, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("Performance Progression: M1 → M2 → M3 → M4", fontsize=12, y=1.02)
    fig.savefig(RESULTS_DIR / "extra_04_layer_progression.png")
    plt.close(fig)
    print("[saved] extra_04_layer_progression.png")


# ── 5. Flat-class problem: all models predict 0% flat ────────────
def plot_class_distribution():
    sub = df[(df["target"] == "direction")].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    recall_cols = ["recall_down", "recall_flat", "recall_up"]
    for col in recall_cols:
        if col not in sub.columns:
            sub[col] = np.nan

    m4_sub = sub[sub["layer"] == "M4"].dropna(subset=recall_cols)
    if not m4_sub.empty:
        models = m4_sub["model"].values
        x = np.arange(len(models))
        w = 0.25
        axes[0].bar(x - w, m4_sub["recall_down"], w, label="Down (-1)", color="#e74c3c")
        axes[0].bar(x, m4_sub["recall_flat"], w, label="Flat (0)", color="#95a5a6")
        axes[0].bar(x + w, m4_sub["recall_up"], w, label="Up (+1)", color="#27ae60")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        axes[0].set_ylabel("Recall")
        axes[0].set_title("Per-Class Recall on M4 — Direction", fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.3)

    prec_cols = ["precision_down", "precision_flat", "precision_up"]
    for col in prec_cols:
        if col not in sub.columns:
            sub[col] = np.nan
    m4_sub2 = sub[sub["layer"] == "M4"].dropna(subset=prec_cols)
    if not m4_sub2.empty:
        models = m4_sub2["model"].values
        x = np.arange(len(models))
        axes[1].bar(x - w, m4_sub2["precision_down"], w, label="Down (-1)", color="#e74c3c")
        axes[1].bar(x, m4_sub2["precision_flat"], w, label="Flat (0)", color="#95a5a6")
        axes[1].bar(x + w, m4_sub2["precision_up"], w, label="Up (+1)", color="#27ae60")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Per-Class Precision on M4 — Direction", fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Class-Level Performance Analysis (Direction Prediction)", fontsize=12, y=1.02)
    fig.savefig(RESULTS_DIR / "extra_05_class_distribution.png")
    plt.close(fig)
    print("[saved] extra_05_class_distribution.png")


# ── 6. Summary dashboard ────────────────────────────────────────
def plot_summary_dashboard():
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")

    configs = [
        (1, "direction", "accuracy", "Direction Accuracy", True),
        (2, "direction", "macro_f1", "Direction Macro-F1", True),
        (3, "price", "rmse", "Price RMSE", False),
        (4, "price", "r2", "Price R²", True),
        (5, "volatility", "rmse", "Volatility RMSE", False),
        (6, "volatility", "r2", "Volatility R²", True),
    ]
    for pos, target, metric, title, higher_better in configs:
        ax = fig.add_subplot(2, 3, pos)
        sub = df[df["target"] == target]
        if metric not in sub.columns or sub.empty:
            ax.set_visible(False)
            continue

        pivot = sub.pivot_table(index="model", columns="layer",
                                values=metric, aggfunc="first")
        layer_order = [l for l in LAYERS if l in pivot.columns]
        pivot = pivot.reindex(columns=layer_order)

        cmap = "YlGn" if higher_better else "YlOrRd_r"
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap,
                    linewidths=0.3, ax=ax, annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.6})
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)

    fig.suptitle("Complete 2D Comparison Dashboard (5 Models × 4 Layers × 3 Targets)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(RESULTS_DIR / "extra_06_summary_dashboard.png")
    plt.close(fig)
    print("[saved] extra_06_summary_dashboard.png")


if __name__ == "__main__":
    print("Generating additional visualizations...\n")
    plot_marginal_improvement()
    plot_best_per_layer()
    plot_model_ranking()
    plot_layer_progression()
    plot_class_distribution()
    plot_summary_dashboard()
    print(f"\nAll extra visualizations saved to: {RESULTS_DIR}")
