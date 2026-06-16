"""
Master runner: execute all models and produce the 2D comparison matrix.
Corresponds to project_plan Section 9 (Two-Dimensional Comparison).
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def run_script(name: str):
    """Run a model script as subprocess."""
    script = SCRIPT_DIR / name
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script)],
                           capture_output=False, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"  [WARNING] {name} exited with code {result.returncode}")
    return result.returncode


def aggregate_results():
    """Load all result CSVs and merge into one table."""
    all_rows = []
    for csv_file in sorted(RESULTS_DIR.glob("*_results.csv")):
        df = pd.read_csv(csv_file)
        all_rows.append(df)
    if not all_rows:
        print("No results found!")
        return pd.DataFrame()
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "all_results_combined.csv", index=False)
    print(f"\nCombined results: {len(combined)} rows → all_results_combined.csv")
    return combined


def generate_comparison_heatmaps(combined: pd.DataFrame):
    """
    Generate the 2D comparison matrix visualization (Section 9.1).
    One heatmap per target × metric.
    """
    if combined.empty:
        return

    target_metrics = {
        "direction": ["accuracy", "macro_f1"],
        "volatility": ["rmse", "r2"],
        "price": ["rmse", "r2"],
    }

    for target, metrics in target_metrics.items():
        sub = combined[combined["target"] == target]
        if sub.empty:
            continue
        for metric in metrics:
            if metric not in sub.columns:
                continue
            pivot = sub.pivot_table(index="model", columns="layer",
                                    values=metric, aggfunc="first")
            layer_order = [l for l in ["M1", "M2", "M3", "M4", "M5"] if l in pivot.columns]
            pivot = pivot.reindex(columns=layer_order)

            fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.6 + 1)))
            cmap = "YlGn" if metric in ["accuracy", "macro_f1", "r2"] else "YlOrRd_r"
            sns.heatmap(pivot, annot=True, fmt=".4f", cmap=cmap,
                        linewidths=0.5, ax=ax)
            ax.set_title(f"{target.title()} — {metric.upper()} (Models × Layers)")
            fig.savefig(RESULTS_DIR / f"heatmap_{target}_{metric}.png",
                        bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"  [saved] heatmap_{target}_{metric}.png")


def generate_m1_vs_m4_chart(combined: pd.DataFrame):
    """Bar chart comparing M1 vs M4 across all models for each target."""
    if combined.empty:
        return

    for target, metric in [("direction", "macro_f1"), ("volatility", "rmse"), ("price", "rmse")]:
        sub = combined[combined["target"] == target]
        if sub.empty or metric not in sub.columns:
            continue

        m1 = sub[sub["layer"] == "M1"].set_index("model")[metric]
        m4 = sub[sub["layer"] == "M4"].set_index("model")[metric]
        models = sorted(set(m1.index) & set(m4.index))
        if not models:
            continue

        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 0.2, [m1.get(m, np.nan) for m in models], 0.35,
               label="M1 (Financial)", color="#2196F3")
        ax.bar(x + 0.2, [m4.get(m, np.nan) for m in models], 0.35,
               label="M4 (All)", color="#FF9800")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"M1 vs M4: {target.title()} — {metric.upper()}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.savefig(RESULTS_DIR / f"m1_vs_m4_{target}_{metric}.png",
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  [saved] m1_vs_m4_{target}_{metric}.png")


def generate_m4_vs_m5_chart(combined: pd.DataFrame):
    """Bar chart comparing M4 vs M5 (GDELT supplement) for Discussion/Appendix."""
    if combined.empty:
        return

    for target, metric in [("direction", "macro_f1"), ("volatility", "rmse"), ("price", "rmse")]:
        sub = combined[combined["target"] == target]
        if sub.empty or metric not in sub.columns:
            continue

        m4 = sub[sub["layer"] == "M4"].set_index("model")[metric]
        m5 = sub[sub["layer"] == "M5"].set_index("model")[metric]
        models = sorted(set(m4.index) & set(m5.index))
        if not models:
            continue

        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 0.2, [m4.get(m, np.nan) for m in models], 0.35,
               label="M4 (NTL+Ship)", color="#FF9800")
        ax.bar(x + 0.2, [m5.get(m, np.nan) for m in models], 0.35,
               label="M5 (M4+GDELT)", color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"M4 vs M5 (GDELT): {target.title()} — {metric.upper()}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.savefig(RESULTS_DIR / f"m4_vs_m5_{target}_{metric}.png",
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  [saved] m4_vs_m5_{target}_{metric}.png")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scripts = [
        "01_baselines.py",
        "02_xgboost_model.py",
        "03_lstm_model.py",
        "04_tft_model.py",
        "05_stgnn_model.py",
    ]

    for script in scripts:
        if not (SCRIPT_DIR / script).exists():
            print(f"  [skip] {script} not found")
            continue
        run_script(script)

    print(f"\n{'='*60}")
    print("Aggregating results...")
    print(f"{'='*60}")

    combined = aggregate_results()

    print("\nGenerating comparison visualizations...")
    generate_comparison_heatmaps(combined)
    generate_m1_vs_m4_chart(combined)
    generate_m4_vs_m5_chart(combined)

    print(f"\n{'='*60}")
    print(f"All done! Results in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
