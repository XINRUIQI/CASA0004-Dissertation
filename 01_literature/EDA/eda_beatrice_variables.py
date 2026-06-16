"""
EDA for Beatrice-recommended variable subsets (M1–M4).
Outputs all figures and summary tables into 01_literature/EDA/.

Variable subsets derived from beatrice_task_literature_matrix.md.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
})

PROJECT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT / "03_data" / "processed" / "weekly_features.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Variable subsets ─────────────────────────────────────────────
# AOI site names for readable labels
AOI_NAMES = {
    "P001": "Rotterdam", "P002": "Fujairah", "P003": "Ras Tanura",
    "P004": "Jurong Island", "P005": "Houston", "P006": "Ningbo",
    "P007": "Jamnagar", "P008": "Basra", "P009": "Ulsan",
    "P010": "Kharg Island", "P011": "Yanbu",
}

# M1 = mechanism-based 10-variable set (post-close-reading, see
# beatrice_task_literature_matrix.md §①).
M1_VARS = [
    "brent_price", "crude_stocks_change", "global_econ_activity",
    "nonoil_industrial_commodity", "futures_spread", "ovx",
    "gpr", "dgs10_change", "gold_return", "commodity_fx",
]

# Clean M2 set: dynamic NTL anomaly (not raw radiance) per AOI, per
# beatrice_task_literature_matrix.md §②.
M2_RS_VARS = [
    "ntl_anomaly_rotterdam",   # Rotterdam (import / refining)
    "ntl_anomaly_fujairah",    # Fujairah (offshore tanker storage)
    "ntl_anomaly_ras_tanura",  # Ras Tanura (crude export)
    "ntl_anomaly_us_gulf",     # Houston / US Gulf
    "ntl_anomaly_ningbo",      # Ningbo (China demand proxy)
]

M3_SHIP_VARS = [
    "pw_hormuz_n_tanker", "pw_hormuz_capacity_tanker",
    "pw_suez_n_tanker", "pw_suez_capacity_tanker",
    "pw_malacca_n_tanker", "pw_malacca_capacity_tanker",
    "pw_all_n_tanker_sum",
]

TARGETS = [
    "target_brent_price_next_1w",
    "target_brent_vol_next_1w",
    "target_brent_direction_next_1w",
]

def pretty(col: str) -> str:
    """Shorten column name for display."""
    for code, name in AOI_NAMES.items():
        col = col.replace(f"_{code}", f"\n({name})")
    return (col
        .replace("ntl_anomaly_", "NTL anom ")
        .replace("ntl_valid_obs_count_", "NTL validobs ")
        .replace("s2_cloud_fraction_", "S2 cloudfrac ")
        .replace("s2_clear_obs_count_", "S2 clearobs ")
        .replace("pw_", "PW ")
        .replace("gfw_", "GFW ")
        .replace("target_brent_", "Target:")
        .replace("_", " "))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df.index.name = "week_ending_friday"
    return df


# ── 1. Summary statistics ───────────────────────────────────────
def summary_stats(df: pd.DataFrame) -> None:
    all_vars = M1_VARS + M2_RS_VARS + M3_SHIP_VARS + TARGETS
    present = [c for c in all_vars if c in df.columns]
    stats = df[present].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    stats["missing"] = df[present].isna().sum()
    stats["missing%"] = (stats["missing"] / len(df) * 100).round(1)
    stats["non_null"] = df[present].notna().sum()
    stats.to_csv(OUT_DIR / "01_summary_statistics.csv")
    print(f"[saved] 01_summary_statistics.csv  ({len(present)} vars)")


# ── 2. Missing data heatmap ─────────────────────────────────────
def missing_heatmap(df: pd.DataFrame) -> None:
    all_vars = M1_VARS + M2_RS_VARS + M3_SHIP_VARS
    present = [c for c in all_vars if c in df.columns]
    yearly = df[present].notna().astype(int)
    yearly["year"] = df.index.year
    by_year = yearly.groupby("year").mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(by_year.T, annot=True, fmt=".0%", cmap="YlGn",
                linewidths=0.3, ax=ax, vmin=0, vmax=1,
                yticklabels=[pretty(c) for c in present])
    ax.set_title("Data Availability by Year (Beatrice Subset)", fontsize=12)
    ax.set_xlabel("Year")
    fig.savefig(OUT_DIR / "02_missing_heatmap.png")
    plt.close(fig)
    print("[saved] 02_missing_heatmap.png")


# ── 3. Time series: M1 variables ────────────────────────────────
def ts_m1(df: pd.DataFrame) -> None:
    present = [c for c in M1_VARS if c in df.columns]
    n = len(present)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, present):
        ax.plot(df.index, df[col], linewidth=0.7, color="steelblue")
        ax.set_ylabel(pretty(col), fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("M1 Financial Variables — Time Series", fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(OUT_DIR / "03_timeseries_M1.png")
    plt.close(fig)
    print("[saved] 03_timeseries_M1.png")


# ── 4. Time series: M2 RS (NTL) ────────────────────────────────
def ts_m2(df: pd.DataFrame) -> None:
    present = [c for c in M2_RS_VARS if c in df.columns]
    fig, axes = plt.subplots(len(present) + 1, 1,
                             figsize=(14, 2.2 * (len(present) + 1)), sharex=True)
    axes[0].plot(df.index, df["brent_price"], linewidth=0.7, color="black")
    axes[0].set_ylabel("Brent Price\n(USD)", fontsize=8)
    axes[0].grid(alpha=0.3)
    for ax, col in zip(axes[1:], present):
        s = df[col].dropna()
        ax.plot(s.index, s.values, linewidth=0.7, color="darkorange")
        ax.set_ylabel(pretty(col), fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("M2 Remote Sensing (NTL) vs Brent Price", fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(OUT_DIR / "04_timeseries_M2_NTL.png")
    plt.close(fig)
    print("[saved] 04_timeseries_M2_NTL.png")


# ── 5. Time series: M3 Shipping ─────────────────────────────────
def ts_m3(df: pd.DataFrame) -> None:
    present = [c for c in M3_SHIP_VARS if c in df.columns]
    fig, axes = plt.subplots(len(present) + 1, 1,
                             figsize=(14, 2.2 * (len(present) + 1)), sharex=True)
    axes[0].plot(df.index, df["brent_price"], linewidth=0.7, color="black")
    axes[0].set_ylabel("Brent Price\n(USD)", fontsize=8)
    axes[0].grid(alpha=0.3)
    for ax, col in zip(axes[1:], present):
        s = df[col].dropna()
        ax.plot(s.index, s.values, linewidth=0.7, color="seagreen")
        ax.set_ylabel(pretty(col), fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("M3 Shipping (PortWatch Tanker) vs Brent Price", fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(OUT_DIR / "05_timeseries_M3_shipping.png")
    plt.close(fig)
    print("[saved] 05_timeseries_M3_shipping.png")


# ── 6. Correlation heatmaps ─────────────────────────────────────
def corr_heatmap(df: pd.DataFrame, cols: list[str], title: str, fname: str) -> None:
    present = [c for c in cols if c in df.columns]
    sub = df[present].dropna()
    if sub.empty or len(sub) < 30:
        print(f"  [skip] {fname} — insufficient overlapping data ({len(sub)} rows)")
        return
    corr = sub.corr()
    labels = [pretty(c) for c in present]
    fig, ax = plt.subplots(figsize=(max(8, len(present) * 0.9),
                                    max(6, len(present) * 0.7)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.3,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 7})
    ax.set_title(title, fontsize=12)
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)
    print(f"[saved] {fname}")


def correlation_analysis(df: pd.DataFrame) -> None:
    corr_heatmap(df, M1_VARS, "M1 Financial — Correlation Matrix",
                 "06a_corr_M1.png")
    corr_heatmap(df, M2_RS_VARS, "M2 RS (NTL) — Correlation Matrix",
                 "06b_corr_M2_NTL.png")
    corr_heatmap(df, M3_SHIP_VARS, "M3 Shipping — Correlation Matrix",
                 "06c_corr_M3_ship.png")
    corr_heatmap(df, M1_VARS + M2_RS_VARS + M3_SHIP_VARS,
                 "M4 All Variables — Correlation Matrix",
                 "06d_corr_M4_all.png")


# ── 7. Correlation with targets ─────────────────────────────────
def corr_with_targets(df: pd.DataFrame) -> None:
    all_vars = M1_VARS + M2_RS_VARS + M3_SHIP_VARS
    present = [c for c in all_vars if c in df.columns]
    tgt_present = [c for c in TARGETS if c in df.columns]
    if not tgt_present:
        print("  [skip] no target variables found")
        return

    records = []
    for feat in present:
        for tgt in tgt_present:
            sub = df[[feat, tgt]].dropna()
            if len(sub) < 30:
                continue
            r = sub[feat].corr(sub[tgt])
            modality = ("M1" if feat in M1_VARS
                        else "M2" if feat in M2_RS_VARS
                        else "M3")
            records.append({"feature": feat, "target": tgt,
                            "corr": r, "abs_corr": abs(r),
                            "n_obs": len(sub), "modality": modality})
    corr_df = pd.DataFrame(records).sort_values("abs_corr", ascending=False)
    corr_df.to_csv(OUT_DIR / "07_correlation_with_targets.csv", index=False)

    fig, axes = plt.subplots(1, len(tgt_present), figsize=(6 * len(tgt_present), 8))
    if len(tgt_present) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, tgt_present):
        sub_df = corr_df[corr_df["target"] == tgt].sort_values("corr")
        colors = ["#2196F3" if m == "M1" else "#FF9800" if m == "M2"
                   else "#4CAF50" for m in sub_df["modality"]]
        ax.barh(range(len(sub_df)), sub_df["corr"], color=colors)
        ax.set_yticks(range(len(sub_df)))
        ax.set_yticklabels([pretty(f) for f in sub_df["feature"]], fontsize=7)
        ax.set_xlabel("Pearson r")
        ax.set_title(pretty(tgt), fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Feature–Target Correlations (Blue=M1, Orange=M2, Green=M3)",
                 fontsize=12, y=1.01)
    fig.savefig(OUT_DIR / "07_correlation_with_targets.png")
    plt.close(fig)
    print("[saved] 07_correlation_with_targets.csv + .png")


# ── 8. Distributions ────────────────────────────────────────────
def distributions(df: pd.DataFrame) -> None:
    all_vars = M1_VARS + M2_RS_VARS + M3_SHIP_VARS
    present = [c for c in all_vars if c in df.columns]
    ncols = 4
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(present):
        s = df[col].dropna()
        axes[i].hist(s, bins=50, color="slategray", edgecolor="white",
                     linewidth=0.3, alpha=0.85)
        axes[i].set_title(pretty(col), fontsize=8)
        axes[i].tick_params(labelsize=6)
    for j in range(len(present), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distributions — Beatrice Variable Subset", fontsize=12, y=1.01)
    fig.savefig(OUT_DIR / "08_distributions.png")
    plt.close(fig)
    print("[saved] 08_distributions.png")


# ── 9. Rolling correlation with brent ───────────────────────────
def rolling_corr(df: pd.DataFrame) -> None:
    key_features = {
        "M1: OVX": "ovx",
        "M1: GPR": "gpr",
        "M1: Stocks Δ": "crude_stocks_change",
        "M2: NTL anom Rotterdam": "ntl_anomaly_rotterdam",
        "M2: NTL anom Fujairah": "ntl_anomaly_fujairah",
        "M3: PW Hormuz tanker": "pw_hormuz_n_tanker",
        "M3: PW Suez tanker": "pw_suez_n_tanker",
    }
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_features)))
    for (label, col), color in zip(key_features.items(), colors):
        if col not in df.columns:
            continue
        rc = df[col].rolling(52, min_periods=26).corr(df["brent_price"])
        ax.plot(df.index, rc, label=label, linewidth=0.9, color=color, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("52-week Rolling Correlation with Brent Price")
    ax.set_title("Rolling Correlation — Key Features vs Brent Price", fontsize=12)
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(OUT_DIR / "09_rolling_correlation.png")
    plt.close(fig)
    print("[saved] 09_rolling_correlation.png")


# ── 10. M4 scatter: top correlated ──────────────────────────────
def scatter_top(df: pd.DataFrame) -> None:
    target = "target_brent_price_next_1w"
    if target not in df.columns:
        return
    candidates = M1_VARS + M2_RS_VARS + M3_SHIP_VARS
    present = [c for c in candidates if c in df.columns]
    corrs = {}
    for c in present:
        sub = df[[c, target]].dropna()
        if len(sub) >= 30:
            corrs[c] = abs(sub[c].corr(sub[target]))
    top6 = sorted(corrs, key=corrs.get, reverse=True)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, top6):
        sub = df[[col, target]].dropna()
        modality = ("M1" if col in M1_VARS else "M2" if col in M2_RS_VARS else "M3")
        c = {"M1": "#2196F3", "M2": "#FF9800", "M3": "#4CAF50"}[modality]
        ax.scatter(sub[col], sub[target], s=4, alpha=0.4, color=c)
        ax.set_xlabel(pretty(col), fontsize=8)
        ax.set_ylabel("Next-Week Brent", fontsize=8)
        ax.set_title(f"r = {corrs[col]:.3f}  [{modality}]", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Top 6 Features by |correlation| with Next-Week Brent Price", fontsize=12)
    fig.savefig(OUT_DIR / "10_scatter_top_features.png")
    plt.close(fig)
    print("[saved] 10_scatter_top_features.png")


# ── Main ────────────────────────────────────────────────────────
def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = load_data()
    print(f"Shape: {df.shape}, Period: {df.index.min()} ~ {df.index.max()}\n")

    summary_stats(df)
    missing_heatmap(df)
    ts_m1(df)
    ts_m2(df)
    ts_m3(df)
    correlation_analysis(df)
    corr_with_targets(df)
    distributions(df)
    rolling_corr(df)
    scatter_top(df)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
