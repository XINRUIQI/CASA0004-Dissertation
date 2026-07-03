"""
B2 - Channel B (M2) mechanism-validation EDA.

Reads the weekly Channel B table (B1 long form) + M1 weekly features and produces
interpretability evidence:
  1) site-type composite anomaly time series per indicator, with event markers
  2) lead-lag cross-correlation of Channel B composites vs Brent log return
     (which weeks Channel B *leads* the market = predictive direction)
  3) contemporaneous correlation of per-site NTL/NDBI anomalies vs key M1 vars
  4) Houston (the only US AOI) anomalies vs US refinery utilisation / exports
  + optional Granger causality on the strongest leading features

INTERPRETATION CAVEAT: M1 fundamentals are *US* (EIA); Channel B AOIs are *global*.
The clean mechanism link is Channel B activity -> global Brent price; only Houston
has a direct geographic match to US fundamentals.

Outputs (-> 03_data/processed/M2/outputs/):
  m2_eda_anom_by_type.png  m2_eda_leadlag_brent.png
  m2_eda_corr_m1.png       m2_eda_houston_us.png
  m2_eda_leadlag_corr.csv

Run:
  python3 03_data/processed/M2/py/eda_m2_mechanism.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "03_data/processed/M2/outputs"
M2_LONG = OUT / "m2_eda_weekly.csv"
M1_CSV = ROOT / "03_data/processed/M1/outputs/m1_weekly_features.csv"

INDEX_ORDER = ["NDVI", "NDWI", "NDBI", "BSI", "NTL"]
EVENTS = {
    "2020-03-13": "COVID crash",
    "2022-02-24": "Russia-Ukraine",
    "2023-10-13": "Red Sea / Houthi",
    "2024-01-12": "Red Sea escalation",
}
M1_VARS = ["brent_log_return", "brent_price", "refinery_utilisation",
           "crude_exports", "crude_imports", "crude_production", "ovx"]
LAGS = list(range(-8, 9))


def load():
    m2 = pd.read_csv(M2_LONG, parse_dates=["week_fri"])
    m1 = pd.read_csv(M1_CSV, parse_dates=["week_ending_friday"]).rename(
        columns={"week_ending_friday": "week_fri"})
    return m2, m1


def composite_anom(m2: pd.DataFrame) -> pd.DataFrame:
    comp = (m2.groupby(["week_fri", "index", "site_type"])["anom"]
            .mean().reset_index())
    comp["col"] = comp["index"] + "_" + comp["site_type"]
    return comp.pivot_table(index="week_fri", columns="col", values="anom")


def leadlag(wide_cb: pd.DataFrame, m1: pd.DataFrame,
            target="brent_log_return") -> pd.DataFrame:
    df = wide_cb.join(m1.set_index("week_fri")[[target]], how="inner")
    rows = {col: {k: df[col].corr(df[target].shift(-k)) for k in LAGS}
            for col in wide_cb.columns}
    return pd.DataFrame(rows).T


def plot_anom_by_type(m2, path):
    comp = (m2.groupby(["week_fri", "index", "site_type"])["anom"]
            .mean().reset_index())
    fig, axes = plt.subplots(len(INDEX_ORDER), 1, figsize=(14, 13), sharex=True)
    for ax, idx in zip(axes, INDEX_ORDER):
        sub = comp[comp["index"] == idx]
        for st, g in sub.groupby("site_type"):
            ax.plot(g["week_fri"], g["anom"], label=st, lw=1)
        for d in EVENTS:
            ax.axvline(pd.Timestamp(d), color="grey", ls="--", lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(idx)
        ax.legend(loc="upper left", fontsize=7, ncol=3)
    ymax = axes[0].get_ylim()[1]
    for d, lbl in EVENTS.items():
        axes[0].text(pd.Timestamp(d), ymax, lbl, rotation=90, va="top",
                     fontsize=6, color="grey")
    axes[0].set_title("Channel B site-type composite anomaly (mean within-site z-score) by indicator")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_leadlag(ll, path):
    disp = ll.copy()
    disp.columns = [f"{k:+d}" for k in ll.columns]
    plt.figure(figsize=(11, 8))
    sns.heatmap(disp, cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"label": "corr with Brent log return"})
    plt.title("Lead-lag: Channel B composite anomaly vs Brent log return\n"
              "lag k>0 = Channel B leads the market (predictive direction)")
    plt.xlabel("lag k (weeks):  anomaly_t  vs  return_{t+k}")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_corr_m1(m2, m1, path):
    sub = m2[m2["index"].isin(["NTL", "NDBI"])].copy()
    sub["col"] = sub["index"] + "_anom_" + sub["short_name"]
    wide = sub.pivot_table(index="week_fri", columns="col", values="anom")
    df = wide.join(m1.set_index("week_fri")[M1_VARS], how="inner")
    corr = df.corr().loc[wide.columns, M1_VARS]
    plt.figure(figsize=(8, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-0.4, vmax=0.4,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"label": "contemporaneous corr"})
    plt.title("Per-site NTL / NDBI anomaly vs M1 fundamentals (contemporaneous)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_houston(m2, m1, path):
    h = m2[(m2["short_name"] == "Houston") & (m2["index"].isin(["NTL", "NDBI"]))]
    hw = h.pivot_table(index="week_fri", columns="index", values="anom")
    m1i = m1.set_index("week_fri")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(hw.index, hw.get("NTL"), color="purple", lw=1, label="Houston NTL anom")
    a0 = axes[0].twinx()
    a0.plot(m1i.index, m1i["refinery_utilisation"], color="orange", lw=1, alpha=0.6)
    axes[0].set_title("Houston NTL anomaly (purple) vs US refinery utilisation (orange)")
    axes[1].plot(hw.index, hw.get("NDBI"), color="brown", lw=1, label="Houston NDBI anom")
    a1 = axes[1].twinx()
    a1.plot(m1i.index, m1i["crude_exports"], color="green", lw=1, alpha=0.6)
    axes[1].set_title("Houston NDBI anomaly (brown) vs US crude exports (green)")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.4)
        ax.set_xlim(hw.index.min(), hw.index.max())
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def granger_top(wide_cb, m1, ll, target="brent_log_return", topn=4):
    from statsmodels.tsa.stattools import grangercausalitytests
    lead = (ll[[k for k in ll.columns if k > 0]].abs().max(axis=1)
            .sort_values(ascending=False))
    df = wide_cb.join(m1.set_index("week_fri")[[target]], how="inner")
    print(f"\n[Granger causality -> {target}, maxlag=4]  (H0: feature does NOT cause)")
    for col in lead.head(topn).index:
        pair = df[[target, col]].dropna()
        if len(pair) < 40:
            print(f"  {col:22s} skipped (n={len(pair)})")
            continue
        try:
            res = grangercausalitytests(pair, maxlag=4, verbose=False)
            pvals = [round(res[l][0]["ssr_ftest"][1], 3) for l in range(1, 5)]
            print(f"  {col:22s} min p={min(pvals):.3f}  (lags1-4: {pvals})")
        except Exception as e:
            print(f"  {col:22s} skipped ({type(e).__name__})")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    m2, m1 = load()
    wide_cb = composite_anom(m2)

    plot_anom_by_type(m2, OUT / "m2_eda_anom_by_type.png")

    ll = leadlag(wide_cb, m1)
    ll.to_csv(OUT / "m2_eda_leadlag_corr.csv")
    plot_leadlag(ll, OUT / "m2_eda_leadlag_brent.png")
    plot_corr_m1(m2, m1, OUT / "m2_eda_corr_m1.png")
    plot_houston(m2, m1, OUT / "m2_eda_houston_us.png")
    granger_top(wide_cb, m1, ll)

    print(f"\nfigures -> {OUT}/m2_eda_*.png")
    print(f"leadlag -> {OUT}/m2_eda_leadlag_corr.csv")
    best = ll.abs().max(axis=1).sort_values(ascending=False).head(6)
    print("\n[strongest |lead-lag corr| with Brent log return]")
    for col, v in best.items():
        k = int(ll.loc[col].abs().idxmax())
        print(f"  {col:18s} max|corr|={v:.3f} at lag {k:+d}")


if __name__ == "__main__":
    main()
