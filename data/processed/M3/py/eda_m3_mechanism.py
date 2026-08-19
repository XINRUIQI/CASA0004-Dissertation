"""
M3 (shipping / maritime-network) mechanism-validation EDA.

Companion to `data/processed/M2/py/eda_m2_mechanism.py`, providing the
shipping-side mechanistic validation that Meeting 03 asked for: compare the
remote-sensing AND shipping proxies to oil prices as exploratory evidence,
before judging them by out-of-sample skill.

Reads the weekly M3 wide table (`m3_weekly_features.csv`) + M1 weekly features
and produces interpretability evidence:
  1) chokepoint-level tanker-capacity activity (within-series z-score) with
     event markers (COVID, Russia-Ukraine, Red Sea / Houthi)
  2) Red Sea re-routing mechanism: Bab el-Mandeb + Suez tanker capacity falling
     while Cape of Good Hope rises after the Oct-2023 / Jan-2024 disruption
  3) lead-lag cross-correlation of shipping *changes* vs Brent log return
     (lag k>0 = shipping leads the market = predictive direction)
  4) contemporaneous correlation of shipping changes vs key M1 fundamentals
     (crude exports/imports/production, refinery utilisation, OVX)
  + optional Granger causality on the strongest leading shipping signals

INTERPRETATION CAVEATS
  - Level series (transit capacity, presence hours, export/import volume) are
    non-stationary; all correlations use weekly log-changes / first differences
    so a "change in shipping vs change in price" reading is stationarity-safe.
  - REVERSE CAUSALITY: Mi et al. (2022) [P016] and Mi et al. (2023) [P017] show
    the documented causal arrow runs oil price -> tanker port-call activity, not
    the reverse. A negative lag (k<0, price leads shipping) is therefore the
    expected sign; any positive-lag (k>0) predictive content is what this study
    would exploit and must be read cautiously.
  - Chokepoint transit is a coarse proxy for oil trade (P070 / P018): tanker =
    liquid-bulk grouping (not only crude), capacity = DWT-equivalent transit
    proxy (not loaded barrels). PortWatch begins 2019; the standard 2019-2026
    window is used throughout for comparability with the modelling protocol.

Outputs (-> data/processed/M3/outputs/):
  m3_eda_chokepoint_activity.png   m3_eda_redsea_rerouting.png
  m3_eda_leadlag_brent.png         m3_eda_corr_m1.png
  m3_eda_leadlag_corr.csv

Run:
  python3 data/processed/M3/py/eda_m3_mechanism.py
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
OUT = ROOT / "data/processed/M3/outputs"
M3_CSV = OUT / "m3_weekly_features.csv"
M1_CSV = ROOT / "data/processed/M1/outputs/m1_weekly_features.csv"

WINDOW_START = "2019-01-01"  # PortWatch begins 2019; matches the modelling window
WINDOW_END = "2025-12-31"    # standard 2019-2025 comparison window (drop partial 2026)
CHOKES = ["hormuz", "suez", "malacca", "mandeb", "panama", "cape"]
CHOKE_LABEL = {
    "hormuz": "Hormuz", "suez": "Suez", "malacca": "Malacca",
    "mandeb": "Bab el-Mandeb", "panama": "Panama", "cape": "Cape of Good Hope",
}
EVENTS = {
    "2020-03-13": "COVID crash",
    "2022-02-24": "Russia-Ukraine",
    "2023-10-13": "Red Sea / Houthi",
    "2024-01-12": "Red Sea escalation",
}
# M1 fundamentals for the contemporaneous mechanism cross-check (US EIA-based).
M1_VARS = ["brent_log_return", "crude_exports", "crude_imports",
           "crude_production", "refinery_utilisation", "ovx"]
LAGS = list(range(-8, 9))

# Curated shipping signals for the lead-lag / M1 correlation panels.
# capacity-weighted tanker transit (P070/P018 prefer this over vessel counts).
LEADLAG_SIGNALS = {
    "pw_hormuz_capacity_tanker": "Hormuz tanker cap.",
    "pw_suez_capacity_tanker": "Suez tanker cap.",
    "pw_malacca_capacity_tanker": "Malacca tanker cap.",
    "pw_mandeb_capacity_tanker": "Bab el-Mandeb tanker cap.",
    "pw_cape_capacity_tanker": "Cape tanker cap.",
    "pw_all_n_tanker_sum": "All-choke tanker transits",
    "pw_exp_hubs_export_vol": "Export-hub tanker vol.",
    "pw_imp_hubs_import_vol": "Import-hub tanker vol.",
    "pw_tanker_exp_imp_log_ratio": "Export/Import log-ratio",
    "gfw_hormuz_total_hours": "Hormuz GFW presence",
    "gfw_suez_total_hours": "Suez GFW presence",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    m3 = pd.read_csv(M3_CSV, parse_dates=["week_ending_friday"]).rename(
        columns={"week_ending_friday": "week_fri"}).set_index("week_fri")
    m1 = pd.read_csv(M1_CSV, parse_dates=["week_ending_friday"]).rename(
        columns={"week_ending_friday": "week_fri"}).set_index("week_fri")
    m3 = m3.loc[(m3.index >= WINDOW_START) & (m3.index <= WINDOW_END)]
    m1 = m1.loc[(m1.index >= WINDOW_START) & (m1.index <= WINDOW_END)]
    return m3, m1


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * np.nan


def weekly_change(s: pd.Series) -> pd.Series:
    """Stationarity-safe weekly change: log-change for strictly-positive level
    series, first difference otherwise (e.g. an already-logged ratio)."""
    s = s.astype(float)
    if s.dropna().gt(0).all() and s.name != "pw_tanker_exp_imp_log_ratio":
        return np.log(s).diff()
    return s.diff()


def m1_change(s: pd.Series) -> pd.Series:
    if s.name == "brent_log_return":
        return s  # already a return
    if s.name == "ovx":
        return s.diff()  # level -> change in implied vol
    pos = s.dropna().gt(0).all()
    return np.log(s).diff() if pos else s.diff()


# --------------------------------------------------------------------------- #
def plot_chokepoint_activity(m3: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for cp in CHOKES:
        col = f"pw_{cp}_capacity_tanker"
        if col in m3:
            ax.plot(m3.index, zscore(m3[col]).rolling(4, min_periods=1).mean(),
                    lw=1.2, label=CHOKE_LABEL[cp])
    for d in EVENTS:
        ax.axvline(pd.Timestamp(d), color="grey", ls="--", lw=0.8)
    ymax = ax.get_ylim()[1]
    for d, lbl in EVENTS.items():
        ax.text(pd.Timestamp(d), ymax, lbl, rotation=90, va="top",
                fontsize=7, color="grey")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("tanker transit capacity\n(within-series z-score, 4w MA)")
    ax.set_title("PortWatch chokepoint tanker-transit capacity (2019-2025)")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_redsea_rerouting(m3: pd.DataFrame, path: Path) -> None:
    """The clearest shipping mechanism: after the Oct-2023 Houthi disruption,
    Bab el-Mandeb + Suez tanker capacity falls while Cape of Good Hope rises."""
    sub = m3.loc[m3.index >= "2022-06-01"]
    fig, ax = plt.subplots(figsize=(14, 6))
    styles = {"mandeb": ("crimson", "-"), "suez": ("darkorange", "-"),
              "cape": ("navy", "-")}
    for cp, (c, ls) in styles.items():
        col = f"pw_{cp}_capacity_tanker"
        if col in sub:
            ax.plot(sub.index, zscore(m3[col]).reindex(sub.index)
                    .rolling(4, min_periods=1).mean(),
                    color=c, ls=ls, lw=1.6, label=CHOKE_LABEL[cp])
    for d in ["2023-10-13", "2024-01-12"]:
        ax.axvline(pd.Timestamp(d), color="grey", ls="--", lw=1.0)
        ax.text(pd.Timestamp(d), ax.get_ylim()[1], EVENTS[d], rotation=90,
                va="top", fontsize=8, color="grey")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("tanker transit capacity\n(within-series z-score, 4w MA)")
    ax.set_title("Red Sea re-routing mechanism: Bab el-Mandeb / Suez fall as "
                 "Cape of Good Hope rises after Oct-2023")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def leadlag(m3: pd.DataFrame, m1: pd.DataFrame,
            target="brent_log_return") -> pd.DataFrame:
    tgt = m1[target]
    rows = {}
    for col, lbl in LEADLAG_SIGNALS.items():
        if col not in m3:
            continue
        chg = weekly_change(m3[col])
        rows[lbl] = {k: chg.corr(tgt.shift(-k)) for k in LAGS}
    return pd.DataFrame(rows).T


def plot_leadlag(ll: pd.DataFrame, path: Path) -> None:
    disp = ll.copy()
    disp.columns = [f"{k:+d}" for k in ll.columns]
    plt.figure(figsize=(12, 7))
    sns.heatmap(disp, cmap="RdBu_r", center=0, vmin=-0.3, vmax=0.3,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"label": "corr with Brent log return"})
    plt.title("Lead-lag: shipping change vs Brent log return\n"
              "lag k>0 = shipping leads the market (predictive); "
              "k<0 = price leads shipping (reverse causality, Mi et al. 2022/23)")
    plt.xlabel("lag k (weeks):  shipping_change_t  vs  return_{t+k}")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_corr_m1(m3: pd.DataFrame, m1: pd.DataFrame, path: Path) -> None:
    ship = pd.DataFrame({lbl: weekly_change(m3[col])
                         for col, lbl in LEADLAG_SIGNALS.items() if col in m3})
    fund = pd.DataFrame({v: m1_change(m1[v]) for v in M1_VARS if v in m1})
    df = ship.join(fund, how="inner")
    corr = df.corr().loc[ship.columns, fund.columns]
    plt.figure(figsize=(8, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-0.4, vmax=0.4,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"label": "contemporaneous corr (weekly changes)"})
    plt.title("Shipping change vs M1 fundamentals (contemporaneous)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def granger_top(m3: pd.DataFrame, m1: pd.DataFrame, ll: pd.DataFrame,
                target="brent_log_return", topn=4) -> None:
    from statsmodels.tsa.stattools import grangercausalitytests
    label_to_col = {lbl: col for col, lbl in LEADLAG_SIGNALS.items()}
    lead = (ll[[k for k in ll.columns if k > 0]].abs().max(axis=1)
            .sort_values(ascending=False))
    tgt = m1[target]
    print(f"\n[Granger causality -> {target}, maxlag=4]  (H0: signal does NOT cause)")
    for lbl in lead.head(topn).index:
        col = label_to_col[lbl]
        pair = pd.DataFrame({target: tgt, col: weekly_change(m3[col])}).dropna()
        if len(pair) < 40:
            print(f"  {lbl:26s} skipped (n={len(pair)})")
            continue
        try:
            res = grangercausalitytests(pair[[target, col]], maxlag=4,
                                        verbose=False)
            pvals = [round(res[l][0]["ssr_ftest"][1], 3) for l in range(1, 5)]
            print(f"  {lbl:26s} min p={min(pvals):.3f}  (lags1-4: {pvals})")
        except Exception as e:  # noqa: BLE001
            print(f"  {lbl:26s} skipped ({type(e).__name__})")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    m3, m1 = load()

    plot_chokepoint_activity(m3, OUT / "m3_eda_chokepoint_activity.png")
    plot_redsea_rerouting(m3, OUT / "m3_eda_redsea_rerouting.png")

    ll = leadlag(m3, m1)
    ll.to_csv(OUT / "m3_eda_leadlag_corr.csv")
    plot_leadlag(ll, OUT / "m3_eda_leadlag_brent.png")
    plot_corr_m1(m3, m1, OUT / "m3_eda_corr_m1.png")
    granger_top(m3, m1, ll)

    print(f"\nfigures -> {OUT}/m3_eda_*.png")
    print(f"leadlag -> {OUT}/m3_eda_leadlag_corr.csv")
    best = ll.abs().max(axis=1).sort_values(ascending=False).head(6)
    print("\n[strongest |lead-lag corr| with Brent log return]")
    for lbl, v in best.items():
        k = int(ll.loc[lbl].abs().idxmax())
        arrow = "shipping leads" if k > 0 else ("price leads" if k < 0 else "same week")
        print(f"  {lbl:26s} max|corr|={v:.3f} at lag {k:+d}  ({arrow})")


if __name__ == "__main__":
    main()
