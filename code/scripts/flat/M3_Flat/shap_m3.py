"""
SHAP analysis for M3 (M1 + Shipping).

Symmetric to m2/shap_m2.py.  Trains Ridge and XGB on a fixed chronological
holdout, then computes SHAP values grouped by shipping data source
(PortWatch vs GFW) and by shipping variable family
(vessel counts, throughput, tanker metrics, etc.).

Outputs (-> results/baselines/Flat/M3_Flat/):
  shap_xgb_by_feature.csv      per base-feature mean|SHAP| (XGB)
  shap_ridge_by_feature.csv    per base-feature mean|SHAP| (Ridge)
  shap_xgb_m3_by_source.csv    M3 features grouped by data source (PW / GFW)
  shap_xgb_m3_by_family.csv    M3 features grouped by variable family
  shap_m3.png                  4-panel overview

Run:
  python3 code/scripts/flat/M3_Flat/shap_m3.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, models, rolling          # noqa: E402

OUT_DIR = data.ROOT / "results/baselines/Flat/M3_Flat"


def strip_lag(name: str) -> str:
    return re.sub(r"_lag\d+$", "", name)


def m3_source(base: str) -> str:
    """Return 'PortWatch', 'GFW', or 'M1' for a base feature name."""
    if base.startswith("pw_"):
        return "PortWatch"
    if base.startswith("gfw_"):
        return "GFW"
    return "M1"


def m3_family(base: str) -> str:
    """Coarse variable family for M3 features."""
    b = base.lower()
    for kw, label in [
        ("tanker", "tanker"),
        ("vessel", "vessel_count"),
        ("throughput", "throughput"),
        ("port_call", "port_call"),
        ("transit", "transit"),
        ("cargo", "cargo"),
        ("avg_size", "vessel_size"),
    ]:
        if kw in b:
            return label
    if base.startswith("pw_") or base.startswith("gfw_"):
        return "shipping_other"
    return "M1_finance"


def build_holdout(train_end, test_start, lookback=4,
                  fill_mode=data.DEFAULT_FILL_MODE):
    df   = data.load_matrix()
    dico = data.load_dict()
    cols = data.select_features(dico, "M3")
    ds   = data.build_dataset(df, cols, lookback, "all",
                              window_start=data.WINDOW_START,
                              window_end=data.WINDOW_END,
                              fill_mode=fill_mode)
    idx = ds["idx"]
    X   = ds["X"]
    r   = ds["r_next"]
    fn  = ds["feat_names"]
    tr  = idx <= pd.Timestamp(train_end)
    te  = (idx >= pd.Timestamp(test_start)) & (idx <= pd.Timestamp(data.WINDOW_END))
    return X[tr], r[tr], X[te], r[te], fn


def fit_tuned(X_tr, r_tr, val_weeks, seed):
    best_alpha, best_xgb = rolling.tune_hyperparams(X_tr, r_tr, val_weeks, seed)
    pipe_r = models.ridge_pipe(best_alpha, seed).fit(X_tr, r_tr)
    pipe_x = models.xgb_pipe(best_xgb,   seed).fit(X_tr, r_tr)
    print(f"  ridge_alpha={best_alpha}  XGB n_est={best_xgb['n_estimators']} "
          f"depth={best_xgb['max_depth']}  train_n={len(X_tr)}")
    return pipe_r, pipe_x


def shap_xgb(pipe, X_tr, X_te, feat_names):
    import shap as _shap
    X_vt  = pipe["vt"].transform(X_te)
    surv  = [feat_names[i] for i in pipe["vt"].get_support(indices=True)]
    sv    = _shap.TreeExplainer(pipe["m"]).shap_values(X_vt)
    return sv, surv


def shap_ridge(pipe, X_tr, X_te, feat_names):
    import shap as _shap
    X_tr_t = pipe[:-1].transform(X_tr)
    X_te_t = pipe[:-1].transform(X_te)
    surv   = [feat_names[i] for i in pipe["vt"].get_support(indices=True)]
    sv     = _shap.LinearExplainer(pipe["m"], X_tr_t).shap_values(X_te_t)
    return sv, surv


def aggregate_by_base(sv, survived):
    df_sv = pd.DataFrame(np.abs(sv), columns=survived)
    df_sv.columns = [strip_lag(c) for c in survived]
    return df_sv.T.groupby(level=0).sum().T.mean(axis=0).sort_values(ascending=False)


def make_plot(feat_xgb, feat_ridge, by_source, by_family, path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for ax, feat, mdl_label, color in [
        (axes[0, 0], feat_xgb,   "XGB",   "tab:red"),
        (axes[0, 1], feat_ridge, "Ridge", "tab:blue"),
    ]:
        top = feat.head(20)
        colors = [color if m3_source(f) != "M1" else "tab:grey" for f in top.index]
        ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1], fontsize=7)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"Top-20 features — {mdl_label}  (colour=M3, grey=M1)")
        ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 0]
    by_source.sort_values(ascending=False).plot.bar(ax=ax, color=["tab:red", "tab:orange", "tab:grey"])
    ax.set_title("M3 importance by data source (XGB)")
    ax.set_ylabel("Sum mean|SHAP|")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=0)

    ax = axes[1, 1]
    by_fam = by_family.sort_values(ascending=False).head(10)
    by_fam.plot.bar(ax=ax, color="tab:red", alpha=0.75)
    ax.set_title("M3 importance by variable family — top-10 (XGB)")
    ax.set_ylabel("Sum mean|SHAP|")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("SHAP importance: M3 (M1 + Shipping, L4_tuned holdout)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="SHAP analysis for M3.")
    ap.add_argument("--train-end",  default="2023-12-31")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--lookback",   type=int, default=4)
    ap.add_argument("--val-weeks",  type=int, default=52)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--fill-mode",  default=data.DEFAULT_FILL_MODE,
                    choices=list(data.FILL_MODES),
                    help="leading-gap treatment: by_family (default; zero for RS "
                         "anomalies, fold median elsewhere), zero or fold_median")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building M3 holdout (train≤{args.train_end}, test≥{args.test_start}) "
          f"| fill={args.fill_mode} …")
    X_tr, r_tr, X_te, r_te, feat_names = build_holdout(
        args.train_end, args.test_start, args.lookback, args.fill_mode)
    print(f"  Train={len(X_tr)}  Test={len(X_te)}  Features={len(feat_names)}")

    print("Fitting …")
    pipe_r, pipe_x = fit_tuned(X_tr, r_tr, args.val_weeks, args.seed)

    print("SHAP …")
    sv_x, surv_x = shap_xgb(pipe_x, X_tr, X_te, feat_names)
    sv_r, surv_r = shap_ridge(pipe_r, X_tr, X_te, feat_names)

    feat_xgb   = aggregate_by_base(sv_x, surv_x)
    feat_ridge = aggregate_by_base(sv_r, surv_r)

    m3_xgb    = {f: v for f, v in feat_xgb.items() if m3_source(f) != "M1"}
    m3_series = pd.Series(m3_xgb).sort_values(ascending=False)

    by_source = m3_series.groupby(m3_series.index.map(m3_source)).sum()
    by_family = feat_xgb.groupby(feat_xgb.index.map(m3_family)).sum()

    feat_xgb_df = feat_xgb.reset_index(); feat_xgb_df.columns = ["feature","mean_abs_shap"]
    feat_xgb_df.to_csv(OUT_DIR / "shap_xgb_by_feature.csv", index=False)
    feat_ridge_df = feat_ridge.reset_index(); feat_ridge_df.columns = ["feature","mean_abs_shap"]
    feat_ridge_df.to_csv(OUT_DIR / "shap_ridge_by_feature.csv", index=False)
    by_source.reset_index().rename(columns={0:"sum_mean_abs_shap"}).to_csv(
        OUT_DIR / "shap_xgb_m3_by_source.csv", index=False)
    by_family.reset_index().rename(columns={0:"sum_mean_abs_shap"}).to_csv(
        OUT_DIR / "shap_xgb_m3_by_family.csv", index=False)

    print("\n── Top-10 M3 features (XGB) ──")
    print(m3_series.head(10).to_string())
    print("\n── By source ──")
    print(by_source.to_string())

    make_plot(feat_xgb, feat_ridge, by_source, by_family, OUT_DIR / "shap_m3.png")
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
