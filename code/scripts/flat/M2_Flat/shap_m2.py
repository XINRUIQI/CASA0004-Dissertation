"""
B3 interpretability: SHAP analysis of M2 (M1 + RS anom) models.

Trains Ridge and XGB on a fixed chronological holdout using the same locked
L4_tuned protocol (lookback=4, inner-val tuning, VarianceThreshold + StandardScaler),
then computes SHAP values on the test set.

Train / test split (default): train 2019-01–2023-12, test 2024-01–2025-12.
This is a post-hoc interpretability run, not the rolling-origin evaluation.
For fair predictive metrics use run_baseline.py.

SHAP aggregation:
  1. Per lagged feature -> sum |SHAP| over lags (lag0..lag3) per base feature
  2. M2 anom features:
       - by RS index  (NTL / NDBI / BSI / NDWI / NDVI)
       - by AOI / site (Houston, Fujairah, …)
  3. Top-N ranking -> shap_topN_anom.csv (consumed by robustness_m2.py)

Outputs (-> results/baselines/Flat/M2_Flat/):
  shap_xgb_by_feature.csv      per base-feature mean|SHAP| (XGB, M2+M1 combined)
  shap_ridge_by_feature.csv    per base-feature mean|SHAP| (Ridge)
  shap_xgb_m2_by_index.csv     M2 columns, summed by RS index (XGB)
  shap_xgb_m2_by_aoi.csv       M2 columns, summed by AOI (XGB)
  shap_topN_anom.csv           top-N M2 anom features by XGB |SHAP| (for robustness_m2)
  shap_anom.png                4-panel overview

Run:
  python3 code/scripts/flat/M2_Flat/shap_m2.py
  python3 code/scripts/flat/M2_Flat/shap_m2.py --top-n 15 --train-end 2022-12-31
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

OUT_DIR = data.ROOT / "results/baselines/Flat/M2_Flat"

RS_INDICES = data.RS_INDICES   # ["NDVI", "NDWI", "NDBI", "BSI", "NTL"]


# ---------------------------------------------------------------------------
# Feature name parsing
# ---------------------------------------------------------------------------

def strip_lag(name: str) -> str:
    return re.sub(r"_lag\d+$", "", name)


def parse_m2_meta(base: str) -> tuple[str | None, str | None]:
    """Return (rs_index, aoi) for an M2 anom base feature, else (None, None)."""
    for idx in RS_INDICES:
        prefix = f"{idx}_anom_"
        if base.startswith(prefix):
            return idx, base[len(prefix):]
    return None, None


# ---------------------------------------------------------------------------
# Build lagged dataset for a fixed holdout split
# ---------------------------------------------------------------------------

def build_holdout(train_end: str, test_start: str, lookback: int = 4,
                  fill_mode: str = data.DEFAULT_FILL_MODE
                  ) -> tuple[np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray,
                             list[str]]:
    df   = data.load_matrix()
    dico = data.load_dict()
    cols = data.select_features(dico, "M2", "anom")    # 34 M1 + 55 M2 = 89 base cols
    ds   = data.build_dataset(df, cols, lookback, "all",
                              window_start=data.WINDOW_START,
                              window_end=data.WINDOW_END,
                              fill_mode=fill_mode)

    idx        = ds["idx"]
    X          = ds["X"]
    r_next     = ds["r_next"]
    feat_names = ds["feat_names"]                      # e.g. brent_price_lag0, NTL_anom_Houston_lag2

    tr_mask = idx <= pd.Timestamp(train_end)
    te_mask = (idx >= pd.Timestamp(test_start)) & (idx <= pd.Timestamp(data.WINDOW_END))

    return (X[tr_mask], r_next[tr_mask],
            X[te_mask], r_next[te_mask],
            feat_names)


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def shap_values_xgb(pipe, X_train: np.ndarray, X_test: np.ndarray,
                    feat_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (shap_matrix, survived_feature_names) for XGBRegressor pipeline."""
    import shap as _shap
    X_tr_vt = pipe["vt"].transform(X_train)
    X_te_vt = pipe["vt"].transform(X_test)
    survived = [feat_names[i] for i in pipe["vt"].get_support(indices=True)]
    explainer = _shap.TreeExplainer(pipe["m"])
    sv = explainer.shap_values(X_te_vt)
    return sv, survived


def shap_values_ridge(pipe, X_train: np.ndarray, X_test: np.ndarray,
                      feat_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (shap_matrix, survived_feature_names) for Ridge pipeline."""
    import shap as _shap
    X_tr_t = pipe[:-1].transform(X_train)   # VT + SC
    X_te_t = pipe[:-1].transform(X_test)
    survived = [feat_names[i] for i in pipe["vt"].get_support(indices=True)]
    explainer = _shap.LinearExplainer(pipe["m"], X_tr_t,
                                      feature_perturbation="interventional")
    sv = explainer.shap_values(X_te_t)
    return sv, survived


# ---------------------------------------------------------------------------
# Aggregate: per-base-feature mean|SHAP| (sum over lags)
# ---------------------------------------------------------------------------

def aggregate_by_base(sv: np.ndarray, survived: list[str]) -> pd.Series:
    """Mean absolute SHAP per base feature (summed over lag variants)."""
    df_sv = pd.DataFrame(np.abs(sv), columns=survived)
    df_sv.columns = [strip_lag(c) for c in survived]  # collapse lag suffix
    return df_sv.T.groupby(level=0).sum().T.mean(axis=0).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Tune + fit a single holdout model
# ---------------------------------------------------------------------------

def fit_tuned(X_tr: np.ndarray, r_tr: np.ndarray,
              val_weeks: int, seed: int) -> tuple:
    """Tune and fit both Ridge and XGB on the training set."""
    best_alpha, best_xgb = rolling.tune_hyperparams(X_tr, r_tr, val_weeks, seed)
    pipe_ridge = models.ridge_pipe(best_alpha, seed).fit(X_tr, r_tr)
    pipe_xgb   = models.xgb_pipe(best_xgb, seed).fit(X_tr, r_tr)
    n_tr = len(X_tr)
    print(f"  Fitted: ridge_alpha={best_alpha}  XGB n_est={best_xgb['n_estimators']} "
          f"depth={best_xgb['max_depth']}  train_n={n_tr}")
    return pipe_ridge, pipe_xgb


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_shap_plot(feat_xgb: pd.Series, feat_ridge: pd.Series,
                   by_index: pd.Series, by_aoi: pd.Series, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Panel A — top-20 XGB features (M2 + M1)
    ax = axes[0, 0]
    top = feat_xgb.head(20)
    colors = ["tab:red" if parse_m2_meta(f)[0] is not None else "tab:grey"
              for f in top.index]
    ax.barh(range(len(top)), top.values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=7)
    ax.set_xlabel("Mean |SHAP| (sum over lags)")
    ax.set_title("Top-20 base features — XGB\n(red=M2 RS, grey=M1 finance)")
    ax.grid(alpha=0.3, axis="x")

    # Panel B — top-20 Ridge features
    ax = axes[0, 1]
    top_r = feat_ridge.head(20)
    colors_r = ["tab:blue" if parse_m2_meta(f)[0] is not None else "tab:grey"
                for f in top_r.index]
    ax.barh(range(len(top_r)), top_r.values[::-1], color=colors_r[::-1])
    ax.set_yticks(range(len(top_r)))
    ax.set_yticklabels(top_r.index[::-1], fontsize=7)
    ax.set_xlabel("Mean |SHAP| (sum over lags)")
    ax.set_title("Top-20 base features — Ridge")
    ax.grid(alpha=0.3, axis="x")

    # Panel C — M2 importance by RS index (XGB)
    ax = axes[1, 0]
    idx_order = by_index.sort_values(ascending=False)
    ax.bar(range(len(idx_order)), idx_order.values,
           color=["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
                 [:len(idx_order)])
    ax.set_xticks(range(len(idx_order)))
    ax.set_xticklabels(idx_order.index, fontsize=9)
    ax.set_ylabel("Sum mean|SHAP| across all AOIs")
    ax.set_title("M2 importance by RS index — XGB")
    ax.grid(alpha=0.3, axis="y")

    # Panel D — M2 importance by AOI (XGB)
    ax = axes[1, 1]
    aoi_order = by_aoi.sort_values(ascending=False)
    ax.bar(range(len(aoi_order)), aoi_order.values, color="tab:red", alpha=0.75)
    ax.set_xticks(range(len(aoi_order)))
    ax.set_xticklabels(aoi_order.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Sum mean|SHAP| across all RS indices")
    ax.set_title("M2 importance by AOI / site — XGB")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("SHAP importance: M2 (M1 + RS anom 55 cols, L4_tuned holdout)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SHAP analysis for M2 (M1 + RS anom).")
    ap.add_argument("--train-end",  default="2023-12-31",
                    help="last date of training set (default 2023-12-31)")
    ap.add_argument("--test-start", default="2024-01-01",
                    help="first date of test set (default 2024-01-01)")
    ap.add_argument("--lookback",   type=int, default=4)
    ap.add_argument("--val-weeks",  type=int, default=52)
    ap.add_argument("--top-n",      type=int, default=20,
                    help="top-N M2 features to write to shap_topN_anom.csv")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--fill-mode",  default=data.DEFAULT_FILL_MODE,
                    choices=list(data.FILL_MODES),
                    help="leading-gap treatment: by_family (default; zero for RS "
                         "anomalies, fold median elsewhere), zero or fold_median")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building holdout dataset (lookback={args.lookback}, "
          f"train≤{args.train_end}, test≥{args.test_start}, "
          f"fill={args.fill_mode}) …")
    X_tr, r_tr, X_te, r_te, feat_names = build_holdout(
        args.train_end, args.test_start, args.lookback, args.fill_mode)
    n_base = len({strip_lag(f) for f in feat_names})
    print(f"  Train={len(X_tr)} weeks  Test={len(X_te)} weeks  "
          f"Lagged features={len(feat_names)} (≈{n_base} base)")

    print("\nTuning + fitting models on training set …")
    pipe_ridge, pipe_xgb = fit_tuned(X_tr, r_tr, args.val_weeks, args.seed)

    # Quick in-sample check
    r_hat_r = pipe_ridge.predict(X_te)
    r_hat_x = pipe_xgb.predict(X_te)
    rmse_r = float(np.sqrt(np.mean((r_hat_r - r_te) ** 2)))
    rmse_x = float(np.sqrt(np.mean((r_hat_x - r_te) ** 2)))
    print(f"  Holdout log-return RMSE — Ridge: {rmse_r:.4f}  XGB: {rmse_x:.4f}")

    print("\nComputing SHAP values …")
    sv_xgb,   surv_xgb   = shap_values_xgb(pipe_xgb,   X_tr, X_te, feat_names)
    sv_ridge, surv_ridge = shap_values_ridge(pipe_ridge, X_tr, X_te, feat_names)
    print(f"  XGB survived features: {len(surv_xgb)}   "
          f"Ridge survived features: {len(surv_ridge)}")

    print("\nAggregating …")
    feat_xgb   = aggregate_by_base(sv_xgb,   surv_xgb)
    feat_ridge = aggregate_by_base(sv_ridge, surv_ridge)

    # Separate M2 anom features
    m2_xgb = {f: v for f, v in feat_xgb.items()
               if parse_m2_meta(f)[0] is not None}
    m2_series = pd.Series(m2_xgb).sort_values(ascending=False)

    by_index = m2_series.groupby(
        m2_series.index.map(lambda f: parse_m2_meta(f)[0])
    ).sum().sort_values(ascending=False)

    by_aoi = m2_series.groupby(
        m2_series.index.map(lambda f: parse_m2_meta(f)[1])
    ).sum().sort_values(ascending=False)

    # Save CSVs
    feat_xgb_df = feat_xgb.reset_index()
    feat_xgb_df.columns = ["feature", "mean_abs_shap"]
    feat_xgb_df.to_csv(OUT_DIR / "shap_xgb_by_feature.csv", index=False)

    feat_ridge_df = feat_ridge.reset_index()
    feat_ridge_df.columns = ["feature", "mean_abs_shap"]
    feat_ridge_df.to_csv(OUT_DIR / "shap_ridge_by_feature.csv", index=False)

    by_index_df = by_index.reset_index()
    by_index_df.columns = ["rs_index", "sum_mean_abs_shap"]
    by_index_df.to_csv(OUT_DIR / "shap_xgb_m2_by_index.csv", index=False)

    by_aoi_df = by_aoi.reset_index()
    by_aoi_df.columns = ["aoi", "sum_mean_abs_shap"]
    by_aoi_df.to_csv(OUT_DIR / "shap_xgb_m2_by_aoi.csv", index=False)

    top_n_df = m2_series.head(args.top_n).reset_index()
    top_n_df.columns = ["feature", "mean_abs_shap"]
    top_n_df.to_csv(OUT_DIR / "shap_topN_anom.csv", index=False)

    # Summary printout
    print("\n── Top-15 features (XGB) ──")
    print(feat_xgb.head(15).to_string())
    print("\n── M2 importance by RS index (XGB) ──")
    print(by_index.to_string())
    print("\n── M2 importance by AOI (XGB, top-10) ──")
    print(by_aoi.head(10).to_string())

    make_shap_plot(feat_xgb, feat_ridge, by_index, by_aoi,
                   OUT_DIR / "shap_anom.png")

    print(f"\nSaved to {OUT_DIR}/:")
    for f in ["shap_xgb_by_feature.csv", "shap_ridge_by_feature.csv",
              "shap_xgb_m2_by_index.csv", "shap_xgb_m2_by_aoi.csv",
              "shap_topN_anom.csv", "shap_anom.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
