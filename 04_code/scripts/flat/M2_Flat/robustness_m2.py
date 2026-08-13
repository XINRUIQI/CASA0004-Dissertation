"""
C2 降维对照实验 — M2 (M1 + RS anom 55 cols) dimensionality reduction ablation.

Addresses P058 critique: "SHAP ≠ PCA; they solve different problems."

Four arms compared under the SAME locked L4 protocol
(lookback=4, min_train=104, retrain_every=13, 2019-2025 window):

  arm            RS features used     pre-processing inside rolling window
  ─────────────  ───────────────────  ───────────────────────────────────────
  all-55         all 55 anom cols     VT → SC → Ridge / XGB
  pca-90         all 55 anom cols     VT → SC → PCA(≥90% var) → Ridge / XGB
  elastic        all 55 anom cols     VT → SC → ElasticNet-select → Ridge / XGB
  shap-top20     top-20 from SHAP     VT → SC → Ridge / XGB (fewer inputs)

For Ridge: fixed alpha=1000 (best from main L4_tuned run).
For XGB: fixed depth=2, n_est=200 (best from SHAP holdout run).
Using fixed hyperparams (not per-fold inner-val) keeps the comparison focused
on the dimensionality-reduction effect, not tuning noise.

Clark-West vs M1: M1 is re-run once on the same common test weeks so the
nested-increment test is exact (same weeks, same protocol).

Outputs (-> 05_outputs/baselines/Flat/M2_Flat/, or --out-dir):
  c2_summary.csv          arm × model RMSE / skill / CW_p table
  c2_overview.png         2-panel: RMSE bars + CW_p bars

Run:
  python3 04_code/scripts/flat/M2_Flat/robustness_m2.py
  python3 04_code/scripts/flat/M2_Flat/robustness_m2.py --top-n 15   # shap-top15 instead of 20
  python3 04_code/scripts/flat/M2_Flat/robustness_m2.py --fill-mode fold_median \
      --out-dir 05_outputs/_experiments/leading_impute/M2_Flat
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics          # noqa: E402

OUT_DIR   = data.ROOT / "05_outputs/baselines/Flat/M2_Flat"
SHAP_CSV  = OUT_DIR / "shap_topN_anom.csv"

# Fixed hyperparams from main L4_tuned run (isolates DimRed effect)
RIDGE_ALPHA = 1000.0
XGB_PARAMS  = dict(n_estimators=200, max_depth=2, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)


# ---------------------------------------------------------------------------
# Pipeline factories (one callable per arm × model)
# ---------------------------------------------------------------------------

def _im():
    """Median imputation fit on the training fold only; no-op under fill_mode='zero'."""
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_ridge_plain(seed):
    return Pipeline([("im", _im()),
                     ("vt", VarianceThreshold(0.0)),
                     ("sc", StandardScaler()),
                     ("m",  Ridge(alpha=RIDGE_ALPHA, random_state=seed))])


def make_ridge_pca(seed):
    return Pipeline([("im",  _im()),
                     ("vt",  VarianceThreshold(0.0)),
                     ("sc",  StandardScaler()),
                     ("pca", PCA(n_components=0.90, random_state=seed)),
                     ("m",   Ridge(alpha=RIDGE_ALPHA, random_state=seed))])


def make_ridge_elastic(seed):
    sel = SelectFromModel(
        ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed),
        threshold="median",
    )
    return Pipeline([("im",  _im()),
                     ("vt",  VarianceThreshold(0.0)),
                     ("sc",  StandardScaler()),
                     ("sel", sel),
                     ("m",   Ridge(alpha=RIDGE_ALPHA, random_state=seed))])


def make_xgb_plain(seed):
    return Pipeline([("im", _im()),
                     ("vt", VarianceThreshold(0.0)),
                     ("m",  XGBRegressor(random_state=seed, n_jobs=4,
                                        objective="reg:squarederror",
                                        **XGB_PARAMS))])


def make_xgb_pca(seed):
    return Pipeline([("im",  _im()),
                     ("vt",  VarianceThreshold(0.0)),
                     ("sc",  StandardScaler()),
                     ("pca", PCA(n_components=0.90, random_state=seed)),
                     ("m",   XGBRegressor(random_state=seed, n_jobs=4,
                                         objective="reg:squarederror",
                                         **XGB_PARAMS))])


def make_xgb_elastic(seed):
    sel = SelectFromModel(
        ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=seed),
        threshold="median",
    )
    return Pipeline([("im",  _im()),
                     ("vt",  VarianceThreshold(0.0)),
                     ("sc",  StandardScaler()),
                     ("sel", sel),
                     ("m",   XGBRegressor(random_state=seed, n_jobs=4,
                                         objective="reg:squarederror",
                                         **XGB_PARAMS))])


# ---------------------------------------------------------------------------
# Minimal rolling-origin loop (fixed-pipeline variant)
# ---------------------------------------------------------------------------

def rolling_fixed(ds: dict, pipe_factory, label: str,
                  min_train: int, retrain_every: int, seed: int) -> pd.DataFrame:
    """Walk-forward backtest with a fixed-hyperparameter pipeline factory.

    pipe_factory(seed) -> unfitted sklearn Pipeline.
    Same protocol as rolling.rolling_origin but no inner-val tuning,
    so all arms run under the same base hyperparams for a fair DimRed comparison.
    """
    idx   = ds["idx"]
    X, r  = ds["X"], ds["r_next"]
    Pt, Pn, rnow = ds["P_t"], ds["P_next"], ds["r_now"]
    n = len(idx)
    pipe = None
    rows = []
    for i in range(n):
        if i < min_train:
            continue
        refit = (pipe is None) or ((i - min_train) % retrain_every == 0)
        if refit:
            pipe = pipe_factory(seed).fit(X[:i], r[:i])
        rhat = float(pipe.predict(X[i:i + 1])[0])
        rows.append({
            "date":           idx[i],
            "P_t":            Pt[i],
            "P_next_actual":  Pn[i],
            "r_actual":       r[i],
            "r_now":          rnow[i],
            "r_hat_M0":       0.0,
            "P_hat_M0":       Pt[i],
            f"r_hat_{label}": rhat,
            f"P_hat_{label}": Pt[i] * np.exp(rhat),
        })
    res = pd.DataFrame(rows).set_index("date")
    res.attrs["label"] = label
    return res


# ---------------------------------------------------------------------------
# Run one arm (M2 variant + M1 baseline on common weeks)
# ---------------------------------------------------------------------------

def run_arm(df, dico, arm_label: str, m2_cols: list[str],
            min_train: int, retrain_every: int, seed: int,
            fill_mode: str = data.DEFAULT_FILL_MODE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (res_m2, res_m1) on common test weeks."""
    m1_cols = data.select_features(dico, "M1")
    m2_all  = m1_cols + m2_cols

    ds_m1 = data.build_dataset(df, m1_cols, lookback=4, feature_mode="all",
                               fill_mode=fill_mode)
    ds_m2 = data.build_dataset(df, m2_all,  lookback=4, feature_mode="all",
                               fill_mode=fill_mode)

    # M1 plain Ridge & XGB (fixed params, no tuning — same as M2 arms for fair DimRed comparison)
    res_m1r = rolling_fixed(ds_m1, make_ridge_plain, "M1_Flat_Ridge", min_train, retrain_every, seed)
    res_m1x = rolling_fixed(ds_m1, make_xgb_plain,  "M1_Flat_XGB",   min_train, retrain_every, seed)
    res_m1  = res_m1r.join(res_m1x[[c for c in res_m1x.columns if "M1_Flat_XGB" in c]])

    # M2 arm Ridge & XGB
    factory_r, factory_x = ARM_FACTORIES[arm_label]
    res_m2r = rolling_fixed(ds_m2, factory_r, f"{arm_label}_Ridge", min_train, retrain_every, seed)
    res_m2x = rolling_fixed(ds_m2, factory_x, f"{arm_label}_XGB",   min_train, retrain_every, seed)
    res_m2  = res_m2r.join(res_m2x[[c for c in res_m2x.columns if arm_label in c and "XGB" in c]])

    common = res_m2.index.intersection(res_m1.index)
    return res_m2.loc[common], res_m1.loc[common]


# ---------------------------------------------------------------------------
# Metrics + CW
# ---------------------------------------------------------------------------

def arm_metrics(res_m2, res_m1, arm_label):
    rows = []
    Pn    = res_m2["P_next_actual"].to_numpy()
    e_m0  = res_m2["P_hat_M0"].to_numpy() - Pn
    rmse_m0 = float(np.sqrt(np.mean(e_m0 ** 2)))
    r_act = res_m2["r_actual"].to_numpy()

    for mdl in ("Ridge", "XGB"):
        col_m2 = f"P_hat_{arm_label}_{mdl}"
        col_m1 = f"P_hat_M1_Flat_{mdl}"
        if col_m2 not in res_m2.columns or col_m1 not in res_m1.columns:
            continue
        P_m2 = res_m2[col_m2].to_numpy()
        P_m1 = res_m1[col_m1].to_numpy()
        rmse_m2 = float(np.sqrt(np.mean((P_m2 - Pn) ** 2)))
        rmse_m1 = float(np.sqrt(np.mean((P_m1 - Pn) ** 2)))
        r_hat   = res_m2[f"r_hat_{arm_label}_{mdl}"].to_numpy()
        dm_stat, dm_p = metrics.dm_test(P_m1 - Pn, P_m2 - Pn)
        # Clark-West only for Ridge: see metrics.incremental_tests.
        cw_stat, cw_p = (metrics.clark_west(Pn, P_m1, P_m2) if mdl == "Ridge"
                         else (np.nan, np.nan))
        rows.append({
            "arm":         arm_label,
            "model":       mdl,
            "test_weeks":  len(Pn),
            "M0_RMSE":     rmse_m0,
            "M1_RMSE":     rmse_m1,
            "M2_RMSE":     rmse_m2,
            "skill_vs_M0": float(1 - rmse_m2 / rmse_m0),
            "DM_stat":     dm_stat,
            "DM_p_vs_M1":  dm_p,
            "CW_stat":     cw_stat,
            "CW_p_vs_M1":  cw_p,
        })
    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(summary: pd.DataFrame, m1_rmse_r: float, m1_rmse_x: float,
              m0_rmse: float, path: Path) -> None:
    arms  = summary["arm"].unique().tolist()
    x     = np.arange(len(arms))
    w     = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, col, ylabel, title, m1_line in [
        (ax1, "M2_RMSE",    "RMSE (USD/barrel)",  "RMSE by arm",        None),
        (ax2, "DM_p_vs_M1", "DM-HLN p vs M1 (one-sided)",
         "DM-HLN p vs M1 by arm (raw p)", None),
    ]:
        r_vals = [summary[(summary.arm == a) & (summary.model == "Ridge")][col].values
                  for a in arms]
        x_vals = [summary[(summary.arm == a) & (summary.model == "XGB")][col].values
                  for a in arms]
        r_vals = [v[0] if len(v) else np.nan for v in r_vals]
        x_vals = [v[0] if len(v) else np.nan for v in x_vals]

        ax.bar(x - w / 2, r_vals, w, color="tab:blue", alpha=0.8, label="Ridge")
        ax.bar(x + w / 2, x_vals, w, color="tab:red",  alpha=0.8, label="XGB")
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        if col == "M2_RMSE":
            ax.axhline(m0_rmse, color="black", ls="--", lw=0.9, label="M0 RW")
            ax.axhline(m1_rmse_r, color="tab:blue", ls=":", lw=0.9, label="M1 Ridge")
            ax.axhline(m1_rmse_x, color="tab:red",  ls=":", lw=0.9, label="M1 XGB")
            ax.legend(fontsize=7)
        if col == "DM_p_vs_M1":
            ax.axhline(0.05, color="black", ls="--", lw=0.9, label="p=0.05")
            ax.legend(fontsize=8)

    fig.suptitle("C2: M2 dimensionality-reduction ablation (L4, fixed hyperparams)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ---------------------------------------------------------------------------
# ARM registry (built after factory functions are defined)
# ---------------------------------------------------------------------------

ARM_FACTORIES = {
    "all-55":      (make_ridge_plain,   make_xgb_plain),
    "pca-90":      (make_ridge_pca,     make_xgb_pca),
    "elastic":     (make_ridge_elastic, make_xgb_elastic),
}


def build_shap_top_factories(top_n: int):
    """shap-top-N arm: same plain pipelines, but fewer input features."""
    return make_ridge_plain, make_xgb_plain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="C2: M2 dimensionality-reduction ablation.")
    ap.add_argument("--top-n",        type=int, default=20,
                    help="N for the shap-top-N arm (default 20)")
    ap.add_argument("--min-train",    type=int, default=104)
    ap.add_argument("--retrain-every",type=int, default=13)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--fill-mode", default=data.DEFAULT_FILL_MODE,
                    choices=list(data.FILL_MODES),
                    help="leading-gap treatment: by_family (default; zero for RS "
                         "anomalies, fold median elsewhere), zero or fold_median")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (keeps main results intact)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df   = data.load_matrix()
    dico = data.load_dict()
    all_m2_cols = data.m2_columns(dico, "anom")           # all 55 anom cols

    # shap-top-N columns
    if not SHAP_CSV.exists():
        raise FileNotFoundError(
            f"{SHAP_CSV} not found — run shap_m2.py first.")
    top_cols = pd.read_csv(SHAP_CSV)["feature"].head(args.top_n).tolist()
    shap_arm = f"shap-top{args.top_n}"

    arm_cols = {
        "all-55":    all_m2_cols,
        "pca-90":    all_m2_cols,
        "elastic":   all_m2_cols,
        shap_arm:    top_cols,
    }

    ARM_FACTORIES[shap_arm] = build_shap_top_factories(args.top_n)

    print(f"C2 ablation | arms={list(arm_cols)} | L4 fixed-hyperparams | "
          f"ridge_alpha={RIDGE_ALPHA} | XGB depth={XGB_PARAMS['max_depth']} "
          f"n_est={XGB_PARAMS['n_estimators']} | fill={args.fill_mode}\n"
          f"Matrix: {df.shape}\n")

    all_rows = []
    m0_rmse = m1_rmse_r = m1_rmse_x = np.nan

    for arm, m2_cols in arm_cols.items():
        t0 = time.time()
        print(f"── arm: {arm} ({len(m2_cols)} M2 features) …")
        res_m2, res_m1 = run_arm(df, dico, arm, m2_cols,
                                  args.min_train, args.retrain_every, args.seed,
                                  args.fill_mode)
        rows = arm_metrics(res_m2, res_m1, arm)
        all_rows.extend(rows)

        # capture M0 and M1 RMSE from first arm (same for all)
        if np.isnan(m0_rmse) and rows:
            m0_rmse   = rows[0]["M0_RMSE"]
            m1_rmse_r = rows[0]["M1_RMSE"]
            m1_rmse_x = next((r["M1_RMSE"] for r in rows if r["model"] == "XGB"), np.nan)

        for r in rows:
            print(f"   {r['model']:5s}  RMSE={r['M2_RMSE']:.3f}  "
                  f"skill={r['skill_vs_M0']:+.1%}  DM_p={r['DM_p_vs_M1']:.3f}")
        print(f"   ({time.time() - t0:.0f}s)\n")

    summary = pd.DataFrame(all_rows)
    csv_path = out_dir / "c2_summary.csv"
    png_path = out_dir / "c2_overview.png"
    summary.to_csv(csv_path, index=False)
    make_plot(summary, m1_rmse_r, m1_rmse_x, m0_rmse, png_path)

    print("=" * 80)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print("=" * 80)
    print(f"\nM0 RMSE: {m0_rmse:.3f}  M1 Ridge: {m1_rmse_r:.3f}  M1 XGB: {m1_rmse_x:.3f}")
    print("DM_p_vs_M1 is the primary one-sided test (DM-HLN); raw p, exploratory "
          "only — these ablation arms are not part of the frozen comparison "
          "families. CW is reported for Ridge only.")
    print(f"\nSaved: {csv_path}\n       {png_path}")


if __name__ == "__main__":
    main()
