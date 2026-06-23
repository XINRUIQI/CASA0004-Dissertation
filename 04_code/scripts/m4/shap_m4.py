"""
SHAP analysis for M4 (M1 + Remote Sensing + Shipping).

Trains Ridge and XGB on a fixed chronological holdout, then computes SHAP
values grouped by modality (M1/M2/M3), by M2 RS index (NDVI/NDWI/etc),
and by M3 shipping source (PortWatch / GFW).

Outputs (-> 05_outputs/baselines/m4/):
  shap_xgb_by_feature.csv      per base-feature mean|SHAP| (XGB)
  shap_ridge_by_feature.csv    per base-feature mean|SHAP| (Ridge)
  shap_m4_by_modality.csv      M1 vs M2 vs M3 modality totals (XGB)
  shap_m4_m2_by_index.csv      M2 features grouped by RS index (XGB)
  shap_m4_m3_by_source.csv     M3 features grouped by shipping source (XGB)
  shap_m4.png                  4-panel overview

Run:
  python3 04_code/scripts/m4/shap_m4.py
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
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, models, rolling          # noqa: E402

OUT_DIR = data.ROOT / "05_outputs/baselines/m4"

RS_INDICES = data.RS_INDICES   # ["NDVI","NDWI","NDBI","BSI","NTL"]


def strip_lag(name: str) -> str:
    return re.sub(r"_lag\d+$", "", name)


def modality_of(base: str) -> str:
    """Classify a base feature into M1 / M2 / M3."""
    if any(base.startswith(idx + "_") for idx in RS_INDICES):
        return "M2"
    if base.startswith("pw_") or base.startswith("gfw_"):
        return "M3"
    return "M1"


def m2_index_of(base: str) -> str | None:
    """Return the RS index prefix for an M2 feature (e.g. 'NTL'), else None."""
    for idx in RS_INDICES:
        if base.startswith(idx + "_"):
            return idx
    return None


def m3_source_of(base: str) -> str | None:
    """Return 'PortWatch', 'GFW', or None for non-M3 features."""
    if base.startswith("pw_"):
        return "PortWatch"
    if base.startswith("gfw_"):
        return "GFW"
    return None


def build_holdout(train_end, test_start, lookback=4):
    df   = data.load_matrix()
    dico = data.load_dict()
    cols = data.select_features(dico, "M4", "anom")
    ds   = data.build_dataset(df, cols, lookback, "all",
                              window_start=data.WINDOW_START,
                              window_end=data.WINDOW_END)
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


def shap_xgb(pipe, X_te, feat_names):
    import shap as _shap
    X_vt = pipe["vt"].transform(X_te)
    surv = [feat_names[i] for i in pipe["vt"].get_support(indices=True)]
    sv   = _shap.TreeExplainer(pipe["m"]).shap_values(X_vt)
    return sv, surv


def shap_ridge(pipe, X_tr, X_te, feat_names):
    import shap as _shap
    X_tr_t = pipe[:-1].transform(X_tr)
    X_te_t = pipe[:-1].transform(X_te)
    surv   = [feat_names[i] for i in pipe["vt"].get_support(indices=True)]
    sv     = _shap.LinearExplainer(pipe["m"], X_tr_t).shap_values(X_te_t)
    return sv, surv


def aggregate_by_base(sv, survived) -> pd.Series:
    df_sv = pd.DataFrame(np.abs(sv), columns=survived)
    df_sv.columns = [strip_lag(c) for c in survived]
    return df_sv.T.groupby(level=0).sum().T.mean(axis=0).sort_values(ascending=False)


_MODALITY_COLORS = {"M1": "tab:grey", "M2": "tab:orange", "M3": "tab:red"}


def make_plot(feat_xgb, feat_ridge, by_modality, m2_by_index, m3_by_source, path):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    for ax, feat, mdl_label in [
        (axes[0, 0], feat_xgb,   "XGB"),
        (axes[0, 1], feat_ridge, "Ridge"),
    ]:
        top = feat.head(20)
        bar_colors = [_MODALITY_COLORS[modality_of(f)] for f in top.index]
        ax.barh(range(len(top)), top.values[::-1], color=bar_colors[::-1])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1], fontsize=7)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"Top-20 features — {mdl_label}  (grey=M1, orange=M2, red=M3)")
        ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 0]
    colors_mod = [_MODALITY_COLORS.get(m, "tab:grey") for m in by_modality.index]
    by_modality.sort_values(ascending=False).plot.bar(ax=ax, color=colors_mod, alpha=0.8)
    ax.set_title("M4 importance by modality (XGB)")
    ax.set_ylabel("Sum mean|SHAP|")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=0)

    ax = axes[1, 1]
    m2_plot = m2_by_index.sort_values(ascending=False)
    m3_plot = m3_by_source.sort_values(ascending=False)
    combined = pd.concat([m2_plot.rename(lambda x: f"M2:{x}"),
                          m3_plot.rename(lambda x: f"M3:{x}")])
    bar_col2 = ["tab:orange" if s.startswith("M2") else "tab:red"
                for s in combined.index]
    combined.plot.bar(ax=ax, color=bar_col2, alpha=0.8)
    ax.set_title("M2 by RS index & M3 by source (XGB)")
    ax.set_ylabel("Sum mean|SHAP|")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.suptitle("SHAP importance: M4 (M1 + RS + Shipping, L4_tuned holdout)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="SHAP analysis for M4.")
    ap.add_argument("--train-end",  default="2023-12-31")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--lookback",   type=int, default=4)
    ap.add_argument("--val-weeks",  type=int, default=52)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building M4 holdout (train≤{args.train_end}, test≥{args.test_start}) …")
    X_tr, r_tr, X_te, r_te, feat_names = build_holdout(
        args.train_end, args.test_start, args.lookback)
    print(f"  Train={len(X_tr)}  Test={len(X_te)}  Features={len(feat_names)}")

    print("Fitting …")
    pipe_r, pipe_x = fit_tuned(X_tr, r_tr, args.val_weeks, args.seed)

    rmse_r = float(np.sqrt(np.mean((pipe_r.predict(X_te) - r_te) ** 2)))
    rmse_x = float(np.sqrt(np.mean((pipe_x.predict(X_te) - r_te) ** 2)))
    print(f"  holdout log-return RMSE  Ridge={rmse_r:.4f}  XGB={rmse_x:.4f}")

    print("SHAP …")
    sv_x, surv_x = shap_xgb(pipe_x, X_te, feat_names)
    sv_r, surv_r = shap_ridge(pipe_r, X_tr, X_te, feat_names)

    feat_xgb   = aggregate_by_base(sv_x, surv_x)
    feat_ridge = aggregate_by_base(sv_r, surv_r)

    by_modality = feat_xgb.groupby(feat_xgb.index.map(modality_of)).sum()

    m2_feats  = feat_xgb[feat_xgb.index.map(modality_of) == "M2"]
    m2_by_idx = m2_feats.groupby(m2_feats.index.map(
        lambda f: m2_index_of(f) or "other")).sum()

    m3_feats     = feat_xgb[feat_xgb.index.map(modality_of) == "M3"]
    m3_by_source = m3_feats.groupby(m3_feats.index.map(
        lambda f: m3_source_of(f) or "other")).sum()

    # save CSVs
    feat_xgb_df = feat_xgb.reset_index()
    feat_xgb_df.columns = ["feature", "mean_abs_shap"]
    feat_xgb_df.to_csv(OUT_DIR / "shap_xgb_by_feature.csv", index=False)

    feat_ridge_df = feat_ridge.reset_index()
    feat_ridge_df.columns = ["feature", "mean_abs_shap"]
    feat_ridge_df.to_csv(OUT_DIR / "shap_ridge_by_feature.csv", index=False)

    by_modality.reset_index().rename(
        columns={0: "sum_mean_abs_shap"}
    ).to_csv(OUT_DIR / "shap_m4_by_modality.csv", index=False)

    m2_by_idx.reset_index().rename(
        columns={0: "sum_mean_abs_shap"}
    ).to_csv(OUT_DIR / "shap_m4_m2_by_index.csv", index=False)

    m3_by_source.reset_index().rename(
        columns={0: "sum_mean_abs_shap"}
    ).to_csv(OUT_DIR / "shap_m4_m3_by_source.csv", index=False)

    print("\n── By modality (XGB) ──")
    print(by_modality.to_string())
    print("\n── M2 by RS index ──")
    print(m2_by_idx.to_string())
    print("\n── M3 by source ──")
    print(m3_by_source.to_string())
    print("\n── Top-10 features (XGB) ──")
    print(feat_xgb.head(10).to_string())

    make_plot(feat_xgb, feat_ridge, by_modality, m2_by_idx, m3_by_source,
              OUT_DIR / "shap_m4.png")
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
