"""
M2 (M1 + Remote Sensing) robustness & feature-contract sweep.

Runs the same rolling-origin engine as run_baseline.py across a grid of
lookback windows AND m2-feature contracts, so all axes of M2 sensitivity
are visible in one table.

Grid axes:
  feature contracts : anom (55, main), literature (4 NTL_anom), level (55)
  lookback windows  : 1 / 4 / 8 weeks  (L4 is the locked main protocol)
  All other settings follow the locked L4_tuned protocol (tune=True,
  retrain_every=13, min_train=104, feature_mode=all).

Outputs (-> 05_outputs/baselines/Flat/M2_Flat/, or --out-dir):
  sweep_m2_summary.csv        full grid (feature_contract x lookback x model)
  sweep_m2_overview.png       RMSE grid heat-map + CW_p heat-map

Run:
  python3 04_code/scripts/flat/M2_Flat/sweep_m2.py
  python3 04_code/scripts/flat/M2_Flat/sweep_m2.py --quick    # retrain_every=26
  python3 04_code/scripts/flat/M2_Flat/sweep_m2.py --fill-mode fold_median \
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics, rolling          # noqa: E402

OUT_DIR   = data.ROOT / "05_outputs/baselines/Flat/M2_Flat"
M1_LABEL  = "M1_Flat"
CONTRACTS = ["anom", "literature", "level"]
LOOKBACKS = [1, 4, 8]


def run_one(df, dico, lookback, m2_feat, min_train, retrain_every, val_weeks, seed,
            fill_mode=data.DEFAULT_FILL_MODE):
    """Run M1 and M2 on the same weeks; return (res_m1, res_m2, met_m1, met_m2)."""
    m2_modality = "M2_Flat"

    cols_m1 = data.select_features(dico, "M1")
    ds_m1   = data.build_dataset(df, cols_m1, lookback, "all", fill_mode=fill_mode)
    res_m1  = rolling.rolling_origin(ds_m1, M1_LABEL, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    cols_m2 = data.select_features(dico, "M2", m2_feat)
    ds_m2   = data.build_dataset(df, cols_m2, lookback, "all", fill_mode=fill_mode)
    res_m2  = rolling.rolling_origin(ds_m2, m2_modality, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    common  = res_m1.index.intersection(res_m2.index)
    res_m1, res_m2 = res_m1.loc[common], res_m2.loc[common]

    met_m1 = metrics.evaluate(res_m1, [f"{M1_LABEL}_Ridge", f"{M1_LABEL}_XGB"])
    met_m2 = metrics.evaluate(res_m2, [f"{m2_modality}_Ridge", f"{m2_modality}_XGB"])
    return res_m1, res_m2, met_m1, met_m2


def run_grid(df, dico, min_train, retrain_every, val_weeks, seed, fill_mode=data.DEFAULT_FILL_MODE):
    rows = []
    for contract in CONTRACTS:
        for lb in LOOKBACKS:
            t0 = time.time()
            res_m1, res_m2, met_m1, met_m2 = run_one(
                df, dico, lb, contract, min_train, retrain_every, val_weeks, seed,
                fill_mode)
            m0_rmse  = float(met_m2.loc["M0_RW", "RMSE"])
            m1r_rmse = float(met_m1.loc["M1_Flat_Ridge", "RMSE"])
            m1x_rmse = float(met_m1.loc["M1_Flat_XGB",   "RMSE"])

            for mdl in ("Ridge", "XGB"):
                inc = metrics.incremental_tests(res_m2, res_m1, "M2_Flat", M1_LABEL, mdl)
                row = {
                    "contract":      contract,
                    "lookback":      lb,
                    "model":         mdl,
                    "test_weeks":    len(res_m2),
                    "M0_RMSE":       m0_rmse,
                    "M1_RMSE":       m1r_rmse if mdl == "Ridge" else m1x_rmse,
                    "M2_RMSE":       float(met_m2.loc[f"M2_Flat_{mdl}", "RMSE"]),
                    "M2_skill_vs_M0":float(met_m2.loc[f"M2_Flat_{mdl}", "RMSE_skill_vs_M0"]),
                    "CW_p_vs_M1":    inc["CW_p_vs_base"],
                    "DM_p_vs_M1":    inc["DM_p_vs_base"],
                }
                rows.append(row)

            elapsed = time.time() - t0
            print(f"  {contract:12s} L={lb}  Ridge={met_m2.loc['M2_Flat_Ridge','RMSE']:.3f} "
                  f"XGB={met_m2.loc['M2_Flat_XGB','RMSE']:.3f}  ({elapsed:.0f}s)")
    return pd.DataFrame(rows)


def _draw_cells(ax, piv, fmt):
    """Tick labels + per-cell value annotations (NaN -> 'n/a')."""
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"L={c}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, "n/a" if pd.isna(v) else fmt.format(v),
                    ha="center", va="center", fontsize=9)


def make_overview(summary, path):
    from matplotlib.colors import TwoSlopeNorm

    row_order = [c for c in ("anom", "literature", "level")
                 if c in summary["contract"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- left: Ridge RMSE (red = higher = worse) --
    ax = axes[0]
    piv = (summary[summary.model == "Ridge"]
           .pivot(index="contract", columns="lookback", values="M2_RMSE")
           .reindex(row_order))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax)
    _draw_cells(ax, piv, "{:.3f}")
    ax.set_title("Ridge RMSE  (red = higher = worse)")

    # -- right: XGB Clark-West p; colour breaks at 0.05 so green = significant --
    ax = axes[1]
    piv = (summary[summary.model == "XGB"]
           .pivot(index="contract", columns="lookback", values="CW_p_vs_M1")
           .reindex(row_order))
    finite = piv.values[np.isfinite(piv.values)]
    vmax = float(finite.max()) if finite.size else 0.1
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.05, vmax=max(vmax, 0.0501))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn_r", norm=norm)
    plt.colorbar(im, ax=ax, label="Clark-West p (colour breaks at 0.05)")
    _draw_cells(ax, piv, "{:.4f}")
    ax.set_title("XGB Clark-West p vs M1  (green = p<0.05 significant)")

    fig.suptitle("M2 sweep: feature contract × lookback", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M2 feature-contract × lookback sweep.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-draw overview from existing summary CSV (skip the sweep)")
    ap.add_argument("--min-train",      type=int, default=104)
    ap.add_argument("--val-weeks",      type=int, default=52)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--fill-mode", default=data.DEFAULT_FILL_MODE,
                    choices=list(data.FILL_MODES),
                    help="leading-gap treatment: by_family (default; zero for RS "
                         "anomalies, fold median elsewhere), zero or fold_median")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (keeps main results intact)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_m2_summary.csv"
    png_path = out_dir / "sweep_m2_overview.png"

    if args.plot_only:
        summary = pd.read_csv(csv_path)
        make_overview(summary, png_path)
        print(f"Re-drew overview from {csv_path}\n       {png_path}")
        return

    retrain_every = 26 if args.quick else 13
    df   = data.load_matrix()
    dico = data.load_dict()
    print(f"M2 sweep ({'quick' if args.quick else 'full'}) | "
          f"contracts={CONTRACTS} × lookbacks={LOOKBACKS} | "
          f"matrix {df.shape} | fill={args.fill_mode}\n")

    summary = run_grid(df, dico, args.min_train, retrain_every, args.val_weeks,
                       args.seed, args.fill_mode)
    summary.to_csv(csv_path, index=False)
    make_overview(summary, png_path)

    print("\n" + "=" * 90)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print("=" * 90)
    print(f"\nSaved: {csv_path}\n       {png_path}")


if __name__ == "__main__":
    main()
