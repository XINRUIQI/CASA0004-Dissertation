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

Outputs (-> 05_outputs/baselines/m2/):
  sweep_m2_summary.csv        full grid (feature_contract x lookback x model)
  sweep_m2_overview.png       RMSE grid heat-map + CW_p heat-map

Run:
  python3 04_code/scripts/m2/sweep_m2.py
  python3 04_code/scripts/m2/sweep_m2.py --quick    # retrain_every=26
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics, rolling          # noqa: E402

OUT_DIR   = data.ROOT / "05_outputs/baselines/m2"
M1_LABEL  = "M1"
CONTRACTS = ["anom", "literature", "level"]
LOOKBACKS = [1, 4, 8]


def run_one(df, dico, lookback, m2_feat, min_train, retrain_every, val_weeks, seed):
    """Run M1 and M2 on the same weeks; return (res_m1, res_m2, met_m1, met_m2)."""
    m2_modality = "M2"

    cols_m1 = data.select_features(dico, M1_LABEL)
    ds_m1   = data.build_dataset(df, cols_m1, lookback, "all")
    res_m1  = rolling.rolling_origin(ds_m1, M1_LABEL, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    cols_m2 = data.select_features(dico, m2_modality, m2_feat)
    ds_m2   = data.build_dataset(df, cols_m2, lookback, "all")
    res_m2  = rolling.rolling_origin(ds_m2, m2_modality, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    common  = res_m1.index.intersection(res_m2.index)
    res_m1, res_m2 = res_m1.loc[common], res_m2.loc[common]

    met_m1 = metrics.evaluate(res_m1, [f"{M1_LABEL}_Ridge", f"{M1_LABEL}_XGB"])
    met_m2 = metrics.evaluate(res_m2, [f"{m2_modality}_Ridge", f"{m2_modality}_XGB"])
    return res_m1, res_m2, met_m1, met_m2


def run_grid(df, dico, min_train, retrain_every, val_weeks, seed):
    rows = []
    for contract in CONTRACTS:
        for lb in LOOKBACKS:
            t0 = time.time()
            res_m1, res_m2, met_m1, met_m2 = run_one(
                df, dico, lb, contract, min_train, retrain_every, val_weeks, seed)
            m0_rmse  = float(met_m2.loc["M0_RW", "RMSE"])
            m1r_rmse = float(met_m1.loc["M1_Ridge", "RMSE"])
            m1x_rmse = float(met_m1.loc["M1_XGB",   "RMSE"])

            for mdl in ("Ridge", "XGB"):
                inc = metrics.incremental_tests(res_m2, res_m1, "M2", M1_LABEL, mdl)
                row = {
                    "contract":      contract,
                    "lookback":      lb,
                    "model":         mdl,
                    "test_weeks":    len(res_m2),
                    "M0_RMSE":       m0_rmse,
                    "M1_RMSE":       m1r_rmse if mdl == "Ridge" else m1x_rmse,
                    "M2_RMSE":       float(met_m2.loc[f"M2_{mdl}", "RMSE"]),
                    "M2_skill_vs_M0":float(met_m2.loc[f"M2_{mdl}", "RMSE_skill_vs_M0"]),
                    "CW_p_vs_M1":    inc["CW_p_vs_base"],
                    "DM_p_vs_M1":    inc["DM_p_vs_base"],
                }
                rows.append(row)

            elapsed = time.time() - t0
            print(f"  {contract:12s} L={lb}  Ridge={met_m2.loc['M2_Ridge','RMSE']:.3f} "
                  f"XGB={met_m2.loc['M2_XGB','RMSE']:.3f}  ({elapsed:.0f}s)")
    return pd.DataFrame(rows)


def make_overview(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mdl, title in zip(axes, ["Ridge", "XGB"],
                               ["Ridge RMSE", "XGB Clark-West p (vs M1)"]):
        sub = summary[summary.model == mdl].pivot(
            index="contract", columns="lookback",
            values="M2_RMSE" if mdl == "Ridge" else "CW_p_vs_M1")
        im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn_r" if mdl == "Ridge" else "RdYlGn")
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels([f"L={c}" for c in sub.columns])
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels(sub.index)
        for i in range(len(sub.index)):
            for j in range(len(sub.columns)):
                ax.text(j, i, f"{sub.values[i, j]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
    fig.suptitle("M2 sweep: feature contract × lookback", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M2 feature-contract × lookback sweep.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--min-train",      type=int, default=104)
    ap.add_argument("--val-weeks",      type=int, default=52)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    retrain_every = 26 if args.quick else 13
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df   = data.load_matrix()
    dico = data.load_dict()
    print(f"M2 sweep ({'quick' if args.quick else 'full'}) | "
          f"contracts={CONTRACTS} × lookbacks={LOOKBACKS} | "
          f"matrix {df.shape}\n")

    summary = run_grid(df, dico, args.min_train, retrain_every, args.val_weeks, args.seed)
    csv_path = OUT_DIR / "sweep_m2_summary.csv"
    png_path = OUT_DIR / "sweep_m2_overview.png"
    summary.to_csv(csv_path, index=False)
    make_overview(summary, png_path)

    print("\n" + "=" * 90)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print("=" * 90)
    print(f"\nSaved: {csv_path}\n       {png_path}")


if __name__ == "__main__":
    main()
