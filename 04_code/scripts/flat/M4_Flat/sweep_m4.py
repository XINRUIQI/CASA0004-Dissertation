"""
M4 (M1 + RS + Shipping) lookback robustness sweep.

Runs the rolling-origin engine across lookback windows for M4 and compares
against M1 (the nested baseline). Symmetric to m3/sweep_m3.py.

Grid:
  lookback: 1 / 4 / 8 weeks  (L4 is the locked main protocol)
  All other settings: L4_tuned (tune=True, retrain_every=13, min_train=104).

Outputs (-> 05_outputs/baselines/Flat/M4_Flat/, or --out-dir):
  sweep_m4_summary.csv
  sweep_m4_overview.png

Run:
  python3 04_code/scripts/flat/M4_Flat/sweep_m4.py
  python3 04_code/scripts/flat/M4_Flat/sweep_m4.py --quick
  python3 04_code/scripts/flat/M4_Flat/sweep_m4.py --fill-mode fold_median \
      --out-dir 05_outputs/_experiments/leading_impute/M4_Flat
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
SRC_DIR = SCRIPTS_DIR.parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics, rolling          # noqa: E402

OUT_DIR   = data.ROOT / "05_outputs/baselines/Flat/M4_Flat"
M1_LABEL  = "M1_Flat"
LOOKBACKS = [1, 4, 8]


def run_one(df, dico, lookback, min_train, retrain_every, val_weeks, seed,
            fill_mode="zero"):
    cols_m1 = data.select_features(dico, "M1")
    ds_m1   = data.build_dataset(df, cols_m1, lookback, "all", fill_mode=fill_mode)
    res_m1  = rolling.rolling_origin(ds_m1, M1_LABEL, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    cols_m4 = data.select_features(dico, "M4", "anom")
    ds_m4   = data.build_dataset(df, cols_m4, lookback, "all", fill_mode=fill_mode)
    res_m4  = rolling.rolling_origin(ds_m4, "M4_Flat", min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    common = res_m1.index.intersection(res_m4.index)
    res_m1, res_m4 = res_m1.loc[common], res_m4.loc[common]
    met_m1 = metrics.evaluate(res_m1, ["M1_Flat_Ridge", "M1_Flat_XGB"])
    met_m4 = metrics.evaluate(res_m4, ["M4_Flat_Ridge", "M4_Flat_XGB"])
    return res_m1, res_m4, met_m1, met_m4


def run_grid(df, dico, min_train, retrain_every, val_weeks, seed, fill_mode="zero"):
    rows = []
    for lb in LOOKBACKS:
        t0 = time.time()
        res_m1, res_m4, met_m1, met_m4 = run_one(
            df, dico, lb, min_train, retrain_every, val_weeks, seed, fill_mode)
        m0_rmse = float(met_m4.loc["M0_RW", "RMSE"])
        for mdl in ("Ridge", "XGB"):
            inc = metrics.incremental_tests(res_m4, res_m1, "M4_Flat", M1_LABEL, mdl)
            rows.append({
                "lookback":     lb,
                "model":        mdl,
                "test_weeks":   len(res_m4),
                "M0_RMSE":      m0_rmse,
                "M1_RMSE":      float(met_m1.loc[f"M1_Flat_{mdl}", "RMSE"]),
                "M4_RMSE":      float(met_m4.loc[f"M4_Flat_{mdl}", "RMSE"]),
                "skill_vs_M0":  float(met_m4.loc[f"M4_Flat_{mdl}", "RMSE_skill_vs_M0"]),
                "CW_p_vs_M1":   inc["CW_p_vs_base"],
                "DM_p_vs_M1":   inc["DM_p_vs_base"],
            })
        elapsed = time.time() - t0
        print(f"  L={lb}  Ridge={met_m4.loc['M4_Flat_Ridge','RMSE']:.3f}  "
              f"XGB={met_m4.loc['M4_Flat_XGB','RMSE']:.3f}  ({elapsed:.0f}s)")
    return pd.DataFrame(rows)


def make_overview(summary, path):
    piv_rmse = summary.pivot(index="lookback", columns="model",
                             values="M4_RMSE").sort_index()
    piv_cw   = summary.pivot(index="lookback", columns="model",
                             values="CW_p_vs_M1").sort_index()
    m0 = summary.groupby("lookback")["M0_RMSE"].first().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -- left: RMSE for both models + M0 reference --
    ax = axes[0]
    ax.plot(piv_rmse.index, piv_rmse["Ridge"], "o-", color="tab:blue", label="M4 Ridge")
    ax.plot(piv_rmse.index, piv_rmse["XGB"],   "s-", color="tab:red",  label="M4 XGB")
    ax.plot(m0.index, m0.values, "k--", lw=0.9, label="M0 (random walk)")
    ax.set_xlabel("lookback (weeks)")
    ax.set_ylabel("RMSE (restored price)")
    ax.set_title("M4 sweep: RMSE vs lookback")
    ax.set_xticks(LOOKBACKS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # -- right: Clark-West p for both models + 0.05 line --
    ax = axes[1]
    ax.plot(piv_cw.index, piv_cw["Ridge"], "o-", color="tab:blue", label="Ridge")
    ax.plot(piv_cw.index, piv_cw["XGB"],   "s-", color="tab:red",  label="XGB")
    ax.axhline(0.05, color="black", ls="--", lw=0.8, label="p=0.05")
    ax.set_xlabel("lookback (weeks)")
    ax.set_ylabel("Clark-West p vs M1")
    ax.set_title("M4 sweep: Clark-West p vs M1 (nested)")
    ax.set_xticks(LOOKBACKS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("M4 (M1 + RS + Shipping) lookback robustness", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M4 lookback robustness sweep.")
    ap.add_argument("--quick", action="store_true",
                    help="retrain_every=26 for speed")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-draw overview from existing summary CSV (skip the sweep)")
    ap.add_argument("--min-train",  type=int, default=104)
    ap.add_argument("--val-weeks",  type=int, default=52)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--fill-mode", default="zero", choices=list(data.FILL_MODES),
                    help="leading-gap treatment: zero (default) or fold_median")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (keeps main results intact)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_m4_summary.csv"
    png_path = out_dir / "sweep_m4_overview.png"

    if args.plot_only:
        summary = pd.read_csv(csv_path)
        make_overview(summary, png_path)
        print(f"Re-drew overview from {csv_path}\n       {png_path}")
        return

    retrain_every = 26 if args.quick else 13
    df   = data.load_matrix()
    dico = data.load_dict()
    print(f"M4 sweep ({'quick' if args.quick else 'full'}) | "
          f"lookbacks={LOOKBACKS} | matrix {df.shape} | fill={args.fill_mode}\n")

    summary = run_grid(df, dico, args.min_train, retrain_every, args.val_weeks,
                       args.seed, args.fill_mode)
    summary.to_csv(csv_path, index=False)
    make_overview(summary, png_path)

    print("\n" + "=" * 80)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\nSaved: {csv_path}\n       {png_path}")


if __name__ == "__main__":
    main()
