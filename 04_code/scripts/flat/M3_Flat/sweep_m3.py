"""
M3 (M1 + Shipping) robustness sweep.

Runs the rolling-origin engine across lookback windows for the M3 modality
(M1 finance + M3 shipping: PortWatch + GFW).  Symmetric to m1/sweep_m1.py.

Grid:
  lookback: 1 / 4 / 8 weeks  (L4 is the locked main protocol)
  All other settings: L4_tuned (tune=True, retrain_every=13, min_train=104).

Outputs (-> 05_outputs/baselines/Flat/M3_Flat/, or --out-dir):
  sweep_m3_summary.csv
  sweep_m3_overview.png

Run:
  python3 04_code/scripts/flat/M3_Flat/sweep_m3.py
  python3 04_code/scripts/flat/M3_Flat/sweep_m3.py --quick
  python3 04_code/scripts/flat/M3_Flat/sweep_m3.py --fill-mode fold_median \
      --out-dir 05_outputs/_experiments/leading_impute/M3_Flat
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

OUT_DIR   = data.ROOT / "05_outputs/baselines/Flat/M3_Flat"
M1_LABEL  = "M1_Flat"
LOOKBACKS = [1, 4, 8]


def run_one(df, dico, lookback, min_train, retrain_every, val_weeks, seed,
            fill_mode=data.DEFAULT_FILL_MODE):
    cols_m1 = data.select_features(dico, "M1")
    ds_m1   = data.build_dataset(df, cols_m1, lookback, "all", fill_mode=fill_mode)
    res_m1  = rolling.rolling_origin(ds_m1, M1_LABEL, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    cols_m3 = data.select_features(dico, "M3")
    ds_m3   = data.build_dataset(df, cols_m3, lookback, "all", fill_mode=fill_mode)
    res_m3  = rolling.rolling_origin(ds_m3, "M3_Flat", min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    common = res_m1.index.intersection(res_m3.index)
    res_m1, res_m3 = res_m1.loc[common], res_m3.loc[common]
    met_m1 = metrics.evaluate(res_m1, ["M1_Flat_Ridge", "M1_Flat_XGB"])
    met_m3 = metrics.evaluate(res_m3, ["M3_Flat_Ridge", "M3_Flat_XGB"])
    return res_m1, res_m3, met_m1, met_m3


def run_grid(df, dico, min_train, retrain_every, val_weeks, seed, fill_mode=data.DEFAULT_FILL_MODE):
    rows = []
    for lb in LOOKBACKS:
        t0 = time.time()
        res_m1, res_m3, met_m1, met_m3 = run_one(
            df, dico, lb, min_train, retrain_every, val_weeks, seed, fill_mode)
        m0_rmse = float(met_m3.loc["M0_RW", "RMSE"])
        for mdl in ("Ridge", "XGB"):
            inc = metrics.incremental_tests(res_m3, res_m1, "M3_Flat", M1_LABEL, mdl)
            rows.append({
                "lookback":     lb,
                "model":        mdl,
                "test_weeks":   len(res_m3),
                "M0_RMSE":      m0_rmse,
                "M1_RMSE":      float(met_m1.loc[f"M1_Flat_{mdl}", "RMSE"]),
                "M3_RMSE":      float(met_m3.loc[f"M3_Flat_{mdl}", "RMSE"]),
                "skill_vs_M0":  float(met_m3.loc[f"M3_Flat_{mdl}", "RMSE_skill_vs_M0"]),
                "CW_p_vs_M1":   inc["CW_p_vs_base"],
                "DM_p_vs_M1":   inc["DM_p_vs_base"],
            })
        elapsed = time.time() - t0
        print(f"  L={lb}  Ridge={met_m3.loc['M3_Flat_Ridge','RMSE']:.3f}  "
              f"XGB={met_m3.loc['M3_Flat_XGB','RMSE']:.3f}  ({elapsed:.0f}s)")
    return pd.DataFrame(rows)


def make_overview(summary, path):
    piv_rmse = summary.pivot(index="lookback", columns="model",
                             values="M3_RMSE").sort_index()
    piv_dm   = summary.pivot(index="lookback", columns="model",
                             values="DM_p_vs_M1").sort_index()
    m0 = summary.groupby("lookback")["M0_RMSE"].first().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -- left: RMSE for both models + M0 reference --
    ax = axes[0]
    ax.plot(piv_rmse.index, piv_rmse["Ridge"], "o-", color="tab:blue", label="M3 Ridge")
    ax.plot(piv_rmse.index, piv_rmse["XGB"],   "s-", color="tab:red",  label="M3 XGB")
    ax.plot(m0.index, m0.values, "k--", lw=0.9, label="M0 (random walk)")
    ax.set_xlabel("lookback (weeks)")
    ax.set_ylabel("RMSE (restored price)")
    ax.set_title("M3 sweep: RMSE vs lookback")
    ax.set_xticks(LOOKBACKS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # -- right: DM p vs M1, the primary test for both learners --
    ax = axes[1]
    ax.plot(piv_dm.index, piv_dm["Ridge"], "o-", color="tab:blue", label="Ridge")
    ax.plot(piv_dm.index, piv_dm["XGB"],   "s-", color="tab:red",  label="XGB")
    ax.axhline(0.05, color="black", ls="--", lw=0.8, label="p=0.05")
    ax.set_xlabel("lookback (weeks)")
    ax.set_ylabel("DM-HLN p vs M1 (one-sided)")
    ax.set_title("M3 sweep: DM-HLN p vs M1 (raw, not Holm-adjusted)")
    ax.set_xticks(LOOKBACKS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("M3 (M1 + Shipping) lookback robustness", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M3 lookback robustness sweep.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-draw overview from existing summary CSV (skip the sweep)")
    ap.add_argument("--min-train",  type=int, default=104)
    ap.add_argument("--val-weeks",  type=int, default=52)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--fill-mode", default=data.DEFAULT_FILL_MODE,
                    choices=list(data.FILL_MODES),
                    help="leading-gap treatment: by_family (default; zero for RS "
                         "anomalies, fold median elsewhere), zero or fold_median")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (keeps main results intact)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_m3_summary.csv"
    png_path = out_dir / "sweep_m3_overview.png"

    if args.plot_only:
        summary = pd.read_csv(csv_path)
        make_overview(summary, png_path)
        print(f"Re-drew overview from {csv_path}\n       {png_path}")
        return

    retrain_every = 26 if args.quick else 13
    df   = data.load_matrix()
    dico = data.load_dict()
    print(f"M3 sweep ({'quick' if args.quick else 'full'}) | "
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
