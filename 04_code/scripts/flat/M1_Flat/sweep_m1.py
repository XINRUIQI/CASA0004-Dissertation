"""
Robustness + tuning sweep for the M1 flat baseline, on the unified merged matrix.

Uses the shared kernel (04_code/src/backtest/) with modality=M1, so the sweep
runs under the exact same engine and data source (merged weekly matrix) as
run_baseline.py. One identical 2019-2026 rolling-origin protocol; writes one
comparison table + overview figure.

Grid:
  Robustness (strict rolling-origin, retrain_every=1, feature_mode=all):
    L1 / L4 / L8 / L12        -> effect of lookback window length
  Tuning / feature engineering (lookback=4):
    L4_returns                -> stationarised M1 level features
    L4_tuned                  -> inner-val tuned Ridge alpha + XGB grid
    L4_returns_tuned          -> both

Note: on the merged matrix (starts 2019-01-04) a larger lookback consumes a few
more warm-up weeks, so the test-week count varies slightly with L; each config's
skill is computed against its OWN M0 and test_weeks is reported per row.

Outputs (-> 05_outputs/baselines/Flat/M1_Flat/, or --out-dir):
  sweep_summary.csv           config x model metrics (+ test_weeks, m0_rmse)
  sweep_overview.png          (a) lookback robustness  (b) tuning experiments

Run:
  python3 04_code/scripts/flat/M1_Flat/sweep_m1.py
  python3 04_code/scripts/flat/M1_Flat/sweep_m1.py --quick   # coarser retrain, faster
  python3 04_code/scripts/flat/M1_Flat/sweep_m1.py --fill-mode fold_median \
      --out-dir 05_outputs/_experiments/leading_impute/M1_Flat
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
SRC_DIR = SCRIPTS_DIR.parent.parent.parent / "src"   # M*_Flat/ -> flat/ -> scripts/ -> src/
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics, rolling          # noqa: E402

OUT_DIR = data.ROOT / "05_outputs/baselines/Flat/M1_Flat"
MODEL_NAMES = ["M1_Flat_Ridge", "M1_Flat_XGB"]

FULL_GRID = [
    dict(name="L1_all", lookback=1, feature_mode="all", tune=False, retrain_every=1),
    dict(name="L4_all", lookback=4, feature_mode="all", tune=False, retrain_every=1),
    dict(name="L8_all", lookback=8, feature_mode="all", tune=False, retrain_every=1),
    dict(name="L12_all", lookback=12, feature_mode="all", tune=False, retrain_every=1),
    dict(name="L4_returns", lookback=4, feature_mode="returns", tune=False, retrain_every=1),
    dict(name="L4_tuned", lookback=4, feature_mode="all", tune=True, retrain_every=13),
    dict(name="L4_returns_tuned", lookback=4, feature_mode="returns", tune=True, retrain_every=13),
]

QUICK_GRID = [
    dict(name="L1_all", lookback=1, feature_mode="all", tune=False, retrain_every=4),
    dict(name="L4_all", lookback=4, feature_mode="all", tune=False, retrain_every=4),
    dict(name="L8_all", lookback=8, feature_mode="all", tune=False, retrain_every=4),
    dict(name="L12_all", lookback=12, feature_mode="all", tune=False, retrain_every=4),
    dict(name="L4_returns", lookback=4, feature_mode="returns", tune=False, retrain_every=4),
    dict(name="L4_tuned", lookback=4, feature_mode="all", tune=True, retrain_every=26),
    dict(name="L4_returns_tuned", lookback=4, feature_mode="returns", tune=True, retrain_every=26),
]


def run_one(df, dico, cfg, min_train, val_weeks, seed, fill_mode=data.DEFAULT_FILL_MODE):
    cols = data.select_features(dico, "M1")
    ds = data.build_dataset(df, cols, cfg["lookback"], cfg["feature_mode"],
                            fill_mode=fill_mode)
    res = rolling.rolling_origin(ds, "M1_Flat", min_train, cfg["retrain_every"],
                                 seed, cfg["tune"], val_weeks)
    met = metrics.evaluate(res, MODEL_NAMES)
    return res, met


def run_grid(grid, min_train, val_weeks, seed, fill_mode=data.DEFAULT_FILL_MODE):
    rows = []
    m0_done = False
    for cfg in grid:
        t0 = time.time()
        res, met = run_one(df_g, dico_g, cfg, min_train, val_weeks, seed, fill_mode)
        m0_rmse = float(met.loc["M0_RW", "RMSE"])
        dt = time.time() - t0
        print(f"  {cfg['name']:18s} feats={res.attrs['n_features']:4d} "
              f"fits={res.attrs['n_fits']:3d} test={len(res)} m0={m0_rmse:.3f} "
              f"Ridge={met.loc['M1_Flat_Ridge','RMSE']:.3f} "
              f"XGB={met.loc['M1_Flat_XGB','RMSE']:.3f} ({dt:.0f}s)")

        if not m0_done:                              # representative M0 row (L1_all)
            r = met.loc["M0_RW"].to_dict()
            r.update(config="(benchmark)", model="M0_RW", lookback=0,
                     feature_mode="-", tune=False, test_weeks=len(res), m0_rmse=m0_rmse)
            rows.append(r)
            m0_done = True

        for name in MODEL_NAMES:
            r = met.loc[name].to_dict()
            r.update(config=cfg["name"], model=name, lookback=cfg["lookback"],
                     feature_mode=cfg["feature_mode"], tune=cfg["tune"],
                     test_weeks=len(res), m0_rmse=m0_rmse)
            rows.append(r)

    cols = ["config", "model", "lookback", "feature_mode", "tune", "test_weeks",
            "m0_rmse", "RMSE", "MAE", "DirAcc", "RMSE_skill_vs_M0",
            "DM_stat", "DM_p_better_than_M0"]
    return pd.DataFrame(rows)[cols]


def make_overview(summary, path):
    m0_line = summary["m0_rmse"].dropna().mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    rob = summary[summary.config.str.match(r"L\d+_all$")].copy()
    rob["L"] = rob["lookback"]
    for name, color in [("M1_Flat_Ridge", "tab:blue"), ("M1_Flat_XGB", "tab:red")]:
        sub = rob[rob.model == name].sort_values("L")
        ax1.plot(sub["L"], sub["RMSE"], "o-", color=color, label=name)
    ax1.axhline(m0_line, color="grey", ls="--", label="M0 random walk (mean)")
    ax1.set_xlabel("lookback (weeks)")
    ax1.set_ylabel("RMSE (USD/barrel)")
    ax1.set_title("(a) Robustness: lookback window length")
    ax1.set_xticks(sorted(rob["L"].unique()))
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    exp_order = ["L4_all", "L4_returns", "L4_tuned", "L4_returns_tuned"]
    exp = summary[summary.config.isin(exp_order)]
    x = range(len(exp_order))
    w = 0.38
    ridge = [exp[(exp.config == c) & (exp.model == "M1_Flat_Ridge")]["RMSE"].iloc[0] for c in exp_order]
    xgb = [exp[(exp.config == c) & (exp.model == "M1_Flat_XGB")]["RMSE"].iloc[0] for c in exp_order]
    ax2.bar([i - w / 2 for i in x], ridge, w, color="tab:blue", label="M1_Flat_Ridge")
    ax2.bar([i + w / 2 for i in x], xgb, w, color="tab:red", label="M1_Flat_XGB")
    ax2.axhline(m0_line, color="grey", ls="--", label="M0 random walk (mean)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(exp_order, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("RMSE (USD/barrel)")
    ax2.set_title("(b) Tuning / feature engineering (lookback=4)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    global df_g, dico_g
    ap = argparse.ArgumentParser(description="Robustness + tuning sweep for M1 baseline (merged matrix).")
    ap.add_argument("--quick", action="store_true", help="faster grid (coarser retrain cadence)")
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--val-weeks", type=int, default=52)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fill-mode", default=data.DEFAULT_FILL_MODE,
                    choices=list(data.FILL_MODES),
                    help="leading-gap treatment: by_family (default; zero for RS "
                         "anomalies, fold median elsewhere), zero or fold_median")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (keeps main results intact)")
    args = ap.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df_g = data.load_matrix()
    dico_g = data.load_dict()
    print(f"Sweep ({'quick' if args.quick else 'full'}) | {len(grid)} configs "
          f"| merged matrix {df_g.shape} | window {data.WINDOW_START}..{data.WINDOW_END} "
          f"| min_train={args.min_train} | fill={args.fill_mode}\n")

    summary = run_grid(grid, args.min_train, args.val_weeks, args.seed, args.fill_mode)
    out_csv = out_dir / "sweep_summary.csv"
    out_png = out_dir / "sweep_overview.png"
    summary.to_csv(out_csv, index=False)
    make_overview(summary, out_png)

    print("\n" + "=" * 96)
    print(summary.drop(columns=["lookback"]).to_string(
        index=False, float_format=lambda x: f"{x:8.4f}"))
    print("=" * 96)
    print("skill>0 beats M0.  DM_p<0.05: significantly better than M0.")
    print(f"\nSaved: {out_csv}\n       {out_png}")


if __name__ == "__main__":
    main()
