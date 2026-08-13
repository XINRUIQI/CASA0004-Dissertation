"""
M4 (M1 + RS + Shipping) leave-one-modality-out (LOMO) robustness.

Tests the incremental value of each added modality within the full fusion:

  arm          Features used             Notes
  ──────────── ──────────────────────── ──────────────────────────────
  full         M1 + M2(anom) + M3       Main M4 configuration
  minus-M2     M1 + M3                  = M3 modality (RS removed)
  minus-M3     M1 + M2(anom)            = M2 modality (shipping removed)
  M1-only      M1                       finance-only baseline

All arms use CW (Clark-West) test vs M1 as the nested base, matching the
protocol of M2/M3 baselines.

Key questions answered:
  - Does M2 still add value when M3 is present? (full vs minus-M2)
  - Does M3 still add value when M2 is present? (full vs minus-M3)
  - How does full M4 compare to each modality alone?

Outputs (-> 05_outputs/baselines/Flat/M4_Flat/, or --out-dir):
  robustness_m4_summary.csv    arm × model (RMSE / CW_p vs M1)
  robustness_m4_overview.png   bar chart

Run:
  python3 04_code/scripts/flat/M4_Flat/robustness_m4.py
  python3 04_code/scripts/flat/M4_Flat/robustness_m4.py --quick
  python3 04_code/scripts/flat/M4_Flat/robustness_m4.py --fill-mode fold_median \
      --out-dir 05_outputs/_experiments/leading_impute/M4_Flat
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

OUT_DIR  = data.ROOT / "05_outputs/baselines/Flat/M4_Flat"
M1_LABEL = "M1_Flat"

ARMS = [
    ("full",     "M1 + M2(anom) + M3"),
    ("minus-M2", "M1 + M3  (RS removed)"),
    ("minus-M3", "M1 + M2(anom)  (shipping removed)"),
    ("M1-only",  "M1  (finance only)"),
]


def select_arm_cols(dico, arm: str) -> list[str]:
    m1 = data.select_features(dico, "M1")
    m2 = data.m2_columns(dico, "anom")
    m3 = dico.loc[dico["modality"] == "M3", "feature"].tolist()
    if arm == "full":
        return m1 + m2 + m3
    if arm == "minus-M2":
        return m1 + m3
    if arm == "minus-M3":
        return m1 + m2
    if arm == "M1-only":
        return m1
    raise ValueError(f"Unknown arm: {arm}")


def run_arm(df, dico, arm, min_train, retrain_every, val_weeks, seed,
            fill_mode=data.DEFAULT_FILL_MODE):
    cols = select_arm_cols(dico, arm)
    if not cols:
        return None, None, None, None

    m1_cols = data.select_features(dico, "M1")
    ds_m1   = data.build_dataset(df, m1_cols, 4, "all", fill_mode=fill_mode)
    res_m1  = rolling.rolling_origin(ds_m1, M1_LABEL, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    label = "M4_Flat"
    ds_arm  = data.build_dataset(df, cols, 4, "all", fill_mode=fill_mode)
    res_arm = rolling.rolling_origin(ds_arm, label, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    common = res_m1.index.intersection(res_arm.index)
    res_m1, res_arm = res_m1.loc[common], res_arm.loc[common]
    met_m1  = metrics.evaluate(res_m1,  ["M1_Flat_Ridge", "M1_Flat_XGB"])
    met_arm = metrics.evaluate(res_arm, ["M4_Flat_Ridge", "M4_Flat_XGB"])
    return res_m1, res_arm, met_m1, met_arm


def make_overview(summary, path):
    arms = [a for a, _ in ARMS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(arms))
    for ax, mdl, col, title in [
        (axes[0], "Ridge", "M4_RMSE",    "RMSE — Ridge"),
        (axes[1], "XGB",   "CW_p_vs_M1", "CW p vs M1 — XGB"),
    ]:
        sub = summary[summary.model == mdl]
        vals = [sub[sub.arm == a][col].values[0] if len(sub[sub.arm == a]) else np.nan
                for a in arms]
        colors = ["tab:green" if a == "full" else
                  "tab:orange" if "M2" in a else
                  "tab:red" if "M3" in a else "tab:grey"
                  for a in arms]
        ax.bar(x, vals, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=18, ha="right")
        ax.set_ylabel(col)
        ax.set_title(f"M4 LOMO: {title}")
        ax.grid(alpha=0.3, axis="y")
        if col == "CW_p_vs_M1":
            ax.axhline(0.05, color="black", ls="--", lw=0.8, label="p=0.05")
            ax.legend(fontsize=8)
    fig.suptitle("M4 leave-one-modality-out: does each modality add value in full fusion?",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M4 leave-one-modality-out ablation.")
    ap.add_argument("--arms", nargs="+",
                    default=[a for a, _ in ARMS])
    ap.add_argument("--quick", action="store_true",
                    help="retrain_every=26 for speed")
    ap.add_argument("--min-train",     type=int, default=104)
    ap.add_argument("--val-weeks",     type=int, default=52)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--fill-mode", default=data.DEFAULT_FILL_MODE,
                    choices=list(data.FILL_MODES),
                    help="leading-gap treatment: by_family (default; zero for RS "
                         "anomalies, fold median elsewhere), zero or fold_median")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (keeps main results intact)")
    args = ap.parse_args()

    retrain_every = 26 if args.quick else 13
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df   = data.load_matrix()
    dico = data.load_dict()
    print(f"M4 LOMO | arms={args.arms} | {'quick' if args.quick else 'full'} | "
          f"matrix {df.shape} | fill={args.fill_mode}\n")

    rows = []
    for arm in args.arms:
        desc = dict(ARMS).get(arm, arm)
        print(f"  running arm: {arm} ({desc}) …")
        t0 = time.time()
        res_m1, res_arm, met_m1, met_arm = run_arm(
            df, dico, arm, args.min_train, retrain_every, args.val_weeks, args.seed,
            args.fill_mode)
        if res_arm is None:
            continue
        n_cols = len(select_arm_cols(dico, arm))
        for mdl in ("Ridge", "XGB"):
            inc = metrics.incremental_tests(res_arm, res_m1, "M4_Flat", M1_LABEL, mdl)
            rows.append({
                "arm":          arm,
                "description":  desc,
                "n_features":   n_cols,
                "model":        mdl,
                "test_weeks":   len(res_arm),
                "M1_RMSE":      float(met_m1.loc[f"M1_Flat_{mdl}", "RMSE"]),
                "M4_RMSE":      float(met_arm.loc[f"M4_Flat_{mdl}", "RMSE"]),
                "skill_vs_M0":  float(met_arm.loc[f"M4_Flat_{mdl}", "RMSE_skill_vs_M0"]),
                "CW_p_vs_M1":   inc["CW_p_vs_base"],
                "DM_p_vs_M1":   inc["DM_p_vs_base"],
            })
        elapsed = time.time() - t0
        print(f"    Ridge={met_arm.loc['M4_Flat_Ridge','RMSE']:.3f}  "
              f"XGB={met_arm.loc['M4_Flat_XGB','RMSE']:.3f}  ({elapsed:.0f}s)")

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "robustness_m4_summary.csv"
    png_path = out_dir / "robustness_m4_overview.png"
    summary.to_csv(csv_path, index=False)
    make_overview(summary, png_path)

    print("\n" + "=" * 90)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\nSaved: {csv_path}\n       {png_path}")


if __name__ == "__main__":
    main()
