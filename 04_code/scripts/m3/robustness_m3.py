"""
M3 (M1 + Shipping) robustness ablations.

Runs leave-one-channel-out (LOCHO) experiments to isolate the relative
contribution of each shipping data source under the locked L4_tuned protocol:

  arm                 Shipping columns used
  ─────────────────── ─────────────────────────────────────────────────────
  core                §11.1 main-model core (GFW 6x4 + PW chokepoints 6x2 + ports 2 = 38)
  full                All M3 shipping columns (PortWatch + GFW combined)
  portwatch-only      Only PortWatch vessel-traffic columns
  gfw-only            Only GFW monthly AIS columns
  gfw-presence        GFW core 24 + 6x mean_presence_hours_per_vessel (presence experiment)
  gfw-aggregate       gfw_all_activity_zmean only (aggregate benchmark; derived, leak-free)
  tanker-only         Tanker-type columns only (portwatch vessel_type filter)

Outputs (-> 05_outputs/baselines/m3/):
  robustness_m3_summary.csv   arm × model (RMSE / CW_p vs M1)
  robustness_m3_overview.png  bar chart

Run:
  python3 04_code/scripts/m3/robustness_m3.py
"""

from __future__ import annotations

import argparse
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

from backtest import data, metrics, rolling          # noqa: E402

OUT_DIR  = data.ROOT / "05_outputs/baselines/m3"
M1_LABEL = "M1"


def select_m3_arm(dico: pd.DataFrame, arm: str) -> list[str]:
    """Select M3 shipping columns for the given arm."""
    m3_all = dico.loc[dico["modality"] == "M3", "feature"].tolist()
    if arm == "full":
        return m3_all
    if arm == "core":                       # §11.1 main-model core (38)
        return data.m3_core_columns(dico)
    if arm == "portwatch-only":
        return [c for c in m3_all if c.startswith("pw_")]
    if arm == "gfw-only":
        return [c for c in m3_all if c.startswith("gfw_")]
    if arm == "gfw-presence":               # GFW core 24 + 6x mean_presence
        return data.gfw_core_columns(dico) + data.gfw_presence_columns(dico)
    if arm == "gfw-aggregate":              # derived z-mean only (injected in build)
        return [data.GFW_ZMEAN_COL]
    if arm == "tanker-only":
        return [c for c in m3_all if "tanker" in c.lower()]
    raise ValueError(f"Unknown arm: {arm}")


def run_arm(df, dico, arm, min_train, retrain_every, val_weeks, seed):
    m3_cols = select_m3_arm(dico, arm)
    if not m3_cols:
        print(f"  WARNING: no columns found for arm '{arm}', skipping.")
        return None, None, None, None

    m1_cols = data.select_features(dico, M1_LABEL)
    ds_m1   = data.build_dataset(df, m1_cols, 4, "all")
    res_m1  = rolling.rolling_origin(ds_m1, M1_LABEL, min_train, retrain_every,
                                     seed, tune=True, val_weeks=val_weeks)

    all_cols = m1_cols + m3_cols
    ds_m3 = data.build_dataset(df, all_cols, 4, "all")
    res_m3 = rolling.rolling_origin(ds_m3, "M3", min_train, retrain_every,
                                    seed, tune=True, val_weeks=val_weeks)

    common = res_m1.index.intersection(res_m3.index)
    res_m1, res_m3 = res_m1.loc[common], res_m3.loc[common]
    met_m1 = metrics.evaluate(res_m1, ["M1_Ridge", "M1_XGB"])
    met_m3 = metrics.evaluate(res_m3, ["M3_Ridge", "M3_XGB"])
    return res_m1, res_m3, met_m1, met_m3


def make_overview(summary, path):
    arms = summary["arm"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(arms))
    for ax, mdl, col, title in [
        (axes[0], "Ridge", "M3_RMSE",    "RMSE — Ridge"),
        (axes[1], "XGB",   "CW_p_vs_M1", "CW p vs M1 — XGB"),
    ]:
        sub = summary[summary.model == mdl]
        vals = [sub[sub.arm == a][col].values[0] if len(sub[sub.arm == a]) else np.nan
                for a in arms]
        ax.bar(x, vals, color="tab:blue" if mdl == "Ridge" else "tab:red", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=20, ha="right")
        ax.set_ylabel(col)
        ax.set_title(f"M3 LOCHO: {title}")
        ax.grid(alpha=0.3, axis="y")
        if col == "CW_p_vs_M1":
            ax.axhline(0.05, color="black", ls="--", lw=0.8)
    fig.suptitle("M3 leave-one-channel-out (PortWatch vs GFW vs full)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M3 leave-one-channel-out ablation.")
    ap.add_argument("--arms", nargs="+",
                    default=["core", "full", "portwatch-only", "gfw-only",
                             "gfw-presence", "gfw-aggregate", "tanker-only"])
    ap.add_argument("--min-train",      type=int, default=104)
    ap.add_argument("--retrain-every",  type=int, default=13)
    ap.add_argument("--val-weeks",      type=int, default=52)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df   = data.load_matrix()
    dico = data.load_dict()
    print(f"M3 LOCHO | arms={args.arms} | L4_tuned | matrix {df.shape}\n")

    rows = []
    for arm in args.arms:
        print(f"  running arm: {arm} …")
        res_m1, res_m3, met_m1, met_m3 = run_arm(
            df, dico, arm, args.min_train, args.retrain_every,
            args.val_weeks, args.seed)
        if res_m3 is None:
            continue
        for mdl in ("Ridge", "XGB"):
            inc = metrics.incremental_tests(res_m3, res_m1, "M3", M1_LABEL, mdl)
            rows.append({
                "arm":         arm,
                "model":       mdl,
                "test_weeks":  len(res_m3),
                "M1_RMSE":     float(met_m1.loc[f"M1_{mdl}", "RMSE"]),
                "M3_RMSE":     float(met_m3.loc[f"M3_{mdl}", "RMSE"]),
                "skill_vs_M0": float(met_m3.loc[f"M3_{mdl}", "RMSE_skill_vs_M0"]),
                "CW_p_vs_M1":  inc["CW_p_vs_base"],
                "DM_p_vs_M1":  inc["DM_p_vs_base"],
            })
        print(f"    Ridge={met_m3.loc['M3_Ridge','RMSE']:.3f}  "
              f"XGB={met_m3.loc['M3_XGB','RMSE']:.3f}")

    summary = pd.DataFrame(rows)
    csv_path = OUT_DIR / "robustness_m3_summary.csv"
    png_path = OUT_DIR / "robustness_m3_overview.png"
    summary.to_csv(csv_path, index=False)
    make_overview(summary, png_path)

    print("\n" + "=" * 80)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\nSaved: {csv_path}\n       {png_path}")


if __name__ == "__main__":
    main()
