"""
Representation-level (deep) fusion baseline — single entry point.

Runs, under the SAME rolling-origin protocol / target / test weeks as the flat
baseline (backtest.data + backtest.rolling), the deep modality encoders:

  Mship  = z_ship (17-node heterogeneous shipping graph GAT+TCN)   [RQ1 shipping]
  Mfin   = z_fin  (finance TCN)                                    [deep finance ref]
  Mfusion= gated(z_fin, z_ship) -> head                           [RQ2 representation arm]
  Mfinrs = gated(z_fin, z_rs)   -> head          [RS incr. over finance, no shipping]
  Mconcat= concat(z_fin,z_rs,z_ship) -> MLP head [encoder-concat rung of RQ2 ladder]

and the flat M1 (Ridge/XGB) on the identical weeks as the nested reference. It
reports RMSE / skill vs M0 / DM (vs M0) and Clark-West nested increments:
  * Mfusion vs Mfin        (does the shipping representation add over finance?)
  * Mfusion vs flat M1     (representation fusion vs flat finance)
  * Mship  vs flat M1      (shipping representation vs flat finance)

This is the deep side of RQ2 (flat vs representation-level fusion): compare the
CW increments / skill here against the flat M3/M4 numbers from run_baseline.py.

Outputs (-> 05_outputs/baselines/deep/):
  deep_metrics.csv        M0 + flat M1 + deep modes, with DM/CW
  deep_cw.csv             the nested Clark-West comparison table
  deep_predictions.csv    per test week, every model's P_hat
  deep_backtest.png       price tracks + RMSE + skill bars

Run:
  python3 04_code/scripts/run_deep_baseline.py                 # ship,fin,fusion
  python3 04_code/scripts/run_deep_baseline.py --modes fusion  # subset
  python3 04_code/scripts/run_deep_baseline.py --epochs 40 --lookback 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics                      # noqa: E402
from models.deep_dataset import build_deep_dataset      # noqa: E402
from models.deep_rolling import CONFIGS, rolling_origin_deep  # noqa: E402

# IMPORTANT: do NOT import backtest.rolling / xgboost here. Loading xgboost in
# the same process as torch segfaults on macOS (duplicate OpenMP runtimes), so
# the flat M1 nested reference is READ from the pre-computed flat baseline
# predictions instead of being recomputed. Run run_baseline.py --modality M1
# first to (re)generate it.

OUT_DIR = data.ROOT / "05_outputs/baselines/deep"
LABELS = {"ship": "Mship", "fin": "Mfin", "rs": "Mrs",
          "fusion": "Mfusion", "finrs": "Mfinrs", "m4rep": "Mfull",
          "m4xattn": "Mxattn", "m4concat": "Mconcat"}


def cw_row(merged: pd.DataFrame, small_col: str, large_col: str, name: str) -> dict:
    y = merged["P_next_actual"].to_numpy()
    ys = merged[f"P_hat_{small_col}"].to_numpy()
    yl = merged[f"P_hat_{large_col}"].to_numpy()
    cw, p = metrics.clark_west(y, ys, yl)
    dm, dmp = metrics.dm_test(yl - y, ys - y)
    return {"comparison": name, "small": small_col, "large": large_col,
            "CW_stat": cw, "CW_p": p, "DM_stat": dm, "DM_p": dmp}


def make_plot(merged, summ, model_cols, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5),
                                        gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax1.plot(merged.index, merged["P_next_actual"], color="black", lw=1.6, label="Actual P(t+1)")
    ax1.plot(merged.index, merged["P_hat_M0"], color="grey", lw=1.0, ls="--", label="M0 RW")
    for col in model_cols:
        ax1.plot(merged.index, merged[f"P_hat_{col}"], lw=1.0, alpha=0.8, label=col)
    ax1.set_title("Next-week Brent: actual vs deep predictions")
    ax1.set_ylabel("USD / barrel"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    bars = summ["RMSE"].dropna()
    ax2.bar(range(len(bars)), bars.values)
    ax2.axhline(bars.get("M0_RW", np.nan), color="grey", ls="--", lw=0.8)
    ax2.set_xticks(range(len(bars))); ax2.set_xticklabels(bars.index, rotation=35, ha="right", fontsize=7)
    ax2.set_title("RMSE on price (lower=better)"); ax2.grid(alpha=0.3, axis="y")

    skill = summ["RMSE_skill_vs_M0"].drop(index=["M0_RW", "Naive_DirPersist"],
                                          errors="ignore").dropna() * 100
    ax3.bar(range(len(skill)), skill.values)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(range(len(skill))); ax3.set_xticklabels(skill.index, rotation=35, ha="right", fontsize=7)
    ax3.set_title("Skill vs M0 (%)  >0 beats RW"); ax3.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep representation-level fusion baseline.")
    ap.add_argument("--modes", default="fin,ship,rs,fusion,finrs,m4rep,m4concat",
                    help="comma list of deep configs "
                         "(fin/ship/rs/fusion/finrs/m4rep/m4xattn/m4concat)")
    ap.add_argument("--lookback", type=int, default=8, help="deep sequence lookback (weeks)")
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--flat-lookback", type=int, default=4, help="flat M1 lookback (protocol)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = data.load_matrix()
    dico = data.load_dict()
    print(f"Merged matrix: {df.shape}  {df.index.min().date()} ~ {df.index.max().date()}")

    # Flat M1 (Ridge/XGB) nested reference: READ the pre-computed flat baseline
    # predictions (see import note; avoids xgboost+torch OpenMP segfault).
    t0 = time.time()
    m1_pred = data.ROOT / "05_outputs/baselines/m1/baseline_predictions.csv"
    if not m1_pred.exists():
        raise SystemExit(f"Missing {m1_pred}.\n"
                         f"Run: python3 04_code/scripts/run_baseline.py --modality M1")
    res_m1 = pd.read_csv(m1_pred, index_col=0, parse_dates=True)
    res_m1.index.name = "date"
    print(f"flat M1 (read baseline_predictions): {len(res_m1)} test weeks "
          f"({res_m1.index.min().date()}~{res_m1.index.max().date()})")

    # Deep dataset (shared across deep modes).
    dds = build_deep_dataset(df, dico, lookback=args.lookback)
    print(f"deep dataset: N={len(dds['idx'])} lookback={args.lookback}\n")

    deep_res = {}
    for mode in modes:
        print(f"[{mode}] rolling-origin deep training...")
        deep_res[mode] = rolling_origin_deep(
            dds, LABELS[mode], mode, min_train=args.min_train,
            retrain_every=args.retrain_every, seed=args.seed, epochs=args.epochs)

    # Align all models to common test weeks.
    common = res_m1.index
    for r in deep_res.values():
        common = common.intersection(r.index)
    common = common.sort_values()
    print(f"\nCommon test weeks: {len(common)} "
          f"({common.min().date()} ~ {common.max().date()})")

    merged = res_m1.loc[common].copy()
    deep_cols = []
    for mode, r in deep_res.items():
        col = f"{LABELS[mode]}_{CONFIGS[mode][1]}"
        merged[f"r_hat_{col}"] = r.loc[common, f"r_hat_{col}"]
        merged[f"P_hat_{col}"] = r.loc[common, f"P_hat_{col}"]
        deep_cols.append(col)

    flat_cols = ["M1_Ridge", "M1_XGB"]
    all_cols = flat_cols + deep_cols
    summ = metrics.evaluate(merged, all_cols)

    # Nested Clark-West comparisons (only those whose columns are present).
    have = set(deep_cols)
    cand = [
        ("Mfull_M4rep", "Mfusion_Fusion", "RS incr. (M4rep vs fin+ship)"),
        ("Mfull_M4rep", "M1_Ridge", "full representation vs flat M1 (M4rep vs M1_Ridge)"),
        ("Mfusion_Fusion", "Mfin_TCN", "shipping incr. (fusion vs fin)"),
        ("Mfinrs_FinRS", "Mfin_TCN", "rs incr. (finrs vs fin)"),
        ("Mfinrs_FinRS", "M1_Ridge", "fin+rs representation vs flat M1 (finrs vs M1_Ridge)"),
        ("Mrs_RS", "M1_Ridge", "rs-rep vs flat M1 (rs vs M1_Ridge)"),
        ("Mship_GNN", "M1_Ridge", "shipping-rep vs flat M1 (ship vs M1_Ridge)"),
        # encoder-concat rung: fusion-mechanism comparisons (read DM_p, non-nested).
        ("Mconcat_M4concat", "M1_Ridge", "concat fusion vs flat M1 (M4concat vs M1_Ridge)"),
        ("Mfull_M4rep", "Mconcat_M4concat", "gating gain (M4rep vs encoder-concat)"),
    ]
    cw_rows = []
    for large, small, name in cand:
        if large in have and (small in have or small.startswith("M1_")):
            cw_rows.append(cw_row(merged, small, large, name))
    cw = pd.DataFrame(cw_rows)

    met_path = OUT_DIR / "deep_metrics.csv"
    cw_path = OUT_DIR / "deep_cw.csv"
    pred_path = OUT_DIR / "deep_predictions.csv"
    summ.to_csv(met_path)
    cw.to_csv(cw_path, index=False)
    keep = (["P_t", "P_next_actual", "r_actual", "r_now", "r_hat_M0", "P_hat_M0"]
            + [f"P_hat_{c}" for c in all_cols] + [f"r_hat_{c}" for c in all_cols])
    merged[keep].to_csv(pred_path)

    print("\n" + "=" * 100)
    print(summ.to_string(float_format=lambda x: f"{x:8.4f}"))
    print("=" * 100)
    if len(cw):
        print("\nNested Clark-West (one-sided p<0.05 => the 'large' model adds significant info):")
        print(cw.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not args.no_plot:
        plot_path = OUT_DIR / "deep_backtest.png"
        make_plot(merged, summ, all_cols, plot_path)
    else:
        plot_path = None

    print(f"\nElapsed {time.time()-t0:.0f}s")
    print(f"Saved: {met_path}\n       {cw_path}\n       {pred_path}"
          + (f"\n       {plot_path}" if plot_path else ""))


if __name__ == "__main__":
    main()
