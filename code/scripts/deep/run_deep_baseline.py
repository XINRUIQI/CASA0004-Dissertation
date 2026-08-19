"""
Representation-level (deep) baseline — single entry point.

Runs, under the SAME rolling-origin protocol as the flat baseline, the deep
modality encoders (display names):

  M_ship_GNN     = z_ship (17-node GAT+TCN)
  M1_Deep        = z_fin (finance TCN)
  M3_Deep_gated  = gated(z_fin, z_ship)          [main RQ2 arm]
  M2_Deep_gated  = gated(z_fin, z_rs)
  M4_Deep_gated  = gated(z_fin, z_rs, z_ship)
  M4_Deep_Concat = concat(…)                     [fusion-ladder floor]

Flat M1_Flat (Ridge/XGB) is READ from baseline_predictions.csv (no xgboost;
avoids macOS torch+OpenMP segfault).

Outputs (-> results/baselines/Deep/):
  _cross/deep_metrics.csv / deep_cw.csv / deep_predictions.csv / deep_backtest.png
  M*_Deep/baseline_metrics.csv / baseline_predictions.csv  (per-tier slim exports)

Run:
  python3 code/scripts/deep/run_deep_baseline.py
  python3 code/scripts/deep/run_deep_baseline.py --modes m3_deep_gated
  # old aliases still work: --modes finship
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics                      # noqa: E402
from model_naming import (                              # noqa: E402
    DEEP_METRICS_ANCHORS, DEEP_MODEL_NAMES, DEEP_TIER_MODELS,
    deep_out_dir, m1_flat_predictions, resolve_deep_config,
)
from models.deep_dataset import build_deep_dataset      # noqa: E402
from models.deep_rolling import CONFIGS, rolling_origin_deep  # noqa: E402

OUT_DIR = deep_out_dir(data.ROOT, "cross")

_BASE_PRED_COLS = ["P_t", "P_next_actual", "r_actual", "r_now", "r_hat_M0", "P_hat_M0"]


def _write_tier_exports(merged: pd.DataFrame, summ: pd.DataFrame,
                        deep_cols: list[str]) -> None:
    """Slim per-tier baseline CSVs (Flat-style names inside M*_Deep/)."""
    for tier, models in DEEP_TIER_MODELS.items():
        present = [m for m in models if m in deep_cols]
        if not present:
            continue
        tdir = deep_out_dir(data.ROOT, tier)
        tdir.mkdir(parents=True, exist_ok=True)
        pred_cols = _BASE_PRED_COLS + [f"P_hat_{c}" for c in present] + [f"r_hat_{c}" for c in present]
        merged[pred_cols].to_csv(tdir / "baseline_predictions.csv")
        idx = [i for i in DEEP_METRICS_ANCHORS if i in summ.index] + present
        summ.loc[[i for i in idx if i in summ.index]].to_csv(tdir / "baseline_metrics.csv")


def compare_row(merged: pd.DataFrame, small_col: str, large_col: str, name: str,
                predictor_set_nested: bool) -> dict:
    """One pairwise Deep comparison, DM-HLN primary.

    d_t = L(small) - L(large), so a positive statistic means the larger
    specification is the more accurate forecast. Clark-West is deliberately not
    reported for these pairs: they change encoders, fusion structure or model
    class, so the MSPE adjustment is not valid for them (see backtest.metrics).
    `predictor_set_nested` only records that the pair adds predictors.

    These rows are descriptive. The frozen comparison families and their
    Holm-adjusted p-values come from scripts/tools/build_test_tables.py.
    """
    y = merged["P_next_actual"].to_numpy()
    ys = merged[f"P_hat_{small_col}"].to_numpy()
    yl = merged[f"P_hat_{large_col}"].to_numpy()
    dm, dmp = metrics.dm_test(ys - y, yl - y)
    return {"comparison": name, "reference": small_col, "candidate": large_col,
            "predictor_set_nested": predictor_set_nested,
            "primary_test": "DM-HLN one-sided", "DM_stat": dm, "DM_p": dmp}


# Descriptive pairwise comparisons written to deep_cw.csv. The frozen families
# live in scripts/tools/build_test_tables.py; these rows are not a substitute.
COMPARISONS = [
    ("M4_Deep_gated", "M3_Deep_gated", "RS incr. (M4 vs M3 gated)", True),
    ("M3_Deep_gated", "M1_Deep", "shipping incr. (M3 vs M1 deep)", True),
    ("M2_Deep_gated", "M1_Deep", "rs incr. (M2 vs M1 deep)", True),
    ("M4_Deep_gated", "M1_Flat_Ridge", "full rep vs M1_Flat", False),
    ("M2_Deep_gated", "M1_Flat_Ridge", "fin+rs rep vs M1_Flat", False),
    ("M_rs_deep", "M1_Flat_Ridge", "rs-rep vs M1_Flat", False),
    ("M_ship_GNN", "M1_Flat_Ridge", "shipping-rep vs M1_Flat", False),
    ("M4_Deep_Concat", "M1_Flat_Ridge", "concat rep vs M1_Flat", False),
    ("M4_Deep_gated", "M4_Deep_Concat", "gating vs concat", False),
]


def comparison_table(merged: pd.DataFrame, deep_cols: list[str]) -> pd.DataFrame:
    have = set(deep_cols)
    rows = [compare_row(merged, small, large, name, nested)
            for large, small, name, nested in COMPARISONS
            if large in have and (small in have or small.startswith("M1_Flat"))]
    return pd.DataFrame(rows)


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
    default_modes = ",".join([
        "m1_deep", "m_ship_gnn", "m_rs_deep",
        "m3_deep_gated", "m2_deep_gated", "m4_deep_gated", "m4_deep_concat",
    ])
    ap = argparse.ArgumentParser(description="Deep representation-level baseline.")
    ap.add_argument("--modes", default=default_modes,
                    help="comma list of deep configs (new or legacy aliases)")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--replot-only", action="store_true",
                    help="Rebuild deep_backtest.png from existing CSVs (no retrain)")
    ap.add_argument("--recompare-only", action="store_true",
                    help="Rebuild deep_cw.csv from the saved weekly predictions "
                         "(no retrain), e.g. after the test definitions change")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    met_path = OUT_DIR / "deep_metrics.csv"
    cw_path = OUT_DIR / "deep_cw.csv"
    pred_path = OUT_DIR / "deep_predictions.csv"
    plot_path = OUT_DIR / "deep_backtest.png"

    if args.replot_only or args.recompare_only:
        if not met_path.exists() or not pred_path.exists():
            raise SystemExit(f"Missing {met_path} or {pred_path}")
        summ = pd.read_csv(met_path, index_col=0)
        merged = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        model_cols = [c[len("P_hat_"):] for c in merged.columns
                      if c.startswith("P_hat_") and c != "P_hat_M0"]
        if args.recompare_only:
            cw = comparison_table(merged, [c for c in model_cols
                                           if not c.startswith("M1_Flat")])
            cw.to_csv(cw_path, index=False)
            print(cw.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print(f"\nRebuilt from {len(merged)} saved weeks: {cw_path}")
            print("Holm-adjusted p-values: "
                  "code/scripts/tools/build_test_tables.py")
        if args.replot_only:
            make_plot(merged, summ, model_cols, plot_path)
            print(f"Replotted: {plot_path}")
        return

    modes = [resolve_deep_config(m.strip())
             for m in args.modes.split(",") if m.strip()]

    df = data.load_matrix()
    dico = data.load_dict()
    print(f"Merged matrix: {df.shape}  {df.index.min().date()} ~ {df.index.max().date()}")

    t0 = time.time()
    m1_pred = m1_flat_predictions(data.ROOT)
    if not m1_pred.exists():
        raise SystemExit(f"Missing {m1_pred}.\n"
                         f"Run: python3 code/scripts/flat/run_baseline.py --modality M1")
    res_m1 = pd.read_csv(m1_pred, index_col=0, parse_dates=True)
    res_m1.index.name = "date"
    print(f"flat M1_Flat (read baseline_predictions): {len(res_m1)} test weeks "
          f"({res_m1.index.min().date()}~{res_m1.index.max().date()})")

    dds = build_deep_dataset(df, dico, lookback=args.lookback)
    print(f"deep dataset: N={len(dds['idx'])} lookback={args.lookback}\n")

    deep_res = {}
    for mode in modes:
        print(f"[{mode}] rolling-origin deep training...")
        name = DEEP_MODEL_NAMES[mode]
        deep_res[mode] = rolling_origin_deep(
            dds, name, mode, min_train=args.min_train,
            retrain_every=args.retrain_every, seed=args.seed, epochs=args.epochs)

    common = res_m1.index
    for r in deep_res.values():
        common = common.intersection(r.index)
    common = common.sort_values()
    print(f"\nCommon test weeks: {len(common)} "
          f"({common.min().date()} ~ {common.max().date()})")

    merged = res_m1.loc[common].copy()
    deep_cols = []
    for mode, r in deep_res.items():
        col = CONFIGS[mode][1]
        merged[f"r_hat_{col}"] = r.loc[common, f"r_hat_{col}"]
        merged[f"P_hat_{col}"] = r.loc[common, f"P_hat_{col}"]
        deep_cols.append(col)

    flat_cols = ["M1_Flat_Ridge", "M1_Flat_XGB"]
    all_cols = flat_cols + deep_cols
    summ = metrics.evaluate(merged, all_cols)

    cw = comparison_table(merged, deep_cols)

    summ.to_csv(met_path)
    cw.to_csv(cw_path, index=False)
    keep = (["P_t", "P_next_actual", "r_actual", "r_now", "r_hat_M0", "P_hat_M0"]
            + [f"P_hat_{c}" for c in all_cols] + [f"r_hat_{c}" for c in all_cols])
    merged[keep].to_csv(pred_path)
    _write_tier_exports(merged, summ, deep_cols)

    print("\n" + "=" * 100)
    print(summ.to_string(float_format=lambda x: f"{x:8.4f}"))
    print("=" * 100)
    if len(cw):
        print("\nModel comparisons (DM-HLN, one-sided: candidate more accurate):")
        print(cw[["comparison", "reference", "candidate", "DM_stat", "DM_p"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("Holm-adjusted p-values: code/scripts/tools/build_test_tables.py")

    if not args.no_plot:
        make_plot(merged, summ, all_cols, plot_path)
    else:
        plot_path = None

    print(f"\nElapsed {time.time()-t0:.0f}s")
    print(f"Saved: {met_path}\n       {cw_path}\n       {pred_path}"
          + (f"\n       {plot_path}" if plot_path else ""))


if __name__ == "__main__":
    main()
