"""
RQ2 fusion-ladder MATRIX: 3 modality combos x 3 fusion mechanisms, under the
SAME rolling-origin protocol as run_deep_baseline (lookback=4, min_train=104,
retrain_every=13, seed=42, epochs=80). Fills every cell of

              Encoder-Concat |  Gated  | Cross-Attention
   Mfinship   finship_concat | finship | finship_xattn
   Mfinrs     finrs_concat   | finrs   | finrs_xattn
   Mfull      m4concat       | m4rep   | m4xattn

Each cell reports DESCRIPTIVE accuracy only (RMSE, skill vs M0, DirAcc) and dumps
its full per-origin forecast path. No p-values are computed here: every forecast-
comparison test comes from the single frozen test table built by
scripts/tools/build_test_tables.py, so that one script owns the choice of test,
the alternative and the multiple-comparison family. The flat M1_Flat predictions
are READ from baseline_predictions.csv (no xgboost import; avoids the macOS
torch+OpenMP segfault, mirroring run_deep_baseline).

Outputs (-> results/baselines/Deep/_cross/):
  deep_fusion_matrix.csv       one row per (combo, fusion): RMSE / skill / DirAcc
  deep_fusion_predictions.csv  long format, one row per (combo, fusion, origin)
  deep_fusion_matrix.png       skill-vs-M0 grouped bars (combo x fusion)

Run:
  python3 code/scripts/deep/run_deep_fusion_matrix.py            # ~15-30 min CPU
  python3 code/scripts/deep/run_deep_fusion_matrix.py --epochs 40
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
from model_naming import deep_out_dir, m1_flat_predictions  # noqa: E402
from models.deep_dataset import build_deep_dataset      # noqa: E402
from models.deep_rolling import CONFIGS, rolling_origin_deep  # noqa: E402

OUT_DIR = deep_out_dir(data.ROOT, "cross")

# (combo display label, fusion display label, CONFIGS key)
CELLS = [
    ("M3_Deep", "concat", "m3_deep_concat"),
    ("M3_Deep", "gated",  "m3_deep_gated"),
    ("M3_Deep", "xattn",  "m3_deep_xattn"),
    ("M2_Deep", "concat", "m2_deep_concat"),
    ("M2_Deep", "gated",  "m2_deep_gated"),
    ("M2_Deep", "xattn",  "m2_deep_xattn"),
    ("M4_Deep", "concat", "m4_deep_concat"),
    ("M4_Deep", "gated",  "m4_deep_gated"),
    ("M4_Deep", "xattn",  "m4_deep_xattn"),
]
COMBOS = ["M3_Deep", "M2_Deep", "M4_Deep"]
FUSIONS = ["concat", "gated", "xattn"]

# Run-to-run agreement is checked against the previous deep_fusion_matrix.csv.
# Seeds are fixed per fold, but CPU reduction order is not pinned, so repeated
# runs agree closely rather than bit-exactly; the observed spread is reported so
# it can be quoted instead of claiming exact reproducibility.
REPRO_TOL = 1e-3      # RMSE (USD/barrel) treated as run-to-run agreement


def main() -> None:
    ap = argparse.ArgumentParser(description="RQ2 fusion-ladder matrix (combo x fusion).")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prev_csv = OUT_DIR / "deep_fusion_matrix.csv"
    prev = (pd.read_csv(prev_csv).set_index(["combo", "fusion"])["RMSE"]
            if prev_csv.exists() else None)
    df = data.load_matrix()
    dico = data.load_dict()
    m1_pred = m1_flat_predictions(data.ROOT)
    if not m1_pred.exists():
        raise SystemExit(f"Missing {m1_pred}.\n"
                         f"Run: python3 code/scripts/flat/run_baseline.py --modality M1")
    res_m1 = pd.read_csv(m1_pred, index_col=0, parse_dates=True)
    res_m1.index.name = "date"

    dds = build_deep_dataset(df, dico, lookback=args.lookback)
    print(f"deep dataset N={len(dds['idx'])} lookback={args.lookback}  "
          f"(seed={args.seed}, epochs={args.epochs})\n")

    t0 = time.time()
    rows, preds = [], []
    for combo, fus, cfg in CELLS:
        res = rolling_origin_deep(
            dds, CONFIGS[cfg][1], cfg, min_train=args.min_train,
            retrain_every=args.retrain_every, seed=args.seed,
            epochs=args.epochs, verbose=False)
        col = CONFIGS[cfg][1]
        common = res.index.intersection(res_m1.index)
        y = res_m1.loc[common, "P_next_actual"].to_numpy()
        e_m0 = res_m1.loc[common, "P_hat_M0"].to_numpy() - y
        rmse_m0 = float(np.sqrt(np.mean(e_m0 ** 2)))
        yhat = res.loc[common, f"P_hat_{col}"].to_numpy()
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        rhat = res.loc[common, f"r_hat_{col}"].to_numpy()
        ract = res_m1.loc[common, "r_actual"].to_numpy()
        row = {"combo": combo, "fusion": fus, "config": cfg, "model": col,
               "RMSE": rmse, "MAE": float(np.mean(np.abs(yhat - y))),
               "skill_vs_M0": 1 - rmse / rmse_m0,
               "DirAcc": metrics.directional_acc(rhat, ract),
               "n_test": len(common)}
        rows.append(row)
        preds.append(pd.DataFrame({
            "forecast_origin": common,
            "target_date": common + pd.Timedelta(days=7),
            "information_set": combo,
            "fusion": fus,
            "config": cfg,
            "model": col,
            "seed": args.seed,
            "P_t": res_m1.loc[common, "P_t"].to_numpy(),
            "P_next_actual": y,
            "P_hat": yhat,
            "r_actual": ract,
            "r_hat": rhat,
        }))
        ref = None if prev is None else prev.get((combo, fus))
        chk = "" if ref is None else f" [prev {ref:.6f} d={rmse-ref:+.2e}]"
        print(f"  {combo:9s} {fus:7s} skill={row['skill_vs_M0']*100:+.2f}% "
              f"DirAcc={row['DirAcc']:.3f} RMSE={rmse:.6f}{chk} "
              f"({time.time()-t0:.0f}s)", flush=True)

    summ = pd.DataFrame(rows)
    out_csv = OUT_DIR / "deep_fusion_matrix.csv"
    summ.to_csv(out_csv, index=False)

    pred_all = pd.concat(preds, ignore_index=True)
    pred_csv = OUT_DIR / "deep_fusion_predictions.csv"
    pred_all.to_csv(pred_csv, index=False)

    n_dates = pred_all.groupby(["information_set", "fusion"])["forecast_origin"].nunique()
    if n_dates.nunique() != 1:
        raise SystemExit(f"cells do not share one forecast calendar:\n{n_dates}")
    print(f"\nforecast paths: {len(CELLS)} cells x {n_dates.iloc[0]} origins "
          f"on one shared calendar -> {pred_csv.name}")
    if prev is not None:
        got = summ.set_index(["combo", "fusion"])["RMSE"]
        delta = (got - prev.reindex(got.index)).dropna()
        worst = delta.abs().idxmax()
        print(f"  run-to-run RMSE agreement over {len(delta)} cells: "
              f"max |delta| = {delta.abs().max():.2e} at {worst[0]} {worst[1]} "
              f"({'within' if delta.abs().max() < REPRO_TOL else 'ABOVE'} "
              f"tolerance {REPRO_TOL:.0e})")

    def _mat(val):
        return (summ.pivot(index="combo", columns="fusion", values=val)
                .reindex(index=COMBOS, columns=FUSIONS))
    print("\n" + "=" * 60)
    print("Skill vs M0 (%)  [>0 beats the random walk, descriptive]:")
    print((_mat("skill_vs_M0") * 100).to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\nRMSE (USD/barrel):")
    print(_mat("RMSE").to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nForecast-comparison tests are NOT reported here. Run:")
    print("  python3 code/scripts/tools/build_test_tables.py")
    print("=" * 60)

    if not args.no_plot:
        _plot(summ, OUT_DIR / "deep_fusion_matrix.png")
        print(f"Saved: {out_csv}\n       {OUT_DIR/'deep_fusion_matrix.png'}")
    else:
        print(f"Saved: {out_csv}")
    print(f"Elapsed {time.time()-t0:.0f}s")


def _plot(summ: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.25
    for j, f in enumerate(FUSIONS):
        vals = []
        for c in COMBOS:
            v = summ[(summ.combo == c) & (summ.fusion == f)]["skill_vs_M0"].values
            vals.append(v[0] * 100 if len(v) else np.nan)
        ax.bar(np.arange(len(COMBOS)) + j * w, vals, width=w, label=f)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(COMBOS)) + w)
    ax.set_xticklabels(COMBOS)
    ax.set_ylabel("skill vs M0 (%)")
    ax.set_title("RQ2 fusion-ladder matrix: skill vs M0 (combo x fusion)")
    ax.legend(title="fusion")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
