"""
RQ2 fusion-ladder MATRIX: 3 modality combos x 3 fusion mechanisms, under the
SAME rolling-origin protocol as run_deep_baseline (lookback=4, min_train=104,
retrain_every=13, seed=42, epochs=80). Fills every cell of

              Encoder-Concat |  Gated  | Cross-Attention
   Mfinship   finship_concat | finship | finship_xattn
   Mfinrs     finrs_concat   | finrs   | finrs_xattn
   Mfull      m4concat       | m4rep   | m4xattn

For each cell it reports RMSE, skill vs M0, DirAcc, the VALID Clark-West p vs M0
(nested -> does it beat the random walk?) and the VALID Diebold-Mariano p vs the
flat M1_Ridge (non-nested model class, where Clark-West would be invalid). The
flat M1 predictions are READ from baseline_predictions.csv (no xgboost import;
avoids the macOS torch+OpenMP segfault, mirroring run_deep_baseline).

Outputs (-> 05_outputs/baselines/deep/):
  deep_fusion_matrix.csv     one row per (combo, fusion) with all metrics
  deep_fusion_matrix.png     skill-vs-M0 grouped bars (combo x fusion)

Run:
  python3 04_code/scripts/run_deep_fusion_matrix.py            # ~15-30 min CPU
  python3 04_code/scripts/run_deep_fusion_matrix.py --epochs 40
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

OUT_DIR = data.ROOT / "05_outputs/baselines/deep"

# (combo display label, fusion display label, CONFIGS key)
CELLS = [
    ("Mfinship", "concat", "finship_concat"),
    ("Mfinship", "gated",  "finship"),
    ("Mfinship", "xattn",  "finship_xattn"),
    ("Mfinrs",   "concat", "finrs_concat"),
    ("Mfinrs",   "gated",  "finrs"),
    ("Mfinrs",   "xattn",  "finrs_xattn"),
    ("Mfull",    "concat", "m4concat"),
    ("Mfull",    "gated",  "m4rep"),
    ("Mfull",    "xattn",  "m4xattn"),
]
COMBOS = ["Mfinship", "Mfinrs", "Mfull"]
FUSIONS = ["concat", "gated", "xattn"]


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
    df = data.load_matrix()
    dico = data.load_dict()
    m1_pred = data.ROOT / "05_outputs/baselines/m1/baseline_predictions.csv"
    if not m1_pred.exists():
        raise SystemExit(f"Missing {m1_pred}.\n"
                         f"Run: python3 04_code/scripts/run_baseline.py --modality M1")
    res_m1 = pd.read_csv(m1_pred, index_col=0, parse_dates=True)
    res_m1.index.name = "date"

    dds = build_deep_dataset(df, dico, lookback=args.lookback)
    print(f"deep dataset N={len(dds['idx'])} lookback={args.lookback}  "
          f"(seed={args.seed}, epochs={args.epochs})\n")

    t0 = time.time()
    rows = []
    for combo, fus, cfg in CELLS:
        res = rolling_origin_deep(
            dds, combo, cfg, min_train=args.min_train,
            retrain_every=args.retrain_every, seed=args.seed,
            epochs=args.epochs, verbose=False)
        col = f"{combo}_{CONFIGS[cfg][1]}"
        common = res.index.intersection(res_m1.index)
        y = res_m1.loc[common, "P_next_actual"].to_numpy()
        e_m0 = res_m1.loc[common, "P_hat_M0"].to_numpy() - y
        rmse_m0 = float(np.sqrt(np.mean(e_m0 ** 2)))
        yhat = res.loc[common, f"P_hat_{col}"].to_numpy()
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        rhat = res.loc[common, f"r_hat_{col}"].to_numpy()
        ract = res_m1.loc[common, "r_actual"].to_numpy()
        # VALID tests: CW vs M0 (nested), DM vs flat M1_Ridge (non-nested).
        _, cw_p_m0 = metrics.clark_west(
            y, res_m1.loc[common, "P_hat_M0"].to_numpy(), yhat)
        _, dm_p_m1 = metrics.dm_test(
            yhat - y, res_m1.loc[common, "P_hat_M1_Ridge"].to_numpy() - y)
        row = {"combo": combo, "fusion": fus, "config": cfg,
               "RMSE": rmse, "skill_vs_M0": 1 - rmse / rmse_m0,
               "DirAcc": metrics.directional_acc(rhat, ract),
               "CW_p_vs_M0": cw_p_m0, "DM_p_vs_M1": dm_p_m1,
               "n_test": len(common)}
        rows.append(row)
        print(f"  {combo:9s} {fus:7s} skill={row['skill_vs_M0']*100:+.2f}% "
              f"DirAcc={row['DirAcc']:.3f} CWvsM0={cw_p_m0:.4f} "
              f"DMvsM1={dm_p_m1:.4f} RMSE={rmse:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

    summ = pd.DataFrame(rows)
    out_csv = OUT_DIR / "deep_fusion_matrix.csv"
    summ.to_csv(out_csv, index=False)

    # pretty matrices (skill % / CW vs M0 / DM vs M1)
    def _mat(val):
        return (summ.pivot(index="combo", columns="fusion", values=val)
                .reindex(index=COMBOS, columns=FUSIONS))
    print("\n" + "=" * 60)
    print("Skill vs M0 (%)  [>0 beats the random walk]:")
    print((_mat("skill_vs_M0") * 100).to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\nCW p vs M0  [<0.05 significantly beats RW]:")
    print(_mat("CW_p_vs_M0").to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nDM p vs flat M1_Ridge  [<0.05 significantly beats flat M1]:")
    print(_mat("DM_p_vs_M1").to_string(float_format=lambda x: f"{x:.4f}"))
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
