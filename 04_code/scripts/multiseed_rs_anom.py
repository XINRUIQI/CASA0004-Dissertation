"""
Multi-seed 3A check: is the m4rep gain from `meanpool_anom` (within-site RS
embedding anomaly) STABLE across seeds, or a single-seed fluke?

Same protocol as run_deep_baseline (lookback=4, min_train=104, retrain_every=13,
epochs=80) and the same flat M0 / M1_Ridge reference. For each seed and rs_kind
it runs the config(s) and reports skill vs M0 / DirAcc / RMSE / CW_p vs M0.
Results are paired by seed (raw vs anom share the identical dataset & seed), so
the decision rule is: adopt anom only if it beats raw on skill in the clear
majority of seeds AND the mean paired gain is positive.

Run:
  python3 04_code/scripts/multiseed_rs_anom.py                       # m4rep, 5 seeds
  python3 04_code/scripts/multiseed_rs_anom.py --configs m4rep,finrs --seeds 42,0,1,2,3
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

from backtest import data, metrics                          # noqa: E402
from models.deep_dataset import build_deep_dataset          # noqa: E402
from models.deep_rolling import CONFIGS, rolling_origin_deep  # noqa: E402

LABELS = {"ship": "Mship", "fin": "Mfin", "rs": "Mrs", "finship": "Mfinship",
          "finrs": "Mfinrs", "m4rep": "Mfull", "m4concat": "Mconcat"}
KINDS = ["meanpool", "meanpool_anom"]
OUT = data.ROOT / "05_outputs/baselines/deep/rs_anom_multiseed.csv"


def _metrics(merged, col, y, ym0, ym1, ract) -> dict:
    yhat = merged[f"P_hat_{col}"].to_numpy()
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    rmse0 = float(np.sqrt(np.mean((ym0 - y) ** 2)))
    rhat = merged[f"r_hat_{col}"].to_numpy()
    return {"RMSE": rmse, "skill_vs_M0": 1 - rmse / rmse0,
            "DirAcc": metrics.directional_acc(rhat, ract),
            "CW_p_vs_M0": metrics.clark_west(y, ym0, yhat)[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="m4rep")
    ap.add_argument("--seeds", default="42,0,1,2,3")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    df = data.load_matrix(); dico = data.load_dict()
    res_m1 = pd.read_csv(
        data.ROOT / "05_outputs/baselines/m1/baseline_predictions.csv",
        index_col=0, parse_dates=True)
    res_m1.index.name = "date"

    t0, rows = time.time(), []
    for kind in KINDS:
        ds = build_deep_dataset(df, dico, lookback=args.lookback, rs_kind=kind)
        for seed in seeds:
            for cfg in configs:
                print(f"[{kind}/{cfg}/seed={seed}] rolling...", flush=True)
                r = rolling_origin_deep(
                    ds, LABELS[cfg], cfg, min_train=args.min_train,
                    retrain_every=args.retrain_every, seed=seed,
                    epochs=args.epochs, verbose=False)
                common = res_m1.index.intersection(r.index).sort_values()
                merged = res_m1.loc[common].copy()
                col = f"{LABELS[cfg]}_{CONFIGS[cfg][1]}"
                merged[f"r_hat_{col}"] = r.loc[common, f"r_hat_{col}"]
                merged[f"P_hat_{col}"] = r.loc[common, f"P_hat_{col}"]
                m = _metrics(merged, col,
                             merged["P_next_actual"].to_numpy(),
                             merged["P_hat_M0"].to_numpy(),
                             merged["P_hat_M1_Ridge"].to_numpy(),
                             merged["r_actual"].to_numpy())
                rows.append({"rs_kind": kind, "config": cfg, "seed": seed, **m})
                print(f"DONE {kind}/{cfg}/seed={seed} "
                      f"skill={m['skill_vs_M0']*100:+.2f}% DirAcc={m['DirAcc']*100:.1f}%",
                      flush=True)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\n" + "=" * 88)
    print(f"MULTI-SEED 3A  raw meanpool vs meanpool_anom  "
          f"(seeds={seeds}, epochs={args.epochs}, lb={args.lookback})")
    print("=" * 88)
    for cfg in configs:
        raw = (out[(out.config == cfg) & (out.rs_kind == "meanpool")]
               .set_index("seed").sort_index())
        an = (out[(out.config == cfg) & (out.rs_kind == "meanpool_anom")]
              .set_index("seed").sort_index())
        print(f"\n[{cfg}] {LABELS[cfg]}   per-seed skill vs M0 (%)  /  DirAcc (%)")
        print(f"  {'seed':>5s} {'raw_skill':>10s} {'anom_skill':>11s} "
              f"{'Δskill':>8s} {'raw_DA':>7s} {'anom_DA':>8s}")
        for s in seeds:
            rs_, as_ = raw.loc[s], an.loc[s]
            print(f"  {s:5d} {rs_.skill_vs_M0*100:10.2f} {as_.skill_vs_M0*100:11.2f} "
                  f"{(as_.skill_vs_M0-rs_.skill_vs_M0)*100:+8.2f} "
                  f"{rs_.DirAcc*100:7.1f} {as_.DirAcc*100:8.1f}")
        dsk = (an["skill_vs_M0"] - raw["skill_vs_M0"]) * 100
        dda = (an["DirAcc"] - raw["DirAcc"]) * 100
        n_better = int((dsk > 0).sum())
        try:
            from scipy import stats
            pp = stats.ttest_rel(an["skill_vs_M0"], raw["skill_vs_M0"]).pvalue
        except Exception:
            pp = float("nan")
        print(f"  ---")
        print(f"  raw  skill  mean={raw['skill_vs_M0'].mean()*100:+.2f}% "
              f"std={raw['skill_vs_M0'].std()*100:.2f}  "
              f"DirAcc mean={raw['DirAcc'].mean()*100:.1f}%")
        print(f"  anom skill  mean={an['skill_vs_M0'].mean()*100:+.2f}% "
              f"std={an['skill_vs_M0'].std()*100:.2f}  "
              f"DirAcc mean={an['DirAcc'].mean()*100:.1f}%")
        print(f"  Δskill mean={dsk.mean():+.2f}%  ΔDirAcc mean={dda.mean():+.1f}pp  "
              f"anom>raw in {n_better}/{len(seeds)} seeds  paired t p={pp:.3f}")
        stable = n_better > len(seeds) / 2 and dsk.mean() > 0
        print(f"  VERDICT: anom {'STABLE improvement -> consider upgrading' if stable else 'NOT stable -> keep raw'}")
    print("\n" + "=" * 88)
    print(f"saved -> {OUT}   elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
