"""
Multi-seed robustness for 3A: m4rep (full gated fin+rs+ship) with raw meanpool
vs within-site anomaly (meanpool_anom) RS embedding, across several seeds, under
the MAIN deep protocol (lb=4, min_train=104, retrain_every=13, epochs=80).

Purpose: the single-seed comparison (compare_rs_anom.py) showed m4rep improved
with anom (skill -1.28%->+0.05%, CW vs M0 0.894->0.042) while the rs/finrs arms
got worse. This script checks whether that m4rep gain is STABLE across seeds
before considering promoting meanpool_anom to the main model. Same flat M0 /
M1_Ridge reference as run_deep_baseline (read from baseline_predictions.csv).

Run:
  python3 04_code/scripts/multiseed_m4rep_anom.py
  python3 04_code/scripts/multiseed_m4rep_anom.py --seeds 42,1,2,3,4 --epochs 80
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

KINDS = ["meanpool", "meanpool_anom"]
CFG = "m4rep"
LABEL = "Mfull"
OUT = data.ROOT / "05_outputs/baselines/deep/rs_anom_multiseed.csv"


def _metrics(merged, col, y, ym0, ym1, ract) -> dict:
    yhat = merged[f"P_hat_{col}"].to_numpy()
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    rmse0 = float(np.sqrt(np.mean((ym0 - y) ** 2)))
    return {"RMSE": rmse, "skill_vs_M0": 1 - rmse / rmse0,
            "DirAcc": metrics.directional_acc(merged[f"r_hat_{col}"].to_numpy(), ract),
            "CW_p_vs_M0": metrics.clark_west(y, ym0, yhat)[1],
            "DM_p_vs_M1": metrics.dm_test(yhat - y, ym1 - y)[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2,3,4")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    df = data.load_matrix(); dico = data.load_dict()
    res_m1 = pd.read_csv(data.ROOT / "05_outputs/baselines/m1/baseline_predictions.csv",
                         index_col=0, parse_dates=True)
    res_m1.index.name = "date"
    col = f"{LABEL}_{CONFIGS[CFG][1]}"

    # dataset does not depend on seed -> build once per kind.
    ds = {k: build_deep_dataset(df, dico, lookback=args.lookback, rs_kind=k)
          for k in KINDS}

    t0 = time.time()
    rows = []
    for seed in seeds:
        for k in KINDS:
            r = rolling_origin_deep(ds[k], LABEL, CFG, min_train=args.min_train,
                                    retrain_every=args.retrain_every, seed=seed,
                                    epochs=args.epochs, verbose=False)
            common = res_m1.index.intersection(r.index).sort_values()
            merged = res_m1.loc[common].copy()
            merged[f"r_hat_{col}"] = r.loc[common, f"r_hat_{col}"]
            merged[f"P_hat_{col}"] = r.loc[common, f"P_hat_{col}"]
            rows.append({"seed": seed, "rs_kind": k, **_metrics(
                merged, col, merged["P_next_actual"].to_numpy(),
                merged["P_hat_M0"].to_numpy(), merged["P_hat_M1_Ridge"].to_numpy(),
                merged["r_actual"].to_numpy())})
            print(f"DONE seed={seed} {k}: skill={rows[-1]['skill_vs_M0']*100:+.2f}% "
                  f"DirAcc={rows[-1]['DirAcc']*100:.1f} CWp={rows[-1]['CW_p_vs_M0']:.3f}",
                  flush=True)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    raw = out[out.rs_kind == "meanpool"].set_index("seed")
    an = out[out.rs_kind == "meanpool_anom"].set_index("seed")
    print("\n" + "=" * 84)
    print(f"MULTI-SEED m4rep  raw vs anom  ({len(seeds)} seeds, epochs={args.epochs}, "
          f"lb={args.lookback})")
    print("=" * 84)
    print(f"{'seed':>5s} {'skill_raw%':>10s} {'skill_anom%':>11s} {'delta%':>8s} "
          f"{'DirAcc_r':>8s} {'DirAcc_a':>8s} {'CWp_raw':>8s} {'CWp_anom':>8s}")
    for s in seeds:
        dr = (an.loc[s, "skill_vs_M0"] - raw.loc[s, "skill_vs_M0"]) * 100
        print(f"{s:>5d} {raw.loc[s,'skill_vs_M0']*100:>10.2f} "
              f"{an.loc[s,'skill_vs_M0']*100:>11.2f} {dr:>+8.2f} "
              f"{raw.loc[s,'DirAcc']*100:>8.1f} {an.loc[s,'DirAcc']*100:>8.1f} "
              f"{raw.loc[s,'CW_p_vs_M0']:>8.3f} {an.loc[s,'CW_p_vs_M0']:>8.3f}")
    print("-" * 84)
    n_better = int((an["skill_vs_M0"] > raw["skill_vs_M0"]).sum())
    print(f"skill  raw : mean {raw['skill_vs_M0'].mean()*100:+.2f}%  "
          f"std {raw['skill_vs_M0'].std()*100:.2f}")
    print(f"skill  anom: mean {an['skill_vs_M0'].mean()*100:+.2f}%  "
          f"std {an['skill_vs_M0'].std()*100:.2f}")
    print(f"DirAcc raw/anom mean: {raw['DirAcc'].mean()*100:.1f}% / "
          f"{an['DirAcc'].mean()*100:.1f}%")
    print(f"CW_p<0.05 vs M0: raw {int((raw['CW_p_vs_M0']<0.05).sum())}/{len(seeds)}  "
          f"anom {int((an['CW_p_vs_M0']<0.05).sum())}/{len(seeds)}")
    print(f"anom beats raw on skill in {n_better}/{len(seeds)} seeds")
    stable = (n_better >= np.ceil(0.8 * len(seeds))
              and an["skill_vs_M0"].mean() > raw["skill_vs_M0"].mean())
    print("\nVERDICT: " + ("STABLE improvement -> candidate to promote anom to main."
                           if stable else
                           "NOT stable -> keep raw as main; anom stays supplementary."))
    print(f"saved -> {OUT}   elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
