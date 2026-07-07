"""
3A comparison: raw Prithvi meanpool embedding vs within-site anomaly
(`meanpool_anom`, see deep_dataset._site_expanding_demean) on the RS-involving
deep configs, under the MAIN deep protocol (lookback=4, min_train=104,
retrain_every=13, epochs=80, seed=42). Same flat M0 / M1_Ridge reference as
run_deep_baseline.py (read from baseline_predictions.csv).

For each config (rs / finrs / m4rep) x rs_kind it reports RMSE, skill vs M0,
DirAcc, CW_p vs M0 (nested), DM_p vs flat M1_Ridge (non-nested), and the learned
gate alpha (m4rep). Prints a raw-vs-anom delta table + verdict so we can keep the
change only if it helps, else roll back.

Run:
  python3 04_code/scripts/compare_rs_anom.py
  python3 04_code/scripts/compare_rs_anom.py --configs rs,m4rep --epochs 80
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
from models.deep_dataset import (apply_scalers,             # noqa: E402
                                 build_deep_dataset, fit_scalers)
from models.deep_rolling import (CONFIGS, _to_tensors,      # noqa: E402
                                 _train_fold, rolling_origin_deep)

LABELS = {"ship": "Mship", "fin": "Mfin", "rs": "Mrs", "fusion": "Mfusion",
          "finrs": "Mfinrs", "m4rep": "Mfull", "m4concat": "Mconcat"}
KINDS = ["meanpool", "meanpool_anom"]
OUT = data.ROOT / "05_outputs/baselines/deep/rs_anom_compare.csv"


def _metrics(merged, col, y, ym0, ym1, ract) -> dict:
    yhat = merged[f"P_hat_{col}"].to_numpy()
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    rmse0 = float(np.sqrt(np.mean((ym0 - y) ** 2)))
    rhat = merged[f"r_hat_{col}"].to_numpy()
    return {"RMSE": rmse, "skill_vs_M0": 1 - rmse / rmse0,
            "DirAcc": metrics.directional_acc(rhat, ract),
            "CW_p_vs_M0": metrics.clark_west(y, ym0, yhat)[1],
            "DM_p_vs_M1": metrics.dm_test(yhat - y, ym1 - y)[1]}


def _gate_alpha(ds, epochs, seed) -> dict:
    import torch
    n = len(ds["idx"])
    mods, _, ftype = CONFIGS["m4rep"]
    sc = fit_scalers(ds, train_n=n)
    model, _, _ = _train_fold(ds, sc, n, mods, seed, epochs, 1e-3, 1e-4, 32, 52,
                              "cpu", {}, ftype)
    X = _to_tensors(apply_scalers(ds, sc, slice(0, n)), "cpu")
    model.eval()
    with torch.no_grad():
        _, info = model(aoi=X["aoi"], choke=X["choke"], adj=X["adj"],
                        fin=X["fin"], rs=X["rs"], rs_mask=X["rs_mask"])
    g = info["gate"].cpu().numpy()
    return dict(zip(info["gate_order"], g.mean(0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="rs,finrs,m4rep")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    df = data.load_matrix(); dico = data.load_dict()
    m1_pred = data.ROOT / "05_outputs/baselines/m1/baseline_predictions.csv"
    res_m1 = pd.read_csv(m1_pred, index_col=0, parse_dates=True)
    res_m1.index.name = "date"

    t0 = time.time()
    rows, gates = [], {}
    for kind in KINDS:
        print(f"\n===== rs_kind = {kind} =====", flush=True)
        ds = build_deep_dataset(df, dico, lookback=args.lookback, rs_kind=kind)
        res = {}
        for cfg in configs:
            print(f"[{kind}/{cfg}] rolling...", flush=True)
            res[cfg] = rolling_origin_deep(
                ds, LABELS[cfg], cfg, min_train=args.min_train,
                retrain_every=args.retrain_every, seed=args.seed,
                epochs=args.epochs, verbose=False)
            print(f"DONE {kind}/{cfg}", flush=True)
        common = res_m1.index
        for r in res.values():
            common = common.intersection(r.index)
        common = common.sort_values()
        merged = res_m1.loc[common].copy()
        y = merged["P_next_actual"].to_numpy()
        ym0 = merged["P_hat_M0"].to_numpy()
        ym1 = merged["P_hat_M1_Ridge"].to_numpy()
        ract = merged["r_actual"].to_numpy()
        for cfg in configs:
            col = f"{LABELS[cfg]}_{CONFIGS[cfg][1]}"
            merged[f"r_hat_{col}"] = res[cfg].loc[common, f"r_hat_{col}"]
            merged[f"P_hat_{col}"] = res[cfg].loc[common, f"P_hat_{col}"]
            rows.append({"rs_kind": kind, "config": cfg,
                         **_metrics(merged, col, y, ym0, ym1, ract)})
        if "m4rep" in configs:
            gates[kind] = _gate_alpha(ds, args.epochs, args.seed)
            print(f"gate[{kind}] {gates[kind]}", flush=True)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\n" + "=" * 92)
    print(f"3A COMPARISON  raw meanpool vs meanpool_anom  (n_common={len(common)}, "
          f"epochs={args.epochs}, lb={args.lookback})")
    print("=" * 92)
    piv = out.pivot(index="config", columns="rs_kind")
    for cfg in configs:
        r = out[(out.config == cfg) & (out.rs_kind == "meanpool")].iloc[0]
        a = out[(out.config == cfg) & (out.rs_kind == "meanpool_anom")].iloc[0]
        d_skill = (a.skill_vs_M0 - r.skill_vs_M0) * 100
        d_dir = (a.DirAcc - r.DirAcc) * 100
        better = (a.skill_vs_M0 > r.skill_vs_M0)
        print(f"\n[{cfg}] {LABELS[cfg]}")
        print(f"  {'metric':14s} {'raw':>10s} {'anom':>10s} {'delta':>10s}")
        print(f"  {'skill_vs_M0%':14s} {r.skill_vs_M0*100:10.2f} "
              f"{a.skill_vs_M0*100:10.2f} {d_skill:+10.2f}")
        print(f"  {'DirAcc%':14s} {r.DirAcc*100:10.1f} {a.DirAcc*100:10.1f} "
              f"{d_dir:+10.1f}")
        print(f"  {'RMSE':14s} {r.RMSE:10.3f} {a.RMSE:10.3f} "
              f"{a.RMSE - r.RMSE:+10.3f}")
        print(f"  {'CW_p_vs_M0':14s} {r.CW_p_vs_M0:10.3f} {a.CW_p_vs_M0:10.3f}")
        print(f"  {'DM_p_vs_M1':14s} {r.DM_p_vs_M1:10.3f} {a.DM_p_vs_M1:10.3f}")
        print(f"  verdict: anom is {'BETTER' if better else 'WORSE/EQUAL'} "
              f"(skill delta {d_skill:+.2f}%)")

    if gates:
        print("\ngate alpha (m4rep, mean over samples):")
        for kind in KINDS:
            if kind in gates:
                g = gates[kind]
                print(f"  {kind:14s} " + "  ".join(
                    f"{k}={v:.3f}" for k, v in g.items()))

    n_better = sum(
        out[(out.config == c) & (out.rs_kind == "meanpool_anom")].iloc[0].skill_vs_M0
        > out[(out.config == c) & (out.rs_kind == "meanpool")].iloc[0].skill_vs_M0
        for c in configs)
    print("\n" + "=" * 92)
    print(f"OVERALL: anom improves skill in {n_better}/{len(configs)} configs. "
          f"{'KEEP 3A.' if n_better > len(configs) / 2 else 'ROLL BACK 3A (no consistent gain).'}")
    print(f"saved -> {OUT}   elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
