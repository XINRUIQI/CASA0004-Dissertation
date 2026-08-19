"""
Multi-seed stability for any Deep configuration, on the fusion-MATRIX protocol.

run_deep_sweep.py reseeds only m3_deep_gated / m4_deep_gated / m4_deep_xattn, and
does so at epochs=60. That left the fusions posting the best single-seed numbers
(cross-attention, and concat as its control) without a seed-averaged figure, and
left the reseeded configurations on a different epoch budget from the headline
results. This script closes both gaps for whichever configurations are asked for.

Protocol is the matrix protocol: epochs=80. rolling_origin_deep's defaults already
equal the sweep's _DEF grid point (d=32, gat=2, tcn=2, dropout=0.1, lr=1e-3,
wd=1e-4), so epochs is the only lever that has to be matched, and matching it is
what makes these seeds poolable with deep_fusion_matrix.csv.

Where seed 42 is included it is re-run rather than copied from the matrix, so its
RMSE gap against deep_fusion_matrix.csv measures whether this script reproduces
that protocol. CPU reduction order is not pinned, so agreement is expected to be
close rather than bit-exact (same caveat as run_deep_fusion_matrix.py).

Each invocation writes its own file, and the per-configuration mean and SD that
Appendix B.4 quotes are NOT computed here: scripts/tools/pool_deep_seeds.py pools
every such file with the matrix and sweep, de-duplicates on (config, seed) and owns
the aggregation, so runs added later never have to be merged by hand.

Outputs (-> results/baselines/Deep/_cross/):
  <--out-name>   one row per (config, seed), columns aligned with
                 deep_sweep_summary.csv plus an explicit epochs column

Run:
  python3 code/scripts/deep/run_deep_multiseed.py                  # ~10 min CPU
  python3 code/scripts/deep/run_deep_multiseed.py \
      --configs m4_deep_gated,m4_deep_xattn --seeds 1,2 --out-name deep_seed_m4_80.csv
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

DEFAULT_CONFIGS = ["m3_deep_xattn", "m3_deep_concat"]

# Grid point shared with run_deep_sweep._DEF, recorded so a row is self-describing.
_DEF = dict(lookback=4, d=32, gat=2, tcn=2, rs_kind="meanpool",
            lr=1e-3, wd=1e-4, dropout=0.1)

REPRO_TOL = 1e-3      # RMSE (USD/barrel), as in run_deep_fusion_matrix.py


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-seed stability for M3_Deep cross-attention and concat.")
    ap.add_argument("--seeds", type=str, default="42,1,2")
    ap.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS),
                    help="comma-separated CONFIGS keys, e.g. m3_deep_gated")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--out-name", type=str, default="deep_seed_m3fusion.csv")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown config(s) {unknown}; valid keys: {sorted(CONFIGS)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mat_csv = OUT_DIR / "deep_fusion_matrix.csv"
    mat = (pd.read_csv(mat_csv).set_index("config")["RMSE"]
           if mat_csv.exists() else None)

    df = data.load_matrix()
    dico = data.load_dict()
    m1_pred = m1_flat_predictions(data.ROOT)
    if not m1_pred.exists():
        raise SystemExit(f"Missing {m1_pred}.\n"
                         f"Run: python3 code/scripts/flat/run_baseline.py --modality M1")
    res_m1 = pd.read_csv(m1_pred, index_col=0, parse_dates=True)
    res_m1.index.name = "date"

    dds = build_deep_dataset(df, dico, lookback=args.lookback)
    runs = [(c, s) for c in configs for s in seeds]
    print(f"deep dataset N={len(dds['idx'])} lookback={args.lookback}")
    print(f"{len(runs)} runs = {len(configs)} configs x {len(seeds)} seeds "
          f"(epochs={args.epochs}, matrix protocol)\n")

    t0 = time.time()
    rows = []
    for k, (cfg, seed) in enumerate(runs):
        res = rolling_origin_deep(
            dds, CONFIGS[cfg][1], cfg, min_train=args.min_train,
            retrain_every=args.retrain_every, seed=seed,
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
        # Exploratory one-sided DM, same convention as run_deep_sweep: reference
        # first, so a small p means this run is the more accurate of the pair.
        _, dm_p_m0 = metrics.dm_test(e_m0, yhat - y)
        _, dm_p_flat_s1 = metrics.dm_test(
            res_m1.loc[common, "P_hat_M1_Flat_Ridge"].to_numpy() - y, yhat - y)
        rows.append({"group": "seed", "config": cfg, "seed": seed,
                     "epochs": args.epochs, **_DEF,
                     "RMSE": rmse, "skill_vs_M0": 1 - rmse / rmse_m0,
                     "DirAcc": metrics.directional_acc(rhat, ract),
                     "DM_p_vs_M0": dm_p_m0, "DM_p_vs_Flat_S1": dm_p_flat_s1,
                     "n_test": len(common)})
        ref = None if (mat is None or seed != 42) else mat.get(cfg)
        chk = "" if ref is None else f" [matrix {ref:.6f} d={rmse-ref:+.2e}]"
        print(f"  [{k+1}/{len(runs)}] {cfg:15s} seed={seed:<3d} "
              f"skill={rows[-1]['skill_vs_M0']*100:+.2f}% "
              f"DirAcc={rows[-1]['DirAcc']:.3f} RMSE={rmse:.6f}{chk} "
              f"({time.time()-t0:.0f}s)", flush=True)

    summ = pd.DataFrame(rows)
    out_csv = OUT_DIR / args.out_name
    summ.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    if mat is not None and 42 in seeds:
        rep = summ[summ["seed"] == 42].set_index("config")["RMSE"]
        delta = (rep - mat.reindex(rep.index)).dropna()
        if len(delta):
            ok = "within" if delta.abs().max() < REPRO_TOL else "ABOVE"
            print(f"Protocol check vs deep_fusion_matrix.csv over {len(delta)} "
                  f"seed=42 cells: max |delta RMSE| = {delta.abs().max():.2e} "
                  f"({ok} tolerance {REPRO_TOL:.0e})")
    print(f"\nMulti-seed stability over seeds {seeds} (epochs={args.epochs}):")
    for c in configs:
        s = summ[summ["config"] == c]
        if not len(s):
            continue
        print(f"  {c:15s}: skill {s['skill_vs_M0'].mean()*100:+.2f}% "
              f"± {s['skill_vs_M0'].std()*100:.2f}  "
              f"(min {s['skill_vs_M0'].min()*100:+.2f}, "
              f"max {s['skill_vs_M0'].max()*100:+.2f})  "
              f"| positive in {(s['skill_vs_M0']>0).sum()}/{len(s)} seeds")
    print("\nPer-run detail:")
    print(summ[["config", "seed", "RMSE", "skill_vs_M0", "DirAcc",
                "DM_p_vs_M0", "DM_p_vs_Flat_S1"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)
    print(f"\nElapsed {time.time()-t0:.0f}s\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
