"""
Robustness sweep for the deep representation-level models: multi-seed +
hyperparameter grid (lookback / d_model / layers), to check the RQ2 Clark-West
conclusion (representation fusion > flat M1) is stable, not a single-seed fluke.

Two experiment groups:
  seed  : {fusion, m4rep} x seeds {42,1,2} at lookback=8, d=32, layers=2
  hyper : fusion at seed=42 over lookback {4,8,12} x d {32,64} (+ layers=1)

For each run we record skill vs M0 and Clark-West p vs flat M1_Ridge on the
common test weeks (flat M1 read from baseline_predictions.csv; no xgboost import).

Outputs (-> 05_outputs/baselines/deep/):
  deep_sweep_summary.csv   one row per run
  deep_sweep.png           skill-by-config (seeds) + skill-by-hyperparam panels

Run:
  python3 04_code/scripts/run_deep_sweep.py               # ~20 min, CPU
  python3 04_code/scripts/run_deep_sweep.py --epochs 40   # faster
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
LABELS = {"fusion": "Mfusion", "m4rep": "Mfull", "ship": "Mship",
          "fin": "Mfin", "rs": "Mrs"}


def experiments() -> list[dict]:
    seed = [dict(group="seed", config=c, seed=s, lookback=8, d=32, gat=2, tcn=2)
            for c in ["fusion", "m4rep"] for s in [42, 1, 2]]
    hyper = [dict(group="hyper", config="fusion", seed=42, lookback=lb, d=d, gat=2, tcn=2)
             for lb in [4, 8, 12] for d in [32, 64]]
    hyper += [dict(group="hyper", config="fusion", seed=42, lookback=8, d=32, gat=1, tcn=1)]
    return seed + hyper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    args = ap.parse_args()

    df = data.load_matrix()
    dico = data.load_dict()
    m1_pred = data.ROOT / "05_outputs/baselines/m1/baseline_predictions.csv"
    res_m1 = pd.read_csv(m1_pred, index_col=0, parse_dates=True)
    res_m1.index.name = "date"

    exps = experiments()
    print(f"Sweep: {len(exps)} runs (epochs={args.epochs})")

    ds_cache: dict[int, dict] = {}
    rows = []
    t0 = time.time()
    for k, e in enumerate(exps):
        lb = e["lookback"]
        if lb not in ds_cache:
            ds_cache[lb] = build_deep_dataset(df, dico, lookback=lb)
        dds = ds_cache[lb]
        mk = {"d": e["d"], "gat_layers": e["gat"], "tcn_layers": e["tcn"]}
        res = rolling_origin_deep(dds, LABELS[e["config"]], e["config"],
                                  min_train=args.min_train,
                                  retrain_every=args.retrain_every,
                                  seed=e["seed"], epochs=args.epochs,
                                  model_kwargs=mk, verbose=False)
        col = f"{LABELS[e['config']]}_{CONFIGS[e['config']][1]}"
        common = res.index.intersection(res_m1.index)
        y = res_m1.loc[common, "P_next_actual"].to_numpy()
        pn = y
        e_m0 = res_m1.loc[common, "P_hat_M0"].to_numpy() - pn
        rmse_m0 = float(np.sqrt(np.mean(e_m0 ** 2)))
        yhat = res.loc[common, f"P_hat_{col}"].to_numpy()
        rmse = float(np.sqrt(np.mean((yhat - pn) ** 2)))
        rhat = res.loc[common, f"r_hat_{col}"].to_numpy()
        ract = res_m1.loc[common, "r_actual"].to_numpy()
        diracc = metrics.directional_acc(rhat, ract)
        _, cw_p = metrics.clark_west(
            y, res_m1.loc[common, "P_hat_M1_Ridge"].to_numpy(), yhat)
        row = {**e, "RMSE": rmse, "skill_vs_M0": 1 - rmse / rmse_m0,
               "DirAcc": diracc, "CW_p_vs_M1": cw_p, "n_test": len(common)}
        rows.append(row)
        print(f"  [{k+1}/{len(exps)}] {e['group']:5s} {e['config']:7s} "
              f"seed={e['seed']} lb={lb} d={e['d']} L={e['gat']} "
              f"skill={row['skill_vs_M0']*100:+.2f}% CW_p={cw_p:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    summ = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "deep_sweep_summary.csv"
    summ.to_csv(out_csv, index=False)

    # --- summary stats ---
    print("\n" + "=" * 78)
    seed_g = summ[summ["group"] == "seed"]
    print("Multi-seed stability (skill vs M0, CW_p vs flat M1):")
    for c in seed_g["config"].unique():
        s = seed_g[seed_g["config"] == c]
        print(f"  {c:7s}: skill {s['skill_vs_M0'].mean()*100:+.2f}% "
              f"± {s['skill_vs_M0'].std()*100:.2f}  | "
              f"CW_p {s['CW_p_vs_M1'].mean():.4f} "
              f"(min {s['CW_p_vs_M1'].min():.4f}, max {s['CW_p_vs_M1'].max():.4f}) "
              f"| CW_p<0.05 in {(s['CW_p_vs_M1']<0.05).sum()}/{len(s)} seeds")
    print("\nHyperparam sweep (fusion, seed=42):")
    hyp = summ[summ["group"] == "hyper"].sort_values(["lookback", "d", "gat"])
    print(hyp[["lookback", "d", "gat", "skill_vs_M0", "DirAcc",
               "CW_p_vs_M1"]].to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)

    _plot(summ, OUT_DIR / "deep_sweep.png")
    print(f"\nElapsed {time.time()-t0:.0f}s\nSaved: {out_csv}\n"
          f"       {OUT_DIR/'deep_sweep.png'}")


def _plot(summ: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    seed_g = summ[summ["group"] == "seed"]
    configs = list(seed_g["config"].unique())
    for i, c in enumerate(configs):
        s = seed_g[seed_g["config"] == c]
        ax1.scatter([i] * len(s), s["skill_vs_M0"] * 100, s=60, alpha=0.7)
        ax1.scatter([i], [s["skill_vs_M0"].mean() * 100], marker="_", s=400, color="k")
    ax1.axhline(0, color="grey", ls="--", lw=0.8)
    ax1.set_xticks(range(len(configs))); ax1.set_xticklabels(configs)
    ax1.set_title("Multi-seed skill vs M0 (%)  (— = mean)")
    ax1.set_ylabel("skill vs M0 (%)"); ax1.grid(alpha=0.3, axis="y")

    hyp = summ[summ["group"] == "hyper"]
    for d in sorted(hyp["d"].unique()):
        h = hyp[(hyp["d"] == d) & (hyp["gat"] == 2)].sort_values("lookback")
        ax2.plot(h["lookback"], h["skill_vs_M0"] * 100, marker="o", label=f"d={d}")
    ax2.axhline(0, color="grey", ls="--", lw=0.8)
    ax2.set_title("Fusion skill vs M0 by lookback / d (seed=42)")
    ax2.set_xlabel("lookback (weeks)"); ax2.set_ylabel("skill vs M0 (%)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
