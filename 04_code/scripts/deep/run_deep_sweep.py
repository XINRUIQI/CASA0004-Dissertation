"""
Robustness sweep for the deep representation-level models: hyperparameter grid
(lookback / d_model / layers), regularisation grid and RS-branch study, to check
that the RQ2 conclusion is not an artefact of the chosen capacity or regularisation,
and to probe the weakest modality (RS).

Three experiment groups, all at seed=42. Reseeding is NOT done here: it lives in
run_deep_multiseed.py, which runs on the matrix epoch budget so its seeds pool with
the headline results, and is aggregated by scripts/tools/pool_deep_seeds.py. A
former "seed" group in this script duplicated those (config, seed) pairs at a lower
budget and was discarded during pooling every time, so it was removed.

  hyper : fusion at seed=42 over lookback {4,8,12} x d {32,64} (+ layers=1)
  rs    : RS branch — meanpool vs cls Prithvi embedding at default reg, plus a
          small lr / weight_decay / dropout grid on meanpool (P1-5): is the
          weakest branch weak because it is under-tuned, or genuinely weak?
  reg   : lr / weight_decay / dropout grid on the MAIN gated fusion (P1-6): is
          the main model's skill sensitive to regularisation, not only RS?

For each run we record skill vs M0 plus two one-sided DM-HLN p values, against M0
and against M1_Flat_Ridge, on the common test weeks (flat M1_Flat read from
baseline_predictions.csv; no xgboost import). Clark-West is deliberately not used:
it presumes the benchmark is a parameter restriction of the candidate, which no Deep
configuration satisfies. Both p values here are exploratory sensitivity readings and
are not the dissertation's reported tests — those come from the single frozen family
built by scripts/tools/build_test_tables.py.

Outputs (-> 05_outputs/baselines/Deep/_cross/):
  deep_sweep_summary.csv   one row per run, incl. the epoch budget and the full grid
  deep_sweep.png           multi-seed (read from deep_seed_pooled.csv) + hyperparam
                           + RS-branch panels

Run:
  python3 04_code/scripts/deep/run_deep_sweep.py --epochs 80   # matrix budget
  python3 04_code/scripts/deep/run_deep_sweep.py               # ~30 min at epochs 60
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
    DEEP_MODEL_NAMES, deep_out_dir, m1_flat_predictions, resolve_deep_config,
)
from models.deep_dataset import build_deep_dataset      # noqa: E402
from models.deep_rolling import CONFIGS, rolling_origin_deep  # noqa: E402

OUT_DIR = deep_out_dir(data.ROOT, "cross")

# default training hyperparameters (mirrors deep_rolling defaults)
_DEF = dict(gat=2, tcn=2, rs_kind="meanpool", lr=1e-3, wd=1e-4, dropout=0.1)


def experiments() -> list[dict]:
    """Return the sweep runs. Every dict carries the full config so a row in
    deep_sweep_summary.csv is self-describing.

    Reseeding is deliberately absent. It used to live here as a "seed" group, but
    those runs were fixed at this script's epoch budget while the headline results
    use the matrix budget, so every one of them was superseded and discarded during
    pooling. Multi-seed evidence now comes from run_deep_multiseed.py, aggregated by
    scripts/tools/pool_deep_seeds.py; keeping a second producer here would only
    re-introduce duplicate (config, seed) rows at a lower budget."""
    hyper = [dict(group="hyper", config="m3_deep_gated", seed=42, lookback=lb, d=d, **_DEF)
             for lb in [4, 8, 12] for d in [32, 64]]
    hyper += [dict(group="hyper", config="m3_deep_gated", seed=42, lookback=4, d=32,
                   **{**_DEF, "gat": 1, "tcn": 1})]
    rs = [dict(group="rs", config="m_rs_deep", seed=42, lookback=4, d=32,
               **{**_DEF, "rs_kind": rk}) for rk in ["meanpool", "cls"]]
    for lr in [1e-3, 3e-4]:
        for wd in [1e-4, 1e-3]:
            for dp in [0.1, 0.3]:
                if (lr, wd, dp) == (1e-3, 1e-4, 0.1):
                    continue
                rs.append(dict(group="rs", config="m_rs_deep", seed=42, lookback=4, d=32,
                               **{**_DEF, "lr": lr, "wd": wd, "dropout": dp}))
    reg = []
    for lr in [1e-3, 3e-4]:
        for wd in [1e-4, 1e-3]:
            for dp in [0.1, 0.3]:
                if (lr, wd, dp) == (1e-3, 1e-4, 0.1):
                    continue
                reg.append(dict(group="reg", config="m3_deep_gated", seed=42, lookback=4,
                                d=32, **{**_DEF, "lr": lr, "wd": wd, "dropout": dp}))
    return hyper + rs + reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    args = ap.parse_args()

    df = data.load_matrix()
    dico = data.load_dict()
    m1_pred = m1_flat_predictions(data.ROOT)
    res_m1 = pd.read_csv(m1_pred, index_col=0, parse_dates=True)
    res_m1.index.name = "date"

    exps = experiments()
    print(f"Sweep: {len(exps)} runs (epochs={args.epochs})")

    ds_cache: dict[tuple, dict] = {}
    rows = []
    t0 = time.time()
    for k, e in enumerate(exps):
        lb, rk = e["lookback"], e["rs_kind"]
        key = (lb, rk)
        if key not in ds_cache:
            ds_cache[key] = build_deep_dataset(df, dico, lookback=lb, rs_kind=rk)
        dds = ds_cache[key]
        mk = {"d": e["d"], "gat_layers": e["gat"], "tcn_layers": e["tcn"],
              "dropout": e["dropout"]}
        res = rolling_origin_deep(dds, DEEP_MODEL_NAMES[resolve_deep_config(e["config"])],
                                  e["config"],
                                  min_train=args.min_train,
                                  retrain_every=args.retrain_every,
                                  seed=e["seed"], epochs=args.epochs,
                                  lr=e["lr"], weight_decay=e["wd"],
                                  model_kwargs=mk, verbose=False)
        col = CONFIGS[resolve_deep_config(e["config"])][1]
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
        # Sensitivity sweep: DM-HLN one-sided against M0, plus the same test
        # against flat S1. The second contrast changes BOTH the pathway and the
        # information set, so it is descriptive only and is not the RQ2 contrast
        # (that one holds the information set fixed; see build_test_tables.py).
        _, dm_p_m0 = metrics.dm_test(e_m0, yhat - y)
        _, dm_p_flat_s1 = metrics.dm_test(
            res_m1.loc[common, "P_hat_M1_Flat_Ridge"].to_numpy() - y, yhat - y)
        # epochs is recorded per row, not left to the argparse default: the pooling
        # step in scripts/tools/pool_deep_seeds.py keys on it to decide which run of
        # a repeated (config, seed) pair to keep, and a CSV that omits it forces that
        # script to guess the protocol from this file's defaults.
        row = {**e, "epochs": args.epochs,
               "RMSE": rmse, "skill_vs_M0": 1 - rmse / rmse_m0,
               "DirAcc": diracc, "DM_p_vs_M0": dm_p_m0,
               "DM_p_vs_Flat_S1": dm_p_flat_s1, "n_test": len(common)}
        rows.append(row)
        print(f"  [{k+1}/{len(exps)}] {e['group']:5s} {e['config']:8s} "
              f"seed={e['seed']} lb={lb} d={e['d']} rs={rk:8s} "
              f"lr={e['lr']:.0e} wd={e['wd']:.0e} dp={e['dropout']} "
              f"skill={row['skill_vs_M0']*100:+.2f}% DMvsM0={dm_p_m0:.4f} "
              f"DMvsFlatS1={dm_p_flat_s1:.4f} ({time.time()-t0:.0f}s)", flush=True)

    summ = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "deep_sweep_summary.csv"
    summ.to_csv(out_csv, index=False)

    # --- summary stats ---
    print("\n" + "=" * 78)
    print("Multi-seed stability is not produced here. See deep_seed_summary.csv "
          "(run_deep_multiseed.py -> pool_deep_seeds.py).")
    print("\nHyperparam sweep (fusion, seed=42):")
    hyp = summ[summ["group"] == "hyper"].sort_values(["lookback", "d", "gat"])
    print(hyp[["lookback", "d", "gat", "skill_vs_M0", "DirAcc",
               "DM_p_vs_M0", "DM_p_vs_Flat_S1"]].to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))

    rs_g = summ[summ["group"] == "rs"]
    if len(rs_g):
        print("\nRS branch (P1-5) — embedding type @ default reg (lr1e-3/wd1e-4/dp0.1):")
        emb = rs_g[(rs_g["lr"] == 1e-3) & (rs_g["wd"] == 1e-4)
                   & (rs_g["dropout"] == 0.1)]
        for _, r in emb.sort_values("rs_kind").iterrows():
            print(f"  {r['rs_kind']:8s}: skill {r['skill_vs_M0']*100:+.2f}%  "
                  f"DirAcc {r['DirAcc']:.3f}  DMvsM0_p {r['DM_p_vs_M0']:.4f}")
        mp = rs_g[rs_g["rs_kind"] == "meanpool"].sort_values(
            "skill_vs_M0", ascending=False)
        print("  meanpool regularisation grid (best skill first):")
        print(mp[["lr", "wd", "dropout", "skill_vs_M0", "DirAcc",
                  "DM_p_vs_M0", "DM_p_vs_Flat_S1"]].to_string(index=False,
                 float_format=lambda x: f"{x:.4f}"))
        mp_def = emb[emb["rs_kind"] == "meanpool"]["skill_vs_M0"].values[0] * 100
        b = mp.iloc[0]
        print(f"  => best meanpool reg: lr={b['lr']:.0e} wd={b['wd']:.0e} "
              f"dp={b['dropout']} skill={b['skill_vs_M0']*100:+.2f}% "
              f"(vs default meanpool {mp_def:+.2f}%)")

    reg_g = summ[summ["group"] == "reg"]
    if len(reg_g):
        print("\nMain fusion regularisation grid (P1-6, seed=42 lb=4; best skill first):")
        rr = reg_g.sort_values("skill_vs_M0", ascending=False)
        print(rr[["lr", "wd", "dropout", "skill_vs_M0", "DirAcc",
                  "DM_p_vs_M0", "DM_p_vs_Flat_S1"]].to_string(index=False,
                 float_format=lambda x: f"{x:.4f}"))
    print("=" * 78)

    _plot(summ, OUT_DIR / "deep_sweep.png")
    print(f"\nElapsed {time.time()-t0:.0f}s\nSaved: {out_csv}\n"
          f"       {OUT_DIR/'deep_sweep.png'}")


def _plot(summ: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

    # Panel 1 is no longer produced by this sweep: reseeding moved to
    # run_deep_multiseed.py. Read the pooled per-seed table so the panel and
    # Appendix B.4 quote the same runs, and degrade to a pointer if it is absent.
    pooled_csv = OUT_DIR / "deep_seed_pooled.csv"
    pooled = pd.read_csv(pooled_csv) if pooled_csv.exists() else None
    if pooled is not None:
        multi = (pooled.groupby("config").filter(lambda g: len(g) >= 3)
                 .sort_values("config"))
    else:
        multi = None
    if multi is not None and len(multi):
        configs = list(dict.fromkeys(multi["config"]))
        for i, c in enumerate(configs):
            s = multi[multi["config"] == c]
            ax1.scatter([i] * len(s), s["skill_vs_M0"] * 100, s=60, alpha=0.7)
            ax1.scatter([i], [s["skill_vs_M0"].mean() * 100], marker="_",
                        s=400, color="k")
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels([c.replace("_deep_", " ") for c in configs], rotation=15)
        ax1.set_title("Multi-seed skill vs M0 (%)  (— = mean)\n"
                      "source: deep_seed_pooled.csv")
    else:
        ax1.text(0.5, 0.5, "no pooled multi-seed table\nrun run_deep_multiseed.py "
                 "then pool_deep_seeds.py", ha="center", va="center",
                 fontsize=9, transform=ax1.transAxes)
        ax1.set_title("Multi-seed skill vs M0 (%)")
    ax1.axhline(0, color="grey", ls="--", lw=0.8)
    ax1.set_ylabel("skill vs M0 (%)"); ax1.grid(alpha=0.3, axis="y")

    hyp = summ[summ["group"] == "hyper"]
    for d in sorted(hyp["d"].unique()):
        h = hyp[(hyp["d"] == d) & (hyp["gat"] == 2)].sort_values("lookback")
        ax2.plot(h["lookback"], h["skill_vs_M0"] * 100, marker="o", label=f"d={d}")
    ax2.axhline(0, color="grey", ls="--", lw=0.8)
    ax2.set_title("Fusion skill vs M0 by lookback / d (seed=42)")
    ax2.set_xlabel("lookback (weeks)"); ax2.set_ylabel("skill vs M0 (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    rs_g = summ[summ["group"] == "rs"].reset_index(drop=True)
    if len(rs_g):
        labels, colors = [], []
        for _, r in rs_g.iterrows():
            is_def = (r["lr"] == 1e-3 and r["wd"] == 1e-4 and r["dropout"] == 0.1)
            if r["rs_kind"] == "cls":
                labels.append("cls\n(default)"); colors.append("tab:orange")
            elif is_def:
                labels.append("mp\n(default)"); colors.append("tab:green")
            else:
                labels.append(f"mp lr{r['lr']:.0e}\nwd{r['wd']:.0e} dp{r['dropout']}")
                colors.append("tab:blue")
        ax3.bar(range(len(rs_g)), rs_g["skill_vs_M0"] * 100, color=colors)
        ax3.axhline(0, color="grey", ls="--", lw=0.8)
        ax3.set_xticks(range(len(rs_g)))
        ax3.set_xticklabels(labels, fontsize=6)
        ax3.set_title("RS branch: meanpool vs cls + reg grid (seed=42)")
        ax3.set_ylabel("skill vs M0 (%)"); ax3.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
