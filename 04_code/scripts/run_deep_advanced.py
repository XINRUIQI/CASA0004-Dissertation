"""
Advanced deep-fusion ablations (RQ2/RQ3 robustness extensions):
  1. Fusion type   : encoder-concat vs gated vs cross-attention (finance-as-query)
                     on M4-rep (the RQ2 fusion ladder).
  2. Modality dropout: on/off (ModDrop-style missing-modality regularisation).
  3. Sub-period     : split the common test into early (2021-2022) vs late
                      (2023-2025) and re-check skill / CW-vs-M0 stability.
  4. Longer window  : min_train=78 (test starts ~2020-08) skill vs M0 on the
                      model's own longer test span.

Flat M1 (Ridge/XGB) read from baseline_predictions.csv (no xgboost import).

Outputs (-> 05_outputs/baselines/deep/):
  deep_advanced_summary.csv   one row per arm (+ subperiod rows)
  deep_advanced.png           skill-by-arm + subperiod bars

Run:
  python3 04_code/scripts/run_deep_advanced.py --epochs 80
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
SPLIT = pd.Timestamp("2023-01-01")


def _skill_cw(res: pd.DataFrame, res_m1: pd.DataFrame, col: str,
              idx: pd.Index) -> dict:
    """skill vs M0 + DirAcc + the VALID tests on weeks `idx`: Clark-West vs M0
    (nested -> beats the random walk?) and Diebold-Mariano vs flat M1_Ridge
    (non-nested model class, where Clark-West would be invalid)."""
    common = res.index.intersection(idx)
    y = res.loc[common, "P_next_actual"].to_numpy()
    ym0 = res.loc[common, "P_hat_M0"].to_numpy()
    e_m0 = ym0 - y
    rmse_m0 = float(np.sqrt(np.mean(e_m0 ** 2)))
    yhat = res.loc[common, f"P_hat_{col}"].to_numpy()
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    rhat = res.loc[common, f"r_hat_{col}"].to_numpy()
    ract = res.loc[common, "r_actual"].to_numpy()
    _, cw0 = metrics.clark_west(y, ym0, yhat)          # valid nested: vs M0 (RW)
    out = {"n_test": len(common), "RMSE": rmse,
           "skill_vs_M0": 1 - rmse / rmse_m0,
           "DirAcc": metrics.directional_acc(rhat, ract),
           "CW_p_vs_M0": cw0}
    m1c = res_m1.index.intersection(common)
    if len(m1c) > 30:
        yv = res_m1.loc[m1c, "P_next_actual"].to_numpy()
        _, dmp = metrics.dm_test(                       # valid non-nested: vs M1
            res.loc[m1c, f"P_hat_{col}"].to_numpy() - yv,
            res_m1.loc[m1c, "P_hat_M1_Ridge"].to_numpy() - yv)
        out["DM_p_vs_M1"] = dmp
    else:
        out["DM_p_vs_M1"] = np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    df = data.load_matrix(); dico = data.load_dict()
    res_m1 = pd.read_csv(data.ROOT / "05_outputs/baselines/m1/baseline_predictions.csv",
                         index_col=0, parse_dates=True)
    res_m1.index.name = "date"
    dds = build_deep_dataset(df, dico, lookback=4)
    print(f"deep dataset N={len(dds['idx'])}\n")

    # arm: (name, config, model_kwargs, min_train)
    arms = [
        ("m4concat", "m4concat", {}, 104),     # encoder-concat rung (RQ2 ladder floor)
        ("m4rep_gated", "m4rep", {}, 104),
        ("m4xattn", "m4xattn", {}, 104),
        ("m4rep_drop0.3", "m4rep", {"modality_dropout": 0.3}, 104),
        ("m4xattn_drop0.3", "m4xattn", {"modality_dropout": 0.3}, 104),
        ("m4rep_longwin(mt78)", "m4rep", {}, 78),
    ]
    t0 = time.time()
    rows, res_store = [], {}
    for name, cfg, mk, mt in arms:
        col = f"{CONFIGS[cfg][1]}"           # model name; label defaults below
        res = rolling_origin_deep(dds, "A", cfg, min_train=mt, seed=42,
                                  epochs=args.epochs, model_kwargs=mk, verbose=False)
        colname = f"A_{CONFIGS[cfg][1]}"
        res_store[name] = (res, colname)
        full = _skill_cw(res, res_m1, colname, res.index)
        rows.append({"arm": name, "config": cfg, "fusion": CONFIGS[cfg][2],
                     "modality_dropout": mk.get("modality_dropout", 0.0),
                     "min_train": mt, "period": "full", **full})
        print(f"  {name:22s} skill={full['skill_vs_M0']*100:+.2f}% "
              f"DirAcc={full['DirAcc']:.3f} CWvsM0={full['CW_p_vs_M0']:.4f} "
              f"DMvsM1={full['DM_p_vs_M1']:.4f} n={full['n_test']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # sub-period stability for the two standard-window fusion arms
    for name in ("m4rep_gated", "m4xattn"):
        res, colname = res_store[name]
        for lab, idx in [("early(<=2022)", res.index[res.index < SPLIT]),
                         ("late(>=2023)", res.index[res.index >= SPLIT])]:
            s = _skill_cw(res, res_m1, colname, idx)
            rows.append({"arm": name, "config": name, "fusion": "",
                         "modality_dropout": 0.0, "min_train": 104,
                         "period": lab, **s})
            print(f"  {name:14s} {lab:14s} skill={s['skill_vs_M0']*100:+.2f}% "
                  f"CWvsM0={s['CW_p_vs_M0']:.4f} n={s['n_test']}")

    summ = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT_DIR / "deep_advanced_summary.csv", index=False)

    # DM: cross-attn vs gated (same weeks, non-nested).
    rg, cg = res_store["m4rep_gated"]
    rx, cx = res_store["m4xattn"]
    cm = rg.index.intersection(rx.index)
    y = rg.loc[cm, "P_next_actual"].to_numpy()
    dm, dmp = metrics.dm_test(rx.loc[cm, f"P_hat_{cx}"].to_numpy() - y,
                              rg.loc[cm, f"P_hat_{cg}"].to_numpy() - y)
    print(f"\nDM cross-attn vs gated: stat={dm:.3f} p={dmp:.3f} "
          f"(<0.5 => xattn more accurate)")

    _plot(summ, OUT_DIR / "deep_advanced.png")
    print(f"\nElapsed {time.time()-t0:.0f}s\n"
          f"Saved: {OUT_DIR/'deep_advanced_summary.csv'}\n"
          f"       {OUT_DIR/'deep_advanced.png'}")


def _plot(summ: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = summ[summ["period"] == "full"]
    sub = summ[summ["period"] != "full"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(len(full)), full["skill_vs_M0"] * 100)
    ax1.axhline(0, color="grey", ls="--", lw=0.8)
    ax1.set_xticks(range(len(full)))
    ax1.set_xticklabels(full["arm"], rotation=30, ha="right", fontsize=7)
    ax1.set_title("Skill vs M0 by arm (full test)"); ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylabel("skill vs M0 (%)")
    if len(sub):
        arms = sub["arm"].unique()
        periods = sub["period"].unique()
        w = 0.35
        for j, p in enumerate(periods):
            vals = [sub[(sub["arm"] == a) & (sub["period"] == p)]["skill_vs_M0"].values[0] * 100
                    for a in arms]
            ax2.bar(np.arange(len(arms)) + j * w, vals, width=w, label=p)
        ax2.axhline(0, color="grey", ls="--", lw=0.8)
        ax2.set_xticks(np.arange(len(arms)) + w / 2)
        ax2.set_xticklabels(arms, fontsize=8)
        ax2.set_title("Sub-period skill vs M0"); ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
