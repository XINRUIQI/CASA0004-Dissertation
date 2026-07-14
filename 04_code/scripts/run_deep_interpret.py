"""
RQ3 interpretability for the deep M4-representation model: extract the per-week
modality GATE weights [finance, rs, shipping] and the node-level site-attention
(shipping 17-node graph + rs 11-AOI) from the walk-forward m4rep run, and plot
them against known supply/geopolitical events.

Mirrors deep_rolling's walk-forward (each test week uses its own fold model, no
look-ahead) but additionally records the fusion info dict.

Outputs (-> 05_outputs/baselines/deep/):
  deep_gate_weekly.csv     week, gate_finance/rs/shipping (+ optional site atts)
  deep_interpret.png       gate time series (stack) + event lines + site-att bars

Run:
  python3 04_code/scripts/run_deep_interpret.py --lookback 8 --epochs 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data                                # noqa: E402
from models.deep_dataset import (apply_scalers, build_deep_dataset,  # noqa: E402
                                 fit_scalers)
from models.deep_rolling import CONFIGS, _to_tensors, _train_fold  # noqa: E402

OUT_DIR = data.ROOT / "05_outputs/baselines/deep"
# (date, short label) — known supply / geopolitical markers within 2021-2025.
EVENTS = [
    ("2022-02-24", "Russia–Ukraine"),
    ("2022-06-01", "EU RU oil ban announced"),
    ("2023-04-02", "OPEC+ surprise cut"),
    ("2023-11-19", "Houthi Red Sea attacks"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = data.load_matrix(); dico = data.load_dict()
    ds = build_deep_dataset(df, dico, lookback=args.lookback)
    modalities, _, _ = CONFIGS["m4rep"]
    node_ids = ds["node_ids"]
    sites = ds["sites"]
    n_aoi = len(sites)

    idx = ds["idx"]
    n = len(idx)
    model = None
    r_mean = r_std = 0.0
    gate_rows, ship_att_acc, rs_att_acc, n_acc = [], np.zeros(17), np.zeros(n_aoi), 0
    for i in range(n):
        if i < args.min_train:
            continue
        if model is None or ((i - args.min_train) % args.retrain_every == 0):
            sc = fit_scalers(ds, train_n=i)
            model, r_mean, r_std = _train_fold(
                ds, sc, i, modalities, args.seed, args.epochs, 1e-3, 1e-4, 32,
                52, "cpu", {"d": 32, "gat_layers": 2, "tcn_layers": 2})
            print(f"  fit @ {idx[i].date()}", flush=True)
        Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), "cpu")
        model.eval()
        with torch.no_grad():
            kw = dict(aoi=Xte["aoi"], choke=Xte["choke"], adj=Xte["adj"],
                      fin=Xte["fin"], rs=Xte["rs"], rs_mask=Xte["rs_mask"])
            _, info = model(**kw)
        g = info["gate"][0].tolist()                     # [fin, rs, ship]
        gate_rows.append({"date": idx[i], "gate_finance": g[0],
                          "gate_rs": g[1], "gate_shipping": g[2]})
        ship_att_acc += info["ship_site_att"][0].numpy()
        rs_att_acc += info["rs_site_att"][0].numpy()
        n_acc += 1

    gate = pd.DataFrame(gate_rows).set_index("date")
    gate.to_csv(OUT_DIR / "deep_gate_weekly.csv")
    ship_att = ship_att_acc / n_acc
    rs_att = rs_att_acc / n_acc

    print("\nMean modality gate:", gate[["gate_finance", "gate_rs",
          "gate_shipping"]].mean().round(3).to_dict())
    print("Top ship nodes:", sorted(zip(node_ids, ship_att.round(3)),
          key=lambda x: -x[1])[:5])
    print("Top rs AOIs:", sorted(zip(sites, rs_att.round(3)),
          key=lambda x: -x[1])[:5])

    _plot(gate, ship_att, rs_att, node_ids, sites, OUT_DIR / "deep_interpret.png")
    print(f"Saved: {OUT_DIR/'deep_gate_weekly.csv'}\n"
          f"       {OUT_DIR/'deep_interpret.png'}")


def _plot(gate, ship_att, rs_att, node_ids, sites, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.stackplot(gate.index, gate["gate_finance"], gate["gate_rs"],
                  gate["gate_shipping"],
                  labels=["finance", "rs", "shipping"], alpha=0.85)
    for d, lab in EVENTS:
        dt = pd.Timestamp(d)
        if gate.index.min() <= dt <= gate.index.max():
            ax0.axvline(dt, color="k", ls="--", lw=0.9)
            ax0.text(dt, 1.01, lab, rotation=30, fontsize=7, ha="left")
    ax0.set_ylim(0, 1); ax0.set_title("M4-rep modality gate weights over time (RQ3)")
    ax0.set_ylabel("gate weight"); ax0.legend(loc="lower left", ncol=3, fontsize=8)

    ax1 = fig.add_subplot(gs[1, 0])
    order = np.argsort(-ship_att)
    ax1.bar(range(len(node_ids)), ship_att[order])
    ax1.set_xticks(range(len(node_ids)))
    ax1.set_xticklabels([node_ids[i] for i in order], rotation=60, fontsize=6)
    ax1.set_title("Shipping site-attention (mean, 17 nodes)")

    ax2 = fig.add_subplot(gs[1, 1])
    order2 = np.argsort(-rs_att)
    ax2.bar(range(len(sites)), rs_att[order2], color="tab:green")
    ax2.set_xticks(range(len(sites)))
    ax2.set_xticklabels([sites[i] for i in order2], rotation=60, fontsize=6)
    ax2.set_title("RS site-attention (mean, 11 AOI)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
