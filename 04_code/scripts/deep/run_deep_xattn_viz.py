"""
RQ3 visualisation for the M4 Cross-Attention model: extract the per-week
cross-attention weights of the finance query over the 28 node/site tokens
(11 RS AOI + 17 shipping graph nodes) from the walk-forward m4xattn run, and
plot which nodes/lanes the financial state attends to over time and on average.

Mirrors deep_rolling's walk-forward (each test week uses its own fold model, no
look-ahead) and additionally records info["xattn_weights"].

Token order (from DeepForecastModel xattn kv concat, modalities [fin,rs,ship]):
  tokens[0:11]  = RS AOI    P001..P011
  tokens[11:28] = shipping  P001..P011 + hormuz/suez/malacca/mandeb/panama/cape

Outputs (-> 05_outputs/baselines/Deep/M4_Deep/):
  deep_xattn_weekly.csv    week x 28 token attention weights
  deep_xattn_viz.png       (a) finance->RS vs ->shipping attention over time
                           (b) mean token attention bar (RS vs shipping colored)
                           (c) week x token heatmap

Run:
  python3 04_code/scripts/deep/run_deep_xattn_viz.py --lookback 4 --epochs 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data                                # noqa: E402
from model_naming import deep_out_dir                    # noqa: E402
from models.deep_dataset import (apply_scalers, build_deep_dataset,  # noqa: E402
                                 fit_scalers)
from models.deep_rolling import CONFIGS, _to_tensors, _train_fold  # noqa: E402

OUT_DIR = deep_out_dir(data.ROOT, "M4")
EVENTS = [
    ("2022-02-24", "Russia–Ukraine"),
    ("2023-04-02", "OPEC+ cut"),
    ("2023-11-19", "Houthi Red Sea"),
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
    modalities, _, ftype = CONFIGS["m4_deep_xattn"]
    sites = ds["sites"]                 # 11 RS AOI
    node_ids = ds["node_ids"]           # 17 shipping nodes
    labels = [f"rs:{s}" for s in sites] + [f"sh:{n}" for n in node_ids]
    n_rs = len(sites)

    idx = ds["idx"]
    n = len(idx)
    model = None
    rows = []
    for i in range(n):
        if i < args.min_train:
            continue
        if model is None or ((i - args.min_train) % args.retrain_every == 0):
            sc = fit_scalers(ds, train_n=i)
            model, _, _ = _train_fold(
                ds, sc, i, modalities, args.seed, args.epochs, 1e-3, 1e-4, 32,
                52, "cpu", {"d": 32, "gat_layers": 2, "tcn_layers": 2}, ftype)
            print(f"  fit @ {idx[i].date()}", flush=True)
        Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), "cpu")
        model.eval()
        with torch.no_grad():
            _, info = model(aoi=Xte["aoi"], choke=Xte["choke"], adj=Xte["adj"],
                            fin=Xte["fin"], rs=Xte["rs"], rs_mask=Xte["rs_mask"])
        w = info["xattn_weights"][0].numpy()          # (28,)
        rows.append({"date": idx[i], **{labels[j]: w[j] for j in range(len(labels))}})

    xw = pd.DataFrame(rows).set_index("date")
    xw.to_csv(OUT_DIR / "deep_xattn_weekly.csv")

    rs_cols = labels[:n_rs]
    sh_cols = labels[n_rs:]
    rs_total = xw[rs_cols].sum(axis=1)
    sh_total = xw[sh_cols].sum(axis=1)
    mean_att = xw.mean(axis=0)

    print("\nMean finance->token attention (share of 28 tokens):")
    print(f"  -> RS total {rs_total.mean():.3f} | -> shipping total {sh_total.mean():.3f}")
    print("Top-8 attended tokens:")
    for name, v in mean_att.sort_values(ascending=False).head(8).items():
        print(f"    {name:14s} {v:.3f}")

    _plot(xw, rs_total, sh_total, mean_att, rs_cols, sh_cols,
          OUT_DIR / "deep_xattn_viz.png")
    print(f"Saved: {OUT_DIR/'deep_xattn_weekly.csv'}\n       {OUT_DIR/'deep_xattn_viz.png'}")


def _plot(xw, rs_total, sh_total, mean_att, rs_cols, sh_cols, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.2])

    # (a) modality-level cross-attention over time.
    ax0 = fig.add_subplot(gs[0])
    ax0.stackplot(xw.index, rs_total, sh_total,
                  labels=["→ RS (11 AOI)", "→ shipping (17 nodes)"],
                  colors=["tab:green", "tab:blue"], alpha=0.85)
    for d, lab in EVENTS:
        dt = pd.Timestamp(d)
        if xw.index.min() <= dt <= xw.index.max():
            ax0.axvline(dt, color="k", ls="--", lw=0.9)
            ax0.text(dt, 1.01, lab, rotation=25, fontsize=7, ha="left")
    ax0.set_ylim(0, 1); ax0.set_ylabel("attention share")
    ax0.set_title("Finance-query cross-attention: RS vs shipping over time (RQ3)")
    ax0.legend(loc="lower left", ncol=2, fontsize=8)

    # (b) mean attention per token (grouped colors).
    ax1 = fig.add_subplot(gs[1])
    order = mean_att.sort_values(ascending=False)
    colors = ["tab:green" if k in rs_cols else "tab:blue" for k in order.index]
    ax1.bar(range(len(order)), order.values, color=colors)
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(order.index, rotation=75, fontsize=6)
    ax1.set_title("Mean finance→token attention (green=RS AOI, blue=shipping node)")
    ax1.set_ylabel("mean weight")

    # (c) heatmap week x token (keep column order rs then shipping).
    ax2 = fig.add_subplot(gs[2])
    cols = rs_cols + sh_cols
    im = ax2.imshow(xw[cols].to_numpy().T, aspect="auto", cmap="magma",
                    extent=[0, len(xw), len(cols), 0])
    ax2.set_yticks(np.arange(len(cols)) + 0.5)
    ax2.set_yticklabels(cols, fontsize=5)
    ax2.axhline(len(rs_cols), color="w", lw=1.0)   # RS | shipping divider
    n_ticks = 6
    tick_pos = np.linspace(0, len(xw) - 1, n_ticks).astype(int)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([xw.index[t].date().isoformat() for t in tick_pos], fontsize=7)
    ax2.set_title("Cross-attention heatmap (week × token; white line = RS/shipping)")
    fig.colorbar(im, ax=ax2, fraction=0.02)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
