"""
RQ3 cross-attention diagnostics: extract the per-week attention weights of the
finance query over the node/site tokens from a walk-forward cross-attention run,
and plot which nodes/lanes the financial state attends to over time and on
average.

RQ3 is entered by any Deep specification with positive out-of-sample RMSE skill
against M0, so this is run for the S4 arm (finance + RS + shipping) and the S3
arm (finance + shipping). Remote-sensing site attention is only defined where the
RS modality is active, so the RS panel comes from the S4 arm.

Mirrors deep_rolling's walk-forward (each test week uses its own fold model, no
look-ahead), with the same architecture defaults as the reported forecasts, and
additionally records info["xattn_weights"].

Token order follows the kv concat in DeepForecastModel, which keeps the CONFIGS
modality order and includes only RS/shipping:
  modalities [fin, rs, ship] -> tokens[0:11] RS AOI, tokens[11:28] shipping nodes
  modalities [fin, ship]     -> tokens[0:17] shipping nodes
  modalities [fin, rs]       -> tokens[0:11] RS AOI

Outputs (-> 05_outputs/baselines/Deep/<tier>/):
  deep_xattn_weekly.csv    week x token attention weights  (S4; S3 -> deep_m3_*)
  deep_xattn_viz.png       (a) attention share over time
                           (b) mean token attention bar
                           (c) week x token heatmap

Run:
  python3 04_code/scripts/deep/run_deep_xattn_viz.py                     # S4
  python3 04_code/scripts/deep/run_deep_xattn_viz.py --config m3_deep_xattn
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
from model_naming import (DEEP_CONFIG_TIER, deep_out_dir,  # noqa: E402
                         resolve_deep_config)
from models.deep_dataset import (apply_scalers, build_deep_dataset,  # noqa: E402
                                 fit_scalers)
from models.deep_rolling import CONFIGS, _to_tensors, _train_fold  # noqa: E402

EVENTS = [
    ("2022-02-24", "Russia–Ukraine"),
    ("2023-04-02", "OPEC+ cut"),
    ("2023-11-19", "Houthi Red Sea"),
]
# M4 keeps the historical unprefixed names; other tiers mirror the gate files.
FILE_PREFIX = {"M4_Deep": "deep_xattn", "M3_Deep": "deep_m3_xattn",
               "M2_Deep": "deep_m2_xattn"}


def token_labels(ds: dict, modalities: list[str]) -> tuple[list[str], int]:
    """kv token labels in model order, plus the number of RS tokens."""
    labels: list[str] = []
    n_rs = 0
    for m in modalities:
        if m == "rs":
            labels += [f"rs:{s}" for s in ds["sites"]]
            n_rs = len(ds["sites"])
        elif m == "ship":
            labels += [f"sh:{n}" for n in ds["node_ids"]]
    return labels, n_rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="m4_deep_xattn",
                    help="cross-attention config, e.g. m3_deep_xattn")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = resolve_deep_config(args.config)
    modalities, model_name, ftype = CONFIGS[cfg]
    if ftype != "xattn":
        raise SystemExit(f"{cfg} uses {ftype} fusion; this script needs xattn.")
    tier = DEEP_CONFIG_TIER[cfg]
    out_dir = deep_out_dir(data.ROOT, tier.split("_")[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = FILE_PREFIX[tier]

    df = data.load_matrix(); dico = data.load_dict()
    ds = build_deep_dataset(df, dico, lookback=args.lookback)
    labels, n_rs = token_labels(ds, modalities)
    print(f"{model_name}: {len(labels)} kv tokens "
          f"({n_rs} RS AOI + {len(labels)-n_rs} shipping nodes)")

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
        kw = dict(aoi=Xte["aoi"], choke=Xte["choke"], adj=Xte["adj"],
                  fin=Xte["fin"])
        if "rs" in Xte:
            kw.update(rs=Xte["rs"], rs_mask=Xte["rs_mask"])
        model.eval()
        with torch.no_grad():
            _, info = model(**kw)
        w = info["xattn_weights"][0].numpy()
        rows.append({"date": idx[i], **{labels[j]: w[j] for j in range(len(labels))}})

    xw = pd.DataFrame(rows).set_index("date")
    csv_path = out_dir / f"{prefix}_weekly.csv"
    png_path = out_dir / f"{prefix}_viz.png"
    xw.to_csv(csv_path)

    rs_cols = labels[:n_rs]
    sh_cols = labels[n_rs:]
    mean_att = xw.mean(axis=0)

    print(f"\nMean finance->token attention (share of {len(labels)} tokens):")
    if rs_cols and sh_cols:
        print(f"  -> RS total {xw[rs_cols].sum(axis=1).mean():.3f} | "
              f"-> shipping total {xw[sh_cols].sum(axis=1).mean():.3f}")
    print("Top-8 attended tokens:")
    for name, v in mean_att.sort_values(ascending=False).head(8).items():
        print(f"    {name:14s} {v:.3f}")

    _plot(xw, mean_att, rs_cols, sh_cols, model_name, png_path)
    print(f"Saved: {csv_path}\n       {png_path}")


def _plot(xw, mean_att, rs_cols, sh_cols, model_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.2])

    # (a) attention share over time. With both modalities present the split is
    # the informative quantity; with one, the top tokens are.
    ax0 = fig.add_subplot(gs[0])
    if rs_cols and sh_cols:
        ax0.stackplot(xw.index, xw[rs_cols].sum(axis=1), xw[sh_cols].sum(axis=1),
                      labels=[f"→ RS ({len(rs_cols)} AOI)",
                              f"→ shipping ({len(sh_cols)} nodes)"],
                      colors=["tab:green", "tab:blue"], alpha=0.85)
        ax0.set_title(f"{model_name}: finance-query cross-attention, "
                      f"RS vs shipping over time (RQ3)")
    else:
        top = mean_att.sort_values(ascending=False).head(6).index.tolist()
        rest = [c for c in xw.columns if c not in top]
        ax0.stackplot(xw.index, *[xw[c] for c in top], xw[rest].sum(axis=1),
                      labels=top + [f"other ({len(rest)})"], alpha=0.85)
        ax0.set_title(f"{model_name}: finance-query cross-attention, "
                      f"top-6 tokens over time (RQ3)")
    for d, lab in EVENTS:
        dt = pd.Timestamp(d)
        if xw.index.min() <= dt <= xw.index.max():
            ax0.axvline(dt, color="k", ls="--", lw=0.9)
            ax0.text(dt, 1.01, lab, rotation=25, fontsize=7, ha="left")
    ax0.set_ylim(0, 1); ax0.set_ylabel("attention share")
    ax0.legend(loc="lower left", ncol=4, fontsize=7)

    # (b) mean attention per token (grouped colors).
    ax1 = fig.add_subplot(gs[1])
    order = mean_att.sort_values(ascending=False)
    colors = ["tab:green" if k in rs_cols else "tab:blue" for k in order.index]
    ax1.bar(range(len(order)), order.values, color=colors)
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(order.index, rotation=75, fontsize=6)
    ax1.set_title(f"Mean finance→token attention, {len(mean_att)} tokens "
                  f"(green=RS AOI, blue=shipping node)")
    ax1.set_ylabel("mean weight")

    # (c) heatmap week x token (keep column order rs then shipping).
    ax2 = fig.add_subplot(gs[2])
    cols = rs_cols + sh_cols
    im = ax2.imshow(xw[cols].to_numpy().T, aspect="auto", cmap="magma",
                    extent=[0, len(xw), len(cols), 0])
    ax2.set_yticks(np.arange(len(cols)) + 0.5)
    ax2.set_yticklabels(cols, fontsize=5)
    if rs_cols and sh_cols:
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
