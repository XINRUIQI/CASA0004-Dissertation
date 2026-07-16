"""
RQ3 interpretability for M3_Deep_gated (finance + shipping) — the deep
specification that clears M0 under the locked protocol.

Extracts per-week modality gate weights [finance, shipping] and mean
shipping-node attention (17 nodes) from walk-forward runs.

Outputs (-> 05_outputs/baselines/Deep/M3_Deep/):
  deep_m3_gate_weekly.csv
  deep_m3_gate_weekly_seed{S}.csv   (if --seeds)
  deep_m3_interpret.png             gate stackplot + shipping attention
  deep_m3_interpret_stability.png   (if multiple seeds)
  deep_m3_gate_stability.csv / corr / events / band_weekly  (multi-seed)

Run:
  python3 04_code/scripts/deep/run_deep_interpret_m3.py --lookback 4 --epochs 80
  python3 04_code/scripts/deep/run_deep_interpret_m3.py --seeds 42,1,2 --lookback 4
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

OUT_DIR = deep_out_dir(data.ROOT, "M3")
CONFIG_KEY = "m3_deep_gated"
EVENTS = [
    ("2022-02-24", "Russia–Ukraine"),
    ("2022-06-01", "EU RU oil ban announced"),
    ("2023-04-02", "OPEC+ surprise cut"),
    ("2023-11-19", "Houthi Red Sea attacks"),
]
EVENT_WINDOW_WEEKS = 8
TOP_K = 5


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return [args.seed]


def _run_one_seed(ds: dict, seed: int, args: argparse.Namespace
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    modalities, _, fusion_type = CONFIGS[CONFIG_KEY]
    idx = ds["idx"]
    n = len(idx)
    model = None
    sc = None
    gate_rows = []
    ship_att_acc = np.zeros(17, dtype=float)
    n_acc = 0

    print(f"\n=== M3_Deep_gated seed={seed} lb={args.lookback} "
          f"epochs={args.epochs} ===", flush=True)
    for i in range(n):
        if i < args.min_train:
            continue
        if model is None or ((i - args.min_train) % args.retrain_every == 0):
            sc = fit_scalers(ds, train_n=i)
            model, _, _ = _train_fold(
                ds, sc, i, modalities, seed, args.epochs, 1e-3, 1e-4, 32,
                52, "cpu", {"d": 32, "gat_layers": 2, "tcn_layers": 2},
                fusion_type=fusion_type)
            print(f"  fit @ {idx[i].date()}", flush=True)
        Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), "cpu")
        model.eval()
        with torch.no_grad():
            kw = dict(aoi=Xte["aoi"], choke=Xte["choke"], adj=Xte["adj"],
                      fin=Xte["fin"])
            _, info = model(**kw)
        order = info.get("gate_order", modalities)
        g = info["gate"][0].tolist()
        gmap = {m: g[j] for j, m in enumerate(order)}
        gate_rows.append({
            "date": idx[i],
            "gate_finance": float(gmap["fin"]),
            "gate_shipping": float(gmap["ship"]),
        })
        ship_att_acc += info["ship_site_att"][0].numpy()
        n_acc += 1

    gate = pd.DataFrame(gate_rows).set_index("date")
    ship_att = ship_att_acc / max(n_acc, 1)
    return gate, ship_att


def _top_names(names, weights: np.ndarray, k: int = TOP_K) -> list[str]:
    order = np.argsort(-weights)
    return [names[i] for i in order[:k]]


def _event_delta(gate: pd.DataFrame, event_date: pd.Timestamp,
                 weeks: int = EVENT_WINDOW_WEEKS) -> float:
    pre = gate.loc[
        (gate.index >= event_date - pd.Timedelta(weeks=weeks))
        & (gate.index < event_date), "gate_shipping"]
    post = gate.loc[
        (gate.index >= event_date)
        & (gate.index < event_date + pd.Timedelta(weeks=weeks)),
        "gate_shipping"]
    if len(pre) < 2 or len(post) < 2:
        return float("nan")
    return float(post.mean() - pre.mean())


def _plot_single(gate: pd.DataFrame, ship_att: np.ndarray,
                 node_ids: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.stackplot(gate.index, gate["gate_finance"], gate["gate_shipping"],
                  labels=["finance", "shipping"],
                  colors=["#1f77b4", "#2ca02c"], alpha=0.85)
    for d, lab in EVENTS:
        dt = pd.Timestamp(d)
        if gate.index.min() <= dt <= gate.index.max():
            ax0.axvline(dt, color="k", ls="--", lw=0.9)
            ax0.text(dt, 1.01, lab, rotation=30, fontsize=7, ha="left")
    ax0.set_ylim(0, 1)
    ax0.set_title("M3_Deep_gated modality gate weights over time")
    ax0.set_ylabel("gate weight")
    ax0.legend(loc="lower left", ncol=2, fontsize=9)

    ax1 = fig.add_subplot(gs[1, 0])
    order = np.argsort(-ship_att)
    ax1.bar(range(len(node_ids)), ship_att[order], color="#1f77b4", alpha=0.9)
    ax1.set_xticks(range(len(node_ids)))
    ax1.set_xticklabels([node_ids[i] for i in order], rotation=60, fontsize=7)
    ax1.set_title("Shipping node attention (mean over scored weeks, 17 nodes)")
    ax1.set_ylabel("attention weight")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _aggregate_and_plot_stability(
        seeds: list[int],
        gates: dict[int, pd.DataFrame],
        ship_atts: dict[int, np.ndarray],
        node_ids: list[str],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Top-k frequency
    rows = []
    mat = np.stack([ship_atts[s] for s in seeds], axis=0)
    mean_w, std_w = mat.mean(0), mat.std(0, ddof=0)
    ranks = np.argsort(np.argsort(-mat, axis=1), axis=1) + 1
    top_sets = [_top_names(node_ids, ship_atts[s], TOP_K) for s in seeds]
    for j, name in enumerate(node_ids):
        freq = sum(1 for tops in top_sets if name in tops)
        rows.append({
            "name": name,
            "att_mean": float(mean_w[j]),
            "att_std": float(std_w[j]),
            "rank_mean": float(ranks[:, j].mean()),
            f"freq_top{TOP_K}": int(freq),
            "n_seeds": len(seeds),
        })
    stability = pd.DataFrame(rows).sort_values(
        [f"freq_top{TOP_K}", "att_mean"], ascending=[False, False])
    stability.to_csv(OUT_DIR / "deep_m3_gate_stability.csv", index=False)

    # correlations
    common = gates[seeds[0]].index
    for s in seeds[1:]:
        common = common.intersection(gates[s].index)
    corr_rows = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            a = gates[s1].loc[common, "gate_shipping"]
            b = gates[s2].loc[common, "gate_shipping"]
            corr_rows.append({
                "seed_a": s1, "seed_b": s2, "n_weeks": len(common),
                "pearson_alpha_shipping": float(a.corr(b, method="pearson")),
                "spearman_alpha_shipping": float(a.corr(b, method="spearman")),
            })
    for s in seeds:
        g = gates[s]
        corr_rows.append({
            "seed_a": s, "seed_b": s, "n_weeks": len(g),
            "pearson_alpha_shipping": 1.0,
            "spearman_alpha_shipping": 1.0,
            "mean_gate_finance": float(g["gate_finance"].mean()),
            "mean_gate_shipping": float(g["gate_shipping"].mean()),
            "std_gate_shipping": float(g["gate_shipping"].std()),
        })
    pd.DataFrame(corr_rows).to_csv(OUT_DIR / "deep_m3_gate_corr.csv",
                                   index=False)

    # events
    ev_rows = []
    for d, lab in EVENTS:
        dt = pd.Timestamp(d)
        deltas = {s: _event_delta(gates[s], dt) for s in seeds}
        signs = [np.sign(v) for v in deltas.values() if np.isfinite(v)]
        n_up = sum(1 for sg in signs if sg > 0)
        n_down = sum(1 for sg in signs if sg < 0)
        n_ok = len(signs)
        same_dir = (n_up == n_ok) or (n_down == n_ok) if n_ok else False
        row = {"event_date": d, "event": lab, "n_seeds_valid": n_ok,
               "n_up": n_up, "n_down": n_down, "same_direction": bool(same_dir)}
        for s in seeds:
            row[f"delta_seed{s}"] = deltas[s]
        ev_rows.append(row)
    pd.DataFrame(ev_rows).to_csv(OUT_DIR / "deep_m3_gate_events.csv",
                                 index=False)

    # band
    band = pd.DataFrame(index=common)
    for col in ("gate_finance", "gate_shipping"):
        stack = np.column_stack([gates[s].loc[common, col].to_numpy()
                                 for s in seeds])
        band[f"{col}_mean"] = stack.mean(1)
        band[f"{col}_std"] = stack.std(1, ddof=0)
        band[f"{col}_lo"] = band[f"{col}_mean"] - band[f"{col}_std"]
        band[f"{col}_hi"] = band[f"{col}_mean"] + band[f"{col}_std"]
    band.to_csv(OUT_DIR / "deep_m3_gate_band_weekly.csv")

    # plot
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    for col, label, color in (("gate_finance", "finance", "#1f77b4"),
                              ("gate_shipping", "shipping", "#2ca02c")):
        m, lo, hi = band[f"{col}_mean"], band[f"{col}_lo"], band[f"{col}_hi"]
        ax0.plot(band.index, m, color=color, lw=1.4, label=label)
        ax0.fill_between(band.index, lo, hi, color=color, alpha=0.22)
    for d, lab in EVENTS:
        dt = pd.Timestamp(d)
        if band.index.min() <= dt <= band.index.max():
            ax0.axvline(dt, color="k", ls="--", lw=0.9)
            ax0.text(dt, 0.98, lab, rotation=30, fontsize=7, ha="left",
                     va="top")
    ax0.set_ylim(0, 1)
    ax0.set_title(f"M3_Deep_gated gates: mean ± cross-seed std (seeds={seeds})")
    ax0.set_ylabel("gate weight")
    ax0.legend(loc="upper right", ncol=2, fontsize=9)

    ax1 = fig.add_subplot(gs[1, 0])
    names = stability["name"].tolist()
    freqs = stability[f"freq_top{TOP_K}"].tolist()
    colors = ["#d62728" if "hormuz" in n.lower() else "#1f77b4" for n in names]
    ax1.bar(range(len(names)), freqs, color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=60, fontsize=7)
    ax1.set_ylim(0, len(seeds) + 0.3)
    ax1.set_ylabel(f"freq in Top-{TOP_K} / {len(seeds)}")
    ax1.set_title(f"Shipping Top-{TOP_K} frequency across seeds")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "deep_m3_interpret_stability.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interpretability for M3_Deep_gated (finance+shipping)")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="",
                    help="comma-separated seeds, e.g. 42,1,2")
    args = ap.parse_args()
    seeds = _parse_seeds(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = data.load_matrix()
    dico = data.load_dict()
    ds = build_deep_dataset(df, dico, lookback=args.lookback)
    node_ids = list(ds["node_ids"])

    gates: dict[int, pd.DataFrame] = {}
    ship_atts: dict[int, np.ndarray] = {}

    for seed in seeds:
        gate, ship_att = _run_one_seed(ds, seed, args)
        gates[seed] = gate
        ship_atts[seed] = ship_att
        gate.to_csv(OUT_DIR / f"deep_m3_gate_weekly_seed{seed}.csv")
        print(f"\n[seed={seed}] Mean gates:",
              gate[["gate_finance", "gate_shipping"]].mean().round(3).to_dict())
        print(f"[seed={seed}] Top ship nodes:",
              _top_names(node_ids, ship_att))

    primary = seeds[0]
    gates[primary].to_csv(OUT_DIR / "deep_m3_gate_weekly.csv")
    out_png = OUT_DIR / "deep_m3_interpret.png"
    _plot_single(gates[primary], ship_atts[primary], node_ids, out_png)
    print(f"\nSaved: {OUT_DIR / 'deep_m3_gate_weekly.csv'}\n"
          f"       {out_png}")

    if len(seeds) > 1:
        _aggregate_and_plot_stability(seeds, gates, ship_atts, node_ids)
        print(f"       {OUT_DIR / 'deep_m3_interpret_stability.png'}")


if __name__ == "__main__":
    main()
