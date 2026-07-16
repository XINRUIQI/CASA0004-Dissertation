"""
RQ3 interpretability for the deep M4-gated representation model: extract the
per-week modality GATE weights [finance, rs, shipping] and node-/site-attention
(shipping 17-node graph + rs 11-AOI) from walk-forward runs.

Supports multi-seed stability (§3.4 of 2026-07-15 progress overview):
  --seeds 42,1,2   run each seed, then aggregate mean±band, correlations,
                   event-window co-movement, and Top-5 frequency.

Outputs (-> 05_outputs/baselines/Deep/M4_Deep/):
  deep_gate_weekly.csv              seed=primary (first seed) weekly gates
  deep_gate_weekly_seed{S}.csv      per-seed weekly gates
  deep_gate_stability.csv           ship/RS Top-k freq + mean attention ranks
  deep_gate_corr.csv                pairwise seed correlations of α_shipping
  deep_gate_events.csv              event-window Δα_shipping by seed
  deep_interpret.png                single-seed stackplot (primary seed)
  deep_interpret_stability.png      mean±band gates + Top-k freq bars

Run:
  python3 04_code/scripts/deep/run_deep_interpret.py --lookback 4 --epochs 80
  python3 04_code/scripts/deep/run_deep_interpret.py --seeds 42,1,2 --lookback 4 --epochs 80
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
# (date, short label) — known supply / geopolitical markers within 2021-2025.
EVENTS = [
    ("2022-02-24", "Russia–Ukraine"),
    ("2022-06-01", "EU RU oil ban announced"),
    ("2023-04-02", "OPEC+ surprise cut"),
    ("2023-11-19", "Houthi Red Sea attacks"),
]
# ± weeks around each event for co-movement check
EVENT_WINDOW_WEEKS = 8
TOP_K = 5


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return [args.seed]


def _run_one_seed(ds: dict, seed: int, args: argparse.Namespace
                  ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Walk-forward gate + mean site attentions for one seed."""
    modalities, _, _ = CONFIGS["m4_deep_gated"]
    n_aoi = len(ds["sites"])
    idx = ds["idx"]
    n = len(idx)
    model = None
    sc = None
    gate_rows = []
    ship_att_acc = np.zeros(17, dtype=float)
    rs_att_acc = np.zeros(n_aoi, dtype=float)
    n_acc = 0

    print(f"\n=== seed={seed} lookback={args.lookback} epochs={args.epochs} ===",
          flush=True)
    for i in range(n):
        if i < args.min_train:
            continue
        if model is None or ((i - args.min_train) % args.retrain_every == 0):
            sc = fit_scalers(ds, train_n=i)
            model, _, _ = _train_fold(
                ds, sc, i, modalities, seed, args.epochs, 1e-3, 1e-4, 32,
                52, "cpu", {"d": 32, "gat_layers": 2, "tcn_layers": 2})
            print(f"  fit @ {idx[i].date()}", flush=True)
        Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), "cpu")
        model.eval()
        with torch.no_grad():
            kw = dict(aoi=Xte["aoi"], choke=Xte["choke"], adj=Xte["adj"],
                      fin=Xte["fin"], rs=Xte["rs"], rs_mask=Xte["rs_mask"])
            _, info = model(**kw)
        g = info["gate"][0].tolist()  # [fin, rs, ship]
        gate_rows.append({"date": idx[i], "gate_finance": g[0],
                          "gate_rs": g[1], "gate_shipping": g[2]})
        ship_att_acc += info["ship_site_att"][0].numpy()
        rs_att_acc += info["rs_site_att"][0].numpy()
        n_acc += 1

    gate = pd.DataFrame(gate_rows).set_index("date")
    ship_att = ship_att_acc / max(n_acc, 1)
    rs_att = rs_att_acc / max(n_acc, 1)
    return gate, ship_att, rs_att


def _top_names(names, weights: np.ndarray, k: int = TOP_K) -> list[str]:
    order = np.argsort(-weights)
    return [names[i] for i in order[:k]]


def _event_delta(gate: pd.DataFrame, event_date: pd.Timestamp,
                 weeks: int = EVENT_WINDOW_WEEKS) -> float:
    """Mean α_shipping in [event, event+weeks) minus mean in [event-weeks, event)."""
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


def _aggregate_stability(
        seeds: list[int],
        gates: dict[int, pd.DataFrame],
        ship_atts: dict[int, np.ndarray],
        rs_atts: dict[int, np.ndarray],
        node_ids: list[str],
        sites: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (stability_df, corr_df, events_df, band_df)."""
    # --- Top-k frequency + mean/std attention ---
    rows = []
    for kind, names, atts in (
            ("ship", node_ids, ship_atts),
            ("rs", sites, rs_atts),
    ):
        # stack attentions: (n_seed, n_nodes)
        mat = np.stack([atts[s] for s in seeds], axis=0)
        mean_w = mat.mean(0)
        std_w = mat.std(0, ddof=0)
        # rank per seed (1 = highest)
        ranks = np.argsort(np.argsort(-mat, axis=1), axis=1) + 1
        top_sets = [_top_names(names, atts[s], TOP_K) for s in seeds]
        for j, name in enumerate(names):
            freq = sum(1 for tops in top_sets if name in tops)
            rows.append({
                "kind": kind,
                "name": name,
                "att_mean": float(mean_w[j]),
                "att_std": float(std_w[j]),
                "rank_mean": float(ranks[:, j].mean()),
                "rank_std": float(ranks[:, j].std(ddof=0)),
                f"freq_top{TOP_K}": int(freq),
                "n_seeds": len(seeds),
            })
    stability = pd.DataFrame(rows).sort_values(
        ["kind", f"freq_top{TOP_K}", "att_mean"], ascending=[True, False, False])

    # --- pairwise α_shipping correlations ---
    ship_series = {s: gates[s]["gate_shipping"] for s in seeds}
    common = ship_series[seeds[0]].index
    for s in seeds[1:]:
        common = common.intersection(ship_series[s].index)
    corr_rows = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            a = ship_series[s1].loc[common]
            b = ship_series[s2].loc[common]
            corr_rows.append({
                "seed_a": s1, "seed_b": s2, "n_weeks": len(common),
                "pearson_alpha_shipping": float(a.corr(b, method="pearson")),
                "spearman_alpha_shipping": float(a.corr(b, method="spearman")),
            })
    # also mean gate stats across seeds
    for s in seeds:
        g = gates[s]
        corr_rows.append({
            "seed_a": s, "seed_b": s, "n_weeks": len(g),
            "pearson_alpha_shipping": 1.0,
            "spearman_alpha_shipping": 1.0,
            "mean_gate_finance": float(g["gate_finance"].mean()),
            "mean_gate_rs": float(g["gate_rs"].mean()),
            "mean_gate_shipping": float(g["gate_shipping"].mean()),
            "std_gate_shipping": float(g["gate_shipping"].std()),
        })
    corr = pd.DataFrame(corr_rows)

    # --- event-window co-movement ---
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
    events = pd.DataFrame(ev_rows)

    # --- weekly mean±std band (aligned dates) ---
    common = gates[seeds[0]].index
    for s in seeds[1:]:
        common = common.intersection(gates[s].index)
    band = pd.DataFrame(index=common)
    for col in ("gate_finance", "gate_rs", "gate_shipping"):
        stack = np.column_stack([gates[s].loc[common, col].to_numpy()
                                 for s in seeds])
        band[f"{col}_mean"] = stack.mean(1)
        band[f"{col}_std"] = stack.std(1, ddof=0)
        band[f"{col}_lo"] = band[f"{col}_mean"] - band[f"{col}_std"]
        band[f"{col}_hi"] = band[f"{col}_mean"] + band[f"{col}_std"]
    return stability, corr, events, band


def _plot_single(gate, ship_att, rs_att, node_ids, sites, path):
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
    ax0.set_ylim(0, 1)
    ax0.set_title("M4-gated modality gate weights over time (RQ3)")
    ax0.set_ylabel("gate weight")
    ax0.legend(loc="lower left", ncol=3, fontsize=8)

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
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_stability(band: pd.DataFrame, stability: pd.DataFrame,
                    seeds: list[int], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])

    # Panel 1: mean ± std band for each modality gate
    ax0 = fig.add_subplot(gs[0, :])
    colors = {"gate_finance": "C0", "gate_rs": "C2", "gate_shipping": "C1"}
    for col, label in (("gate_finance", "finance"),
                       ("gate_rs", "rs"),
                       ("gate_shipping", "shipping")):
        m, lo, hi = band[f"{col}_mean"], band[f"{col}_lo"], band[f"{col}_hi"]
        ax0.plot(band.index, m, color=colors[col], lw=1.4, label=label)
        ax0.fill_between(band.index, lo, hi, color=colors[col], alpha=0.22)
    for d, lab in EVENTS:
        dt = pd.Timestamp(d)
        if band.index.min() <= dt <= band.index.max():
            ax0.axvline(dt, color="k", ls="--", lw=0.9)
            ax0.text(dt, ax0.get_ylim()[1] * 0.98 if ax0.get_ylim()[1] > 0 else 0.9,
                     lab, rotation=30, fontsize=7, ha="left", va="top")
    ax0.set_ylim(0, 1)
    ax0.set_title(
        f"M4-gated modality gates: mean ± cross-seed std "
        f"(seeds={seeds}, lookback=4)")
    ax0.set_ylabel("gate weight α")
    ax0.legend(loc="upper right", ncol=3, fontsize=8)

    # Panel 2–3: Top-k frequency for ship / rs
    for col_i, kind, title in (
            (0, "ship", f"Shipping Top-{TOP_K} frequency across seeds"),
            (1, "rs", f"RS AOI Top-{TOP_K} frequency across seeds"),
    ):
        ax = fig.add_subplot(gs[1, col_i])
        sub = stability[stability["kind"] == kind].sort_values(
            [f"freq_top{TOP_K}", "att_mean"], ascending=[False, False])
        names = sub["name"].tolist()
        freqs = sub[f"freq_top{TOP_K}"].tolist()
        colors_bar = [
            "#d62728" if (kind == "ship" and "hormuz" in n.lower())
            else ("#2ca02c" if kind == "rs" else "#1f77b4")
            for n in names]
        ax.bar(range(len(names)), freqs, color=colors_bar)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, fontsize=6)
        ax.set_ylim(0, len(seeds) + 0.3)
        ax.set_ylabel(f"freq in Top-{TOP_K} / {len(seeds)}")
        ax.set_title(title)
        ax.axhline(len(seeds), color="gray", ls=":", lw=0.8)

    # Panel bottom: mean attention (ship left, rs right) for context
    ax_s = fig.add_subplot(gs[2, 0])
    sub_s = stability[stability["kind"] == "ship"].sort_values(
        "att_mean", ascending=False)
    ax_s.bar(range(len(sub_s)), sub_s["att_mean"],
             yerr=sub_s["att_std"], capsize=2, color="#1f77b4", alpha=0.85)
    ax_s.set_xticks(range(len(sub_s)))
    ax_s.set_xticklabels(sub_s["name"].tolist(), rotation=60, fontsize=6)
    ax_s.set_title("Ship attention mean±std across seeds")

    ax_r = fig.add_subplot(gs[2, 1])
    sub_r = stability[stability["kind"] == "rs"].sort_values(
        "att_mean", ascending=False)
    ax_r.bar(range(len(sub_r)), sub_r["att_mean"],
             yerr=sub_r["att_std"], capsize=2, color="#2ca02c", alpha=0.85)
    ax_r.set_xticks(range(len(sub_r)))
    ax_r.set_xticklabels(sub_r["name"].tolist(), rotation=60, fontsize=6)
    ax_r.set_title("RS attention mean±std across seeds")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42,
                    help="single seed (ignored if --seeds set)")
    ap.add_argument("--seeds", type=str, default="",
                    help="comma-separated seeds, e.g. 42,1,2")
    args = ap.parse_args()
    seeds = _parse_seeds(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = data.load_matrix()
    dico = data.load_dict()
    ds = build_deep_dataset(df, dico, lookback=args.lookback)
    node_ids = list(ds["node_ids"])
    sites = list(ds["sites"])

    gates: dict[int, pd.DataFrame] = {}
    ship_atts: dict[int, np.ndarray] = {}
    rs_atts: dict[int, np.ndarray] = {}

    for seed in seeds:
        gate, ship_att, rs_att = _run_one_seed(ds, seed, args)
        gates[seed] = gate
        ship_atts[seed] = ship_att
        rs_atts[seed] = rs_att

        gate.to_csv(OUT_DIR / f"deep_gate_weekly_seed{seed}.csv")
        print(f"\n[seed={seed}] Mean modality gate:",
              gate[["gate_finance", "gate_rs", "gate_shipping"]]
              .mean().round(3).to_dict())
        print(f"[seed={seed}] Top ship:",
              _top_names(node_ids, ship_att))
        print(f"[seed={seed}] Top rs:",
              _top_names(sites, rs_att))

    # Primary seed outputs (back-compat with walkthrough / docs)
    primary = seeds[0]
    gates[primary].to_csv(OUT_DIR / "deep_gate_weekly.csv")
    _plot_single(gates[primary], ship_atts[primary], rs_atts[primary],
                 node_ids, sites, OUT_DIR / "deep_interpret.png")

    if len(seeds) == 1:
        print(f"\nSaved: {OUT_DIR / 'deep_gate_weekly.csv'}\n"
              f"       {OUT_DIR / 'deep_interpret.png'}")
        return

    stability, corr, events, band = _aggregate_stability(
        seeds, gates, ship_atts, rs_atts, node_ids, sites)
    stability.to_csv(OUT_DIR / "deep_gate_stability.csv", index=False)
    corr.to_csv(OUT_DIR / "deep_gate_corr.csv", index=False)
    events.to_csv(OUT_DIR / "deep_gate_events.csv", index=False)
    band.to_csv(OUT_DIR / "deep_gate_band_weekly.csv")
    _plot_stability(band, stability, seeds,
                    OUT_DIR / "deep_interpret_stability.png")

    # Console summary for writing
    print("\n========== RQ3 multi-seed stability ==========")
    pair = corr[corr["seed_a"] != corr["seed_b"]]
    if len(pair):
        print("α_shipping pairwise correlations:")
        print(pair[["seed_a", "seed_b", "pearson_alpha_shipping",
                    "spearman_alpha_shipping"]].to_string(index=False))
    print("\nEvent-window Δα_shipping (post−pre, ±"
          f"{EVENT_WINDOW_WEEKS}w):")
    print(events.to_string(index=False))

    ship_top = stability[stability["kind"] == "ship"].sort_values(
        [f"freq_top{TOP_K}", "att_mean"], ascending=[False, False]).head(8)
    rs_top = stability[stability["kind"] == "rs"].sort_values(
        [f"freq_top{TOP_K}", "att_mean"], ascending=[False, False]).head(8)
    print(f"\nShip Top-{TOP_K} frequency:")
    print(ship_top[["name", "att_mean", "rank_mean", f"freq_top{TOP_K}"]]
          .to_string(index=False))
    hormuz = stability[(stability["kind"] == "ship")
                       & (stability["name"].str.lower().str.contains("hormuz"))]
    if len(hormuz):
        h = hormuz.iloc[0]
        print(f"\n>>> Hormuz freq_top{TOP_K} = "
              f"{int(h[f'freq_top{TOP_K}'])}/{len(seeds)} "
              f"(att_mean={h['att_mean']:.4f}, rank_mean={h['rank_mean']:.2f})")
    print(f"\nRS Top-{TOP_K} frequency:")
    print(rs_top[["name", "att_mean", "rank_mean", f"freq_top{TOP_K}"]]
          .to_string(index=False))

    print(f"\nSaved:\n"
          f"  {OUT_DIR / 'deep_gate_stability.csv'}\n"
          f"  {OUT_DIR / 'deep_gate_corr.csv'}\n"
          f"  {OUT_DIR / 'deep_gate_events.csv'}\n"
          f"  {OUT_DIR / 'deep_gate_band_weekly.csv'}\n"
          f"  {OUT_DIR / 'deep_interpret_stability.png'}\n"
          f"  (+ per-seed deep_gate_weekly_seed*.csv)")


if __name__ == "__main__":
    main()
