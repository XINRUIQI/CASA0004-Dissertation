"""
Frozen-checkpoint chokepoint-node ablation for M3_Deep_gated, seed 42.

Each OOS week uses that week's checkpoint (no retrain). After fold
standardisation, zero ALL local shipping channels on the masked chokepoint
node(s). Adjacency is unchanged. AOI features and other nodes are untouched
(no global/aggregate columns live on these tensors).

Arms: each of the 6 chokepoints, plus joint Hormuz+Suez+Mandeb+Cape.

Outputs:
  deep_m3_ablate_nodes_weekly.csv
  period_node_ablation_seed42.csv

Run:
  python3 04_code/scripts/deep/run_deep_ablate_m3_nodes.py
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

CHOKE_ORDER = ["hormuz", "suez", "malacca", "mandeb", "panama", "cape"]
CHOKE_LABEL = {
    "hormuz": "Hormuz", "suez": "Suez", "malacca": "Malacca",
    "mandeb": "Mandeb", "panama": "Panama", "cape": "Cape",
}
JOINT_IDS = ["hormuz", "suez", "mandeb", "cape"]
JOINT_NAME = "Hormuz+Suez+Mandeb+Cape"

PERIOD_ORDER = [
    "full",
    "year_2021", "year_2022", "year_2023", "year_2024", "year_2025",
    "event_russia_ukraine", "event_eu_ru_oil_ban",
    "event_opec_plus", "event_red_sea",
]
PERIOD_LABEL = {
    "full": "full OOS",
    "year_2021": "2021", "year_2022": "2022", "year_2023": "2023",
    "year_2024": "2024", "year_2025": "2025",
    "event_russia_ukraine": "Russia–Ukraine",
    "event_eu_ru_oil_ban": "EU RU oil ban",
    "event_opec_plus": "OPEC+",
    "event_red_sea": "Red Sea",
}
DROP_PERIODS = {"early", "late"}


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mask_choke_nodes(X: dict, choke_pos: dict[str, int],
                     node_ids: list[str]) -> dict:
    """Zero local choke features for the listed node ids; adj/AOI unchanged."""
    out = dict(X)
    choke = X["choke"].clone()
    for nid in node_ids:
        choke[:, :, choke_pos[nid], :] = 0
    out["choke"] = choke
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Frozen chokepoint-node ablation for M3_Deep_gated")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-weeks", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = data.load_matrix()
    dico = data.load_dict()
    ds = build_deep_dataset(df, dico, lookback=args.lookback, with_rs=False)
    node_ids = list(ds["node_ids"])
    n_aoi = ds["aoi"].shape[2]
    choke_nodes = node_ids[n_aoi:]
    if choke_nodes != CHOKE_ORDER:
        raise ValueError(f"choke order {choke_nodes} != {CHOKE_ORDER}")
    choke_pos = {nid: i for i, nid in enumerate(choke_nodes)}

    arms: list[tuple[str, list[str]]] = (
        [("intact", [])]
        + [(nid, [nid]) for nid in CHOKE_ORDER]
        + [("joint4", list(JOINT_IDS))]
    )

    modalities, _, fusion_type = CONFIGS[CONFIG_KEY]
    idx = ds["idx"]
    n = len(idx)
    oos_pos = [i for i in range(n) if i >= args.min_train]
    if args.max_weeks and args.max_weeks > 0:
        oos_pos = oos_pos[:args.max_weeks]
    n_oos = len(oos_pos)
    print(f"=== M3 node ablation seed={args.seed} n_oos={n_oos} "
          f"epochs={args.epochs} ===", flush=True)
    print("  mask: scaled choke[node, :] -> 0; adj/AOI/other nodes unchanged",
          flush=True)

    model = None
    sc = None
    r_mean = r_std = 0.0
    rows = []
    for k, i in enumerate(oos_pos):
        if model is None or ((i - args.min_train) % args.retrain_every == 0):
            sc = fit_scalers(ds, train_n=i)
            model, r_mean, r_std = _train_fold(
                ds, sc, i, modalities, args.seed, args.epochs, 1e-3, 1e-4, 32,
                52, args.device, {"d": 32, "gat_layers": 2, "tcn_layers": 2},
                fusion_type=fusion_type)
            model.eval()
            print(f"  fit @ {idx[i].date()} train={i}", flush=True)

        Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), args.device)
        p_t = float(ds["P_t"][i])
        y = float(ds["P_next"][i])
        rec = {"date": idx[i], "P_next_actual": y, "P_hat_M0": p_t}
        model.eval()
        with torch.no_grad():
            for arm, nids in arms:
                Xa = (Xte if arm == "intact"
                      else mask_choke_nodes(Xte, choke_pos, nids))
                out = model(aoi=Xa["aoi"], choke=Xa["choke"],
                            adj=Xa["adj"], fin=Xa["fin"])[0]
                rhat = float(out.item()) * r_std + r_mean
                rec[f"P_hat_{arm}"] = p_t * np.exp(rhat)
        rows.append(rec)
        if (k + 1) % 13 == 0 or k + 1 == n_oos:
            print(f"    node-ablate {k + 1}/{n_oos}  {idx[i].date()}",
                  flush=True)

    weekly = pd.DataFrame(rows)
    weekly.to_csv(OUT_DIR / "deep_m3_ablate_nodes_weekly.csv", index=False)

    memb = pd.read_csv(OUT_DIR / "deep_m3_shap_period_membership.csv",
                       parse_dates=["date"])
    memb = memb[~memb["period_id"].isin(DROP_PERIODS)]
    merged = weekly.merge(memb, on="date", how="inner")

    mask_arms = [(nid, CHOKE_LABEL[nid]) for nid in CHOKE_ORDER]
    mask_arms.append(("joint4", JOINT_NAME))

    out_rows = []
    for pid, sub in merged.groupby("period_id"):
        yv = sub["P_next_actual"].to_numpy()
        rmse0 = _rmse(sub["P_hat_intact"].to_numpy(), yv)
        for arm, label in mask_arms:
            rmse_m = _rmse(sub[f"P_hat_{arm}"].to_numpy(), yv)
            out_rows.append({
                "period": PERIOD_LABEL[pid],
                "period_id": pid,
                "n_weeks": int(len(sub)),
                "masked_node": label,
                "original_RMSE": rmse0,
                "masked_RMSE": rmse_m,
                "delta_RMSE": rmse_m - rmse0,
                "delta_RMSE_pct": float(100.0 * (rmse_m / rmse0 - 1.0)),
            })

    out = pd.DataFrame(out_rows)
    out["period_id"] = pd.Categorical(out["period_id"], PERIOD_ORDER, ordered=True)
    node_order = [CHOKE_LABEL[n] for n in CHOKE_ORDER] + [JOINT_NAME]
    out["masked_node"] = pd.Categorical(out["masked_node"], node_order, ordered=True)
    out = out.sort_values(["period_id", "masked_node"])
    keep = ["period", "masked_node", "n_weeks",
            "original_RMSE", "masked_RMSE", "delta_RMSE", "delta_RMSE_pct"]
    out[keep].to_csv(OUT_DIR / "period_node_ablation_seed42.csv", index=False)

    show = out[keep].copy()
    print("\nΔRMSE>0 => mask hurts (node was useful).", flush=True)
    print(show.to_string(
        index=False,
        formatters={
            "original_RMSE": "{:.4f}".format,
            "masked_RMSE": "{:.4f}".format,
            "delta_RMSE": "{:+.5f}".format,
            "delta_RMSE_pct": "{:+.3f}".format,
        }), flush=True)
    print(f"\nSaved {OUT_DIR / 'period_node_ablation_seed42.csv'}", flush=True)


if __name__ == "__main__":
    main()
