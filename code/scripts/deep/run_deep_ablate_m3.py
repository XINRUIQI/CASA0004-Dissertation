"""
Frozen-checkpoint input ablation for M3_Deep_gated, seed 42.

Walk-forward matches run_deep_shap_m3.py. Each OOS week uses that week's
checkpoint; ablations zero the scaled GFW / PortWatch / all-shipping channels
(training-neutral z = 0). Models are not retrained for ablation. Adjacency
is left unchanged.

Arms
  intact      unmasked S3 (same run, so deltas are paired)
  gfw         mask gfw_* on AOI and chokepoint nodes
  portwatch   mask pw_* on AOI and chokepoint nodes
  shipping    mask gfw_* + pw_* + sar_*  (frozen shipping knockout)

Periods: full OOS, calendar years, four event windows. No early/late split.

Outputs (-> results/baselines/Deep/M3_Deep/):
  deep_m3_ablate_weekly.csv
  deep_m3_ablate_period.csv

Run:
  python3 code/scripts/deep/run_deep_ablate_m3.py
  python3 code/scripts/deep/run_deep_ablate_m3.py --max-weeks 2 --epochs 5
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
M1_PRED = deep_out_dir(data.ROOT, "M1") / "baseline_predictions.csv"
CONFIG_KEY = "m3_deep_gated"

ARMS = ("intact", "gfw", "portwatch", "shipping")
PERIOD_ORDER = [
    "full",
    "year_2021", "year_2022", "year_2023", "year_2024", "year_2025",
    "event_russia_ukraine", "event_eu_ru_oil_ban",
    "event_opec_plus", "event_red_sea",
]
DROP_PERIODS = {"early", "late"}


def _prefixes(arm: str) -> tuple[str, ...]:
    if arm == "gfw":
        return ("gfw_",)
    if arm == "portwatch":
        return ("pw_",)
    if arm == "shipping":
        return ("gfw_", "pw_", "sar_")
    raise ValueError(arm)


def mask_tensors(X: dict, aoi_names: list[str], choke_names: list[str],
                 arm: str) -> dict:
    """Zero scaled feature channels; clone so the intact tensors stay intact."""
    if arm == "intact":
        return X
    pref = _prefixes(arm)
    out = dict(X)
    aoi_idx = [i for i, n in enumerate(aoi_names) if n.startswith(pref)]
    choke_idx = [i for i, n in enumerate(choke_names) if n.startswith(pref)]
    if aoi_idx:
        aoi = X["aoi"].clone()
        aoi[..., aoi_idx] = 0
        out["aoi"] = aoi
    if choke_idx:
        choke = X["choke"].clone()
        choke[..., choke_idx] = 0
        out["choke"] = choke
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _skill_pct(rmse_m: float, rmse_ref: float) -> float:
    return float(100.0 * (1.0 - rmse_m / rmse_ref))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Frozen GFW/PortWatch ablation for M3_Deep_gated")
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
    aoi_names = list(ds["aoi_feature_names"])
    choke_names = list(ds["choke_feature_names"])
    modalities, _, fusion_type = CONFIGS[CONFIG_KEY]
    idx = ds["idx"]
    n = len(idx)
    device = args.device

    oos_pos = [i for i in range(n) if i >= args.min_train]
    if args.max_weeks and args.max_weeks > 0:
        oos_pos = oos_pos[:args.max_weeks]
    n_oos = len(oos_pos)
    print(f"=== M3_Deep_gated ablation seed={args.seed} lb={args.lookback} "
          f"epochs={args.epochs} n_oos={n_oos} arms={list(ARMS)} ===",
          flush=True)
    print("  mask: scaled channels -> 0; adj unchanged; no retrain", flush=True)
    print(f"  GFW aoi={[x for x in aoi_names if x.startswith('gfw_')]} "
          f"choke={[x for x in choke_names if x.startswith('gfw_')]}", flush=True)
    print(f"  PW  aoi={[x for x in aoi_names if x.startswith('pw_')]} "
          f"choke={[x for x in choke_names if x.startswith('pw_')]}", flush=True)

    model = None
    sc = None
    r_mean = r_std = 0.0
    rows = []

    for k, i in enumerate(oos_pos):
        if model is None or ((i - args.min_train) % args.retrain_every == 0):
            sc = fit_scalers(ds, train_n=i)
            model, r_mean, r_std = _train_fold(
                ds, sc, i, modalities, args.seed, args.epochs, 1e-3, 1e-4, 32,
                52, device, {"d": 32, "gat_layers": 2, "tcn_layers": 2},
                fusion_type=fusion_type)
            model.eval()
            print(f"  fit @ {idx[i].date()} train={i}", flush=True)

        Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), device)
        p_t = float(ds["P_t"][i])
        y = float(ds["P_next"][i])
        rec = {
            "date": idx[i],
            "P_t": p_t,
            "P_next_actual": y,
            "P_hat_M0": p_t,
        }
        model.eval()
        with torch.no_grad():
            for arm in ARMS:
                Xa = mask_tensors(Xte, aoi_names, choke_names, arm)
                out = model(aoi=Xa["aoi"], choke=Xa["choke"],
                            adj=Xa["adj"], fin=Xa["fin"])[0]
                rhat = float(out.item()) * r_std + r_mean
                rec[f"r_hat_{arm}"] = rhat
                rec[f"P_hat_{arm}"] = p_t * np.exp(rhat)
        rows.append(rec)
        if (k + 1) % 13 == 0 or k + 1 == n_oos:
            print(f"    ablate {k + 1}/{n_oos}  {idx[i].date()}", flush=True)

    weekly = pd.DataFrame(rows)
    weekly.to_csv(OUT_DIR / "deep_m3_ablate_weekly.csv", index=False)

    memb = pd.read_csv(OUT_DIR / "deep_m3_shap_period_membership.csv",
                       parse_dates=["date"])
    memb = memb[~memb["period_id"].isin(DROP_PERIODS)]
    periods = pd.read_csv(OUT_DIR / "deep_m3_shap_lock_periods.csv")
    labels = periods[["period_id", "label", "kind"]]

    m1 = pd.read_csv(M1_PRED, parse_dates=["date"]).set_index("date")
    weekly_i = weekly.set_index("date")
    common = weekly_i.index.intersection(m1.index)
    s1 = m1.loc[common, "P_hat_M1_Deep"]

    merged = weekly.merge(memb, on="date", how="inner")
    out_rows = []
    for pid, sub in merged.groupby("period_id"):
        yv = sub["P_next_actual"].to_numpy()
        rmse_m0 = _rmse(sub["P_hat_M0"].to_numpy(), yv)
        rmse_intact = _rmse(sub["P_hat_intact"].to_numpy(), yv)
        dts = pd.DatetimeIndex(sub["date"])
        s1_idx = dts.intersection(s1.index)
        rmse_s1 = (_rmse(s1.loc[s1_idx].to_numpy(),
                         weekly_i.loc[s1_idx, "P_next_actual"].to_numpy())
                   if len(s1_idx) == len(sub) else np.nan)
        row = {
            "period_id": pid,
            "n_weeks": int(len(sub)),
            "RMSE_M0": rmse_m0,
            "RMSE_DeepS1": rmse_s1,
            "RMSE_intact": rmse_intact,
            "skill_intact_vs_M0_pct": _skill_pct(rmse_intact, rmse_m0),
            "skill_intact_vs_S1_pct": (_skill_pct(rmse_intact, rmse_s1)
                                       if np.isfinite(rmse_s1) else np.nan),
        }
        for arm in ARMS:
            if arm == "intact":
                continue
            rmse_a = _rmse(sub[f"P_hat_{arm}"].to_numpy(), yv)
            row[f"RMSE_{arm}"] = rmse_a
            row[f"delta_RMSE_{arm}"] = rmse_a - rmse_intact
            row[f"damage_{arm}_pct"] = float(100.0 * (rmse_a / rmse_intact - 1.0))
        out_rows.append(row)

    period = pd.DataFrame(out_rows).merge(labels, on="period_id", how="left")
    period = period[~period["period_id"].isin(DROP_PERIODS)]
    period["period_id"] = pd.Categorical(
        period["period_id"], PERIOD_ORDER, ordered=True)
    period = period.sort_values("period_id")
    front = ["period_id", "label", "kind", "n_weeks",
             "RMSE_M0", "RMSE_DeepS1", "RMSE_intact",
             "skill_intact_vs_M0_pct", "skill_intact_vs_S1_pct"]
    rest = [c for c in period.columns if c not in front]
    period = period[front + rest]
    period.to_csv(OUT_DIR / "deep_m3_ablate_period.csv", index=False)

    show = period[["period_id", "n_weeks", "RMSE_intact",
                   "RMSE_gfw", "damage_gfw_pct",
                   "RMSE_portwatch", "damage_portwatch_pct",
                   "RMSE_shipping", "damage_shipping_pct",
                   "skill_intact_vs_S1_pct"]].copy()
    print("\nDamage = 100*(RMSE_masked / RMSE_intact - 1); "
          "positive means the masked source was helping.", flush=True)
    print(show.to_string(
        index=False,
        formatters={
            "RMSE_intact": "{:.3f}".format,
            "RMSE_gfw": "{:.3f}".format,
            "RMSE_portwatch": "{:.3f}".format,
            "RMSE_shipping": "{:.3f}".format,
            "damage_gfw_pct": "{:+.2f}".format,
            "damage_portwatch_pct": "{:+.2f}".format,
            "damage_shipping_pct": "{:+.2f}".format,
            "skill_intact_vs_S1_pct": "{:+.2f}".format,
        }), flush=True)
    print(f"\nSaved:\n  {OUT_DIR / 'deep_m3_ablate_weekly.csv'}\n"
          f"  {OUT_DIR / 'deep_m3_ablate_period.csv'}", flush=True)


if __name__ == "__main__":
    main()
