"""
Period / group / node aggregation of seed-42 M3_Deep_gated input SHAP.

Uses locked membership from run_deep_shap_m3.py. Does not retrain.

Per week:
  I_{t,g}     = sum_{features in g, lags} |phi|
  S_{t,g}     = sum_{features in g, lags}  phi     (signed; can cancel)
  I_{t,node}  = sum_{features on node, lags} |phi|
  I_{t,f}     = sum_lags |phi| for a (channel, node, feature)

Then mean over weeks in each locked period. Shares are of that period's
mean total |SHAP| (or of finance / shipping subtotal).

Outputs (-> 05_outputs/baselines/Deep/M3_Deep/):
  deep_m3_shap_weekly_group.csv
  deep_m3_shap_weekly_node.csv
  deep_m3_shap_period_group.csv
  deep_m3_shap_period_modality.csv
  deep_m3_shap_period_node.csv
  deep_m3_shap_period_feature.csv     mean I_{t,f} by period
  deep_m3_att_period_node.csv         mean GAT node attention by period

Run:
  python3 04_code/scripts/deep/agg_deep_shap_m3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data                                # noqa: E402
from model_naming import deep_out_dir                    # noqa: E402

OUT_DIR = deep_out_dir(data.ROOT, "M3")

FIN_GROUPS = ["oil_price_basis", "financial_macro", "eia_fundamentals", "gpr"]
SHIP_GROUPS = ["GFW", "PortWatch_chokepoint", "PortWatch_directional", "SAR"]
ALL_GROUPS = FIN_GROUPS + SHIP_GROUPS
MODALITY = {g: ("finance" if g in FIN_GROUPS else "shipping") for g in ALL_GROUPS}

PERIOD_ORDER = [
    "full", "early", "late",
    "year_2021", "year_2022", "year_2023", "year_2024", "year_2025",
    "event_russia_ukraine", "event_eu_ru_oil_ban",
    "event_opec_plus", "event_red_sea",
]


def _load() -> tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame,
                     pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(pd.read_csv(OUT_DIR / "deep_m3_shap_dates.csv")["date"])
    phi = np.load(OUT_DIR / "deep_m3_shap_values.npy")
    fmap = pd.read_csv(OUT_DIR / "deep_m3_shap_feature_map.csv")
    memb = pd.read_csv(OUT_DIR / "deep_m3_shap_period_membership.csv",
                       parse_dates=["date"])
    periods = pd.read_csv(OUT_DIR / "deep_m3_shap_lock_periods.csv")
    att = pd.read_csv(OUT_DIR / "deep_m3_node_attention_weekly.csv",
                      parse_dates=["date"])
    if phi.shape != (len(dates), len(fmap)):
        raise ValueError(f"shape mismatch phi={phi.shape} dates={len(dates)} "
                         f"fmap={len(fmap)}")
    fmap["node_id"] = fmap["node_id"].fillna("").astype(str)
    return dates, phi, fmap, memb, periods, att


def weekly_by_key(dates: pd.DatetimeIndex, phi: np.ndarray,
                  fmap: pd.DataFrame, key: str) -> pd.DataFrame:
    abs_phi = np.abs(phi)
    rows = []
    for val, sub in fmap.groupby(key, dropna=False):
        if key == "group" and val not in ALL_GROUPS:
            continue
        if key == "node_id" and (val == "" or pd.isna(val)):
            continue
        cols = sub["feature_index"].to_numpy(dtype=int)
        rows.append(pd.DataFrame({
            "date": dates,
            key: val,
            "shap_abs": abs_phi[:, cols].sum(axis=1),
            "shap_signed": phi[:, cols].sum(axis=1),
        }))
    out = pd.concat(rows, ignore_index=True)
    if key == "group":
        out["modality"] = out["group"].map(MODALITY)
        out["group"] = pd.Categorical(out["group"], ALL_GROUPS, ordered=True)
        out = out.sort_values(["date", "group"])
    return out


def weekly_feature(dates: pd.DatetimeIndex, phi: np.ndarray,
                   fmap: pd.DataFrame) -> pd.DataFrame:
    """I_{t,f}: sum over lags, keep (channel, node_id, feature, group)."""
    keys = ["channel", "node_id", "feature", "group"]
    abs_phi = np.abs(phi)
    blocks = []
    for meta, sub in fmap.groupby(keys, dropna=False):
        rec = {k: v for k, v in zip(keys, meta)}
        cols = sub["feature_index"].to_numpy(dtype=int)
        blocks.append(pd.DataFrame({
            "date": dates,
            **rec,
            "shap_abs": abs_phi[:, cols].sum(axis=1),
            "shap_signed": phi[:, cols].sum(axis=1),
        }))
    return pd.concat(blocks, ignore_index=True)


def period_mean(weekly: pd.DataFrame, memb: pd.DataFrame,
                keys: list[str], value_cols: tuple[str, ...] = ("shap_abs", "shap_signed")
                ) -> pd.DataFrame:
    merged = weekly.merge(memb, on="date", how="inner")
    g = (merged.groupby(["period_id"] + keys, observed=True)[list(value_cols)]
         .mean()
         .reset_index()
         .rename(columns={c: f"mean_{c}" for c in value_cols}))
    n = memb.groupby("period_id").size().rename("n_weeks")
    g = g.merge(n, on="period_id", how="left")
    return g


def add_shares(period_group: pd.DataFrame) -> pd.DataFrame:
    tot = (period_group.groupby("period_id")["mean_shap_abs"]
           .sum()
           .rename("mean_total"))
    out = period_group.merge(tot, on="period_id")
    out["share_total"] = out["mean_shap_abs"] / out["mean_total"]
    mod = (out.groupby(["period_id", "modality"])["mean_shap_abs"]
           .transform("sum"))
    out["share_within_modality"] = out["mean_shap_abs"] / mod
    out["period_id"] = pd.Categorical(out["period_id"], PERIOD_ORDER, ordered=True)
    out["group"] = pd.Categorical(out["group"], ALL_GROUPS, ordered=True)
    return out.sort_values(["period_id", "group"]).drop(columns=["mean_total"])


def period_modality(period_group: pd.DataFrame) -> pd.DataFrame:
    g = (period_group.groupby(["period_id", "n_weeks", "modality"], observed=True)
         .agg(mean_shap_abs=("mean_shap_abs", "sum"),
              mean_shap_signed=("mean_shap_signed", "sum"))
         .reset_index())
    tot = g.groupby("period_id")["mean_shap_abs"].transform("sum")
    g["share_total"] = g["mean_shap_abs"] / tot
    g["period_id"] = pd.Categorical(g["period_id"], PERIOD_ORDER, ordered=True)
    g["modality"] = pd.Categorical(g["modality"], ["finance", "shipping"],
                                   ordered=True)
    return g.sort_values(["period_id", "modality"])


def add_node_shares(period_node: pd.DataFrame) -> pd.DataFrame:
    tot = period_node.groupby("period_id")["mean_shap_abs"].transform("sum")
    out = period_node.copy()
    out["share_shipping"] = out["mean_shap_abs"] / tot
    out["period_id"] = pd.Categorical(out["period_id"], PERIOD_ORDER, ordered=True)
    return out.sort_values(["period_id", "mean_shap_abs"],
                           ascending=[True, False])


def attach_labels(df: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    lab = periods[["period_id", "label", "kind"]].copy()
    out = df.merge(lab, on="period_id", how="left")
    cols = ["period_id", "label", "kind", "n_weeks"]
    rest = [c for c in out.columns if c not in cols]
    return out[cols + rest]


def main() -> None:
    dates, phi, fmap, memb, periods, att = _load()
    print(f"Loaded phi {phi.shape}  weeks {dates.min().date()}–{dates.max().date()}",
          flush=True)

    w_group = weekly_by_key(dates, phi, fmap, "group")
    w_node = weekly_by_key(dates, phi, fmap, "node_id")
    w_group.to_csv(OUT_DIR / "deep_m3_shap_weekly_group.csv", index=False)
    w_node.to_csv(OUT_DIR / "deep_m3_shap_weekly_node.csv", index=False)

    p_group = add_shares(period_mean(w_group, memb, ["group", "modality"]))
    p_group = attach_labels(p_group, periods)
    p_group.to_csv(OUT_DIR / "deep_m3_shap_period_group.csv", index=False)

    p_mod = period_modality(p_group)
    p_mod = attach_labels(p_mod, periods)
    p_mod.to_csv(OUT_DIR / "deep_m3_shap_period_modality.csv", index=False)

    p_node = add_node_shares(period_mean(w_node, memb, ["node_id"]))
    p_att = period_mean(att.rename(columns={"node_attention": "shap_abs"}),
                        memb, ["node_id"], value_cols=("shap_abs",))
    p_att = p_att.rename(columns={"mean_shap_abs": "mean_node_attention"})
    p_node = p_node.merge(p_att[["period_id", "node_id", "mean_node_attention"]],
                          on=["period_id", "node_id"], how="left")
    p_node = attach_labels(p_node, periods)
    p_node.to_csv(OUT_DIR / "deep_m3_shap_period_node.csv", index=False)

    att_p = attach_labels(p_att.rename(columns={"mean_node_attention": "mean_attention"}),
                          periods)
    att_p["period_id"] = pd.Categorical(att_p["period_id"], PERIOD_ORDER, ordered=True)
    att_p = att_p.sort_values(["period_id", "mean_attention"],
                              ascending=[True, False])
    att_p.to_csv(OUT_DIR / "deep_m3_att_period_node.csv", index=False)

    w_feat = weekly_feature(dates, phi, fmap)
    p_feat = period_mean(w_feat, memb, ["channel", "node_id", "feature", "group"])
    p_feat = attach_labels(p_feat, periods)
    p_feat = p_feat.sort_values(["period_id", "mean_shap_abs"],
                                ascending=[True, False])
    p_feat.to_csv(OUT_DIR / "deep_m3_shap_period_feature.csv", index=False)

    print("\n=== modality share by period ===", flush=True)
    show = p_mod.pivot(index=["period_id", "n_weeks"], columns="modality",
                       values="share_total")
    print(show.round(4).to_string(), flush=True)
    print("\n=== group share of total |SHAP| (full / early / late) ===", flush=True)
    sub = p_group[p_group["period_id"].isin(["full", "early", "late"])]
    print(sub.pivot(index="group", columns="period_id",
                    values="share_total")[["full", "early", "late"]]
          .round(4).to_string(), flush=True)
    print("\nSaved:", flush=True)
    for name in [
        "deep_m3_shap_weekly_group.csv",
        "deep_m3_shap_weekly_node.csv",
        "deep_m3_shap_period_group.csv",
        "deep_m3_shap_period_modality.csv",
        "deep_m3_shap_period_node.csv",
        "deep_m3_shap_period_feature.csv",
        "deep_m3_att_period_node.csv",
    ]:
        print(f"  {OUT_DIR / name}", flush=True)


if __name__ == "__main__":
    main()
