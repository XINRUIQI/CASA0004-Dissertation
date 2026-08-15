"""
Export the two RQ3 inspection CSVs (seed 42):

  node_weekly_seed42.csv
  period_ablation_seed42.csv

No retraining. Drops the early/late split. Node-level masks were not run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data                                # noqa: E402
from model_naming import deep_out_dir                    # noqa: E402

OUT_DIR = deep_out_dir(data.ROOT, "M3")
DROP_PERIODS = {"early", "late"}

NODE_LABEL = {
    "P001": "Rotterdam", "P002": "Fujairah", "P003": "Ras Tanura",
    "P004": "Jurong", "P005": "Houston", "P006": "Ningbo-Zhoushan",
    "P007": "Jamnagar", "P008": "Basra", "P009": "Ulsan",
    "P010": "Kharg", "P011": "Yanbu",
    "hormuz": "Hormuz", "suez": "Suez", "malacca": "Malacca",
    "mandeb": "Mandeb", "panama": "Panama", "cape": "Cape",
}
CHOKE = {"hormuz", "suez", "malacca", "mandeb", "panama", "cape"}
PERIOD_LABEL = {
    "full": "full OOS",
    "year_2021": "2021", "year_2022": "2022", "year_2023": "2023",
    "year_2024": "2024", "year_2025": "2025",
    "event_russia_ukraine": "Russia–Ukraine",
    "event_eu_ru_oil_ban": "EU RU oil ban",
    "event_opec_plus": "OPEC+",
    "event_red_sea": "Red Sea",
}
MASK_LABEL = {"gfw": "GFW", "portwatch": "PortWatch", "shipping": "all shipping"}


def main() -> None:
    shap = pd.read_csv(OUT_DIR / "deep_m3_shap_weekly_node.csv",
                       parse_dates=["date"])
    att = pd.read_csv(OUT_DIR / "deep_m3_node_attention_weekly.csv",
                      parse_dates=["date"])
    memb = pd.read_csv(OUT_DIR / "deep_m3_shap_period_membership.csv",
                       parse_dates=["date"])
    memb = memb[~memb["period_id"].isin(DROP_PERIODS)].copy()
    memb["period"] = memb["period_id"].map(PERIOD_LABEL)

    tot = shap.groupby("date")["shap_abs"].transform("sum")
    shap = shap.copy()
    shap["node_shap_abs"] = shap["shap_abs"]
    shap["node_shap_share"] = shap["shap_abs"] / tot
    shap["node_shap_signed"] = shap["shap_signed"]

    att = att.copy()
    att_tot = att.groupby("date")["node_attention"].transform("sum")
    att["attention_share"] = att["node_attention"] / att_tot

    weekly = shap.merge(att, on=["date", "node_id"], how="inner")
    weekly["node"] = weekly["node_id"].map(NODE_LABEL)
    weekly["node_type"] = weekly["node_id"].map(
        lambda x: "chokepoint" if x in CHOKE else "aoi")
    if weekly["node"].isna().any():
        raise ValueError("unmapped node_id")

    long = weekly.merge(memb, on="date", how="inner")
    long = long.rename(columns={"date": "target_date"})
    cols = [
        "target_date", "period", "node_id", "node", "node_type",
        "node_shap_abs", "node_shap_share", "node_shap_signed",
        "node_attention", "attention_share",
    ]
    long = long[cols].sort_values(["period", "target_date", "node_type", "node"])
    node_path = OUT_DIR / "node_weekly_seed42.csv"
    long.to_csv(node_path, index=False)

    ablate = pd.read_csv(OUT_DIR / "deep_m3_ablate_period.csv")
    ablate = ablate[~ablate["period_id"].isin(DROP_PERIODS)].copy()
    rows = []
    for _, r in ablate.iterrows():
        for arm, name in MASK_LABEL.items():
            rows.append({
                "period": PERIOD_LABEL[r["period_id"]],
                "period_id": r["period_id"],
                "n_weeks": int(r["n_weeks"]),
                "mask_type": "group",
                "mask_name": name,
                "original_RMSE": r["RMSE_intact"],
                "masked_RMSE": r[f"RMSE_{arm}"],
                "delta_RMSE": r[f"delta_RMSE_{arm}"],
                "delta_RMSE_pct": r[f"damage_{arm}_pct"],
                "RMSE_M0": r["RMSE_M0"],
                "RMSE_S1": r["RMSE_DeepS1"],
                "RMSE_S3": r["RMSE_intact"],
            })
    ab = pd.DataFrame(rows)
    ab_path = OUT_DIR / "period_ablation_seed42.csv"
    ab.to_csv(ab_path, index=False)

    print("period date ranges (exclusive of early/late):")
    for pid, lab in PERIOD_LABEL.items():
        d = memb.loc[memb["period_id"] == pid, "date"]
        if d.empty:
            continue
        print(f"  {lab:20s}  n={len(d):3d}  {d.min().date()} – {d.max().date()}")
    print(f"\n{node_path}  rows={len(long)}")
    print(f"{ab_path}  rows={len(ab)}")
    print("node masks: none run (group masks only: GFW, PortWatch, all shipping)")


if __name__ == "__main__":
    main()
