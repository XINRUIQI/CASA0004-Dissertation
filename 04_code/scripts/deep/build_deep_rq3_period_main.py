"""
Cut seed-42 weekly gates by the locked SHAP periods, then join period RMSE
(Deep gated S3 vs Deep S1 vs M0) and period SHAP shares into one main table.

No retraining. Predictions come from the saved walk-forward CSVs; gates from
deep_m3_gate_weekly.csv (interpret run, seed 42); SHAP from agg_deep_shap_m3.py.

Skill = 100 * (1 - RMSE_S3 / RMSE_ref) on reconstructed prices.

Outputs (-> 05_outputs/baselines/Deep/M3_Deep/):
  deep_m3_gate_period.csv
  deep_m3_rq3_period_main.csv

Run:
  python3 04_code/scripts/deep/build_deep_rq3_period_main.py
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
M1_PRED = deep_out_dir(data.ROOT, "M1") / "baseline_predictions.csv"
M3_PRED = OUT_DIR / "baseline_predictions.csv"
GATE = OUT_DIR / "deep_m3_gate_weekly.csv"

S3_COL = "P_hat_M3_Deep_gated"
S1_COL = "P_hat_M1_Deep"
M0_COL = "P_hat_M0"
Y_COL = "P_next_actual"

PERIOD_ORDER = [
    "full",
    "year_2021", "year_2022", "year_2023", "year_2024", "year_2025",
    "event_russia_ukraine", "event_eu_ru_oil_ban",
    "event_opec_plus", "event_red_sea",
]
GROUP_ORDER = [
    "oil_price_basis", "financial_macro", "eia_fundamentals", "gpr",
    "GFW", "PortWatch_chokepoint", "PortWatch_directional", "SAR",
]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _skill_pct(rmse_m: float, rmse_ref: float) -> float:
    return float(100.0 * (1.0 - rmse_m / rmse_ref))


def main() -> None:
    memb = pd.read_csv(OUT_DIR / "deep_m3_shap_period_membership.csv",
                       parse_dates=["date"])
    periods = pd.read_csv(OUT_DIR / "deep_m3_shap_lock_periods.csv")
    gate = pd.read_csv(GATE, parse_dates=["date"])
    m3 = pd.read_csv(M3_PRED, parse_dates=["date"]).set_index("date")
    m1 = pd.read_csv(M1_PRED, parse_dates=["date"]).set_index("date")
    shap_mod = pd.read_csv(OUT_DIR / "deep_m3_shap_period_modality.csv")
    shap_grp = pd.read_csv(OUT_DIR / "deep_m3_shap_period_group.csv")

    dates = pd.DatetimeIndex(
        pd.read_csv(OUT_DIR / "deep_m3_shap_dates.csv")["date"])
    common = dates.intersection(m3.index).intersection(m1.index).intersection(
        pd.DatetimeIndex(gate["date"]))
    if len(common) != 257:
        print(f"WARNING: common weeks={len(common)} (expected 257)", flush=True)
    m3 = m3.loc[common]
    m1 = m1.loc[common]
    if not np.allclose(m3[Y_COL].to_numpy(), m1[Y_COL].to_numpy()):
        raise ValueError("M1 and M3 P_next_actual do not match on common weeks")
    if not np.allclose(m3[M0_COL].to_numpy(), m1[M0_COL].to_numpy()):
        raise ValueError("M1 and M3 P_hat_M0 do not match on common weeks")

    y = m3[Y_COL]
    panel = pd.DataFrame({
        "date": m3.index,
        "y": y.to_numpy(),
        "m0": m3[M0_COL].to_numpy(),
        "s1": m1[S1_COL].to_numpy(),
        "s3": m3[S3_COL].to_numpy(),
    })
    gmerge = gate.set_index("date")[["gate_finance", "gate_shipping"]]
    panel = panel.merge(gmerge, left_on="date", right_index=True, how="inner")
    panel = panel.merge(memb, on="date", how="inner")

    rmse_rows = []
    gate_rows = []
    for pid, sub in panel.groupby("period_id"):
        rmse_m0 = _rmse(sub["m0"].to_numpy(), sub["y"].to_numpy())
        rmse_s1 = _rmse(sub["s1"].to_numpy(), sub["y"].to_numpy())
        rmse_s3 = _rmse(sub["s3"].to_numpy(), sub["y"].to_numpy())
        rmse_rows.append({
            "period_id": pid,
            "n_weeks": int(len(sub)),
            "RMSE_M0": rmse_m0,
            "RMSE_DeepS1": rmse_s1,
            "RMSE_DeepS3": rmse_s3,
            "skill_S3_vs_M0_pct": _skill_pct(rmse_s3, rmse_m0),
            "skill_S3_vs_S1_pct": _skill_pct(rmse_s3, rmse_s1),
        })
        gate_rows.append({
            "period_id": pid,
            "n_weeks": int(len(sub)),
            "gate_finance": float(sub["gate_finance"].mean()),
            "gate_shipping": float(sub["gate_shipping"].mean()),
        })
    rmse_df = pd.DataFrame(rmse_rows)
    gate_df = pd.DataFrame(gate_rows)

    labels = periods[["period_id", "label", "kind"]]
    gate_out = (gate_df.merge(labels, on="period_id")
                .assign(period_id=lambda d: pd.Categorical(
                    d["period_id"], PERIOD_ORDER, ordered=True))
                .sort_values("period_id"))
    gate_out = gate_out[["period_id", "label", "kind", "n_weeks",
                         "gate_finance", "gate_shipping"]]
    gate_out.to_csv(OUT_DIR / "deep_m3_gate_period.csv", index=False)

    shap_wide = shap_mod.pivot(
        index=["period_id", "n_weeks"], columns="modality",
        values=["mean_shap_abs", "share_total"]).reset_index()
    shap_wide.columns = [
        "period_id" if c == ("period_id", "") else
        "n_weeks" if c == ("n_weeks", "") else
        f"shap_I_{c[1]}" if c[0] == "mean_shap_abs" else
        f"shap_share_{c[1]}"
        for c in shap_wide.columns.to_flat_index()
    ]
    grp_share = shap_grp.pivot(
        index="period_id", columns="group", values="share_total").reset_index()
    grp_share = grp_share.rename(
        columns={g: f"shap_share_{g}" for g in GROUP_ORDER})
    grp_I = shap_grp.pivot(
        index="period_id", columns="group", values="mean_shap_abs").reset_index()
    grp_I = grp_I.rename(columns={g: f"shap_I_{g}" for g in GROUP_ORDER})

    main_df = (rmse_df
               .merge(gate_out.drop(columns=["n_weeks"]), on="period_id")
               .merge(shap_wide.drop(columns=["n_weeks"]), on="period_id")
               .merge(grp_share, on="period_id")
               .merge(grp_I, on="period_id"))
    main_df["period_id"] = pd.Categorical(
        main_df["period_id"], PERIOD_ORDER, ordered=True)
    main_df = main_df.sort_values("period_id")
    front = ["period_id", "label", "kind", "n_weeks",
             "RMSE_M0", "RMSE_DeepS1", "RMSE_DeepS3",
             "skill_S3_vs_M0_pct", "skill_S3_vs_S1_pct",
             "gate_finance", "gate_shipping",
             "shap_I_finance", "shap_I_shipping",
             "shap_share_finance", "shap_share_shipping"]
    rest = [c for c in main_df.columns if c not in front]
    main_df = main_df[front + rest]
    main_df.to_csv(OUT_DIR / "deep_m3_rq3_period_main.csv", index=False)

    show = main_df[["period_id", "n_weeks", "RMSE_M0", "RMSE_DeepS1",
                    "RMSE_DeepS3", "skill_S3_vs_M0_pct", "skill_S3_vs_S1_pct",
                    "gate_finance", "gate_shipping",
                    "shap_share_finance", "shap_share_shipping"]].copy()
    print(show.to_string(
        index=False,
        formatters={
            "RMSE_M0": "{:.3f}".format,
            "RMSE_DeepS1": "{:.3f}".format,
            "RMSE_DeepS3": "{:.3f}".format,
            "skill_S3_vs_M0_pct": "{:+.2f}".format,
            "skill_S3_vs_S1_pct": "{:+.2f}".format,
            "gate_finance": "{:.3f}".format,
            "gate_shipping": "{:.3f}".format,
            "shap_share_finance": "{:.3f}".format,
            "shap_share_shipping": "{:.3f}".format,
        }), flush=True)
    print(f"\nSaved:\n  {OUT_DIR / 'deep_m3_gate_period.csv'}\n"
          f"  {OUT_DIR / 'deep_m3_rq3_period_main.csv'}", flush=True)


if __name__ == "__main__":
    main()
