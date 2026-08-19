# -*- coding: utf-8 -*-
"""Side-by-side comparison of the zero vs fold_median leading-gap treatment."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT / "results" / "baselines" / "Flat"
NEW = ROOT / "results" / "_experiments" / "leading_impute"

FILES = {
    "M1": "M1_Flat/baseline_metrics.csv",
    "M2": "M2_Flat/baseline_metrics_anom.csv",
    "M3": "M3_Flat/baseline_metrics.csv",
    "M4": "M4_Flat/baseline_metrics_anom.csv",
}
COLS = ["RMSE", "MAE", "DirAcc", "RMSE_skill_vs_M0", "CW_p_vs_M1"]

rows = []
for mod, rel in FILES.items():
    a = pd.read_csv(MAIN / rel, index_col=0)
    b = pd.read_csv(NEW / rel, index_col=0)
    keep = [i for i in a.index if i.startswith(f"{mod}_Flat")]
    for i in keep:
        rec = {"config": mod, "model": i}
        for c in COLS:
            rec[f"{c}_zero"] = a.loc[i, c]
            rec[f"{c}_median"] = b.loc[i, c]
            rec[f"{c}_delta"] = b.loc[i, c] - a.loc[i, c]
        rows.append(rec)

cmp = pd.DataFrame(rows).set_index(["config", "model"])
out = NEW / "comparison_zero_vs_fold_median.csv"
cmp.to_csv(out)

show = cmp[["RMSE_zero", "RMSE_median", "RMSE_delta",
            "RMSE_skill_vs_M0_zero", "RMSE_skill_vs_M0_median",
            "DirAcc_zero", "DirAcc_median",
            "CW_p_vs_M1_zero", "CW_p_vs_M1_median"]]
pd.set_option("display.width", 200)
print(show.to_string(float_format=lambda x: f"{x:9.4f}"))
print(f"\nmax |dRMSE| = {cmp['RMSE_delta'].abs().max():.4f} USD/bbl")
print(f"Saved: {out}")
