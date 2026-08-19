# -*- coding: utf-8 -*-
"""Side-by-side comparison of the two sweep rounds (zero vs fold_median)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "results" / "_experiments" / "leading_impute"
Z, M = EXP / "sweeps_zero", EXP / "sweeps_median"

SPECS = [
    ("M1", "sweep_summary.csv",    ["config", "model"], "RMSE"),
    ("M2", "sweep_m2_summary.csv", ["contract", "lookback", "model"], "M2_RMSE"),
    ("M3", "sweep_m3_summary.csv", ["lookback", "model"], "M3_RMSE"),
    ("M4", "sweep_m4_summary.csv", ["lookback", "model"], "M4_RMSE"),
]

pd.set_option("display.width", 220)
frames = []
for mod, fname, keys, rmse_col in SPECS:
    a = pd.read_csv(Z / fname).set_index(keys)
    b = pd.read_csv(M / fname).set_index(keys)
    cw = "CW_p_vs_M1" if "CW_p_vs_M1" in a.columns else None
    out = pd.DataFrame({
        "RMSE_zero":   a[rmse_col],
        "RMSE_median": b[rmse_col],
        "dRMSE":       b[rmse_col] - a[rmse_col],
    })
    if cw:
        out["CW_p_zero"] = a[cw]
        out["CW_p_median"] = b[cw]
        out["CW_flip"] = (a[cw] < 0.05) != (b[cw] < 0.05)
    out = out.reset_index()
    out.insert(0, "sweep", mod)
    frames.append(out)
    print(f"===== {mod} =====")
    print(out.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print()

allc = pd.concat(frames, ignore_index=True)
allc.to_csv(EXP / "comparison_sweeps_zero_vs_median.csv", index=False)

d = allc["dRMSE"].abs()
print(f"cells compared        : {len(allc)}")
print(f"max |dRMSE|           : {d.max():.4f} USD/bbl  "
      f"({allc.loc[d.idxmax(), 'sweep']} "
      f"{allc.loc[d.idxmax(), ['config', 'contract', 'lookback', 'model']].dropna().to_dict()})")
print(f"median |dRMSE|        : {d.median():.4f} USD/bbl")
if "CW_flip" in allc:
    flips = allc[allc["CW_flip"].fillna(False)]
    print(f"CW significance flips : {len(flips)} of {allc['CW_flip'].notna().sum()}")
    if len(flips):
        print(flips[["sweep", "contract", "lookback", "model",
                     "CW_p_zero", "CW_p_median"]].to_string(index=False))
print(f"\nSaved: {EXP / 'comparison_sweeps_zero_vs_median.csv'}")
