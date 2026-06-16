"""
Merge the M1 "to-build" weekly variables into the unified weekly anchor.

Reads:
    03_data/processed/weekly_time_index.csv      (anchor, built by build_weekly_time_index.py)
    03_data/processed/m1_to_build_weekly.csv     (built by build_m1_to_build.py)

Joins the 8 new M1 variables on the Friday-ending weekly index and writes the
result back to weekly_time_index.csv. Existing columns with the same name are
overwritten (idempotent: safe to re-run).

Usage:
    python merge_m1_to_build.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
ANCHOR = OUT_DIR / "weekly_time_index.csv"
M1_NEW = OUT_DIR / "m1_to_build_weekly.csv"


def main():
    anchor = pd.read_csv(ANCHOR, index_col=0, parse_dates=[0])
    new = pd.read_csv(M1_NEW, index_col=0, parse_dates=[0])

    new = new.reindex(anchor.index)               # align to anchor weeks
    anchor = anchor.drop(columns=[c for c in new.columns if c in anchor.columns])
    merged = anchor.join(new, how="left")

    merged.to_csv(ANCHOR)

    print(f"Merged {len(new.columns)} M1 variables into {ANCHOR.name}")
    print(f"Shape: {merged.shape}   Period: {merged.index.min().date()} ~ {merged.index.max().date()}")
    print("\nNewly added columns:")
    for c in new.columns:
        nn = merged[c].notna().sum()
        print(f"  {c:30s}  {nn:5d} / {len(merged):5d}  ({nn/len(merged)*100:5.1f}%)")


if __name__ == "__main__":
    main()
