"""
Add `pw_{choke}_avg_tanker_size` (mean DWT-capacity per tanker transit) to the
main weekly feature matrix and the M3 feature group.

Rationale (P070 concept-priority): tanker capacity weighted by loading and
average vessel size rank above raw tanker count; this is a pure physical,
leakage-safe derived feature (= capacity_tanker / n_tanker).

Derived directly from existing columns in weekly_features.csv, so no other
modality (M1 / M2 / text / targets) is touched. Also keeps
weekly_features.parquet and feature_groups.json in sync.

Usage:
    python add_avg_tanker_size.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"

CSV_PATH = PROC_DIR / "weekly_features.csv"
PARQUET_PATH = PROC_DIR / "weekly_features.parquet"
GROUPS_PATH = PROC_DIR / "feature_groups.json"

CHOKEPOINTS = ["cape", "hormuz", "malacca", "mandeb", "panama", "suez"]


def add_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Insert pw_{cp}_avg_tanker_size after pw_all_tanker_share (if present)."""
    new_cols: list[str] = []
    for cp in CHOKEPOINTS:
        cap = f"pw_{cp}_capacity_tanker"
        n = f"pw_{cp}_n_tanker"
        out = f"pw_{cp}_avg_tanker_size"
        if cap not in df.columns or n not in df.columns:
            print(f"  [skip] {cp}: source columns missing")
            continue
        df[out] = (df[cap] / df[n]).replace([np.inf, -np.inf], np.nan)
        new_cols.append(out)

    # Reorder: place new cols right after the shipping block anchor
    anchor = "pw_all_tanker_share"
    if anchor in df.columns:
        cols = [c for c in df.columns if c not in new_cols]
        idx = cols.index(anchor) + 1
        ordered = cols[:idx] + new_cols + cols[idx:]
        df = df[ordered]
    return df, new_cols


def main():
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    before = df.shape[1]

    df, new_cols = add_columns(df)
    if not new_cols:
        print("No columns added (sources missing). Aborting without write.")
        return

    df.to_csv(CSV_PATH)
    try:
        df.to_parquet(PARQUET_PATH)
    except Exception as e:  # parquet optional
        print(f"  [warn] parquet not written: {e}")

    # Update feature group M3_add_shipping
    with open(GROUPS_PATH) as f:
        groups = json.load(f)

    key = "M3_add_shipping"
    if key in groups:
        existing = groups[key]
        for c in new_cols:
            if c not in existing:
                # insert after the matching capacity_tanker_4w_ma if present, else append
                anchor = c.replace("_avg_tanker_size", "_capacity_tanker_4w_ma")
                if anchor in existing:
                    existing.insert(existing.index(anchor) + 1, c)
                else:
                    existing.append(c)
        groups[key] = existing
        with open(GROUPS_PATH, "w") as f:
            json.dump(groups, f, indent=2)

    print(f"Added {len(new_cols)} columns: {new_cols}")
    print(f"weekly_features.csv: {before} -> {df.shape[1]} columns, rows={len(df)}")
    print(f"M3_add_shipping now has {len(groups.get(key, []))} features")
    # Quick sanity: non-null counts on latest available
    latest = df.dropna(subset=new_cols, how="all").tail(1)
    if not latest.empty:
        print(f"Latest non-null week: {latest.index[-1].date()}")
        print(latest[new_cols].T.to_string())


if __name__ == "__main__":
    main()
