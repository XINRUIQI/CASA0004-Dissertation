"""
Sync newly-added shipping columns from weekly_shipping_features.csv into the
main weekly_features matrix (csv + parquet) and the M3_add_shipping group.

Only ADDS shipping columns (gfw_* / pw_*) that are not already present in
weekly_features.csv; never drops or alters other modalities. Idempotent.

Usage:
    python sync_shipping_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"

FEATURES_CSV = PROC_DIR / "weekly_features.csv"
FEATURES_PARQUET = PROC_DIR / "weekly_features.parquet"
SHIPPING_CSV = PROC_DIR / "weekly_shipping_features.csv"
GROUPS_PATH = PROC_DIR / "feature_groups.json"


def main():
    feats = pd.read_csv(FEATURES_CSV, index_col=0, parse_dates=True)
    ship = pd.read_csv(SHIPPING_CSV, index_col=0, parse_dates=True)

    ship_cols = [c for c in ship.columns if c.startswith(("gfw_", "pw_"))]
    new_cols = [c for c in ship_cols if c not in feats.columns]
    if not new_cols:
        print("No new shipping columns to add. weekly_features is up to date.")
        return

    before = feats.shape[1]
    # Left-join new columns on the weekly index (Fridays); pre-coverage -> NaN
    feats = feats.join(ship[new_cols], how="left")

    # Group all shipping columns together: move new cols next to the existing block
    existing_ship = [c for c in feats.columns if c.startswith(("gfw_", "pw_")) and c not in new_cols]
    if existing_ship:
        anchor = existing_ship[-1]
        cols = [c for c in feats.columns if c not in new_cols]
        idx = cols.index(anchor) + 1
        ordered = cols[:idx] + new_cols + cols[idx:]
        feats = feats[ordered]

    feats.to_csv(FEATURES_CSV)
    try:
        feats.to_parquet(FEATURES_PARQUET)
    except Exception as e:
        print(f"  [warn] parquet not written: {e}")

    # Update feature_groups M3_add_shipping
    with open(GROUPS_PATH) as f:
        groups = json.load(f)
    key = "M3_add_shipping"
    grp = groups.get(key, [])
    for c in new_cols:
        if c not in grp:
            grp.append(c)
    groups[key] = grp
    with open(GROUPS_PATH, "w") as f:
        json.dump(groups, f, indent=2)

    print(f"Added {len(new_cols)} new shipping columns:")
    for c in new_cols:
        nn = feats[c].notna().sum()
        print(f"  {c:38s} non-null={nn}")
    print(f"weekly_features.csv: {before} -> {feats.shape[1]} columns")
    print(f"M3_add_shipping now has {len(grp)} features")


if __name__ == "__main__":
    main()
