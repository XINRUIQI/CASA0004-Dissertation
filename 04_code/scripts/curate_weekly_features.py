"""
Curate weekly_features.csv to the literature-aligned feature set.

Three operations (see beatrice_task_literature_matrix.md):
  1. M1 — keep only the new 10-variable mechanism set; drop the old/downgraded
     and raw-extra market/macro/EIA columns.
  2. M2 — drop the old 110 raw remote-sensing columns (opt_* / ntl_ntl_*) and
     merge in the clean 11-AOI M2 set (ntl_anomaly / ntl_valid_obs_count /
     s2_clear_obs_count / s2_cloud_fraction) on week_ending_friday.
  3. Rewrite feature_groups.json M1_market_macro (10) and M2 group (44 clean).

Leaves M3 shipping, GDELT, targets, and remaining availability flags untouched.

Usage:
    python curate_weekly_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"

WEEKLY_CSV = PROC_DIR / "weekly_features.csv"
WEEKLY_PARQUET = PROC_DIR / "weekly_features.parquet"
M2_CLEAN_CSV = PROC_DIR / "weekly_m2_clean_features.csv"
GROUPS_JSON = PROC_DIR / "feature_groups.json"

# ── M1: the new 10-variable mechanism set (kept) ─────────────────────
M1_KEEP = [
    "brent_price",
    "crude_stocks_change",
    "global_econ_activity",
    "nonoil_industrial_commodity",
    "futures_spread",
    "ovx",
    "gpr",
    "dgs10_change",
    "gold_return",
    "commodity_fx",
]

# ── M1: old / downgraded / raw-extra columns to delete ───────────────
M1_DROP = [
    # Brent/WTI price derivatives (targets already materialised separately)
    "wti_price", "brent_wti_spread", "brent_return_pct", "wti_return_pct",
    "brent_log_return", "brent_direction", "brent_vol_4w", "brent_vol_12w",
    # EIA supply / product extras
    "crude_stocks_excl_spr", "cushing_stocks", "crude_production",
    "crude_imports", "crude_exports", "refinery_crude_input",
    "refinery_utilisation", "gasoline_supplied", "distillate_supplied",
    "jet_fuel_supplied", "cushing_stocks_change", "net_crude_trade",
    # Financial extras (downgraded per close-reading)
    "sp500", "vix", "dollar_index", "treasury_10y", "fed_funds_rate",
    "sp500_return_pct",
    # availability flags dangling on deleted columns
    "avail_sp500", "avail_dollar_index",
]


def main() -> None:
    df = pd.read_csv(WEEKLY_CSV, index_col=0, parse_dates=True)
    df.index.name = "week_ending_friday"
    n0 = df.shape[1]
    print(f"Loaded weekly_features.csv: {df.shape}")

    # 1. Drop old M1 columns ------------------------------------------------
    m1_dropped = [c for c in M1_DROP if c in df.columns]
    df = df.drop(columns=m1_dropped)
    print(f"  M1 dropped: {len(m1_dropped)} cols")

    # 2. Drop old raw M2 RS columns (opt_* / ntl_ntl_*) ---------------------
    old_m2 = [c for c in df.columns if c.startswith("opt_") or c.startswith("ntl_ntl_")]
    df = df.drop(columns=old_m2)
    print(f"  M2 old raw dropped: {len(old_m2)} cols")

    # 3. Merge clean 11-AOI M2 set ------------------------------------------
    m2 = pd.read_csv(M2_CLEAN_CSV, index_col=0, parse_dates=True)
    m2.index.name = "week_ending_friday"
    # guard against accidental duplicate columns
    dup = [c for c in m2.columns if c in df.columns]
    if dup:
        df = df.drop(columns=dup)
    df = df.join(m2, how="left")
    m2_cols = list(m2.columns)
    print(f"  M2 clean merged: {len(m2_cols)} cols")

    print(f"Result: {df.shape}  ({n0} -> {df.shape[1]} columns)")

    # Sanity: all kept M1 still present
    missing = [c for c in M1_KEEP if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: expected M1 columns missing: {missing}")

    # Write back ------------------------------------------------------------
    df.to_csv(WEEKLY_CSV)
    print(f"Wrote {WEEKLY_CSV}")
    try:
        df.to_parquet(WEEKLY_PARQUET)
        print(f"Wrote {WEEKLY_PARQUET}")
    except Exception as e:  # pragma: no cover - parquet engine optional
        print(f"  [skip parquet] {e}")

    # 4. Update feature_groups.json ----------------------------------------
    groups = json.loads(GROUPS_JSON.read_text())
    groups["M1_market_macro"] = M1_KEEP
    # drop any previous RS groups, install the single clean M2 group
    for k in ("M2_add_rs", "M2_rs_clean"):
        groups.pop(k, None)
    # place M2_rs_clean before M3_add_shipping for readability
    rebuilt = {}
    for k, v in groups.items():
        if k == "M3_add_shipping":
            rebuilt["M2_rs_clean"] = m2_cols
        rebuilt[k] = v
    if "M2_rs_clean" not in rebuilt:
        rebuilt["M2_rs_clean"] = m2_cols
    GROUPS_JSON.write_text(json.dumps(rebuilt, indent=2) + "\n")
    print(f"Updated {GROUPS_JSON}: " + ", ".join(f"{k}={len(v)}" for k, v in rebuilt.items()))


if __name__ == "__main__":
    main()
