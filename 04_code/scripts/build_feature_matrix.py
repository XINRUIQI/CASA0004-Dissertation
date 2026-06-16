"""
Build the unified weekly feature matrix by merging all modality files.

This applies the literature-aligned curation (see
01_literature/beatrice_task_literature_matrix.md):
  - M1 is restricted to the new 10-variable mechanism set (M1_KEEP); the old /
    downgraded / raw-extra market, EIA and financial columns are dropped.
  - M2 uses the clean 11-AOI remote-sensing set (weekly_m2_clean_features.csv,
    built by 03_data/raw/04_sentinel2/build_m2_clean_features.py) — dynamic NTL
    anomalies + observation-quality variables — NOT the old 110 raw RS columns.

Merges:
  - weekly_time_index.csv         (market + macro + EIA; filtered to M1_KEEP)
  - weekly_m2_clean_features.csv  (clean M2: ntl_anomaly / valid_obs / s2_*)
  - weekly_shipping_features.csv  (GFW + PortWatch chokepoints, if available)
  - weekly_text_features.csv      (GDELT + NLP, if available)

Adds target variables and modality availability flags.

Output:
  03_data/processed/weekly_features.parquet
  03_data/processed/weekly_features.csv
  03_data/processed/feature_groups.json

Usage:
    python build_feature_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"
RAW_MARKET_DIR = PROJECT_ROOT / "03_data" / "raw" / "01_market_financial"

# ── M1: new 10-variable mechanism set (post-close-reading) ───────────
# Everything else in weekly_time_index.csv (wti_price, brent_* derivatives,
# raw EIA supply/product series, sp500/vix/dollar_index/treasury_10y/
# fed_funds_rate, sp500_return_pct, ...) is intentionally dropped.
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
# Availability flags retained from the market anchor (others dropped).
MARKET_AVAIL_KEEP = ["avail_market", "avail_eia_weekly"]


def load_weekly(filename: str) -> pd.DataFrame:
    """Load a processed weekly CSV, set index to datetime."""
    path = PROC_DIR / filename
    if not path.exists():
        print(f"  [skip] {filename} not found")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "week_ending_friday"
    print(f"  [ok]   {filename}: {df.shape}")
    return df


def load_brent_daily() -> pd.Series:
    """Load daily Brent prices from EIA xls for realized vol computation."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    price_dir = RAW_MARKET_DIR / "1A Oil Price"
    xls_files = list(price_dir.glob("EIA_brent_spot_price_daily*.xls"))
    if not xls_files:
        raise FileNotFoundError(f"No Brent daily xls in {price_dir}")

    df = pd.read_excel(xls_files[0], sheet_name="Data 1", header=None, skiprows=3)
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).set_index("date").sort_index()
    return df["price"]


def compute_next_week_realized_vol(weekly_index: pd.DatetimeIndex) -> pd.Series:
    """
    Compute realized volatility for the next week (next 5 trading days).

    For each Friday in weekly_index, find daily log returns from the following
    Monday through Friday and compute their standard deviation.
    """
    brent_daily = load_brent_daily()
    daily_log_ret = np.log(brent_daily / brent_daily.shift(1)).dropna()

    vol_values = []
    for friday in weekly_index:
        next_monday = friday + pd.Timedelta(days=3)
        next_friday = friday + pd.Timedelta(days=7)
        week_returns = daily_log_ret.loc[next_monday:next_friday]
        if len(week_returns) >= 3:
            vol_values.append(week_returns.std())
        else:
            vol_values.append(np.nan)

    return pd.Series(vol_values, index=weekly_index, name="target_brent_vol_next_1w")


def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward-looking target variables for prediction."""
    df = df.copy()

    # Next-week price level (regression target)
    df["target_brent_price_next_1w"] = df["brent_price"].shift(-1)

    # Next-week realized volatility (std of daily log returns in next 5 trading days)
    print("  Computing next-week realized volatility from daily data...")
    next_vol = compute_next_week_realized_vol(df.index)
    df["target_brent_vol_next_1w"] = next_vol

    # Next-week price direction: 3-class (up=1, flat=0, down=-1)
    # "flat" defined as absolute weekly return <= 0.5%
    next_return = df["brent_price"].shift(-1) / df["brent_price"] - 1
    df["target_brent_direction_next_1w"] = np.where(
        next_return > 0.005, 1, np.where(next_return < -0.005, -1, 0)
    )
    df.loc[df["target_brent_price_next_1w"].isna(), "target_brent_direction_next_1w"] = np.nan

    return df


def add_modality_flags(df: pd.DataFrame, rs_cols: list, ship_cols: list, text_cols: list) -> pd.DataFrame:
    """Add binary flags indicating which modalities are available per week."""
    df = df.copy()

    df["avail_remote_sensing"] = 0
    if rs_cols:
        rs_any = df[rs_cols].notna().any(axis=1)
        df["avail_remote_sensing"] = rs_any.astype(int)

    df["avail_shipping"] = 0
    if ship_cols:
        ship_any = df[ship_cols].notna().any(axis=1)
        df["avail_shipping"] = ship_any.astype(int)

    df["avail_text"] = 0
    if text_cols:
        text_any = df[text_cols].notna().any(axis=1)
        df["avail_text"] = text_any.astype(int)

    return df


def main():
    print("Loading weekly feature files...\n")

    market = load_weekly("weekly_time_index.csv")
    rs = load_weekly("weekly_m2_clean_features.csv")
    shipping = load_weekly("weekly_shipping_features.csv")
    text = load_weekly("weekly_text_features.csv")

    # Curate M1: keep only the new 10-variable set + retained availability flags
    if not market.empty:
        keep = [c for c in (M1_KEEP + MARKET_AVAIL_KEEP) if c in market.columns]
        dropped = [c for c in market.columns if c not in keep]
        market = market[keep]
        print(f"  M1 curated: kept {len(keep)} cols, dropped {len(dropped)} old/raw cols")
        missing = [c for c in M1_KEEP if c not in market.columns]
        if missing:
            raise SystemExit(f"ERROR: expected M1 columns missing from anchor: {missing}")

    # ------------------------------------------------------------------
    # Merge all modalities on the market time index
    # ------------------------------------------------------------------
    print("\nMerging features...")
    weekly = market.copy()

    rs_cols = []
    if not rs.empty:
        rs_cols = rs.columns.tolist()
        weekly = weekly.join(rs, how="left")
        print(f"  + remote sensing: {len(rs_cols)} columns")

    ship_cols = []
    if not shipping.empty:
        ship_cols = shipping.columns.tolist()
        weekly = weekly.join(shipping, how="left")
        print(f"  + shipping: {len(ship_cols)} columns")

    text_cols = []
    if not text.empty:
        text_cols = text.columns.tolist()
        weekly = weekly.join(text, how="left")
        print(f"  + text: {len(text_cols)} columns")

    # ------------------------------------------------------------------
    # Add targets and modality flags
    # ------------------------------------------------------------------
    weekly = add_target_variables(weekly)
    weekly = add_modality_flags(weekly, rs_cols, ship_cols, text_cols)

    # ------------------------------------------------------------------
    # Define feature groups for ablation experiments
    # ------------------------------------------------------------------
    feature_groups = {
        "M1_market_macro": [c for c in M1_KEEP if c in market.columns],
        "M2_rs_clean": rs_cols,
        "M3_add_shipping": ship_cols,
    }

    # Save feature group definitions
    import json
    groups_path = PROC_DIR / "feature_groups.json"
    with open(groups_path, "w") as f:
        json.dump(feature_groups, f, indent=2)
    print(f"\nFeature groups saved: {groups_path}")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    out_parquet = PROC_DIR / "weekly_features.parquet"
    out_csv = PROC_DIR / "weekly_features.csv"

    weekly.to_parquet(out_parquet)
    weekly.to_csv(out_csv)

    print(f"\n{'='*60}")
    print(f"Output (parquet): {out_parquet}")
    print(f"Output (csv):     {out_csv}")
    print(f"Shape:  {weekly.shape}")
    print(f"Period: {weekly.index.min()} ~ {weekly.index.max()}")
    print(f"Weeks:  {len(weekly)}")

    print(f"\nFeature group sizes:")
    for name, cols in feature_groups.items():
        print(f"  {name:25s}: {len(cols):3d} features")

    print(f"\nTarget variables:")
    for col in weekly.columns:
        if col.startswith("target_"):
            non_null = weekly[col].notna().sum()
            print(f"  {col}: {non_null}/{len(weekly)} non-null")

    print(f"\nModality availability:")
    for col in weekly.columns:
        if col.startswith("avail_"):
            pct = weekly[col].mean() * 100
            print(f"  {col}: {pct:.1f}% of weeks")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
