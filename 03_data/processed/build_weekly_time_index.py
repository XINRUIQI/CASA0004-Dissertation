"""
Build the unified weekly time index for Brent oil price forecasting.

Reads raw EIA (.xls) and FRED/Yahoo (.csv) data,
aligns everything to a Friday-ending weekly frequency over 2006-01 ~ 2025-12,
and outputs a single processed CSV as the anchor for all downstream features.

Usage:
    python build_weekly_time_index.py

Output:
    03_data/processed/weekly_time_index.csv
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "03_data" / "raw" / "01_market_financial"
EIA_WEEKLY_DIR = RAW_DIR / "1C EIA Weekly Petroleum Status Report"
FRED_DIR = RAW_DIR / "1D Macro-financial control variables"
OUT_DIR = PROJECT_ROOT / "03_data" / "processed"

STUDY_START = "2006-01-01"
STUDY_END = "2025-12-31"


def read_eia_xls(filepath: str | Path, value_name: str) -> pd.DataFrame:
    """Read an EIA .xls file (sheet 'Data 1', rows 3+ are date|value)."""
    df = pd.read_excel(filepath, sheet_name="Data 1", header=None, skiprows=3)
    df.columns = ["date", value_name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.set_index("date").sort_index()


def read_fred_csv(filepath: str | Path, value_name: str) -> pd.DataFrame:
    """Read a FRED CSV (columns: observation_date, VALUE)."""
    df = pd.read_csv(filepath)
    col_value = [c for c in df.columns if c != "observation_date"][0]
    df = df.rename(columns={"observation_date": "date", col_value: value_name})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.set_index("date").sort_index()


def resample_daily_to_weekly(df: pd.DataFrame, method: str = "last") -> pd.DataFrame:
    """Resample daily data to Friday-ending weekly frequency."""
    if method == "last":
        return df.resample("W-FRI").last()
    elif method == "mean":
        return df.resample("W-FRI").mean()
    else:
        raise ValueError(f"Unknown method: {method}")


def align_weekly_to_friday(df: pd.DataFrame) -> pd.DataFrame:
    """Align weekly EIA data (various weekdays) to nearest Friday."""
    df = df.copy()
    df.index = df.index + pd.to_timedelta((4 - df.index.dayofweek) % 7, unit="D")
    return df[~df.index.duplicated(keep="last")]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Target variable: Brent daily price → weekly
    # ------------------------------------------------------------------
    print("Reading Brent daily price...")
    brent_daily = read_eia_xls(
        RAW_DIR / "1A Oil Price" / "EIA_brent_spot_price_daily205051987_18052026_raw.xls",
        "brent_price",
    )

    print("Reading WTI daily price...")
    wti_daily = read_eia_xls(
        RAW_DIR / "1B Benchmark comparison" / "EIA_WTI_cushing_crude_price_daily_02011986-18052026.xls",
        "wti_price",
    )

    brent_weekly = resample_daily_to_weekly(brent_daily, method="last")
    wti_weekly = resample_daily_to_weekly(wti_daily, method="last")

    price_weekly = brent_weekly.join(wti_weekly, how="outer")
    price_weekly["brent_wti_spread"] = price_weekly["brent_price"] - price_weekly["wti_price"]
    price_weekly["brent_return_pct"] = price_weekly["brent_price"].pct_change() * 100
    price_weekly["wti_return_pct"] = price_weekly["wti_price"].pct_change() * 100
    price_weekly["brent_log_return"] = np.log(price_weekly["brent_price"] / price_weekly["brent_price"].shift(1))
    price_weekly["brent_direction"] = (price_weekly["brent_return_pct"] > 0).astype(int)
    price_weekly["brent_vol_4w"] = price_weekly["brent_log_return"].rolling(4).std()
    price_weekly["brent_vol_12w"] = price_weekly["brent_log_return"].rolling(12).std()

    print(f"  Price weekly: {price_weekly.shape}")

    # ------------------------------------------------------------------
    # 2. EIA Weekly Petroleum Status Report (10 series)
    # ------------------------------------------------------------------
    eia_weekly_files = {
        "crude_stocks_excl_spr": "EIA_commercial_crude_stocks_weekly_20081982-15052026.xls",
        "cushing_stocks": "EIA_cushing_crude_stocks_weekly_29042004-15052026.xls",
        "crude_production": "EIA_crude_production_weekly_07011983-15052026.xls",
        "crude_imports": "EIA_crude_imports_weekly_05011990-15052026.xls",
        "crude_exports": "EIA_crude_exports_weekly_08021991-15052026.xls",
        "refinery_crude_input": "EIA_refinery_crude_input_weekly_20081982-15052026.xls",
        "refinery_utilisation": "EIA_refinery_utilisation_weekly_02111990-15052026.xls",
        "gasoline_supplied": "EIA_gasoline_supplied_weekly_08021991-15052026.xls",
        "distillate_supplied": "EIA_distillate_supplied_weekly_08021991-15052026.xls",
        "jet_fuel_supplied": "EIA_jet_fuel_supplied_weekly-08021991-15052026.xls",
    }

    eia_frames = []
    for col_name, filename in eia_weekly_files.items():
        filepath = EIA_WEEKLY_DIR / filename
        print(f"Reading EIA weekly: {col_name}...")
        df = read_eia_xls(filepath, col_name)
        df = align_weekly_to_friday(df)
        eia_frames.append(df)

    eia_weekly = eia_frames[0]
    for df in eia_frames[1:]:
        eia_weekly = eia_weekly.join(df, how="outer")

    eia_weekly["crude_stocks_change"] = eia_weekly["crude_stocks_excl_spr"].diff()
    eia_weekly["cushing_stocks_change"] = eia_weekly["cushing_stocks"].diff()
    eia_weekly["net_crude_trade"] = eia_weekly["crude_imports"] - eia_weekly["crude_exports"]

    print(f"  EIA weekly: {eia_weekly.shape}")

    # ------------------------------------------------------------------
    # 3. FRED macro-financial daily → weekly
    # ------------------------------------------------------------------
    yahoo_sp500 = FRED_DIR / "Yahoo_sp500_daily_20060101_20251231.csv"
    if yahoo_sp500.exists():
        sp500_file = "Yahoo_sp500_daily_20060101_20251231.csv"
    else:
        sp500_file = "FRED_sp500_SP500_daily_23052016_22052026.csv"

    fred_files = {
        "sp500": sp500_file,
        "vix": "FRED_vix_close_VIXCLS_daily_02011990_21052026.csv",
        "dollar_index": "FRED_dollar_index_DTWEXBGS_daily_02012006_15052026.csv",
        "treasury_10y": "FRED_10year_treasury_yield_DGS10_daily_02011962_21052026.csv",
        "fed_funds_rate": "FRED_effective_federal_funds_rate_DFF_daily_01071954_21052026.csv",
    }

    fred_frames = []
    for col_name, filename in fred_files.items():
        filepath = FRED_DIR / filename
        print(f"Reading FRED: {col_name}...")
        df = read_fred_csv(filepath, col_name)
        df_weekly = resample_daily_to_weekly(df, method="last")
        fred_frames.append(df_weekly)

    fred_weekly = fred_frames[0]
    for df in fred_frames[1:]:
        fred_weekly = fred_weekly.join(df, how="outer")

    fred_weekly["sp500_return_pct"] = fred_weekly["sp500"].pct_change() * 100

    print(f"  FRED weekly: {fred_weekly.shape}")

    # ------------------------------------------------------------------
    # 4. Merge all into unified weekly index
    # ------------------------------------------------------------------
    print("\nMerging all sources...")

    weekly = price_weekly.copy()
    weekly = weekly.join(eia_weekly, how="outer")
    weekly = weekly.join(fred_weekly, how="outer")

    weekly = weekly.loc[STUDY_START:STUDY_END]

    weekly.index.name = "week_ending_friday"

    # ------------------------------------------------------------------
    # 5. Add modality availability flags
    # ------------------------------------------------------------------
    weekly["avail_market"] = weekly["brent_price"].notna().astype(int)
    weekly["avail_eia_weekly"] = weekly["crude_stocks_excl_spr"].notna().astype(int)
    weekly["avail_sp500"] = weekly["sp500"].notna().astype(int)
    weekly["avail_dollar_index"] = weekly["dollar_index"].notna().astype(int)

    # ------------------------------------------------------------------
    # 6. Summary and output
    # ------------------------------------------------------------------
    out_path = OUT_DIR / "weekly_time_index.csv"
    weekly.to_csv(out_path)

    print(f"\n{'='*60}")
    print(f"Output: {out_path}")
    print(f"Shape:  {weekly.shape}")
    print(f"Period: {weekly.index.min()} ~ {weekly.index.max()}")
    print(f"Weeks:  {len(weekly)}")
    print(f"\nColumns ({len(weekly.columns)}):")
    for col in weekly.columns:
        non_null = weekly[col].notna().sum()
        pct = non_null / len(weekly) * 100
        print(f"  {col:35s}  {non_null:5d} / {len(weekly):5d}  ({pct:5.1f}%)")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
