"""
Aggregate daily GDELT oil-disruption and transport-disruption features
to Friday-ending weekly frequency for the M2 (text/NLP) modality.

Input:
    03_data/raw/03_news/GDET/gdelt_oil_disruption_daily_calibrated_20060101-20251231.csv
    03_data/raw/03_news/GDET/gdelt_transport_disruption_daily_calibrated_20060101-20251231.csv

Output:
    03_data/processed/weekly_text_features.csv

Usage:
    python aggregate_gdelt_to_weekly.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "03_data" / "raw" / "03_news" / "GDET"
OUT_DIR = PROJECT_ROOT / "03_data" / "processed"

STUDY_START = "2006-01-01"
STUDY_END = "2025-12-31"

SUM_COLS_GEO = [
    "gdelt_oil_geo_event_count",
    "gdelt_oil_geo_total_mentions",
    "gdelt_negative_event_count",
    "gdelt_conflict_event_count",
    "gdelt_sanction_event_count",
    "gdelt_key_oil_region_event_count",
]

MEAN_COLS_GEO = [
    "gdelt_oil_geo_avg_tone",
    "gdelt_oil_geo_avg_goldstein",
]

SUM_COLS_TD = [
    "gdelt_transport_disruption_event_count",
    "gdelt_transport_disruption_total_mentions",
    "gdelt_transport_negative_event_count",
    "gdelt_transport_unrest_conflict_event_count",
    "gdelt_transport_sanction_event_count",
    "gdelt_chokepoint_event_count",
]

MEAN_COLS_TD = [
    "gdelt_transport_disruption_avg_tone",
    "gdelt_transport_disruption_avg_goldstein",
]


def load_gdelt_daily(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df.drop(columns=["gdelt_version"], errors="ignore")
    df = df.loc[STUDY_START:STUDY_END]
    return df


def resample_to_weekly(df: pd.DataFrame, sum_cols: list, mean_cols: list) -> pd.DataFrame:
    """Resample daily features to Friday-ending weekly, using sum for counts
    and mean for tones/scores."""
    agg_dict = {}
    for c in sum_cols:
        if c in df.columns:
            agg_dict[c] = "sum"
    for c in mean_cols:
        if c in df.columns:
            agg_dict[c] = "mean"

    weekly = df.resample("W-FRI").agg(agg_dict)
    return weekly


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages, week-over-week changes, and ratios."""
    df = df.copy()

    # ---- Oil-disruption derived features ----
    if "gdelt_oil_geo_event_count" in df.columns:
        ec = "gdelt_oil_geo_event_count"
        df["gdelt_oil_geo_event_count_4w_ma"] = df[ec].rolling(4, min_periods=1).mean()
        df["gdelt_oil_geo_event_count_wow_pct"] = df[ec].pct_change() * 100

        # Negative / conflict share of total events
        if "gdelt_negative_event_count" in df.columns:
            df["gdelt_oil_geo_negative_share"] = (
                df["gdelt_negative_event_count"] / df[ec].replace(0, np.nan)
            )
        if "gdelt_conflict_event_count" in df.columns:
            df["gdelt_oil_geo_conflict_share"] = (
                df["gdelt_conflict_event_count"] / df[ec].replace(0, np.nan)
            )

    # Tone rolling average
    if "gdelt_oil_geo_avg_tone" in df.columns:
        df["gdelt_oil_geo_avg_tone_4w_ma"] = (
            df["gdelt_oil_geo_avg_tone"].rolling(4, min_periods=1).mean()
        )

    # ---- Transport-disruption derived features ----
    if "gdelt_transport_disruption_event_count" in df.columns:
        td_ec = "gdelt_transport_disruption_event_count"
        df["gdelt_transport_event_count_4w_ma"] = df[td_ec].rolling(4, min_periods=1).mean()
        df["gdelt_transport_event_count_wow_pct"] = df[td_ec].pct_change() * 100

        if "gdelt_transport_negative_event_count" in df.columns:
            df["gdelt_transport_negative_share"] = (
                df["gdelt_transport_negative_event_count"] / df[td_ec].replace(0, np.nan)
            )

    if "gdelt_transport_disruption_avg_tone" in df.columns:
        df["gdelt_transport_avg_tone_4w_ma"] = (
            df["gdelt_transport_disruption_avg_tone"].rolling(4, min_periods=1).mean()
        )

    # ---- Cross-domain composite ----
    geo_ec = df.get("gdelt_oil_geo_event_count")
    td_ec = df.get("gdelt_transport_disruption_event_count")
    if geo_ec is not None and td_ec is not None:
        df["gdelt_combined_event_count"] = geo_ec + td_ec

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GDELT daily features...\n")

    geo = load_gdelt_daily("gdelt_oil_disruption_daily_calibrated_20060101-20251231.csv")
    print(f"  Oil disruption: {geo.shape} ({geo.index.min().date()} ~ {geo.index.max().date()})")

    td = load_gdelt_daily("gdelt_transport_disruption_daily_calibrated_20060101-20251231.csv")
    print(f"  Transport disruption: {td.shape} ({td.index.min().date()} ~ {td.index.max().date()})")

    print("\nResampling to weekly (Friday-ending)...")
    geo_weekly = resample_to_weekly(geo, SUM_COLS_GEO, MEAN_COLS_GEO)
    td_weekly = resample_to_weekly(td, SUM_COLS_TD, MEAN_COLS_TD)

    weekly = geo_weekly.join(td_weekly, how="outer")
    print(f"  Merged weekly: {weekly.shape}")

    print("Adding derived features...")
    weekly = add_derived_features(weekly)
    print(f"  Final: {weekly.shape}")

    weekly.index.name = "week_ending_friday"

    out_path = OUT_DIR / "weekly_text_features.csv"
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
        print(f"  {col:50s}  {non_null:5d} / {len(weekly):5d}  ({pct:5.1f}%)")
    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
