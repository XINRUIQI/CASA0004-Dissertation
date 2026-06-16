"""
Merge CV-enhanced features (EMODnet density + FRT fill level) into
the unified weekly feature matrix.

Input:
  03_data/processed/emodnet_density_monthly.csv
  03_data/processed/frt_fill_level_monthly.csv
  03_data/processed/weekly_features.csv

Output:
  03_data/processed/weekly_features.csv  (updated in place)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
PROC = PROJECT / "03_data" / "processed"
WEEKLY_CSV = PROC / "weekly_features.csv"


def merge_monthly_to_weekly(weekly: pd.DataFrame, monthly_csv: Path,
                            prefix: str, cols_to_keep: list[str] | None = None
                            ) -> tuple[pd.DataFrame, list[str]]:
    """Forward-fill monthly data and join to weekly index."""
    if not monthly_csv.exists():
        print(f"  [skip] {monthly_csv.name} not found")
        return weekly, []

    monthly = pd.read_csv(monthly_csv, index_col=0, parse_dates=True)
    monthly = monthly.sort_index()

    if cols_to_keep:
        monthly = monthly[[c for c in cols_to_keep if c in monthly.columns]]

    weekly_resampled = monthly.resample("W-FRI").ffill()

    new_cols = [c for c in weekly_resampled.columns if c not in weekly.columns]
    if not new_cols:
        existing = [c for c in weekly_resampled.columns if c in weekly.columns]
        print(f"  [update] {monthly_csv.name}: updating {len(existing)} existing columns")
        weekly.update(weekly_resampled[existing])
        return weekly, existing

    weekly = weekly.join(weekly_resampled[new_cols], how="left")
    print(f"  [added] {monthly_csv.name}: {len(new_cols)} new columns "
          f"({weekly_resampled.index.min().date()} ~ {weekly_resampled.index.max().date()})")
    return weekly, new_cols


def main():
    print("Loading weekly features...")
    weekly = pd.read_csv(WEEKLY_CSV, index_col=0, parse_dates=True)
    print(f"  Original shape: {weekly.shape}")

    all_new_cols = []

    emodnet_keep = [
        "emodnet_mean_rotterdam", "emodnet_max_rotterdam",
        "emodnet_mean_suez", "emodnet_max_suez",
    ]
    weekly, new = merge_monthly_to_weekly(
        weekly, PROC / "emodnet_density_monthly.csv", "emodnet", emodnet_keep)
    all_new_cols.extend(new)

    frt_keep = ["fill_level", "n_tanks"]
    weekly, new = merge_monthly_to_weekly(
        weekly, PROC / "frt_fill_level_monthly.csv", "frt", frt_keep)
    for c in new:
        if "fill_level" in c:
            weekly.rename(columns={c: "frt_fill_level_fujairah"}, inplace=True)
            all_new_cols.append("frt_fill_level_fujairah")
        elif "n_tanks" in c:
            weekly.rename(columns={c: "frt_n_tanks_fujairah"}, inplace=True)
            all_new_cols.append("frt_n_tanks_fujairah")
        else:
            all_new_cols.append(c)

    weekly.to_csv(WEEKLY_CSV)
    print(f"\n[saved] {WEEKLY_CSV}")
    print(f"  Updated shape: {weekly.shape}")

    print(f"\nNew CV-enhanced features:")
    for c in all_new_cols:
        if c in weekly.columns:
            nn = weekly[c].notna().sum()
            print(f"  {c:40s}  non-null={nn}/{len(weekly)}")


if __name__ == "__main__":
    main()
