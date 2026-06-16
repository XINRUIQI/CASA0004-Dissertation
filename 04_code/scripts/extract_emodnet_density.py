"""
Extract vessel density statistics from EMODnet monthly GeoTIFF rasters
at Rotterdam (P001) and Suez Canal entrance.

Input:  03_data/raw/05_shipping/emodnet_vessel_density_monthly_2017-2025/*.tif
Output: 03_data/processed/emodnet_density_monthly.csv
"""
from __future__ import annotations
from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

PROJECT = Path(__file__).resolve().parents[2]
TIF_DIR = PROJECT / "03_data" / "raw" / "05_shipping" / "emodnet_vessel_density_monthly_2017-2025"
OUT_CSV = PROJECT / "03_data" / "processed" / "emodnet_density_monthly.csv"

SITES = {
    "rotterdam": {"lon": 4.145, "lat": 51.950, "radius_m": 15_000},
    "suez":      {"lon": 32.37, "lat": 30.58,  "radius_m": 20_000},
}

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def extract_patch_stats(tif_path: Path, cx_3035: float, cy_3035: float,
                        radius_m: float) -> dict:
    with rasterio.open(tif_path) as src:
        left = cx_3035 - radius_m
        right = cx_3035 + radius_m
        bottom = cy_3035 - radius_m
        top = cy_3035 + radius_m

        try:
            window = from_bounds(left, bottom, right, top, src.transform)
        except Exception:
            return {}

        data = src.read(1, window=window)

    valid = data[data > 0]
    if len(valid) == 0:
        return {"mean": np.nan, "max": np.nan, "std": np.nan, "sum": np.nan,
                "n_valid_pixels": 0}

    return {
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "sum": float(np.sum(valid)),
        "n_valid_pixels": int(len(valid)),
    }


def parse_date_from_filename(fname: str) -> datetime | None:
    m = re.search(r"(\d{8})", fname)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    return None


def main():
    tif_files = sorted(TIF_DIR.glob("vesseldensity_*.tif"))
    print(f"Found {len(tif_files)} EMODnet TIF files")

    site_coords_3035 = {}
    for site_name, info in SITES.items():
        cx, cy = transformer.transform(info["lon"], info["lat"])
        site_coords_3035[site_name] = (cx, cy, info["radius_m"])
        print(f"  {site_name}: ({info['lon']}, {info['lat']}) -> EPSG:3035 ({cx:.0f}, {cy:.0f})")

    rows = []
    for tif_path in tif_files:
        date = parse_date_from_filename(tif_path.name)
        if date is None:
            continue

        for site_name, (cx, cy, radius) in site_coords_3035.items():
            stats = extract_patch_stats(tif_path, cx, cy, radius)
            if not stats:
                continue
            row = {"date": date, "site": site_name}
            row.update(stats)
            rows.append(row)

        if len(rows) % 20 == 0:
            print(f"  processed {tif_path.name}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data extracted!")
        return

    wide = df.pivot(index="date", columns="site")
    wide.columns = [f"emodnet_{stat}_{site}" for stat, site in wide.columns]
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    wide.to_csv(OUT_CSV)
    print(f"\n[saved] {OUT_CSV}")
    print(f"Shape: {wide.shape}")
    print(f"Period: {wide.index.min().date()} ~ {wide.index.max().date()}")
    print(f"\nColumns:")
    for c in wide.columns:
        nn = wide[c].notna().sum()
        print(f"  {c}: {nn} non-null")


if __name__ == "__main__":
    main()
