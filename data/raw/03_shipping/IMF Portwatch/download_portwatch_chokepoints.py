"""
Download IMF PortWatch chokepoint transit data.

PortWatch provides daily transit calls and capacity for 29 critical maritime
passages from 2019-01-01 onward. This script fetches the 6 oil-relevant
chokepoints and outputs a daily CSV.

Data source: https://portwatch.imf.org/
ArcGIS FeatureServer: services9.arcgis.com (org: weJ1QsnbMYJlCHdG)

Usage:
    python download_portwatch_chokepoints.py

Output:
    data/raw/05_shipping/portwatch_chokepoints_daily.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "05_shipping"

OIL_CHOKEPOINTS = {
    "chokepoint6": "Strait of Hormuz",
    "chokepoint1": "Suez Canal",
    "chokepoint5": "Malacca Strait",
    "chokepoint2": "Panama Canal",
    "chokepoint4": "Bab el-Mandeb Strait",
    "chokepoint7": "Cape of Good Hope",
}

FEATURE_SERVICE_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0/query"
)

PAGE_SIZE = 1000


def fetch_all_records(where_clause: str) -> list[dict]:
    """Fetch all records matching a WHERE clause, paginating as needed."""
    all_records: list[dict] = []
    offset = 0

    while True:
        params = {
            "where": where_clause,
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json",
            "resultRecordCount": PAGE_SIZE,
            "resultOffset": offset,
        }
        url = f"{FEATURE_SERVICE_URL}?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())

        features = data.get("features", [])
        if not features:
            break

        all_records.extend(f["attributes"] for f in features)
        offset += len(features)

        if len(features) < PAGE_SIZE:
            break

    return all_records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    port_ids = "','".join(OIL_CHOKEPOINTS.keys())
    where = f"portid IN ('{port_ids}')"
    print(f"Fetching 6 oil chokepoints from PortWatch...")
    print(f"  WHERE: {where}")

    records = fetch_all_records(where)
    print(f"  Retrieved {len(records)} total records")

    if not records:
        print("\nNo data retrieved. Check network or API status.")
        return

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["portname", "date"]).reset_index(drop=True)

    out_path = OUT_DIR / "portwatch_chokepoints_daily.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Chokepoints: {df['portname'].nunique()}")
    for name in df["portname"].unique():
        n = len(df[df["portname"] == name])
        print(f"    {name}: {n} days")


if __name__ == "__main__":
    main()
