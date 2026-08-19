"""
Download Global Fishing Watch AIS Vessel Presence data for oil chokepoints.

Uses the GFW 4Wings Report API to fetch monthly vessel presence hours
around 6 oil-critical chokepoints, covering 2012-01 to 2025-12.
Merges with any existing data to avoid re-downloading.

GFW classifies vessels as FISHING/CARGO/CARRIER/PASSENGER/BUNKER/OTHER/etc.
Oil tankers fall under OTHER or BUNKER (GFW is fishing-focused), so we
download ALL vessel types and aggregate total presence as a maritime
activity proxy. The output includes per-type breakdowns for analysis.

Prerequisites:
    1. Register at https://globalfishingwatch.org/our-apis/
    2. Set env var GFW_API_TOKEN or write token to .gfw_token file

Output:
    data/raw/03_shipping/GFW/gfw_chokepoint_vessel_presence_monthly.csv
"""

from __future__ import annotations

import json
import os
import ssl
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUT_DIR.parents[2]
TOKEN_FILE = OUT_DIR / ".gfw_token"

BASE_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-presence:latest"

CHOKEPOINT_POLYGONS = {
    "Strait of Hormuz": [[55.5, 25.5], [57.5, 25.5], [57.5, 27.5], [55.5, 27.5], [55.5, 25.5]],
    "Suez Canal": [[32.0, 29.5], [33.5, 29.5], [33.5, 31.5], [32.0, 31.5], [32.0, 29.5]],
    "Malacca Strait": [[99.5, 0.5], [104.5, 0.5], [104.5, 4.5], [99.5, 4.5], [99.5, 0.5]],
    "Bab el-Mandeb": [[42.5, 12.0], [44.0, 12.0], [44.0, 13.5], [42.5, 13.5], [42.5, 12.0]],
    "Cape of Good Hope": [[17.0, -35.5], [20.5, -35.5], [20.5, -33.0], [17.0, -33.0], [17.0, -35.5]],
    "Panama Canal": [[-80.5, 8.5], [-79.0, 8.5], [-79.0, 10.0], [-80.5, 10.0], [-80.5, 8.5]],
}

YEARS = list(range(2012, 2026))


def get_token() -> str:
    token = os.environ.get("GFW_API_TOKEN", "").strip()
    if token:
        return token
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    raise RuntimeError(
        "GFW API token not found.\n"
        "  export GFW_API_TOKEN='your_token'\n"
        f"  or: echo 'your_token' > {TOKEN_FILE}"
    )


def fetch_presence(
    token: str, geojson_coords: list, date_start: str, date_end: str
) -> list[dict]:
    """Fetch monthly vessel presence grouped by VESSEL_ID for a polygon."""
    ctx = ssl.create_default_context()

    url = (
        f"{BASE_URL}"
        f"?spatial-resolution=LOW"
        f"&temporal-resolution=MONTHLY"
        f"&spatial-aggregation=true"
        f"&group-by=VESSEL_ID"
        f"&datasets[0]={DATASET}"
        f"&date-range={date_start},{date_end}"
        f"&format=JSON"
    )

    geojson = {"type": "Polygon", "coordinates": [geojson_coords]}
    body = json.dumps({"geojson": geojson}).encode("utf-8")

    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        },
        method="POST",
    )

    with urlopen(req, timeout=120, context=ctx) as resp:
        data = json.loads(resp.read().decode())

    entries = data.get("entries", [])
    if not entries:
        return []

    dataset_key = list(entries[0].keys())[0]
    return entries[0].get(dataset_key, [])


def aggregate_records(records: list[dict], chokepoint: str) -> pd.DataFrame:
    """Aggregate vessel-level records to monthly chokepoint-level features."""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    agg = (
        df.groupby("date")
        .agg(
            total_hours=("hours", "sum"),
            total_vessels=("vesselId", "nunique"),
            cargo_hours=("hours", lambda x: x[df.loc[x.index, "vesselType"] == "CARGO"].sum()),
            bunker_hours=("hours", lambda x: x[df.loc[x.index, "vesselType"] == "BUNKER"].sum()),
            other_hours=("hours", lambda x: x[df.loc[x.index, "vesselType"] == "OTHER"].sum()),
            fishing_hours=("hours", lambda x: x[df.loc[x.index, "vesselType"] == "FISHING"].sum()),
            passenger_hours=("hours", lambda x: x[df.loc[x.index, "vesselType"] == "PASSENGER"].sum()),
            cargo_vessels=("vesselType", lambda x: (x == "CARGO").sum()),
            bunker_vessels=("vesselType", lambda x: (x == "BUNKER").sum()),
            other_vessels=("vesselType", lambda x: (x == "OTHER").sum()),
        )
        .reset_index()
    )
    agg["chokepoint"] = chokepoint
    return agg


def load_existing() -> pd.DataFrame:
    """Load existing CSV to skip already-downloaded year/chokepoint combos."""
    out_path = OUT_DIR / "gfw_chokepoint_vessel_presence_monthly.csv"
    if not out_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(out_path)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    token = get_token()
    print(f"GFW API token loaded (length={len(token)})")

    existing = load_existing()
    existing_keys: set[tuple[str, int]] = set()
    if not existing.empty:
        existing["_year"] = pd.to_datetime(existing["date"], format="%Y-%m").dt.year
        existing_keys = set(zip(existing["chokepoint"], existing["_year"]))
        existing = existing.drop(columns=["_year"])
        print(f"  Existing data: {len(existing)} rows, skipping already-downloaded years")

    all_frames = [existing] if not existing.empty else []
    new_count = 0

    for cp_name, coords in CHOKEPOINT_POLYGONS.items():
        for year in YEARS:
            if (cp_name, year) in existing_keys:
                continue

            date_start = f"{year}-01-01"
            date_end = f"{year + 1}-01-01"
            label = f"{cp_name} [{year}]"
            print(f"  Fetching {label}...", end=" ", flush=True)

            try:
                records = fetch_presence(token, coords, date_start, date_end)
                if records:
                    agg = aggregate_records(records, cp_name)
                    all_frames.append(agg)
                    new_count += len(agg)
                    print(f"{len(records)} vessel-records -> {len(agg)} monthly rows")
                else:
                    print("no data")
            except HTTPError as e:
                err_body = e.read().decode()[:200] if hasattr(e, "read") else ""
                print(f"HTTP {e.code}: {err_body}")
            except Exception as e:
                print(f"Error: {e}")

            time.sleep(1.5)

    if not all_frames:
        print("\nNo data retrieved.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["chokepoint", "date"]).reset_index(drop=True)

    out_path = OUT_DIR / "gfw_chokepoint_vessel_presence_monthly.csv"
    combined.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"  Total rows: {len(combined)} (new: {new_count})")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Chokepoints: {combined['chokepoint'].nunique()}")
    for name in sorted(combined["chokepoint"].unique()):
        sub = combined[combined["chokepoint"] == name]
        print(f"    {name}: {len(sub)} months, {sub['total_hours'].sum():.0f} total hours")


if __name__ == "__main__":
    main()
