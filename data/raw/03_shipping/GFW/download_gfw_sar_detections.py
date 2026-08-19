"""
Download GFW Sentinel-1 SAR vessel detections for the 6 oil chokepoints and the
11 oil-infrastructure AOIs, split into AIS-matched vs. unmatched ("dark")
detections. Dark-vessel counts (unmatched SAR detections) are the Stage-2
edge/node feature motivated by P057 (Paolo et al. 2024, Nature) and P061 (GFW
SAR method/data doc).

API (confirmed 2026-07-03 against api-doc.globalfishingwatch.org):
  POST https://gateway.api.globalfishingwatch.org/v3/4wings/report
  query: datasets[0]=public-global-sar-presence:latest,
         temporal-resolution=MONTHLY, spatial-resolution=LOW,
         date-range=<start>,<end>, format=JSON,
         filters[0]=matched='false'   (dark)  /  matched='true' (matched)
  body : {"geojson": <Polygon>}
Unit = detections (counts), coverage 2017 -> ~5 days ago.

For each region we issue 3 requests: total (no filter), dark (matched='false'),
matched (matched='true'); response is a list of grid-cell rows which we sum by
month.

Prerequisites:
    GFW API token in env GFW_API_TOKEN or in ./.gfw_token

Usage:
    python download_gfw_sar_detections.py                 # 2019-2025
    python download_gfw_sar_detections.py --start 2017 --end 2025 --buffer-deg 0.3

Output (same directory as this script):
    gfw_sar_detections_monthly.csv
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUT_DIR.parents[3]
TOKEN_FILE = OUT_DIR / ".gfw_token"
AOI_CSV = (PROJECT_ROOT / "data" / "raw" / "02_sentinel2"
           / "aoi_oil_infrastructure.csv")

BASE_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-sar-presence:latest"

# Same 6 oil chokepoint boxes as download_gfw_vessel_presence.py
CHOKEPOINT_POLYGONS = {
    "Strait of Hormuz": [[55.5, 25.5], [57.5, 25.5], [57.5, 27.5], [55.5, 27.5], [55.5, 25.5]],
    "Suez Canal": [[32.0, 29.5], [33.5, 29.5], [33.5, 31.5], [32.0, 31.5], [32.0, 29.5]],
    "Malacca Strait": [[99.5, 0.5], [104.5, 0.5], [104.5, 4.5], [99.5, 4.5], [99.5, 0.5]],
    "Bab el-Mandeb": [[42.5, 12.0], [44.0, 12.0], [44.0, 13.5], [42.5, 13.5], [42.5, 12.0]],
    "Cape of Good Hope": [[17.0, -35.5], [20.5, -35.5], [20.5, -33.0], [17.0, -33.0], [17.0, -35.5]],
    "Panama Canal": [[-80.5, 8.5], [-79.0, 8.5], [-79.0, 10.0], [-80.5, 10.0], [-80.5, 8.5]],
}

FILTER_VARIANTS = {
    "total": None,
    "dark": "matched='false'",
    "matched": "matched='true'",
}


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


def load_aois() -> list[tuple[str, str, list[list[float]]]]:
    """Return (region_id, region_name, polygon) for the 11 AOIs."""
    df = pd.read_csv(AOI_CSV)
    out = []
    for _, r in df.iterrows():
        lon, lat = float(r["lon"]), float(r["lat"])
        out.append((r["site_id"], r["site_short"], (lon, lat)))
    return out


def square(lon: float, lat: float, h: float) -> list[list[float]]:
    return [[lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h],
            [lon - h, lat + h], [lon - h, lat - h]]


def fetch_sar(token: str, coords: list[list[float]], date_start: str,
              date_end: str, filt: str | None) -> pd.DataFrame:
    """Fetch SAR detections for one polygon; sum grid cells by month."""
    ctx = ssl.create_default_context()
    url = (
        f"{BASE_URL}"
        f"?spatial-resolution=LOW"
        f"&temporal-resolution=MONTHLY"
        f"&datasets[0]={DATASET}"
        f"&date-range={date_start},{date_end}"
        f"&format=JSON"
    )
    if filt is not None:
        url += f"&filters[0]={quote(filt)}"

    geojson = {"type": "Polygon", "coordinates": [coords]}
    body = json.dumps({"geojson": geojson}).encode("utf-8")
    req = Request(url, data=body, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    }, method="POST")

    for attempt in range(4):
        try:
            with urlopen(req, timeout=180, context=ctx) as resp:
                data = json.loads(resp.read().decode())
            break
        except HTTPError as e:
            if e.code in (429,) or e.code >= 500:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    else:
        raise RuntimeError("repeated HTTP errors")

    entries = data.get("entries", [])
    if not entries:
        return pd.DataFrame(columns=["date", "detections"])
    key = list(entries[0].keys())[0]
    rows = entries[0].get(key, [])
    if not rows:
        return pd.DataFrame(columns=["date", "detections"])
    df = pd.DataFrame(rows)
    if "detections" not in df.columns:
        return pd.DataFrame(columns=["date", "detections"])
    # Normalise the monthly bucket to YYYY-MM and sum all cells in the region.
    df["date"] = df["date"].astype(str).str.slice(0, 7)
    return df.groupby("date", as_index=False)["detections"].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--buffer-deg", type=float, default=0.25)
    args = ap.parse_args()

    token = get_token()
    print(f"GFW token loaded (length={len(token)})")
    # SAR /4wings/report allows a maximum of 366 days per request, so we pull
    # one calendar year at a time and concatenate.
    years = list(range(args.start, args.end + 1))

    regions: list[tuple[str, str, str, list[list[float]]]] = []
    for name, coords in CHOKEPOINT_POLYGONS.items():
        regions.append(("chokepoint", name, name, coords))
    for site_id, short, (lon, lat) in load_aois():
        regions.append(("aoi", site_id, short, square(lon, lat, args.buffer_deg)))

    print(f"Fetching SAR detections for {len(regions)} regions "
          f"({args.start}-{args.end}), 3 variants each...")

    merged: dict[tuple[str, str, str, str], dict] = {}
    for rtype, rid, rname, coords in regions:
        per_variant = {}
        for vname, filt in FILTER_VARIANTS.items():
            parts, ok = [], True
            for y in years:
                try:
                    parts.append(fetch_sar(token, coords, f"{y}-01-01",
                                           f"{y + 1}-01-01", filt))
                except Exception as e:
                    ok = False
                    print(f"  {rtype:10} {rid:18} {vname:8} [{y}]: ERROR {e}")
                time.sleep(0.3)
            if parts:
                df = (pd.concat(parts, ignore_index=True)
                      .groupby("date", as_index=False)["detections"].sum())
            else:
                df = pd.DataFrame(columns=["date", "detections"])
            per_variant[vname] = df
            n = int(df["detections"].sum()) if len(df) else 0
            print(f"  {rtype:10} {rid:18} {vname:8}: {n} detections over "
                  f"{len(df)} months{'' if ok else ' (partial)'}", flush=True)

        months = set()
        for df in per_variant.values():
            months.update(df["date"].tolist())
        for m in sorted(months):
            key = (rtype, rid, rname, m)
            merged.setdefault(key, {})
            for vname, df in per_variant.items():
                val = df.loc[df["date"] == m, "detections"]
                merged[key][vname] = int(val.iloc[0]) if len(val) else 0

    if not merged:
        print("\nNo SAR data retrieved.")
        return

    rows = []
    for (rtype, rid, rname, m), vals in merged.items():
        rows.append({
            "region_type": rtype,
            "region_id": rid,
            "region_name": rname,
            "month": m,
            "detections_total": vals.get("total", 0),
            "detections_dark": vals.get("dark", 0),
            "detections_matched": vals.get("matched", 0),
        })
    out = pd.DataFrame(rows).sort_values(["region_type", "region_id", "month"])
    out["dark_share"] = (out["detections_dark"]
                         / out["detections_total"].replace(0, pd.NA))

    out_path = OUT_DIR / "gfw_sar_detections_monthly.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Rows: {len(out)}  Regions: {out['region_id'].nunique()}  "
          f"Months: {out['month'].nunique()}")
    print(f"  Date range: {out['month'].min()} to {out['month'].max()}")
    for rid in out["region_id"].unique():
        sub = out[out["region_id"] == rid]
        print(f"    {rid:18}: total={sub['detections_total'].sum():7d} "
              f"dark={sub['detections_dark'].sum():6d} "
              f"matched={sub['detections_matched'].sum():7d}")


if __name__ == "__main__":
    main()
