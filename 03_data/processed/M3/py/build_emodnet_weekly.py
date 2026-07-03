"""
M3 Stage-2 — EMODnet monthly vessel-density rasters -> per-region zonal stats ->
leakage-safe WEEKLY table, for the 11 AOIs and 6 chokepoints.

Reads (raw layer):
  raw/03_shipping/emodnet_vessel_density_monthly_2017-2025/vesseldensity_10_YYYYMMDD.tif
  raw/02_sentinel2/aoi_oil_infrastructure.csv

Writes (processed/M3/outputs):
  m3_emodnet_density_weekly.csv    long: (week, region_type, region_id) x density stats

CRITICAL COVERAGE CAVEAT
------------------------
EMODnet Human Activities vessel-density is a EUROPEAN product: the rasters are
in EPSG:3035 (ETRS89-LAEA) and only cover European seas (probe bounds ~ Europe +
Mediterranean + Black Sea). Of our regions, effectively ONLY Rotterdam (and,
marginally, the Suez/Mediterranean edge) fall inside the grid; every Gulf / East
Asia / Americas AOI and most chokepoints are OUTSIDE coverage and return NaN.
EMODnet is therefore a Rotterdam-only cross-validation add-on, not a global
graph feature. Rasters also stop at 2024-12 (no 2025).

Method:
  * Build each region polygon in WGS84 (AOI = square around centre; chokepoints
    = the same boxes used by the GFW scripts), reproject to EPSG:3035 ONCE.
  * For each monthly raster: skip regions whose bbox does not intersect the
    raster (fast NaN, no read); otherwise rasterio.mask -> drop nodata (-9999)
    and negatives -> mean / max / sum / valid-pixel count.
  * Monthly -> month-end -> ffill onto the W-FRI union -> publication lag.

Usage:
    python build_emodnet_weekly.py
    python build_emodnet_weekly.py --buffer-deg 0.25 --lag 8
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom

PY_DIR = Path(__file__).resolve().parent
M3_DIR = PY_DIR.parent
DATA_DIR = M3_DIR.parents[1]
EMODNET_DIR = (DATA_DIR / "raw" / "03_shipping"
               / "emodnet_vessel_density_monthly_2017-2025")
AOI_CSV = DATA_DIR / "raw" / "02_sentinel2" / "aoi_oil_infrastructure.csv"
OUT_DIR = M3_DIR / "outputs"
OUT_PATH = OUT_DIR / "m3_emodnet_density_weekly.csv"

STUDY_START = "2019-01-01"
STUDY_END = "2026-12-31"
# EMODnet monthly density is published with a long delay; 8 weeks is a
# conservative release lag (CLI-overridable).
EMODNET_LAG_WEEKS = 8
NODATA = -9999.0

CHOKEPOINT_POLYGONS = {
    "Strait of Hormuz": [[55.5, 25.5], [57.5, 25.5], [57.5, 27.5], [55.5, 27.5], [55.5, 25.5]],
    "Suez Canal": [[32.0, 29.5], [33.5, 29.5], [33.5, 31.5], [32.0, 31.5], [32.0, 29.5]],
    "Malacca Strait": [[99.5, 0.5], [104.5, 0.5], [104.5, 4.5], [99.5, 4.5], [99.5, 0.5]],
    "Bab el-Mandeb": [[42.5, 12.0], [44.0, 12.0], [44.0, 13.5], [42.5, 13.5], [42.5, 12.0]],
    "Cape of Good Hope": [[17.0, -35.5], [20.5, -35.5], [20.5, -33.0], [17.0, -33.0], [17.0, -35.5]],
    "Panama Canal": [[-80.5, 8.5], [-79.0, 8.5], [-79.0, 10.0], [-80.5, 10.0], [-80.5, 8.5]],
}
CHOKE_SHORT = {
    "Strait of Hormuz": "hormuz", "Suez Canal": "suez", "Malacca Strait": "malacca",
    "Bab el-Mandeb": "mandeb", "Cape of Good Hope": "cape", "Panama Canal": "panama",
}


def square(lon: float, lat: float, h: float) -> list[list[float]]:
    return [[lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h],
            [lon - h, lat + h], [lon - h, lat - h]]


def geom_bbox(geom: dict) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for ring in geom["coordinates"]:
        for x, y in ring:
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def bbox_intersects(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def build_regions(buffer_deg: float) -> list[dict]:
    """Return region dicts with WGS84 + reprojected-3035 geometry and bbox."""
    regions = []
    for name, coords in CHOKEPOINT_POLYGONS.items():
        regions.append({"region_type": "chokepoint", "region_id": name,
                        "region_short": CHOKE_SHORT[name],
                        "geom_wgs": {"type": "Polygon", "coordinates": [coords]}})
    df = pd.read_csv(AOI_CSV).sort_values("site_id")
    for _, r in df.iterrows():
        regions.append({
            "region_type": "aoi", "region_id": r["site_id"],
            "region_short": r["site_short"],
            "geom_wgs": {"type": "Polygon",
                         "coordinates": [square(float(r["lon"]), float(r["lat"]), buffer_deg)]},
        })
    # Reproject once to the raster CRS (EPSG:3035) and cache bbox.
    for reg in regions:
        g3035 = transform_geom("EPSG:4326", "EPSG:3035", reg["geom_wgs"])
        reg["geom_3035"] = g3035
        reg["bbox_3035"] = geom_bbox(g3035)
    return regions


def zonal_stats(ds, geom_3035, raster_bbox, region_bbox) -> dict:
    """Masked mean/max/sum/valid-px for a region; NaN if outside coverage."""
    empty = {"emodnet_density_mean": np.nan, "emodnet_density_max": np.nan,
             "emodnet_density_sum": np.nan, "emodnet_n_valid_px": 0}
    if not bbox_intersects(region_bbox, raster_bbox):
        return empty
    try:
        out, _ = rio_mask(ds, [geom_3035], crop=True, nodata=NODATA, filled=True)
    except Exception:
        return empty
    a = out[0].astype("float64")
    valid = a[(a != NODATA) & (a >= 0)]
    if valid.size == 0:
        return empty
    return {"emodnet_density_mean": float(valid.mean()),
            "emodnet_density_max": float(valid.max()),
            "emodnet_density_sum": float(valid.sum()),
            "emodnet_n_valid_px": int(valid.size)}


def month_from_name(path: Path) -> str | None:
    m = re.search(r"(\d{4})(\d{2})\d{2}", path.stem)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description="EMODnet vessel-density weekly zonal stats.")
    ap.add_argument("--buffer-deg", type=float, default=0.25)
    ap.add_argument("--lag", type=int, default=EMODNET_LAG_WEEKS)
    ap.add_argument("--no-lag", action="store_true")
    args = ap.parse_args()
    lag = 0 if args.no_lag else args.lag

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tifs = sorted(EMODNET_DIR.glob("vesseldensity_10_*.tif"))
    if not tifs:
        print(f"No EMODnet rasters in {EMODNET_DIR}")
        return
    regions = build_regions(args.buffer_deg)
    print(f"EMODnet zonal stats: {len(tifs)} rasters x {len(regions)} regions "
          f"(buffer {args.buffer_deg} deg, lag {lag}w)")

    rows = []
    covered = set()
    for k, tif in enumerate(tifs):
        month = month_from_name(tif)
        if month is None:
            continue
        with rasterio.open(tif) as ds:
            rb = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)
            for reg in regions:
                s = zonal_stats(ds, reg["geom_3035"], rb, reg["bbox_3035"])
                if s["emodnet_n_valid_px"] > 0:
                    covered.add(reg["region_id"])
                rows.append({"month": month, "region_type": reg["region_type"],
                             "region_id": reg["region_id"],
                             "region_short": reg["region_short"], **s})
        if (k + 1) % 24 == 0:
            print(f"  ...{k + 1}/{len(tifs)} rasters", flush=True)

    monthly = pd.DataFrame(rows)
    monthly["month_end"] = pd.to_datetime(monthly["month"], format="%Y-%m") + pd.offsets.MonthEnd(0)
    print(f"\nRegions with ANY valid EMODnet pixel: {sorted(covered) or 'NONE'}")

    # Monthly -> W-FRI union -> ffill -> publication lag.
    m_start, m_end = monthly["month_end"].min(), monthly["month_end"].max()
    # EMODnet stops at 2024-12: do NOT ffill stale values across 2025-2026.
    # Extend only enough for the lag shift to place the last real month, then
    # stop, so post-coverage weeks are honestly NaN (data no longer available).
    union_end = min(pd.Timestamp(STUDY_END), m_end + pd.Timedelta(weeks=lag + 2))
    union = pd.date_range(m_start, union_end, freq="W-FRI")
    union.name = "week_ending_friday"

    stat_cols = ["emodnet_density_mean", "emodnet_density_max",
                 "emodnet_density_sum", "emodnet_n_valid_px"]
    frames = []
    for rid, grp in monthly.groupby("region_id"):
        g = grp.set_index("month_end").sort_index()
        w = g[stat_cols].reindex(union, method="ffill")
        if lag:
            w = w.shift(lag)
        w["region_type"] = grp["region_type"].iloc[0]
        w["region_id"] = rid
        w["region_short"] = grp["region_short"].iloc[0]
        frames.append(w)

    weekly = pd.concat(frames).reset_index(names="week_ending_friday")
    weekly = weekly[(weekly["week_ending_friday"] >= STUDY_START) &
                    (weekly["week_ending_friday"] <= STUDY_END)]
    lead = ["week_ending_friday", "region_type", "region_id", "region_short"]
    weekly = weekly[lead + [c for c in weekly.columns if c not in lead]]
    weekly.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Output: {OUT_PATH}")
    print(f"Shape:  {weekly.shape}  weeks {weekly['week_ending_friday'].min().date()} "
          f"~ {weekly['week_ending_friday'].max().date()}")
    print(f"Monthly raster coverage: {monthly['month'].min()} ~ {monthly['month'].max()} "
          f"({monthly['month'].nunique()} months)")
    cov = weekly[weekly["emodnet_density_mean"].notna()]
    print(f"\nWeekly non-null density by region (Europe-only expected):")
    if len(cov):
        for rid, g in cov.groupby("region_id"):
            print(f"  {g['region_type'].iloc[0]:10s} {rid:18s}: "
                  f"{len(g):4d} weeks, mean_density={g['emodnet_density_mean'].mean():.2f}")
    else:
        print("  (none)")
    print(f"{'='*60}\nDone.")


if __name__ == "__main__":
    main()
