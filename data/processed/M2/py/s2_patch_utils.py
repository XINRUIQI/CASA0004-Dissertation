"""
Channel A — Sentinel-2 patch index helpers.

Shared by audit / embedding precompute scripts. Patches listed in
``s2_patch_exclusions.csv`` or with all-zero pixels are treated as missing
(valid_mask=0) even when a .tif file exists on disk.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

ROOT = Path(__file__).resolve().parents[4]
PATCH_DIR = ROOT / "data/raw/02_sentinel2/Channel A/s2_patches"
MANIFEST_CSV = PATCH_DIR / "S2_patches_manifest_ALL.csv"
EXCLUSIONS_CSV = ROOT / "data/raw/02_sentinel2/Channel A/s2_patch_exclusions.csv"

MIN_NONZERO_FRAC = 0.005  # require >=0.5% nonzero pixels (catches empty GEE exports)


def patch_filename(site_id: str, site_name: str, month: str) -> str:
    return f"S2_{site_id}_{site_name}_{month}.tif"


def load_exclusions() -> dict[tuple[str, str], dict[str, str]]:
    """Return {(site_id, month): row_dict} for manually excluded patches."""
    if not EXCLUSIONS_CSV.exists():
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    with open(EXCLUSIONS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["site_id"], row["month"])
            out[key] = row
    return out


def load_manifest() -> list[dict[str, str]]:
    with open(MANIFEST_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def has_valid_pixels(path: Path) -> bool:
    if rasterio is None:
        raise ImportError("rasterio is required to validate patch pixels")
    if not path.exists():
        return False
    with rasterio.open(path) as ds:
        if ds.count != 6:
            return False
        data = ds.read()
    if data.size == 0:
        return False
    return float(np.mean(data != 0)) >= MIN_NONZERO_FRAC


def is_patch_usable(
    site_id: str,
    site_name: str,
    month: str,
    *,
    exclusions: dict[tuple[str, str], dict[str, str]] | None = None,
    patch_dir: Path = PATCH_DIR,
) -> bool:
    """True only when the patch should enter Channel A (embedding) pipeline."""
    exclusions = exclusions if exclusions is not None else load_exclusions()
    if (site_id, month) in exclusions:
        return False
    path = patch_dir / patch_filename(site_id, site_name, month)
    return has_valid_pixels(path)


def build_patch_index(
    *,
    patch_dir: Path = PATCH_DIR,
    exclusions: dict[tuple[str, str], dict[str, str]] | None = None,
) -> list[dict]:
    """
    One row per manifest (site, month) with exported=1.

    Columns include file_exists, excluded, pixel_valid, valid_mask (final usable flag).
    """
    exclusions = exclusions if exclusions is not None else load_exclusions()
    rows: list[dict] = []
    for r in load_manifest():
        if int(r["exported"]) != 1:
            continue
        sid, sname, month = r["site_id"], r["site_name"], r["month"]
        fn = patch_filename(sid, sname, month)
        path = patch_dir / fn
        excluded = (sid, month) in exclusions
        file_exists = path.exists()
        pixel_valid = False
        if file_exists and not excluded:
            try:
                pixel_valid = has_valid_pixels(path)
            except Exception:
                pixel_valid = False
        usable = file_exists and not excluded and pixel_valid
        rows.append(
            {
                "site_id": sid,
                "site_name": sname,
                "site_type": r["site_type"],
                "month": month,
                "year": r["year"],
                "filename": fn,
                "file_exists": int(file_exists),
                "excluded": int(excluded),
                "pixel_valid": int(pixel_valid),
                "valid_mask": int(usable),
                "n_scenes": r["n_scenes"],
                "mean_cloud": r["mean_cloud"],
                "patch_px": r["patch_px"],
                "crs": r["crs"],
                "exclude_reason": exclusions.get((sid, month), {}).get("reason", ""),
            }
        )
    return rows
