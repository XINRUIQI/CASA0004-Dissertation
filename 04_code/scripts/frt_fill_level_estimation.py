"""
Floating Roof Tank (FRT) fill-level estimation from Sentinel-2 patches.

Pipeline:
  1. Load S2 GeoTIFF patch (4 bands: B2, B3, B4, B8 at 10m)
  2. Detect circular tank structures via Hough Circle Transform
  3. Extract shadow ratio inside each detected circle
  4. Estimate fill level = 1 - shadow_ratio
  5. Output monthly CSV with average fill level

Input:  03_data/raw/04_sentinel2/frt_patches/*.tif  (from GEE export)
Output: 03_data/processed/frt_fill_level_monthly.csv

Can also run in DEMO mode with synthetic test patches if real S2 data
is not yet downloaded.
"""
from __future__ import annotations
from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import rasterio

PROJECT = Path(__file__).resolve().parents[2]
PATCH_DIR = PROJECT / "03_data" / "raw" / "04_sentinel2" / "frt_patches"
OUT_CSV = PROJECT / "03_data" / "processed" / "frt_fill_level_monthly.csv"
OUT_VIS_DIR = PROJECT / "03_data" / "processed" / "frt_visualizations"

HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
MIN_RADIUS_PX = 3    # 30m at 10m/px
MAX_RADIUS_PX = 10   # 100m at 10m/px
SHADOW_L_THRESH = 80  # LAB L-channel threshold for shadow detection


def load_s2_patch(tif_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load S2 patch, return (rgb_uint8, nir) or None."""
    with rasterio.open(tif_path) as src:
        bands = src.read()  # (4, H, W) = B2, B3, B4, B8

    if bands.shape[0] < 4:
        return None

    b2, b3, b4, b8 = bands[0], bands[1], bands[2], bands[3]
    rgb = np.stack([b4, b3, b2], axis=-1)  # R=B4, G=B3, B=B2
    rgb_clipped = np.clip(rgb / 3000.0 * 255, 0, 255).astype(np.uint8)

    return rgb_clipped, b8


def detect_tanks(gray: np.ndarray) -> np.ndarray:
    """Detect circular structures using Hough Circle Transform."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=MIN_RADIUS_PX * 3,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS_PX, maxRadius=MAX_RADIUS_PX,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.int32)
    return np.round(circles[0]).astype(np.int32)


def compute_shadow_ratio(rgb: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Compute shadow ratio inside a circular region using LAB L-channel."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    circle_pixels = l_channel[mask > 0]
    if len(circle_pixels) == 0:
        return np.nan

    shadow_pixels = circle_pixels[circle_pixels < SHADOW_L_THRESH]
    return len(shadow_pixels) / len(circle_pixels)


def estimate_fill_level(rgb: np.ndarray) -> dict:
    """Full pipeline: detect tanks -> compute shadow -> estimate fill."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    tanks = detect_tanks(gray)

    if len(tanks) == 0:
        return {"n_tanks": 0, "avg_shadow_ratio": np.nan, "fill_level": np.nan}

    shadow_ratios = []
    for cx, cy, r in tanks:
        sr = compute_shadow_ratio(rgb, cx, cy, r)
        if not np.isnan(sr):
            shadow_ratios.append(sr)

    if not shadow_ratios:
        return {"n_tanks": len(tanks), "avg_shadow_ratio": np.nan, "fill_level": np.nan}

    avg_sr = np.mean(shadow_ratios)
    return {
        "n_tanks": len(tanks),
        "avg_shadow_ratio": avg_sr,
        "fill_level": 1.0 - avg_sr,
    }


def visualize_detection(rgb: np.ndarray, tanks: np.ndarray, date_label: str,
                        out_dir: Path) -> None:
    """Save a visualization of detected tanks for QA."""
    vis = rgb.copy()
    for cx, cy, r in tanks:
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 1)
        cv2.circle(vis, (cx, cy), 1, (0, 0, 255), 1)
    out_path = out_dir / f"frt_detection_{date_label}.png"
    cv2.imwrite(str(out_path), vis)


def parse_date_from_filename(fname: str) -> datetime | None:
    m = re.search(r"(\d{4})_(\d{2})", fname)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), 1)
    m2 = re.search(r"(\d{8})", fname)
    if m2:
        return datetime.strptime(m2.group(1), "%Y%m%d")
    return None


def create_demo_patches(out_dir: Path, n_months: int = 24) -> None:
    """Generate synthetic test patches for pipeline validation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    for i in range(n_months):
        year = 2023 + i // 12
        month = (i % 12) + 1
        img = np.ones((200, 200, 3), dtype=np.uint8) * 180

        fill_pct = 0.3 + 0.5 * np.sin(i / 6.0 * np.pi)
        fill_pct = np.clip(fill_pct, 0.1, 0.95)

        for tx, ty, tr in [(80, 80, 7), (130, 70, 6), (100, 140, 8)]:
            cv2.circle(img, (tx, ty), tr, (200, 200, 200), -1)
            cv2.circle(img, (tx, ty), tr, (100, 100, 100), 1)
            shadow_angle = np.random.uniform(0, 2 * np.pi)
            shadow_extent = int(tr * (1 - fill_pct))
            if shadow_extent > 0:
                sx = tx + int(shadow_extent * np.cos(shadow_angle))
                sy = ty + int(shadow_extent * np.sin(shadow_angle))
                cv2.ellipse(img, (tx, ty), (tr, shadow_extent),
                            np.degrees(shadow_angle), 0, 360, (40, 40, 40), -1)

        fname = out_dir / f"FRT_Fujairah_{year}_{month:02d}.tif"
        with rasterio.open(
            fname, "w", driver="GTiff", height=200, width=200,
            count=4, dtype="float32",
        ) as dst:
            for band_idx in range(3):
                dst.write(img[:, :, band_idx].astype(np.float32) * 12, band_idx + 1)
            dst.write(img[:, :, 0].astype(np.float32) * 15, 4)

    print(f"[demo] Created {n_months} synthetic patches in {out_dir}")


def main():
    if not PATCH_DIR.exists() or len(list(PATCH_DIR.glob("*.tif"))) == 0:
        print("No S2 patches found. Creating demo patches for pipeline validation...")
        create_demo_patches(PATCH_DIR)

    tif_files = sorted(PATCH_DIR.glob("*.tif"))
    print(f"Processing {len(tif_files)} patches from {PATCH_DIR}")

    OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for tif_path in tif_files:
        date = parse_date_from_filename(tif_path.name)
        if date is None:
            continue

        result = load_s2_patch(tif_path)
        if result is None:
            continue
        rgb, nir = result

        info = estimate_fill_level(rgb)

        tanks = detect_tanks(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))
        date_label = date.strftime("%Y_%m")
        visualize_detection(rgb, tanks, date_label, OUT_VIS_DIR)

        row = {"date": date, "site": "fujairah"}
        row.update(info)
        rows.append(row)

        status = f"tanks={info['n_tanks']}, fill={info['fill_level']:.2f}" if not np.isnan(info['fill_level']) else "no tanks"
        print(f"  {tif_path.name}: {status}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results!")
        return

    df = df.set_index("date").sort_index()
    df.to_csv(OUT_CSV)
    print(f"\n[saved] {OUT_CSV}")
    print(f"Shape: {df.shape}")
    print(f"Period: {df.index.min()} ~ {df.index.max()}")
    print(f"Average fill level: {df['fill_level'].mean():.3f}")
    print(f"Tanks detected per image: {df['n_tanks'].mean():.1f}")


if __name__ == "__main__":
    main()
