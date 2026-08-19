"""
Chapter 3 figures: study-site map and expanding-window evaluation design.

Figure 3.3 needs Natural Earth vectors under
data/raw/00_spatial_anchors/naturalearth/: ne_110m_land for the world
panel and ne_10m_land for the Persian Gulf inset.

    python code/scripts/figures/make_method_figures.py [--only 3.2 3.3 3.4]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
NE_DIR = ROOT / "data" / "raw" / "00_spatial_anchors" / "naturalearth"
FLAT_PRED = (ROOT / "results" / "baselines" / "Flat" / "M1_Flat"
             / "baseline_predictions.csv")
FEATURE_MATRIX = (ROOT / "data" / "processed" / "merge" / "outputs"
                  / "weekly_feature_matrix.csv")
OUT_DIR = ROOT / "results" / "figures"

# Backtest settings shared by Flat and Deep (run_baseline.py /
# run_deep_baseline.py argparse defaults).
LOOKBACK = 4
MIN_TRAIN = 104
RETRAIN_EVERY = 13
VAL_WEEKS = 52

# Appendix A.2.1, (lon, lat).
AOI = {
    "P001": ("Rotterdam", "port", 4.145, 51.950),
    "P002": ("Fujairah", "terminal", 56.356, 25.199),
    "P003": ("Ras Tanura", "terminal", 50.157, 26.643),
    "P004": ("Jurong Island", "refinery", 103.708, 1.274),
    "P005": ("Houston", "port", -95.100, 29.736),
    "P006": ("Ningbo-\nZhoushan", "port", 121.982, 29.935),
    "P007": ("Jamnagar", "refinery", 69.860, 22.345),
    "P008": ("Al Basrah Terminal", "terminal", 48.810, 29.681),
    "P009": ("Ulsan", "refinery", 129.343, 35.433),
    "P010": ("Kharg Island", "terminal", 50.324, 29.231),
    "P011": ("Yanbu", "terminal", 38.229, 23.961),
}

# EIA World Oil Transit Chokepoints, representative transit coordinates.
CHOKE = {
    "hormuz": ("Strait of Hormuz", 56.25, 26.57),
    "suez": ("Suez Canal", 32.35, 30.42),
    "malacca": ("Strait of Malacca", 100.40, 2.50),
    "mandeb": ("Bab el-Mandeb", 43.40, 12.60),
    "panama": ("Panama Canal", -79.55, 9.08),
    "cape": ("Cape of Good Hope", 18.47, -34.36),
}

# Label offsets in degrees: id -> (dx, dy, ha, va) for the world panel.
LABEL_OFF = {
    "P001": (0, 3.4, "center", "bottom"),
    "P004": (2.6, -3.4, "left", "top"),
    "P005": (-3.0, 1.8, "right", "bottom"),
    "P006": (-1.6, -2.2, "right", "top"),
    "P007": (0.5, -3.6, "center", "top"),
    "P009": (3.2, 2.2, "left", "bottom"),
    "P011": (-3.0, -1.6, "right", "top"),
    "suez": (-3.2, 2.6, "right", "bottom"),
    "malacca": (-3.0, -3.0, "right", "top"),
    "mandeb": (-3.2, -2.0, "right", "top"),
    "panama": (-3.4, 2.0, "right", "bottom"),
    "cape": (2.6, -2.6, "left", "top"),
}

# Persian Gulf inset: crowded sites get their own offsets.
INSET_OFF = {
    "P002": (0.45, -0.35, "left", "top"),
    "P003": (-0.45, -0.30, "right", "top"),
    "P008": (-0.55, 0.42, "right", "bottom"),
    "P010": (0.40, 0.25, "left", "bottom"),
    "hormuz": (0.45, 0.35, "left", "bottom"),
}
INSET_BOUNDS = (46.5, 22.5, 60.0, 32.0)  # lon0, lat0, lon1, lat1

TYPE_MARKER = {"port": "o", "terminal": "s", "refinery": "^"}

C = {
    "land": "#EFEDE8",
    "coast": "#C9C4BB",
    "border": "#DCD7CE",
    "aoi": "#2E5A88",
    "choke": "#D1622B",
    "train": "#8FA8C8",
    "warm": "#C9C4BB",
    "val": "#E7A33E",
    "test": "#D1622B",
    "grey": "#9A9A9A",
}

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def save(fig, *names: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{name}.{ext}")
        print(f"  saved {name}.png / .pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3.3 - study sites
# --------------------------------------------------------------------------
def _basemap(ax, bounds, land_shp: Path, *, coast_lw: float = 0.4,
             draw_borders: bool = True) -> None:
    import geopandas as gpd

    lon0, lat0, lon1, lat1 = bounds
    pad = 2.0
    land = gpd.read_file(
        land_shp, bbox=(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad))
    land.plot(ax=ax, facecolor=C["land"], edgecolor=C["coast"],
              linewidth=coast_lw, zorder=0)
    borders = NE_DIR / "ne_110m_admin_0_boundary_lines_land.shp"
    if draw_borders and borders.exists():
        gpd.read_file(
            borders, bbox=(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad)
        ).plot(ax=ax, color=C["border"], linewidth=0.35, zorder=1)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / math.cos(math.radians((lat0 + lat1) / 2)))


def _inset_scalebar(ax, lon: float, lat: float, length_km: int = 200) -> None:
    """Short geographic scale bar at inset latitude (Plate Carrée)."""
    km_per_deg = 111.32 * math.cos(math.radians(lat))
    dlon = length_km / km_per_deg
    tick = 0.12
    color = C["grey"]
    ax.plot([lon, lon + dlon], [lat, lat], color=color, linewidth=1.5,
            solid_capstyle="butt", zorder=7, clip_on=True)
    for x in (lon, lon + dlon):
        ax.plot([x, x], [lat - tick, lat + tick], color=color,
                linewidth=1.15, solid_capstyle="butt", zorder=7, clip_on=True)
    ax.text(lon + dlon / 2, lat + 0.22, f"{length_km} km", ha="center",
            va="bottom", fontsize=6.0, color=color, zorder=7)


def _style_zoom_indicator(indicator) -> None:
    """Inset connectors as thin dashed guides, not route-like solid lines."""
    ls = (0, (2.4, 2.2))
    color = "#B4B4B4"
    lw = 0.4
    if isinstance(indicator, tuple):
        rect, connectors = indicator
        artists = [rect, *connectors]
    else:
        artists = [indicator.rectangle, *list(indicator.connectors)]
    for a in artists:
        if a is None:
            continue
        a.set_linestyle(ls)
        a.set_linewidth(lw)
        a.set_alpha(0.8)
        if hasattr(a, "set_edgecolor"):
            a.set_edgecolor(color)
        if hasattr(a, "set_color"):
            a.set_color(color)


def _draw_nodes(ax, offsets, label_sites=True, fontsize=7.5, size=42) -> None:
    for sid, (name, stype, lon, lat) in AOI.items():
        ax.scatter(lon, lat, marker=TYPE_MARKER[stype], s=size, color=C["aoi"],
                   edgecolor="white", linewidth=0.7, zorder=5)
        if label_sites and sid in offsets:
            dx, dy, ha, va = offsets[sid]
            ax.annotate(name, (lon + dx, lat + dy), ha=ha, va=va,
                        fontsize=fontsize, color="#1F3E5F", zorder=6)
    for cid, (name, lon, lat) in CHOKE.items():
        ax.scatter(lon, lat, marker="D", s=size + 8, color=C["choke"],
                   edgecolor="white", linewidth=0.7, zorder=5)
        if cid in offsets:
            dx, dy, ha, va = offsets[cid]
            ax.annotate(name, (lon + dx, lat + dy), ha=ha, va=va,
                        fontsize=fontsize, color="#8A3D16", zorder=6,
                        fontweight="bold")


def fig33_site_map() -> None:
    land_110 = NE_DIR / "ne_110m_land.shp"
    land_10 = NE_DIR / "ne_10m_land.shp"
    if not land_110.exists():
        raise FileNotFoundError(
            f"Natural Earth 110m land not found under {NE_DIR}. "
            "Download ne_110m_land.zip from naciscdn.org and unzip it there.")
    if not land_10.exists():
        raise FileNotFoundError(
            f"Natural Earth 10m land not found under {NE_DIR}. "
            "Download ne_10m_land.zip from naciscdn.org and unzip it there.")

    fig, ax = plt.subplots(figsize=(9.6, 5.35))
    _basemap(ax, (-135, -48, 152, 68), land_110)
    _draw_nodes(ax, LABEL_OFF)

    axi = ax.inset_axes([0.035, 0.035, 0.29, 0.33])
    _basemap(axi, INSET_BOUNDS, land_10, coast_lw=0.55, draw_borders=False)
    _draw_nodes(axi, INSET_OFF, fontsize=6.6, size=34)
    # 200 km ≈ 15% of inset width at ~23°N; 500 km would crowd the panel.
    _inset_scalebar(axi, lon=47.85, lat=23.05, length_km=200)
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_edgecolor("#9A9A9A")
        sp.set_linewidth(0.6)
    axi.set_title("Persian Gulf — enlarged view", fontsize=7.5, pad=2.5)
    _style_zoom_indicator(
        ax.indicate_inset_zoom(axi, edgecolor="#B4B4B4", linewidth=0.4,
                               alpha=0.8))

    handles = [
        Line2D([], [], marker="o", linestyle="", color=C["aoi"],
               markeredgecolor="white", markersize=7, label="Port"),
        Line2D([], [], marker="^", linestyle="", color=C["aoi"],
               markeredgecolor="white", markersize=7, label="Refinery"),
        Line2D([], [], marker="s", linestyle="", color=C["aoi"],
               markeredgecolor="white", markersize=7, label="Export terminal"),
        Line2D([], [], marker="D", linestyle="", color=C["choke"],
               markeredgecolor="white", markersize=7, label="Chokepoint"),
    ]
    ax.legend(handles=handles, loc="lower right", ncol=1,
              bbox_to_anchor=(1.0, 0.02))

    ax.set_xticks(range(-120, 151, 30))
    ax.set_yticks(range(-30, 61, 30))
    ax.set_xticklabels([f"{abs(v)}\u00b0{'W' if v < 0 else 'E' if v else ''}"
                        for v in range(-120, 151, 30)], fontsize=7.5)
    ax.set_yticklabels([f"{abs(v)}\u00b0{'S' if v < 0 else 'N' if v else ''}"
                        for v in range(-30, 61, 30)], fontsize=7.5)
    ax.tick_params(length=2.5, pad=1.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(0.7)
    save(fig, "fig_3_3_study_sites_map", "fig_3_3_study_sites_map2")


# --------------------------------------------------------------------------
# Figures 3.2 and 3.4 - expanding-window evaluation
# --------------------------------------------------------------------------
def fig32_expanding_window() -> None:
    """Section 3.2: the estimation/evaluation split on the calendar."""
    test_dates = pd.to_datetime(
        pd.read_csv(FLAT_PRED, usecols=[0]).iloc[:, 0]).sort_values()
    data_start = pd.to_datetime(
        pd.read_csv(FEATURE_MATRIX, usecols=[0]).iloc[0, 0])
    n_test = len(test_dates)
    n_blocks = math.ceil(n_test / RETRAIN_EVERY)
    week = pd.Timedelta(weeks=1)

    fig, ax = plt.subplots(figsize=(9.6, 4.4))

    # -- every refit block on the calendar ---------------------------------
    for b in range(n_blocks):
        lo = b * RETRAIN_EVERY
        hi = min(lo + RETRAIN_EVERY, n_test)
        t_first, t_last = test_dates.iloc[lo], test_dates.iloc[hi - 1]
        y = n_blocks - 1 - b

        ax.barh(y, (t_first - data_start).days, left=data_start, height=0.62,
                color=C["train"], edgecolor="white", linewidth=0.4, zorder=2)
        ax.barh(y, (t_last + week - t_first).days, left=t_first, height=0.62,
                color=C["test"], edgecolor="white", linewidth=0.4, zorder=3)
        if b % 4 == 0 or b == n_blocks - 1:
            ax.text(data_start - pd.Timedelta(days=30), y, f"fit {b + 1}",
                    ha="right", va="center", fontsize=7, color="#555555")

    warm_end = test_dates.iloc[0]
    ax.axvline(warm_end, color="#444444", linewidth=1.0, linestyle="--", zorder=4)
    ax.annotate(f"first test week: {warm_end.date()}",
                xy=(warm_end, n_blocks - 0.35),
                xytext=(warm_end, n_blocks + 1.45),
                ha="center", va="bottom", fontsize=7.5, color="#444444",
                arrowprops=dict(arrowstyle="-", color="#444444", linewidth=0.8))
    ax.text(data_start + (warm_end - data_start) / 2, n_blocks - 0.35,
            f"initial training\n{MIN_TRAIN} weeks", ha="center", va="bottom",
            fontsize=7.5, color="#444444")
    ax.text(test_dates.iloc[int(n_test * 0.62)], n_blocks - 0.35,
            f"{n_test} test weeks\nrefit every {RETRAIN_EVERY} weeks "
            f"({n_blocks} fits)", ha="center", va="bottom", fontsize=7.5,
            color=C["test"])

    ax.set_ylim(-0.8, n_blocks + 2.6)
    ax.set_yticks([])
    ax.set_xlim(data_start - pd.Timedelta(days=250),
                test_dates.iloc[-1] + pd.Timedelta(days=60))
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Expanding-window backtest: the training set grows, "
                 "the test block always lies ahead of it", loc="left")
    ax.legend(handles=[
        mpl.patches.Patch(color=C["train"], label="training weeks (expanding)"),
        mpl.patches.Patch(color=C["test"],
                          label=f"test block ({RETRAIN_EVERY} weekly forecast "
                                "origins, model held fixed)"),
    ], ncol=2, loc="lower left", bbox_to_anchor=(0.0, -0.20))

    save(fig, "fig_3_2_expanding_window")


def fig34_forecast_origin() -> None:
    """Section 3.6.1: what one re-estimation origin looks like."""
    fig, axb = plt.subplots(figsize=(9.6, 2.4))

    total = 130
    val0 = total - VAL_WEEKS
    axb.add_patch(Rectangle((0, 0.55), val0, 0.3, color=C["train"], zorder=2))
    axb.add_patch(Rectangle((val0, 0.55), VAL_WEEKS, 0.3, color=C["val"], zorder=2))
    axb.text(val0 / 2, 0.70, "training fold\n(all weeks with a realised target)",
             ha="center", va="center", fontsize=7.5, color="white")
    axb.text(val0 + VAL_WEEKS / 2, 0.70,
             f"inner validation\n(last {VAL_WEEKS} weeks)",
             ha="center", va="center", fontsize=7.5, color="white")

    lb0 = total + 2
    axb.add_patch(Rectangle((lb0, 0.55), LOOKBACK, 0.3, facecolor="white",
                            edgecolor=C["aoi"], linewidth=1.2, hatch="///",
                            zorder=2))
    axb.text(lb0 + LOOKBACK / 2, 0.98,
             f"features:\n{LOOKBACK}-week lookback\n(weeks t-{LOOKBACK - 1} to t)",
             ha="center", va="bottom", fontsize=7.5, color=C["aoi"])

    tgt = lb0 + LOOKBACK + 6
    axb.plot(tgt, 0.70, marker="o", markersize=8, color=C["test"], zorder=3)
    axb.annotate("", xy=(tgt - 1.2, 0.70), xytext=(lb0 + LOOKBACK + 0.6, 0.70),
                 arrowprops=dict(arrowstyle="->", color=C["test"], linewidth=1.3))
    axb.text(tgt + 2.5, 0.70, "predict $r_{t+1}$\n(one week ahead)",
             ha="left", va="center", fontsize=7.5, color=C["test"])

    axb.annotate("", xy=(val0 + VAL_WEEKS, 0.42), xytext=(0, 0.42),
                 arrowprops=dict(arrowstyle="<->", color="#666666", linewidth=0.9))
    axb.text(total / 2, 0.34, "training fold at a scheduled re-estimation; "
             "parameters held fixed until the next re-estimation",
             ha="center", va="top", fontsize=7.5, color="#555555")

    axb.set_xlim(-6, tgt + 34)
    axb.set_ylim(0.15, 1.22)
    axb.axis("off")
    axb.set_title("Anatomy of one re-estimation origin: nested inner "
                  "validation, four-week input and one-week-ahead target",
                  loc="left")

    save(fig, "fig_3_4_forecast_origin")


FIGURES = {
    "3.2": fig32_expanding_window,
    "3.3": fig33_site_map,
    "3.4": fig34_forecast_origin,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()
    for k in args.only or sorted(FIGURES):
        print(f"[{k}] {FIGURES[k].__name__}")
        FIGURES[k]()
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
