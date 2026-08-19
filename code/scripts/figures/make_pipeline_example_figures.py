"""
Chapter 3 figures: Flat vs Deep pipeline and the Jurong worked example.

  1. Flat vs Deep data pipeline  — same three blocks, two ingest paths
  2. Worked example (Figure 3.4) — Jurong Island at Friday 14 Mar 2025
                                   three panels; no ingest-architecture boxes

P004 Deep patch is the locked refinery spec (5.12 km / 512 px), not the
snapped GeoTIFF bounds. Voyage counts are read from the lagged graph table
at the origin week (GFW voyage as-of = week ending 28 Feb).

    python code/scripts/figures/make_pipeline_example_figures.py
    python code/scripts/figures/make_pipeline_example_figures.py --only pipeline
    python code/scripts/figures/make_pipeline_example_figures.py --only example
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "figures"
THESIS_FIG = ROOT / "thesis" / "figures"
PATCH_TIF = (
    ROOT / "data" / "raw" / "02_sentinel2" / "Channel A" / "s2_patches"
    / "S2_P004_Jurong_2025_01.tif"
)
PATCH_MANIFEST = PATCH_TIF.parent / "S2_patches_manifest_ALL.csv"
FEATURE_MATRIX = (
    ROOT / "data" / "processed" / "merge" / "outputs" / "weekly_feature_matrix.csv"
)
EDGE_CSV = ROOT / "data" / "processed" / "M3" / "outputs" / "m3_graph_edges_weekly.csv"
NODES_CSV = ROOT / "data" / "processed" / "M3" / "outputs" / "m3_graph_nodes_weekly.csv"

ORIGIN_WEEK = "2025-03-14"

C = {
    "fin": "#2E5A88",
    "rs": "#2E7D5B",
    "ship": "#D1622B",
    "fin_bg": "#EEF3F7",
    "rs_bg": "#EAF4EE",
    "ship_bg": "#F8EEE8",
    "flat": "#2E5A88",
    "flat_bg": "#D4E4F2",
    "deep": "#D1622B",
    "deep_bg": "#F5D8C8",
    "band": "#F4F2EE",
    "box": "#FFFFFF",
    "line": "#8A8A8A",
    "ink": "#22303C",
    "muted": "#5C5C5C",
    "eval": "#4A4A4A",
    "eval_bg": "#F3E4C8",
    "aoi": "#2E5A88",
    "choke": "#D1622B",
}

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name: str, *aliases: str, pad: float = 0.16,
         thesis: bool = False, bbox: str | None = "tight") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": bbox}
    if bbox == "tight":
        kw["pad_inches"] = pad
    for ext in ("png", "pdf"):
        dest = OUT_DIR / f"{name}.{ext}"
        fig.savefig(dest, **kw)
        for alias in aliases:
            shutil.copy2(dest, OUT_DIR / f"{alias}.{ext}")
        if thesis and THESIS_FIG.exists():
            shutil.copy2(dest, THESIS_FIG / f"{name}.{ext}")
    plt.close(fig)
    extra = f"  (+ {', '.join(aliases)})" if aliases else ""
    print(f"  saved {name}.png / .pdf{extra}")


def box(ax, x0, y0, x1, y1, *, face=C["box"], edge=C["line"], lw=1.0,
        ls="solid", radius=1.2, z=2):
    p = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, xy_from, xy_to, *, color=C["line"], lw=1.1, ls="solid",
          style="-|>", rad=0.0, z=4, mutation=11):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=mutation,
        color=color, linewidth=lw, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}"))


def text(ax, x, y, s, *, size=8, color=C["ink"], weight="normal",
         ha="center", va="center", z=5, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, linespacing=1.32, style=style)


def panel_header(ax, letter: str, title: str, subtitle: str) -> None:
    """Shared (A)/(B)/(C) header: bold title, muted subtitle, left-aligned."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.88, f"({letter})  {title}", transform=ax.transAxes,
            fontsize=10.0, fontweight="bold", color=C["ink"], ha="left",
            va="top")
    ax.text(0.0, 0.08, subtitle, transform=ax.transAxes, fontsize=8.0,
            color=C["muted"], ha="left", va="bottom", style="italic",
            linespacing=1.28)


# --------------------------------------------------------------------------
# 1. Flat vs Deep pipeline
# --------------------------------------------------------------------------
def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 10.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    text(ax, 50, 98.6,
         "Same three data blocks, two ingest paths  ·  S4 shown",
         size=11.0, weight="bold")
    text(ax, 50, 96.15,
         "S2 and S3 omit the unused block.  S1 is finance only and skips fusion.",
         size=7.3, color=C["muted"], style="italic")

    # Shared sources
    sources = [
        (3.5, 32.5, C["fin_bg"], C["fin"], "Financial time series",
         "EIA · FRED · futures · macro · GPR"),
        (35.5, 64.5, C["rs_bg"], C["rs"], "Remote sensing  ·  11 AOIs",
         "Sentinel-2  ·  VIIRS night-time lights (Flat only)"),
        (67.5, 96.5, C["ship_bg"], C["ship"], "Shipping  ·  11 AOIs + 6 chokepoints",
         "IMF PortWatch  ·  Global Fishing Watch"),
    ]
    src_cx = []
    for x0, x1, face, edge, head, body in sources:
        box(ax, x0, 87.6, x1, 94.6, face=face, edge=edge, lw=1.25)
        cx = (x0 + x1) / 2
        src_cx.append(cx)
        text(ax, cx, 92.55, head, size=8.2, weight="bold", color=edge)
        text(ax, cx, 89.55, body, size=6.8, color=C["muted"])

    # Split rail
    for cx in src_cx:
        ax.plot([cx, cx], [87.6, 85.6], color=C["line"], lw=1.05, zorder=3)
    ax.plot([src_cx[0], src_cx[-1]], [85.6, 85.6], color=C["line"], lw=1.05,
            zorder=3)
    ax.plot([18.5, 18.5], [85.6, 83.9], color=C["line"], lw=1.05, zorder=3)
    ax.plot([81.5, 81.5], [85.6, 83.9], color=C["line"], lw=1.05, zorder=3)
    arrow(ax, (18.5, 83.9), (18.5, 82.4), color=C["flat"], lw=1.15)
    arrow(ax, (81.5, 83.9), (81.5, 82.4), color=C["deep"], lw=1.15)

    # Column panels
    box(ax, 2.2, 18.8, 48.4, 82.4, face=C["flat_bg"], edge=C["flat"],
        lw=1.35, z=1)
    box(ax, 51.6, 18.8, 97.8, 82.4, face=C["deep_bg"], edge=C["deep"],
        lw=1.35, z=1)

    box(ax, 2.2, 76.6, 48.4, 82.4, face=C["flat"], edge=C["flat"], lw=0, radius=1.0)
    box(ax, 51.6, 76.6, 97.8, 82.4, face=C["deep"], edge=C["deep"], lw=0, radius=1.0)
    text(ax, 25.3, 80.7, "Flat family  ·  early feature fusion",
         size=9.4, color="white", weight="bold")
    text(ax, 25.3, 78.15, "one weekly table  ·  modality structure discarded",
         size=6.8, color="#E8F0F8")
    text(ax, 74.7, 80.7, "Deep family  ·  representation-level fusion",
         size=9.4, color="white", weight="bold")
    text(ax, 74.7, 78.15, "separate encoders  ·  structure retained until fusion",
         size=6.8, color="#FBE8DC")

    # Stage labels inside each column
    def stage_label(x, y, s, color):
        text(ax, x, y, s, size=6.5, color=color, weight="bold", ha="left",
             style="italic")

    # --- Flat body ---
    stage_label(4.0, 74.7, "How each block is stored", C["flat"])
    flat_bits = [
        (4.0, 18.2, C["fin"], "Finance",
         "weekly columns\nlag 0 for daily series"),
        (18.8, 33.0, C["rs"], "Remote sensing",
         "5 km circular buffer\nNDVI, NDWI, NDBI, BSI + NTL"),
        (33.6, 46.6, C["ship"], "Shipping",
         "PortWatch + GFW as\nweekly tabular features"),
    ]
    for x0, x1, edge, head, body in flat_bits:
        box(ax, x0, 64.4, x1, 73.6, face="white", edge=edge, lw=1.15)
        cx = (x0 + x1) / 2
        text(ax, cx, 71.55, head, size=7.4, color=edge, weight="bold")
        text(ax, cx, 67.7, body, size=6.5, color=C["muted"])

    for x0, x1, *_ in flat_bits:
        arrow(ax, ((x0 + x1) / 2, 64.4), ((x0 + x1) / 2, 61.8),
              color=C["flat"], lw=1.0)

    stage_label(4.0, 60.3, "Assembly", C["flat"])
    box(ax, 4.0, 47.4, 46.6, 59.4, face="white", edge=C["flat"], lw=1.2)
    text(ax, 25.3, 56.55, "Merge on the Friday calendar",
         size=8.0, weight="bold", color=C["flat"])
    text(ax, 25.3, 53.35,
         "one row per week  ·  ~260 columns  ·  site identity becomes prefixes",
         size=6.7, color=C["muted"])
    text(ax, 25.3, 49.85,
         "flatten the last 4 weeks into one row at each forecast origin",
         size=6.7, color=C["ink"])

    arrow(ax, (25.3, 47.4), (25.3, 44.8), color=C["flat"], lw=1.1)

    stage_label(4.0, 43.3, "Learner  ·  no modality structure left", C["flat"])
    box(ax, 4.0, 32.6, 24.4, 42.4, face="white", edge=C["flat"], lw=1.15)
    text(ax, 14.2, 39.55, "Ridge", size=8.4, weight="bold", color=C["flat"])
    text(ax, 14.2, 35.7, "L2-regularised\nlinear comparator",
         size=6.5, color=C["muted"])
    box(ax, 26.2, 32.6, 46.6, 42.4, face="white", edge=C["flat"], lw=1.15)
    text(ax, 36.4, 39.55, "XGBoost", size=8.4, weight="bold", color=C["flat"])
    text(ax, 36.4, 35.7, "gradient-boosted trees\nnon-linear comparator",
         size=6.5, color=C["muted"])

    arrow(ax, (25.3, 32.6), (25.3, 29.8), color=C["flat"], lw=1.1)

    box(ax, 4.0, 20.4, 46.6, 29.8, face="white", edge=C["flat"], lw=1.05)
    text(ax, 25.3, 27.15, r"$\hat{r}_{t+1|t}$  then  $\hat{P}_{t+1|t}=P_t\exp(\hat{r})$",
         size=7.8, weight="bold", color=C["flat"])
    text(ax, 25.3, 23.15, "same reconstruction as Deep  ·  no graph, no patch",
         size=6.6, color=C["muted"])

    # --- Deep body ---
    stage_label(53.4, 74.7, "How each block is stored", C["deep"])
    deep_bits = [
        (53.4, 67.4, C["fin"], "Finance",
         "4-week sequence\nkept as a tensor"),
        (68.0, 82.0, C["rs"], "Remote sensing",
         "site-specific square patch\nfrozen Prithvi 1024-d"),
        (82.6, 96.0, C["ship"], "Shipping",
         "weekly 17-node graph\nvoyage + corridor edges"),
    ]
    for x0, x1, edge, head, body in deep_bits:
        box(ax, x0, 64.4, x1, 73.6, face="white", edge=edge, lw=1.15)
        cx = (x0 + x1) / 2
        text(ax, cx, 71.55, head, size=7.4, color=edge, weight="bold")
        text(ax, cx, 67.7, body, size=6.5, color=C["muted"])

    for x0, x1, *_ in deep_bits:
        arrow(ax, ((x0 + x1) / 2, 64.4), ((x0 + x1) / 2, 61.8),
              color=C["deep"], lw=1.0)

    stage_label(53.4, 60.3, "Encoders, then fusion", C["deep"])
    box(ax, 53.4, 47.4, 96.0, 59.4, face="white", edge=C["deep"], lw=1.2)
    text(ax, 74.7, 56.55, "Modality-specific encoders",
         size=8.0, weight="bold", color=C["deep"])
    text(ax, 74.7, 53.35,
         "causal TCN   ·   temporal then site attention   ·   GAT",
         size=6.7, color=C["muted"])
    text(ax, 74.7, 49.85,
         r"gated fusion (main):  $z=\sum_i \alpha_i z_i$   ·   concat / x-attn as alternatives",
         size=6.6, color=C["ink"])

    arrow(ax, (74.7, 47.4), (74.7, 44.8), color=C["deep"], lw=1.1)

    stage_label(53.4, 43.3, "Learner  ·  one vector per origin", C["deep"])
    box(ax, 53.4, 32.6, 96.0, 42.4, face="white", edge=C["deep"], lw=1.2)
    text(ax, 74.7, 39.55, "Regression head", size=8.4, weight="bold", color=C["deep"])
    text(ax, 74.7, 35.7, "MLP trained by MSE on the one-week log return",
         size=6.8, color=C["muted"])

    arrow(ax, (74.7, 32.6), (74.7, 29.8), color=C["deep"], lw=1.1)

    box(ax, 53.4, 20.4, 96.0, 29.8, face="white", edge=C["deep"], lw=1.05)
    text(ax, 74.7, 27.15, r"$\hat{r}_{t+1|t}$  then  $\hat{P}_{t+1|t}=P_t\exp(\hat{r})$",
         size=7.8, weight="bold", color=C["deep"])
    text(ax, 74.7, 23.15, "same reconstruction as Flat  ·  structure used only inside",
         size=6.6, color=C["muted"])

    # Shared evaluation
    arrow(ax, (25.3, 18.8), (25.3, 16.6), color=C["eval"], lw=1.1)
    arrow(ax, (74.7, 18.8), (74.7, 16.6), color=C["eval"], lw=1.1)
    ax.plot([25.3, 74.7], [16.6, 16.6], color=C["eval"], lw=1.1, zorder=3)
    arrow(ax, (50.0, 16.6), (50.0, 15.2), color=C["eval"], lw=1.1)

    box(ax, 8.0, 2.2, 92.0, 15.2, face=C["eval_bg"], edge=C["eval"], lw=1.25)
    text(ax, 50, 12.55, "Shared expanding-window evaluation  ·  versus M0",
         size=9.0, weight="bold", color=C["eval"])
    text(ax, 50, 8.85,
         "same 257 Friday origins  ·  same 4-week lookback  ·  refit every 13 weeks  ·  no look-ahead",
         size=7.2, color=C["muted"])
    text(ax, 50, 5.15,
         "RMSE and skill vs M0 from reconstructed prices.  "
         "Paired Flat vs Deep on the same information set addresses RQ2.",
         size=7.0, color=C["ink"])

    save(fig, "fig_preview_flat_vs_deep_pipeline")


# --------------------------------------------------------------------------
# 2. Worked example — Jurong Island (Figure 3.4)
# --------------------------------------------------------------------------
def _fmt_signed(v: float) -> str:
    return f"+{v:.2f}" if v >= 0 else f"−{abs(v):.2f}"


def _p004_jan_meta() -> dict:
    """Locked Channel A spec from the export manifest, not GeoTIFF bounds."""
    man = pd.read_csv(PATCH_MANIFEST)
    row = man[(man["site_id"] == "P004") & (man["month"] == "2025_01")].iloc[0]
    half = int(row["patch_half_m"])
    return {
        "flat_buffer_km": 5.0,  # aoi_oil_infrastructure.csv; Channel B circle
        "patch_half_m": half,
        "patch_km": 2.0 * half / 1000.0,
        "patch_px": int(row["patch_px"]),
        "n_scenes": int(row["n_scenes"]),
        "cloud_pct": int(round(float(row["mean_cloud"]))),
    }


def _rgb_patch() -> np.ndarray:
    """True-colour preview from B4, B3, B2. Returns (H, W, 3) in 0–1."""
    import rasterio

    with rasterio.open(PATCH_TIF) as ds:
        rgb = ds.read([3, 2, 1]).astype(np.float32)
    rgb = np.moveaxis(rgb, 0, -1)
    mask = rgb > 0
    out = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[..., i]
        valid = band[mask[..., i]]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[..., i] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def _voyage_week(week: str) -> pd.DataFrame:
    """Voyages from the lagged graph table (week = forecast origin)."""
    names = {
        "P001": "Rotterdam", "P002": "Fujairah", "P003": "Ras Tanura",
        "P004": "Jurong", "P005": "Houston", "P006": "Ningbo",
        "P007": "Jamnagar", "P008": "Basra", "P009": "Ulsan",
        "P010": "Kharg", "P011": "Yanbu",
    }
    df = pd.read_csv(EDGE_CSV)
    sub = df[(df["week_ending_friday"] == week)
             & ((df["from_site"] == "P004") | (df["to_site"] == "P004"))
             & (df["n_voyages"] > 0)].copy()
    sub["from_n"] = sub["from_site"].map(names)
    sub["to_n"] = sub["to_site"].map(names)
    return sub.sort_values("n_voyages", ascending=False)


def _jurong_flat_scalars(week: str) -> tuple[float, list[tuple[str, str]]]:
    fm = pd.read_csv(FEATURE_MATRIX)
    fm["week_ending_friday"] = pd.to_datetime(fm["week_ending_friday"])
    row = fm[fm["week_ending_friday"] == pd.Timestamp(week)].iloc[0]
    pairs = [
        ("NDVI anomaly", "NDVI_anom_Jurong"),
        ("NDWI anomaly", "NDWI_anom_Jurong"),
        ("NDBI anomaly", "NDBI_anom_Jurong"),
        ("BSI anomaly", "BSI_anom_Jurong"),
        ("NTL anomaly", "NTL_anom_Jurong"),
    ]
    return float(row["brent_price"]), [
        (lab, _fmt_signed(float(row[col]))) for lab, col in pairs
    ]


def _jurong_portwatch(week: str) -> int:
    df = pd.read_csv(NODES_CSV)
    row = df[(df["site_id"] == "P004") & (df["week_ending_friday"] == week)].iloc[0]
    return int(row["pw_portcalls_tanker"])


def _draw_resize_square(ax, x_center: float, y_bottom: float, y_top: float,
                        label: str = "224 × 224") -> None:
    """Two stacked squares: back is an empty frame, front holds the label."""
    pos = ax.get_position()
    fig = ax.figure
    ax_w = pos.width * fig.get_figwidth()
    ax_h = pos.height * fig.get_figheight()
    side_y = min(0.30, y_top - y_bottom - 0.04)
    side_x = side_y * (ax_h / ax_w)
    ox, oy = side_x * 0.14, side_y * 0.14
    x0 = x_center - (side_x + ox) / 2
    y0 = y_bottom + 0.5 * ((y_top - y_bottom) - (side_y + oy))
    ax.add_patch(Rectangle(
        (x0 + ox, y0 + oy), side_x, side_y, facecolor="none",
        edgecolor=C["rs"], linewidth=1.05, zorder=3))
    ax.add_patch(Rectangle(
        (x0, y0), side_x, side_y, facecolor="white", edgecolor=C["rs"],
        linewidth=1.15, zorder=4))
    ax.text(x0 + side_x / 2, y0 + side_y / 2, label, ha="center", va="center",
            fontsize=8.2, color=C["rs"], zorder=5, fontweight="bold")


def fig_example() -> None:
    meta = _p004_jan_meta()
    rgb = _rgb_patch()
    # Lagged graph table at the origin; GFW voyage as-of is week ending 28 Feb.
    voy = _voyage_week(ORIGIN_WEEK)
    _, scalars = _jurong_flat_scalars(ORIGIN_WEEK)
    pw_calls = _jurong_portwatch(ORIGIN_WEEK)
    patch_km = meta["patch_km"]
    buf_km = meta["flat_buffer_km"]

    fig = plt.figure(figsize=(11.6, 8.05), facecolor="white")
    gs = fig.add_gridspec(
        5, 2,
        height_ratios=[0.34, 0.22, 1.22, 0.22, 1.22],
        width_ratios=[1.06, 1.14],
        hspace=0.07, wspace=0.14,
        left=0.045, right=0.975, top=0.94, bottom=0.04)
    ax_top = fig.add_subplot(gs[0, :])
    ax_a_h = fig.add_subplot(gs[1, 0])
    ax_map = fig.add_subplot(gs[2:, 0])
    ax_b_h = fig.add_subplot(gs[1, 1])
    ax_rs = fig.add_subplot(gs[2, 1])
    ax_c_h = fig.add_subplot(gs[3, 1])
    ax_ship = fig.add_subplot(gs[4, 1])
    ax_top.axis("off")
    ax_rs.axis("off")
    ax_ship.axis("off")

    ax_top.text(
        0.0, 0.82,
        "Figure 3.4   Worked example  ·  Jurong Island (P004), forecast origin 14 March 2025",
        transform=ax_top.transAxes, fontsize=11.2, color=C["ink"],
        fontweight="bold", ha="left", va="top")
    ax_top.text(
        0.0, 0.22,
        "Latest eligible observations under the source-specific availability lags.",
        transform=ax_top.transAxes, fontsize=8.0, color=C["muted"],
        ha="left", va="top", style="italic")

    panel_header(
        ax_a_h, "a",
        "Spatial supports  ·  January 2025 Sentinel-2 composite",
        "Under the locked-lag rule, the February 2025 composite was not yet eligible at this forecast origin.")
    panel_header(
        ax_b_h, "b",
        "Remote-sensing representations",
        "January inputs were eligible from 15 February and carried across the four-week input window.")
    panel_header(
        ax_c_h, "c",
        "Shipping graph inputs around Jurong",
        "GFW voyage edges: week ending 28 February\n"
        "PortWatch tanker calls: week ending 7 March.")

    # ---- map: patch + two footprints (locked spec, not snapped raster) ----
    half = patch_km / 2.0
    # Extra bottom pad so the in-axes legend sits below the 5 km circle.
    frame = 7.15
    ax_map.set_xlim(-frame, frame)
    ax_map.set_ylim(-frame, frame)
    ax_map.set_aspect("equal", adjustable="box", anchor="C")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for sp in ax_map.spines.values():
        sp.set_color("#C8C4BC")
        sp.set_linewidth(0.7)
    ax_map.set_facecolor("#F4F2EE")
    ax_map.imshow(rgb, origin="upper", extent=(-half, half, -half, half),
                  zorder=1, interpolation="nearest")

    ax_map.add_patch(Circle(
        (0, 0), buf_km, fill=False, edgecolor=C["flat"],
        linewidth=1.7, linestyle=(0, (5, 2.2)), zorder=4))
    ax_map.add_patch(Rectangle(
        (-half, -half), patch_km, patch_km, fill=False, edgecolor=C["rs"],
        linewidth=1.7, zorder=4))
    ax_map.scatter([0], [0], s=28, color=C["ink"], zorder=5,
                   edgecolor="white", linewidth=0.6)

    bar_y = -frame + 0.48
    ax_map.plot([-6.05, -5.05], [bar_y, bar_y], color=C["ink"], lw=1.6,
                solid_capstyle="butt", zorder=6)
    ax_map.text(-5.55, bar_y - 0.16, "1 km", ha="center", va="top",
                fontsize=8.0, color=C["ink"])

    # Data-coordinate legend: equal-aspect shrinks the axes box, so an
    # axes-fraction legend would land in the white gap below the square.
    ax_map.legend(
        handles=[
            Line2D([], [], color=C["flat"], lw=1.8, linestyle=(0, (5, 2.2)),
                   label=f"Flat  ·  {buf_km:.0f} km radius buffer"),
            Line2D([], [], color=C["rs"], lw=1.8,
                   label=f"Deep  ·  {patch_km:.2f} km square patch"),
            Line2D([], [], marker="o", linestyle="", color=C["ink"],
                   markeredgecolor="white", markersize=6.0,
                   label="AOI centre"),
        ],
        loc="lower right",
        bbox_to_anchor=(frame - 0.18, -frame + 0.16),
        bbox_transform=ax_map.transData,
        fontsize=8.0, frameon=True, framealpha=0.96,
        facecolor="white", edgecolor="#E4E0D8", fancybox=False,
        borderpad=0.38, labelspacing=0.28, handlelength=1.55,
        handletextpad=0.45, borderaxespad=0.0)

    # ---- (B) remote sensing numbers ----
    ax_rs.set_xlim(0, 1)
    ax_rs.set_ylim(0, 1)
    box(ax_rs, 0.00, 0.00, 1.00, 1.00, face=C["rs_bg"], edge=C["rs"],
        lw=1.15, radius=0.018, z=1)

    x_flat = 0.035
    text(ax_rs, x_flat, 0.925, "Flat",
         size=9.0, color=C["flat"], weight="bold", ha="left")
    text(ax_rs, x_flat, 0.835, "Sentinel-2 + VIIRS scalar anomalies",
         size=8.0, color=C["muted"], ha="left")
    text(ax_rs, x_flat, 0.755,
         f"{buf_km:.0f} km buffer  ·  five site-level scalars",
         size=8.0, color=C["muted"], ha="left")
    y0 = 0.64
    for i, (lab, val) in enumerate(scalars):
        y = y0 - i * 0.105
        text(ax_rs, x_flat, y, lab, size=8.2, color=C["ink"], ha="left")
        text(ax_rs, 0.46, y, val, size=8.2, color=C["ink"], ha="right",
             weight="bold")

    x_deep = 0.52
    text(ax_rs, x_deep, 0.925, "Deep",
         size=9.0, color=C["rs"], weight="bold", ha="left")
    text(ax_rs, x_deep, 0.835, "Sentinel-2 representation",
         size=8.0, color=C["muted"], ha="left")
    text(ax_rs, x_deep, 0.74,
         "Sentinel-2 patch  →  resize to 224 × 224",
         size=8.0, color=C["muted"], ha="left")
    text(ax_rs, x_deep, 0.655,
         "→  frozen Prithvi  →  mean pooling",
         size=8.0, color=C["muted"], ha="left")
    text(ax_rs, x_deep, 0.57,
         "→  1024-d embedding",
         size=8.0, color=C["muted"], ha="left")
    _draw_resize_square(ax_rs, 0.74, 0.08, 0.50)

    # ---- (C) shipping graph inputs: dynamic edges / static edge / node attr ----
    ax_ship.set_xlim(0, 1)
    ax_ship.set_ylim(0, 1)
    box(ax_ship, 0.00, 0.00, 1.00, 1.00, face=C["ship_bg"], edge=C["ship"],
        lw=1.15, radius=0.018, z=1)
    box(ax_ship, 0.018, 0.04, 0.545, 0.96, face="#FFFCFB", edge="#E8D8CE",
        lw=0.7, radius=0.014, z=2)
    box(ax_ship, 0.56, 0.52, 0.982, 0.96, face="#FFFCFB", edge="#E8D8CE",
        lw=0.7, radius=0.014, z=2)
    box(ax_ship, 0.56, 0.04, 0.982, 0.48, face="#FFFCFB", edge="#E8D8CE",
        lw=0.7, radius=0.014, z=2)

    n_lanes = len(voy)
    n_voy = int(voy["n_voyages"].sum())
    x_l = 0.04
    text(ax_ship, x_l, 0.90, "Dynamic GFW voyage edges",
         size=9.0, color=C["ship"], weight="bold", ha="left")
    text(ax_ship, x_l, 0.80,
         f"{n_lanes} directed lanes  ·  {n_voy} voyages",
         size=8.0, color=C["muted"], ha="left")
    text(ax_ship, x_l, 0.72, "busiest four shown",
         size=8.0, color=C["muted"], ha="left")

    text(ax_ship, x_l, 0.61, "Origin → destination",
         size=8.0, color=C["muted"], ha="left")
    text(ax_ship, 0.52, 0.61, "Voyages",
         size=8.0, color=C["muted"], ha="right")
    ax_ship.plot([x_l, 0.52], [0.565, 0.565], color="#E0D0C6", lw=0.6, zorder=4)
    n_show = 4
    top_lanes = voy.head(n_show)
    for i, (_, r) in enumerate(top_lanes.iterrows()):
        y = 0.48 - i * 0.10
        text(ax_ship, x_l, y, f"{r.from_n}  →  {r.to_n}",
             size=8.2, color=C["ink"], ha="left")
        text(ax_ship, 0.52, y, f"{int(r.n_voyages)}",
             size=8.2, color=C["ink"], ha="right", weight="bold")

    x_r = 0.585
    text(ax_ship, x_r, 0.90, "Fixed corridor edge",
         size=9.0, color=C["ship"], weight="bold", ha="left")
    dot_x, dia_x, y_n = 0.70, 0.86, 0.70
    ax_ship.scatter([dot_x], [y_n], s=92, color=C["ship"], zorder=6,
                    edgecolor="white", linewidth=0.7)
    ax_ship.plot([dot_x + 0.028, dia_x - 0.028], [y_n, y_n],
                 color="#8FA8C8", lw=1.9, zorder=3)
    ax_ship.scatter([dia_x], [y_n], s=80, color=C["choke"], marker="D",
                    zorder=6, edgecolor="white", linewidth=0.6)
    text(ax_ship, dot_x, y_n - 0.08, "Jurong",
         size=7.8, color=C["ship"], weight="bold")
    text(ax_ship, dia_x, y_n - 0.08, "Malacca",
         size=7.8, color="#8A3D16", weight="bold")

    text(ax_ship, x_r, 0.40, "Selected node attribute",
         size=9.0, color=C["ship"], weight="bold", ha="left")
    text(ax_ship, x_r, 0.26, "Jurong",
         size=8.5, color=C["ink"], ha="left")
    text(ax_ship, x_r, 0.14,
         f"PortWatch tanker calls  ·  {pw_calls}",
         size=8.5, color=C["ink"], ha="left")

    save(fig, "fig_3_4_worked_example_jurong",
         "fig_preview_worked_example_jurong",
         pad=0.36, thesis=True, bbox="tight")


FIGURES = {
    "pipeline": fig_pipeline,
    "example": fig_example,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()
    for k in args.only or ["pipeline", "example"]:
        print(f"[{k}] {FIGURES[k].__name__}")
        FIGURES[k]()
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
