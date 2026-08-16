"""
Preview figures for Chapter 3 (not yet numbered in the thesis).

  1. Flat vs Deep data pipeline  — same three blocks, two ingest paths
  2. Worked example              — Jurong Island at Friday 14 Mar 2025
                                   (same origin as Figure 3.7)

Colours match Figures 3.1 / 3.5 / 3.6 / 3.7.

    python 04_code/scripts/figures/make_pipeline_example_figures.py
    python 04_code/scripts/figures/make_pipeline_example_figures.py --only pipeline
    python 04_code/scripts/figures/make_pipeline_example_figures.py --only example
"""

from __future__ import annotations

import argparse
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
OUT_DIR = ROOT / "05_outputs" / "figures"
PATCH_TIF = (
    ROOT / "03_data" / "raw" / "02_sentinel2" / "Channel A" / "s2_patches"
    / "S2_P004_Jurong_2025_01.tif"
)
EMB_NPY = ROOT / "03_data" / "processed" / "M2" / "outputs" / "s2_prithvi_emb_meanpool.npy"
EMB_IDX = ROOT / "03_data" / "processed" / "M2" / "outputs" / "s2_prithvi_emb_index.csv"
EDGE_CSV = ROOT / "03_data" / "processed" / "M3" / "outputs" / "m3_graph_edges_weekly.csv"

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


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", pad_inches=0.16)
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


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
# 2. Worked example — Jurong Island
# --------------------------------------------------------------------------
def _rgb_patch() -> tuple[np.ndarray, float]:
    """True-colour preview from B4, B3, B2. Returns (H, W, 3) in 0–1 and side km."""
    import rasterio

    with rasterio.open(PATCH_TIF) as ds:
        # bands: B2, B3, B4, B8A, B11, B12  → RGB = B4, B3, B2
        rgb = ds.read([3, 2, 1]).astype(np.float32)
        side_m = float(ds.bounds.right - ds.bounds.left)
    side_km = side_m / 1000.0
    rgb = np.moveaxis(rgb, 0, -1)
    # Sentinel-2 SR is scaled ~0–10000; stretch on finite nonzero pixels.
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
    return out, side_km


def _voyage_week(week: str) -> pd.DataFrame:
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


def _jurong_embedding() -> np.ndarray:
    idx = pd.read_csv(EMB_IDX)
    row = idx[(idx["site_id"] == "P004") & (idx["month"] == "2025_01")].iloc[0]
    emb = np.load(EMB_NPY)
    return emb[int(row["emb_row"])]


def fig_example() -> None:
    rgb, side_km = _rgb_patch()
    emb = _jurong_embedding()
    voy = _voyage_week("2025-02-28")  # GFW voyage lag +2 w relative to 14 Mar

    fig = plt.figure(figsize=(11.4, 8.70))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.08, 1.05, 0.78], width_ratios=[1.08, 1.16],
        hspace=0.28, wspace=0.12,
        left=0.045, right=0.985, top=0.865, bottom=0.042)
    ax_map = fig.add_subplot(gs[0:2, 0])
    ax_rs = fig.add_subplot(gs[0, 1])
    ax_ship = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[2, :])
    ax_rs.axis("off")
    ax_ship.axis("off")
    ax_c.axis("off")

    fig.suptitle(
        "Worked example  ·  Jurong Island (P004) at Friday 14 March 2025",
        x=0.045, ha="left", fontsize=11.2, color=C["ink"], fontweight="bold")
    fig.text(
        0.045, 0.892,
        "Same origin as Figure 3.7.  Brent $71.94/bbl.  "
        "Values are the latest eligible observations under the locked lags, "
        "not the calendar week of the origin.",
        fontsize=7.4, color=C["muted"], ha="left", style="italic")

    # ---- map: patch + two footprints ----
    half = side_km / 2.0
    canvas = 5.85
    ax_map.set_xlim(-canvas, canvas)
    ax_map.set_ylim(-canvas - 0.15, canvas)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for sp in ax_map.spines.values():
        sp.set_color("#C8C4BC")
        sp.set_linewidth(0.7)
    ax_map.set_facecolor("#F4F2EE")
    ax_map.imshow(rgb, origin="upper", extent=(-half, half, -half, half),
                  zorder=1, interpolation="nearest")

    circ = Circle((0, 0), 5.0, fill=False, edgecolor=C["flat"],
                  linewidth=1.7, linestyle=(0, (5, 2.2)), zorder=4)
    ax_map.add_patch(circ)
    ax_map.add_patch(Rectangle(
        (-half, -half), side_km, side_km, fill=False, edgecolor=C["rs"],
        linewidth=1.7, zorder=4))
    ax_map.scatter([0], [0], s=28, color=C["ink"], zorder=5,
                   edgecolor="white", linewidth=0.6)

    ax_map.plot([-5.35, -4.35], [-5.45, -5.45], color=C["ink"], lw=1.6,
                solid_capstyle="butt", zorder=6)
    ax_map.text(-4.85, -5.58, "1 km", ha="center", va="top", fontsize=6.8,
                color=C["ink"])

    ax_map.set_title(
        "(A)  Two spatial supports on the January 2025 composite",
        loc="left", fontsize=9.0, color=C["ink"], pad=4)
    ax_map.text(
        0.0, 1.012,
        "True-colour Sentinel-2 (B4–B3–B2).  "
        "February is not yet eligible at this origin.",
        transform=ax_map.transAxes, fontsize=6.6, color=C["muted"],
        ha="left", va="bottom")

    ax_map.legend(handles=[
        Line2D([], [], color=C["flat"], lw=1.8, linestyle=(0, (5, 2.2)),
               label="Flat  ·  5 km radius buffer"),
        Line2D([], [], color=C["rs"], lw=1.8,
               label=f"Deep  ·  {side_km:.2f} km square patch"),
        Line2D([], [], marker="o", linestyle="", color=C["ink"],
               markeredgecolor="white", markersize=5.5,
               label="AOI centre  ·  1.274°N, 103.708°E"),
    ], loc="lower right", fontsize=6.6, frameon=True, framealpha=0.94,
       edgecolor="#E4E0D8", fancybox=False, borderpad=0.40,
       handlelength=1.6)

    # ---- (B) remote sensing numbers ----
    ax_rs.set_xlim(0, 1)
    ax_rs.set_ylim(0, 1)
    box(ax_rs, 0.00, 0.00, 1.00, 1.00, face=C["rs_bg"], edge=C["rs"],
        lw=1.15, radius=0.018, z=1)
    text(ax_rs, 0.035, 0.90, "(B)  Remote sensing at this origin",
         size=9.0, weight="bold", color=C["rs"], ha="left")
    text(ax_rs, 0.035, 0.78,
         "Eligible product: January composite (available from 15 Feb).  "
         "Carried forward through the four-week window.",
         size=6.6, color=C["muted"], ha="left")

    # mini table
    rows = [
        ("NDVI anomaly", "+0.56"),
        ("NDWI anomaly", "−0.40"),
        ("NDBI anomaly", "−0.93"),
        ("BSI anomaly", "−0.76"),
        ("NTL anomaly", "+0.36"),
    ]
    text(ax_rs, 0.035, 0.64, "Flat  ·  five scalars from the 5 km buffer",
         size=7.1, color=C["flat"], weight="bold", ha="left")
    y0 = 0.54
    for i, (lab, val) in enumerate(rows):
        y = y0 - i * 0.075
        text(ax_rs, 0.05, y, lab, size=6.7, color=C["ink"], ha="left")
        text(ax_rs, 0.46, y, val, size=6.7, color=C["ink"], ha="right",
             weight="bold")

    text(ax_rs, 0.55, 0.64, "Deep  ·  frozen 1024-d embedding",
         size=7.1, color=C["rs"], weight="bold", ha="left")
    text(ax_rs, 0.55, 0.545,
         "Prithvi-EO-2.0-300M  ·  mean-pool\n"
         f"{rgb.shape[0]} px chip  ·  1 scene  ·  37% cloud",
         size=6.4, color=C["muted"], ha="left")

    # sparkline of first 48 dims
    sl = emb[:48]
    xs = np.linspace(0.55, 0.96, len(sl))
    ys = 0.18 + 0.22 * (sl - sl.min()) / (sl.max() - sl.min() + 1e-9)
    ax_rs.plot(xs, ys, color=C["rs"], lw=0.95, zorder=4)
    ax_rs.fill_between(xs, 0.18, ys, color=C["rs"], alpha=0.18, zorder=3)
    text(ax_rs, 0.755, 0.10, "first 48 of 1 024 dimensions  (not a forecast)",
         size=6.2, color=C["muted"])

    # ---- (C) shipping neighborhood ----
    ax_ship.set_xlim(0, 1)
    ax_ship.set_ylim(0, 1)
    box(ax_ship, 0.00, 0.00, 1.00, 1.00, face=C["ship_bg"], edge=C["ship"],
        lw=1.15, radius=0.018, z=1)
    text(ax_ship, 0.035, 0.91, "(C)  Shipping neighbourhood  ·  lag +2 weeks",
         size=9.0, weight="bold", color=C["ship"], ha="left")
    text(ax_ship, 0.035, 0.795,
         "Latest eligible voyage week: 28 Feb 2025.  "
         "16 directed lanes  ·  255 voyages involving Jurong.",
         size=6.6, color=C["muted"], ha="left")

    # Ranked lane table — less collision-prone than a labelled star.
    text(ax_ship, 0.05, 0.68, "Busiest directed lanes",
         size=7.0, color=C["ship"], weight="bold", ha="left")
    text(ax_ship, 0.05, 0.60, "Origin → destination",
         size=6.2, color=C["muted"], ha="left")
    text(ax_ship, 0.50, 0.60, "Voyages",
         size=6.2, color=C["muted"], ha="right")
    ax_ship.plot([0.05, 0.50], [0.565, 0.565], color="#E0D0C6", lw=0.6)
    top_lanes = voy.head(6)
    for i, (_, r) in enumerate(top_lanes.iterrows()):
        y = 0.50 - i * 0.068
        text(ax_ship, 0.05, y, f"{r.from_n}  →  {r.to_n}",
             size=6.6, color=C["ink"], ha="left")
        text(ax_ship, 0.50, y, f"{int(r.n_voyages)}",
             size=6.6, color=C["ink"], ha="right", weight="bold")
    n_rest = len(voy) - 6
    rest_v = int(voy.iloc[6:]["n_voyages"].sum()) if n_rest else 0
    if n_rest:
        text(ax_ship, 0.05, 0.50 - 6 * 0.068,
             f"+ {n_rest} other lanes  ({rest_v} voyages)",
             size=6.2, color=C["muted"], ha="left")

    # Malacca corridor, kept as a place rather than a voyage count.
    mx, my = 0.78, 0.48
    ax_ship.scatter([0.64], [my], s=86, color=C["ship"], zorder=6,
                    edgecolor="white", linewidth=0.7)
    ax_ship.text(0.64, my - 0.09, "Jurong", fontsize=6.6, color=C["ship"],
                 ha="center", va="top", fontweight="bold")
    ax_ship.plot([0.685, 0.735], [my, my], color="#8FA8C8", lw=1.8, zorder=3)
    ax_ship.scatter([mx], [my], s=74, color=C["choke"], marker="D",
                    zorder=6, edgecolor="white", linewidth=0.6)
    ax_ship.text(mx, my + 0.09, "Malacca", fontsize=7.0, color="#8A3D16",
                 ha="center", va="bottom", fontweight="bold")
    ax_ship.text(mx, my - 0.09, "fixed corridor\nevery week",
                 fontsize=6.0, color=C["muted"], ha="center", va="top")
    text(ax_ship, 0.71, 0.18,
         "PortWatch lag +1 w: 524 tanker\ncalls in week t−1 (7 Mar).",
         size=6.4, color=C["ink"])

    # ---- (D) how each family ingests Jurong ----
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    box(ax_c, 0.00, 0.06, 0.492, 0.96, face=C["flat_bg"], edge=C["flat"],
        lw=1.2, radius=0.012, z=1)
    box(ax_c, 0.508, 0.06, 1.00, 0.96, face=C["deep_bg"], edge=C["deep"],
        lw=1.2, radius=0.012, z=1)
    text(ax_c, 0.246, 0.84, "(D)  Flat ingest  ·  Jurong becomes columns",
         size=8.2, weight="bold", color=C["flat"])
    text(ax_c, 0.246, 0.58,
         "The five anomalies sit among ~260 weekly features.\n"
         "Because the January composite is carried forward,\n"
         "the four lookback weeks are identical for this site.\n"
         "Shipping series are prefixed columns, not neighbours.",
         size=6.8, color=C["ink"])
    text(ax_c, 0.246, 0.22,
         "Ridge / XGBoost never see a patch or a graph.",
         size=6.6, color=C["muted"], style="italic")

    text(ax_c, 0.754, 0.84, "(D)  Deep ingest  ·  Jurong stays a place",
         size=8.2, weight="bold", color=C["deep"])
    text(ax_c, 0.754, 0.58,
         "The 1024-d vector is one of 11 AOI tokens.\n"
         "Temporal then site attention can up-weight Jurong.\n"
         "On the graph it is 1 of 17 nodes, tied to Malacca\n"
         "and to the directed lanes in (C).",
         size=6.8, color=C["ink"])
    text(ax_c, 0.754, 0.22,
         "Gated fusion later mixes z_rs and z_ship with finance.",
         size=6.6, color=C["muted"], style="italic")

    save(fig, "fig_preview_worked_example_jurong")


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
