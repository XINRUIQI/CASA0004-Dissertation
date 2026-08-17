"""
Chapter 3 schematic figures (Sections 3.3 and 3.4.2).

  Figure 3.5  two edge classes on the 17-node shipping map
  Figure 3.7  as-of alignment at one Friday forecast origin

Figure 3.5 uses the same geographic frame, markers and colours as Figure 3.3.

    python 04_code/scripts/figures/make_schematic_figures.py
    python 04_code/scripts/figures/make_schematic_figures.py --only 3.5
    python 04_code/scripts/figures/make_schematic_figures.py --only 3.7
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "05_outputs" / "figures"
EDGE_CSV = (ROOT / "03_data" / "processed" / "M3" / "outputs"
            / "m3_graph_edges_weekly.csv")
NE_DIR = ROOT / "03_data" / "raw" / "00_spatial_anchors" / "naturalearth"

# As-of origin: a Friday in mid-March so the previous month's composite is
# complete but not yet eligible (month-end + 15 d), and the current month is
# still open. Appendix A.3: PUB_LAG_DAYS = 15, EIA/PortWatch +1 week.
ASOF_ORIGIN = pd.Timestamp("2025-03-14")
PUB_LAG_DAYS = 15
EIA_LAG_WEEKS = 1

# Appendix A.2.1, (label, type, lon, lat). Same coordinates as Figure 3.3.
AOI = {
    "P001": ("Rotterdam", "port", 4.145, 51.950),
    "P002": ("Fujairah", "terminal", 56.356, 25.199),
    "P003": ("Ras Tanura", "terminal", 50.157, 26.643),
    "P004": ("Singapore/Jurong", "refinery", 103.708, 1.274),
    "P005": ("Houston", "port", -95.100, 29.736),
    "P006": ("Ningbo", "port", 121.982, 29.935),
    "P007": ("Jamnagar", "refinery", 69.860, 22.345),
    "P008": ("Basra", "terminal", 48.810, 29.681),
    "P009": ("Ulsan", "refinery", 129.343, 35.433),
    "P010": ("Kharg", "terminal", 50.324, 29.231),
    "P011": ("Yanbu", "terminal", 38.229, 23.961),
}

# EIA World Oil Transit Chokepoints, representative transit coordinates.
CHOKE = {
    "hormuz": ("Hormuz", 56.25, 26.57),
    "suez": ("Suez", 32.35, 30.42),
    "malacca": ("Malacca", 100.40, 2.50),
    "mandeb": ("Bab el-Mandeb", 43.40, 12.60),
    "panama": ("Panama", -79.55, 9.08),
    "cape": ("Cape", 18.47, -34.36),
}

# Appendix A.4.2: 13 undirected AOI–chokepoint corridor edges.
CORRIDOR = {
    "hormuz": ["P002", "P003", "P007", "P008", "P010"],
    "suez": ["P001", "P011"],
    "malacca": ["P004", "P006", "P009"],
    "mandeb": ["P011"],
    "cape": ["P001"],
    "panama": ["P005"],
}

MAP_EXTENT = (-135.0, -48.0, 152.0, 68.0)  # lon0, lat0, lon1, lat1
INSET_BOUNDS = (46.5, 22.5, 60.0, 32.0)
GULF_AOI = {"P002", "P003", "P008", "P010"}

# Label offsets in degrees: id -> (dx, dy, ha, va).
LABEL_OFF = {
    "P001": (0.0, 4.2, "center", "bottom"),
    "P004": (3.4, -5.2, "left", "top"),
    "P005": (-3.6, 2.4, "right", "bottom"),
    "P006": (-8.0, -4.6, "right", "top"),   # Ningbo: lower left
    "P007": (-3.8, -6.6, "right", "top"),
    "P009": (6.4, 5.2, "left", "bottom"),   # Ulsan: upper right
    "P011": (-6.0, -3.8, "right", "top"),
    "suez": (-3.4, 2.8, "right", "bottom"),
    "malacca": (-3.4, -3.2, "right", "top"),
    "mandeb": (-3.4, -2.2, "right", "top"),
    "panama": (-3.6, 2.2, "right", "bottom"),
    "cape": (2.8, -2.8, "left", "top"),
}
# Inset leaders: unit direction, then scaled to INSET_LEAD_LEN.
INSET_LEAD_LEN = 0.42
INSET_DIR = {
    "P002": (1.0, -1.0),
    "P003": (-1.0, -1.0),
    "P008": (-1.0, 1.0),
    "P010": (1.0, 1.0),
    "hormuz": (1.0, 0.12),
}
INSET_ALIGN = {
    "P002": ("left", "top"),
    "P003": ("right", "top"),
    "P008": ("right", "bottom"),
    "P010": ("left", "bottom"),
    "hormuz": ("left", "center"),
}
HALO = [pe.withStroke(linewidth=2.8, foreground="white")]
VOYAGE_LW_REF = 100.0  # voyages at which line width saturates
WIDTH_LEGEND = (10, 50, 100)

TYPE_MARKER = {"port": "o", "terminal": "s", "refinery": "^"}

C = {
    "aoi": "#2E5A88",
    "choke": "#D1622B",
    "voyage": "#D1622B",
    "corridor": "#4A6D96",
    "land": "#EFEDE8",
    "coast": "#C9C4BB",
    "border": "#DCD7CE",
    "grey": "#9A9A9A",
    "fin": "#2E5A88",
    "rs": "#2E7D5B",
    "ship": "#D1622B",
    "fin_bg": "#EEF3F7",
    "rs_bg": "#EAF4EE",
    "ship_bg": "#F8EEE8",
    "band": "#F4F2EE",
    "box": "#FFFFFF",
    "line": "#8A8A8A",
    "ink": "#22303C",
    "muted": "#5C5C5C",
    "held": "#C9C4BB",
    "blocked": "#B55A4A",
    "look": "#D9E4EF",
}

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
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


# --------------------------------------------------------------------------
# Figure 3.5 — two edge classes on the study-site map
# --------------------------------------------------------------------------
def _load_voyage_edges() -> pd.DataFrame:
    df = pd.read_csv(EDGE_CSV)
    df["week_ending_friday"] = pd.to_datetime(df["week_ending_friday"])
    pos = df[df["n_voyages"] > 0].copy()
    if pos.empty:
        raise ValueError(f"No voyage edges in {EDGE_CSV}")
    return pos


def _median_density_week(pos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    """Week whose directed-lane count is closest to the sample median.

    Ties: closest weekly voyage total to the sample-median total, then earliest
    Friday. No site is favoured.
    """
    n_lanes = pos.groupby("week_ending_friday").size()
    tot = pos.groupby("week_ending_friday")["n_voyages"].sum()
    med_n = float(n_lanes.median())
    med_tot = float(tot.median())
    delta_n = (n_lanes - med_n).abs()
    cands = n_lanes.index[delta_n == delta_n.min()]
    tot_c = tot.loc[cands]
    delta_t = (tot_c - med_tot).abs()
    week = tot_c.index[int(delta_t.to_numpy().argmin())]
    sub = pos[pos["week_ending_friday"] == week].copy()
    return sub, week, int(round(med_n))


def _aoi_xy(sid: str) -> np.ndarray:
    return np.array(AOI[sid][2:4], dtype=float)


def _choke_xy(cid: str) -> np.ndarray:
    return np.array(CHOKE[cid][1:3], dtype=float)


def _in_bounds(xy: np.ndarray, bounds) -> bool:
    lon0, lat0, lon1, lat1 = bounds
    return lon0 <= xy[0] <= lon1 and lat0 <= xy[1] <= lat1


def _trim_curve_ends(pts: np.ndarray, origin_xy, dest_xy, *,
                     gap_deg: float) -> np.ndarray:
    """Stop the polyline gap_deg (lon/lat) short of each node centre."""
    origin = np.asarray(origin_xy, float)
    dest = np.asarray(dest_xy, float)

    def far(xy, node) -> bool:
        return float(np.linalg.norm(np.asarray(xy, float) - node)) >= gap_deg

    i0 = 0
    for i in range(len(pts)):
        if far(pts[i], origin):
            i0 = i
            break
    i1 = len(pts) - 1
    for i in range(len(pts) - 1, -1, -1):
        if far(pts[i], dest):
            i1 = i
            break
    if i1 <= i0 + 1:
        return pts
    return pts[i0:i1 + 1]


def _arrowhead_triangle(ax, pts: np.ndarray, *, color, alpha, head_deg: float) -> None:
    """Fixed-size triangular head in data coordinates; does not scale with line width."""
    tip = np.asarray(pts[-1], float)
    back = np.asarray(pts[max(len(pts) - 8, 0)], float)
    v = tip - back
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return
    u = v / n
    left = np.array([-u[1], u[0]])
    base = tip - u * head_deg
    half = 0.55 * head_deg
    tri = np.vstack([tip, base + left * half, base - left * half])
    ax.add_patch(Polygon(
        tri, closed=True, facecolor=color, edgecolor="none",
        alpha=alpha, zorder=4, clip_on=True))


def _shorten_ll(p0, p1, pad: float):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    dist = np.linalg.norm(v)
    if dist < 1e-8:
        return p0, p1
    pad = min(pad, 0.22 * dist)
    u = v / dist
    return p0 + u * pad, p1 - u * pad


def _voyage_lw(n: float) -> float:
    frac = math.sqrt(min(float(n), VOYAGE_LW_REF) / VOYAGE_LW_REF)
    return 0.55 + 2.55 * frac


def _voyage_alpha(n: float) -> float:
    frac = math.sqrt(min(float(n), VOYAGE_LW_REF) / VOYAGE_LW_REF)
    return 0.64 + 0.28 * frac


def _perp_offset(p0, p1, sign: float, *, inset: bool):
    """Parallel shift so opposite directions do not occupy the same chord."""
    if sign == 0:
        return p0, p1
    v = np.asarray(p1, float) - np.asarray(p0, float)
    nrm = np.array([-v[1], v[0]], dtype=float)
    nlen = float(np.linalg.norm(nrm))
    if nlen < 1e-9:
        return p0, p1
    nrm = nrm / nlen
    d = 0.10 if inset else float(np.clip(0.02 * nlen, 0.85, 1.8))
    return p0 + sign * d * nrm, p1 + sign * d * nrm


def _quad_bezier(p0, p1, sign: float, *, bulge: float, n: int = 48) -> np.ndarray:
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    v = p1 - p0
    dist = float(np.linalg.norm(v))
    if sign == 0 or dist < 1e-8:
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1.0 - t) * p0 + t * p1
    nrm = np.array([-v[1], v[0]], dtype=float)
    nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
    ctrl = 0.5 * (p0 + p1) + sign * bulge * dist * nrm
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * ctrl + t ** 2 * p1


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


def _mark_gulf_box(ax) -> None:
    """Locator rectangle only: light grey stroke, no fill, below graph edges."""
    lon0, lat0, lon1, lat1 = INSET_BOUNDS
    ax.add_patch(Rectangle(
        (lon0, lat0), lon1 - lon0, lat1 - lat0,
        fill=False, facecolor="none", edgecolor="#C5C0B6",
        linewidth=0.9, linestyle="solid", zorder=2, clip_on=True))


def _style_map_axes(ax, *, show_xlabel: bool = True) -> None:
    ax.set_xticks(range(-120, 151, 30))
    ax.set_yticks(range(-30, 61, 30))
    ax.set_yticklabels(
        [f"{abs(v)}\u00b0{'S' if v < 0 else 'N' if v else ''}"
         for v in range(-30, 61, 30)], fontsize=8.0)
    if show_xlabel:
        ax.set_xticklabels(
            [f"{abs(v)}\u00b0{'W' if v < 0 else 'E' if v else ''}"
             for v in range(-120, 151, 30)], fontsize=8.0)
        ax.tick_params(length=2.4, pad=1.4)
    else:
        ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
        ax.tick_params(axis="y", length=2.4, pad=1.4)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(0.7)


def _draw_aoi_nodes(ax, offsets, *, fontsize=9.0, size=58, labels=True) -> None:
    for sid, (name, stype, lon, lat) in AOI.items():
        ax.scatter(lon, lat, marker=TYPE_MARKER[stype], s=size, color=C["aoi"],
                   edgecolor="white", linewidth=0.85, zorder=10)
        if labels and sid in offsets:
            dx, dy, ha, va = offsets[sid]
            ax.annotate(name, (lon + dx, lat + dy), ha=ha, va=va,
                        fontsize=fontsize, color="#1F3E5F", zorder=11,
                        path_effects=HALO)


def _draw_choke_nodes(ax, offsets, *, fontsize=9.0, size=70, labels=True,
                      only=None) -> None:
    for cid, (name, lon, lat) in CHOKE.items():
        if only is not None and cid not in only:
            continue
        ax.scatter(lon, lat, marker="D", s=size, color=C["choke"],
                   edgecolor="white", linewidth=0.85, zorder=10)
        if labels and cid in offsets:
            dx, dy, ha, va = offsets[cid]
            ax.annotate(name, (lon + dx, lat + dy), ha=ha, va=va,
                        fontsize=fontsize, color="#8A3D16", zorder=11,
                        fontweight="bold", path_effects=HALO)


def _inset_label(ax, key: str, name: str, lon: float, lat: float, *,
                 color: str, fontsize: float = 8.0, fontweight: str = "normal"):
    vec = np.array(INSET_DIR[key], dtype=float)
    vec = vec / np.linalg.norm(vec) * INSET_LEAD_LEN
    ha, va = INSET_ALIGN[key]
    ax.text(
        lon + vec[0], lat + vec[1], name,
        ha=ha, va=va, fontsize=fontsize, color=color, fontweight=fontweight,
        zorder=11, path_effects=HALO,
    )


def _draw_inset_nodes(ax, *, chokes: bool) -> None:
    for sid in GULF_AOI:
        name, stype, lon, lat = AOI[sid]
        ax.scatter(lon, lat, marker=TYPE_MARKER[stype], s=56, color=C["aoi"],
                   edgecolor="white", linewidth=0.85, zorder=10)
        _inset_label(ax, sid, name, lon, lat, color="#1F3E5F")
    if chokes:
        name, lon, lat = CHOKE["hormuz"]
        ax.scatter(lon, lat, marker="D", s=64, color=C["choke"],
                   edgecolor="white", linewidth=0.85, zorder=10)
        _inset_label(ax, "hormuz", name, lon, lat, color="#8A3D16",
                     fontweight="bold")


def _draw_voyage_edges(ax, voy: pd.DataFrame, *, keep=None,
                       inset=False) -> None:
    pairs = set(zip(voy["from_site"], voy["to_site"]))
    bulge = 0.22 if inset else 0.15
    # Stop just outside the marker so arrowheads sit close to the node.
    gap_deg = 0.20 if inset else 1.85
    head_deg = 0.20 if inset else 1.45
    for _, row in voy.sort_values("n_voyages").iterrows():
        a, b = row["from_site"], row["to_site"]
        if keep is not None and not ({a, b} <= keep):
            continue
        n = float(row["n_voyages"])
        origin, dest = _aoi_xy(a), _aoi_xy(b)
        sign = 0.0
        if (b, a) in pairs:
            sign = 1.0 if a < b else -1.0
        p0, p1 = _perp_offset(origin, dest, sign, inset=inset)
        pts = _quad_bezier(p0, p1, sign, bulge=bulge, n=64)
        pts = _trim_curve_ends(pts, origin, dest, gap_deg=gap_deg)
        if len(pts) < 2:
            continue
        lw = _voyage_lw(n)
        alpha = _voyage_alpha(n)
        ax.plot(pts[:, 0], pts[:, 1], color=C["voyage"], linewidth=lw,
                alpha=alpha, zorder=3, solid_capstyle="butt",
                solid_joinstyle="round", clip_on=True)
        _arrowhead_triangle(ax, pts, color=C["voyage"], alpha=alpha,
                            head_deg=head_deg)


def _draw_corridor_edges(ax, *, pad: float, keep_ends=None) -> None:
    for cp, sites in CORRIDOR.items():
        cxy = _choke_xy(cp)
        for sid in sites:
            axy = _aoi_xy(sid)
            if keep_ends is not None:
                if not (_in_bounds(cxy, keep_ends) and _in_bounds(axy, keep_ends)):
                    continue
            p0, p1 = _shorten_ll(cxy, axy, pad)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C["corridor"],
                    linewidth=2.30, alpha=0.95, zorder=2, solid_capstyle="round")


def _add_gulf_inset(ax, land_10: Path, *, voyage=None, corridor=False):
    axi = ax.inset_axes([0.035, 0.035, 0.29, 0.36])
    _basemap(axi, INSET_BOUNDS, land_10, coast_lw=0.55, draw_borders=False)
    if voyage is not None:
        _draw_voyage_edges(axi, voyage, keep=GULF_AOI, inset=True)
        _draw_inset_nodes(axi, chokes=False)
    if corridor:
        _draw_corridor_edges(axi, pad=0.14, keep_ends=INSET_BOUNDS)
        _draw_inset_nodes(axi, chokes=True)
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_edgecolor("#9A9A9A")
        sp.set_linewidth(0.7)
    axi.set_title("Persian Gulf — enlarged view", fontsize=8.0, pad=2.4)
    _mark_gulf_box(ax)
    return axi


def _draw_legend(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        (0.035, "o", C["aoi"], "Port AOI"),
        (0.185, "s", C["aoi"], "Terminal AOI"),
        (0.370, "^", C["aoi"], "Refinery AOI"),
        (0.555, "D", C["choke"], "Chokepoint"),
    ]
    for x, mk, color, lab in nodes:
        ax.scatter([x], [0.72], marker=mk, s=58, color=color,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x + 0.018, 0.72, lab, ha="left", va="center",
                fontsize=8.3, color=C["ink"])
    ax.plot([0.72, 0.805], [0.72, 0.72], color=C["corridor"], linewidth=2.30,
            solid_capstyle="round")
    ax.text(0.818, 0.72, "Corridor edge", ha="left", va="center",
            fontsize=8.3, color=C["ink"])

    ax.text(0.02, 0.26, "Voyage edge — weekly count  (width \u221d \u221an)",
            ha="left", va="center", fontsize=8.3, color=C["ink"])
    xs = [0.46, 0.64, 0.82]
    for x, n in zip(xs, WIDTH_LEGEND):
        ax.annotate(
            "", xy=(x + 0.14, 0.26), xytext=(x, 0.26),
            arrowprops=dict(arrowstyle="-|>", color=C["voyage"],
                            lw=_voyage_lw(n), mutation_scale=12),
        )
        ax.text(x + 0.07, 0.04, f"{n} voyages", ha="center", va="bottom",
                fontsize=8.3, color=C["muted"])


def fig36_edge_classes() -> None:
    land_110 = NE_DIR / "ne_110m_land.shp"
    land_10 = NE_DIR / "ne_10m_land.shp"
    if not land_110.exists() or not land_10.exists():
        raise FileNotFoundError(
            f"Natural Earth land not found under {NE_DIR}.")

    pos = _load_voyage_edges()
    voy, week, med_n = _median_density_week(pos)
    week_str = week.strftime("%-d %B %Y")

    fig = plt.figure(figsize=(10.2, 9.35))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.00, 1.00, 0.20],
        hspace=0.22, left=0.055, right=0.99, top=0.965, bottom=0.04)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[2, 0])

    _basemap(ax_a, MAP_EXTENT, land_110)
    _draw_voyage_edges(ax_a, voy)
    _add_gulf_inset(ax_a, land_10, voyage=voy)
    _style_map_axes(ax_a, show_xlabel=False)
    _draw_aoi_nodes(ax_a, LABEL_OFF)
    ax_a.set_title(
        "(a)  Dynamic voyage edges (directed and weighted)",
        loc="left", fontsize=10.0, color=C["ink"], pad=8)

    _basemap(ax_b, MAP_EXTENT, land_110)
    _draw_corridor_edges(ax_b, pad=0.70)
    _add_gulf_inset(ax_b, land_10, corridor=True)
    _style_map_axes(ax_b, show_xlabel=True)
    _draw_aoi_nodes(ax_b, LABEL_OFF)
    _draw_choke_nodes(ax_b, LABEL_OFF)
    ax_b.set_title(
        "(b)  Fixed corridor edges (undirected and unweighted)",
        loc="left", fontsize=10.0, color=C["ink"], pad=8)

    _draw_legend(ax_leg)

    save(fig, "fig_3_5_edge_classes")
    print(f"  panel (a) week {week_str}; {len(voy)} lanes "
          f"(sample median {med_n})")


# --------------------------------------------------------------------------
# Figure 3.7 — as-of alignment
# --------------------------------------------------------------------------
def _availability(month_end: pd.Timestamp) -> pd.Timestamp:
    return month_end + pd.Timedelta(days=PUB_LAG_DAYS)


def fig37_asof_alignment() -> None:
    t = ASOF_ORIGIN
    target = t + pd.Timedelta(weeks=1)
    look_fridays = [t - pd.Timedelta(weeks=k) for k in (3, 2, 1, 0)]
    jan_end = pd.Timestamp("2025-01-31")
    feb_end = pd.Timestamp("2025-02-28")
    jan_avail = _availability(jan_end)
    feb_avail = _availability(feb_end)

    x0 = pd.Timestamp("2025-02-14")
    x1 = pd.Timestamp("2025-03-26")

    fig = plt.figure(figsize=(10.8, 5.70))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.22, 0.78], wspace=0.02,
                          left=0.02, right=0.98, top=0.82, bottom=0.16)
    ax_lab = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=ax_lab)
    ax_lab.axis("off")
    ax_lab.set_ylim(-0.15, 3.55)
    ax.set_ylim(-0.15, 3.55)

    def dnum(ts):
        return mdates.date2num(pd.Timestamp(ts))

    ax.set_xlim(dnum(x0), dnum(x1))
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=7.6, length=3.5, color="#AAAAAA")
    ax.set_axisbelow(True)

    # lookback band (weeks t-3 to t)
    ax.axvspan(dnum(look_fridays[0] - pd.Timedelta(days=6)), dnum(t),
               color=C["look"], alpha=0.55, zorder=0, lw=0)
    ax.axvline(dnum(t), color=C["ink"], linewidth=1.2, zorder=5)
    ax.axvline(dnum(target), color=C["blocked"], linewidth=1.0,
               linestyle=(0, (4, 2.2)), zorder=5)

    ax.text(dnum(t), 3.42, "forecast origin  t\nFriday 14 Mar 2025",
            ha="center", va="bottom", fontsize=7.6, color=C["ink"],
            fontweight="bold")
    ax.text(dnum(target), 3.42, "target  t+1",
            ha="center", va="bottom", fontsize=7.2, color=C["blocked"])
    ax.text(dnum(look_fridays[1]), 3.28,
            "four-week input window  (weeks t−3 to t)",
            ha="center", va="top", fontsize=7.2, color="#3D5A73")

    rows = [
        (2.55, "Daily finance", "Friday last  ·  lag 0", C["fin"], C["fin_bg"]),
        (1.55, "EIA / PortWatch", "+1 week buffer", C["ship"], C["ship_bg"]),
        (0.55, "Monthly remote sensing",
         "month-end + 15 days,\nthen carry forward", C["rs"], C["rs_bg"]),
    ]
    ax_lab.set_xlim(0, 1)
    for y, title, sub, edge, face in rows:
        ax.axhspan(y - 0.42, y + 0.42, color=face, zorder=1)
        ax_lab.text(0.04, y + 0.14, title, fontsize=8.3, color=edge,
                    fontweight="bold", va="center", ha="left")
        ax_lab.text(0.04, y - 0.16, sub, fontsize=6.7, color=C["muted"],
                    va="center", ha="left", linespacing=1.25)

    def week_bar(y, friday, *, color, hatch=None, lw=0.8, edge="white"):
        start = friday - pd.Timedelta(days=6)
        ax.add_patch(Rectangle(
            (dnum(start), y - 0.20), 6.0, 0.40,
            facecolor=color, edgecolor=edge, linewidth=lw, hatch=hatch,
            zorder=3))

    for fri in look_fridays:
        week_bar(2.55, fri, color=C["fin"])
    ax.text(dnum(t) + 0.55, 2.55, "week t is in",
            fontsize=6.8, color=C["fin"], va="center", ha="left")

    for fri in look_fridays[:-1]:
        week_bar(1.55, fri, color=C["ship"])
    week_bar(1.55, t, color="white", hatch="///", edge=C["ship"], lw=1.05)
    ax.text(dnum(t) - 3.0, 1.55, "withheld",
            fontsize=6.6, color=C["ship"], va="center", ha="center", zorder=4)
    ax.text(dnum(look_fridays[-2]) - 3.0, 1.88, "latest used: week t−1",
            fontsize=6.8, color=C["ship"], ha="center", va="bottom")

    # January composite: eligible from 15 Feb, carried through t.
    ax.add_patch(Rectangle(
        (dnum(jan_avail), 0.55 - 0.20),
        dnum(t) - dnum(jan_avail), 0.40,
        facecolor=C["rs"], edgecolor="white", linewidth=0.6, zorder=3))
    ax.text((dnum(max(jan_avail, x0)) + dnum(t)) / 2, 0.55,
            "January composite  (carried forward)",
            fontsize=7.0, color="white", ha="center", va="center", zorder=4)

    # February becomes eligible the day after t; March is still open.
    ax.plot([dnum(feb_avail), dnum(feb_avail)], [0.55 - 0.42, 0.55 + 0.55],
            color=C["rs"], linewidth=0.95, linestyle=(0, (3, 1.8)), zorder=4)
    ax.scatter([dnum(feb_avail)], [0.55 + 0.55], s=18, color=C["rs"],
               zorder=5, clip_on=False)
    ax.text(dnum(feb_avail) + 0.35, 0.55 + 0.55,
            "February eligible 15 Mar\n(after this origin)",
            fontsize=6.6, color=C["rs"], ha="left", va="bottom")
    ax.text(dnum(t) + 0.55, 0.18,
            "Not in the input at t:  February (embargoed)  ·  March (not yet observed)",
            fontsize=7.0, color=C["blocked"], ha="left", va="center")

    ax.legend(handles=[
        Patch(facecolor=C["fin"], edgecolor="none", label="Used at origin t"),
        Patch(facecolor="white", edgecolor=C["line"], hatch="///",
              label="Observed but not yet eligible"),
        Patch(facecolor=C["look"], edgecolor="none",
              label="Four-week lookback"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=3,
       frameon=False)

    fig.suptitle(
        "As-of alignment at one Friday forecast origin\n"
        "The current month’s satellite composite is not in the input; "
        "neither is the previous month until month-end + 15 days.",
        x=0.02, ha="left", fontsize=10.2, color=C["ink"],
    )

    save(fig, "fig_3_7_asof_alignment")


FIGURES = {
    "3.5": fig36_edge_classes,
    "3.7": fig37_asof_alignment,
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
