"""
Chapter 3 schematic figures (Sections 3.3 and 3.4.2).

  Figure 3.6  two edge classes in the 17-node shipping graph
  Figure 3.7  as-of alignment at one Friday forecast origin

Colours match Figures 3.1 / 3.3 / 3.5.

    python 04_code/scripts/figures/make_schematic_figures.py
    python 04_code/scripts/figures/make_schematic_figures.py --only 3.6
    python 04_code/scripts/figures/make_schematic_figures.py --only 3.7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "05_outputs" / "figures"
EDGE_CSV = (ROOT / "03_data" / "processed" / "M3" / "outputs"
            / "m3_graph_edges_weekly.csv")

# As-of origin: a Friday in mid-March so the previous month's composite is
# complete but not yet eligible (month-end + 15 d), and the current month is
# still open. Appendix A.3: PUB_LAG_DAYS = 15, EIA/PortWatch +1 week.
ASOF_ORIGIN = pd.Timestamp("2025-03-14")
PUB_LAG_DAYS = 15
EIA_LAG_WEEKS = 1

AOI = {
    "P001": ("Rotterdam", "port"),
    "P002": ("Fujairah", "terminal"),
    "P003": ("Ras Tanura", "terminal"),
    "P004": ("Jurong", "refinery"),
    "P005": ("Houston", "port"),
    "P006": ("Ningbo", "port"),
    "P007": ("Jamnagar", "refinery"),
    "P008": ("Basra", "terminal"),
    "P009": ("Ulsan", "refinery"),
    "P010": ("Kharg", "terminal"),
    "P011": ("Yanbu", "terminal"),
}

# Matrix axis order: type groups (port / terminal / refinery), west→east within type.
AOI_ORDER = [
    "P005", "P001", "P006",          # ports
    "P011", "P008", "P010", "P003", "P002",  # terminals
    "P007", "P004", "P009",          # refineries
]

CHOKE = {
    "hormuz": "Hormuz",
    "suez": "Suez",
    "malacca": "Malacca",
    "mandeb": "Bab el-Mandeb",
    "panama": "Panama",
    "cape": "Cape",
}

# Appendix A.4.2: 13 undirected AOI–chokepoint corridor links.
CORRIDOR = {
    "hormuz": ["P002", "P003", "P007", "P008", "P010"],
    "suez": ["P001", "P011"],
    "malacca": ["P004", "P006", "P009"],
    "mandeb": ["P011"],
    "cape": ["P001"],
    "panama": ["P005"],
}

# Schematic coordinates shared by both panels of Figure 3.6 (not a map).
# Gulf AOIs fan around Hormuz so Ras Tanura–Fujairah–Jurong is a triangle
# and the five Hormuz corridor links do not sit on top of one another.
AOI_XY = {
    "P005": (8, 68),    # Houston
    "P001": (24, 90),   # Rotterdam
    "P011": (16, 52),   # Yanbu
    "P008": (36, 86),   # Basra (NW of Hormuz)
    "P010": (40, 58),   # Kharg (W of Hormuz)
    "P003": (26, 28),   # Ras Tanura (SW)
    "P002": (78, 48),   # Fujairah (SE, outside the Gulf)
    "P007": (46, 8),    # Jamnagar (S; below the RT–Jurong trunk)
    "P004": (84, 16),   # Jurong
    "P006": (94, 54),   # Ningbo
    "P009": (92, 82),   # Ulsan
}

CHOKE_XY = {
    "panama": (8, 46),
    "cape": (50, 94),
    "suez": (22, 70),
    "mandeb": (12, 38),
    "hormuz": (64, 72),
    "malacca": (86, 36),
}

# Label (dx, dy, ha, va) relative to node.
AOI_LAB = {
    "P005": (0, 6.8, "center", "bottom"),
    "P001": (-7.2, 0, "right", "center"),
    "P011": (7.2, 1.2, "left", "center"),
    "P008": (0, 6.6, "center", "bottom"),
    "P010": (-7.2, 0, "right", "center"),
    "P003": (0, -6.6, "center", "top"),
    "P002": (7.4, 1.2, "left", "center"),
    "P007": (-7.4, 0, "right", "center"),
    "P004": (7.4, 0, "left", "center"),
    "P006": (7.2, 0, "left", "center"),
    "P009": (0, 6.6, "center", "bottom"),
}

CHOKE_LAB = {
    "panama": (7.8, 0, "left", "center"),
    "cape": (7.8, 0, "left", "center"),
    "suez": (-7.8, 0, "right", "center"),
    "mandeb": (0, -6.4, "center", "top"),
    "hormuz": (0, 6.6, "center", "bottom"),
    "malacca": (7.8, 0, "left", "center"),
}

TYPE_MARKER = {"port": "o", "terminal": "s", "refinery": "^"}

C = {
    "aoi": "#2E5A88",
    "choke": "#D1622B",
    "voyage": "#D1622B",
    "corridor": "#8FA8C8",
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
# Figure 3.6 — two edge classes
# --------------------------------------------------------------------------
def _load_voyage_all() -> tuple[pd.DataFrame, int, pd.Timestamp, pd.Timestamp]:
    """Directed AOI→AOI lanes summed over the full voyage sample."""
    df = pd.read_csv(EDGE_CSV)
    df["week_ending_friday"] = pd.to_datetime(df["week_ending_friday"])
    pos = df[df["n_voyages"] > 0].copy()
    if pos.empty:
        raise ValueError(f"No voyage edges in {EDGE_CSV}")
    agg = (pos.groupby(["from_site", "to_site"], as_index=False)["n_voyages"]
           .sum())
    n_weeks = int(pos["week_ending_friday"].nunique())
    t0 = pos["week_ending_friday"].min()
    t1 = pos["week_ending_friday"].max()
    return agg, n_weeks, t0, t1


def _voyage_log_matrix(voy: pd.DataFrame, order: list[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(order)}
    n = len(order)
    mat = np.full((n, n), np.nan)
    for _, row in voy.iterrows():
        mat[idx[row["from_site"]], idx[row["to_site"]]] = np.log1p(
            float(row["n_voyages"]))
    return mat


def _draw_aoi_nodes(ax, *, labels=True):
    for sid, (name, stype) in AOI.items():
        x, y = AOI_XY[sid]
        ax.scatter(x, y, marker=TYPE_MARKER[stype], s=78, color=C["aoi"],
                   edgecolor="white", linewidth=0.8, zorder=6)
        if labels:
            dx, dy, ha, va = AOI_LAB[sid]
            ax.text(x + dx, y + dy, name, fontsize=7.0, color="#1F3E5F",
                    ha=ha, va=va, zorder=7)


def _draw_choke_nodes(ax):
    for cid, name in CHOKE.items():
        x, y = CHOKE_XY[cid]
        ax.scatter(x, y, marker="D", s=92, color=C["choke"],
                   edgecolor="white", linewidth=0.8, zorder=6)
        dx, dy, ha, va = CHOKE_LAB[cid]
        ax.text(x + dx, y + dy, name, fontsize=7.0, color="#8A3D16",
                ha=ha, va=va, zorder=7, fontweight="bold")


def _style_network(ax, title, subtitle):
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 102)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=9.5, color=C["ink"], pad=8)
    ax.text(0.0, 99.2, subtitle, fontsize=7.2, color=C["muted"],
            ha="left", va="top", transform=ax.transData, linespacing=1.35)


def _draw_voyage_heatmap(ax, cax, mat: np.ndarray, n_weeks: int) -> None:
    n = mat.shape[0]
    vmax = float(np.nanmax(mat))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "voyage", ["#F7D7C4", "#E89A6A", "#D1622B", "#8A3D16"])
    cmap.set_bad("#F4F2EE")

    ax.imshow(mat, origin="upper", cmap=cmap, vmin=0, vmax=vmax,
              aspect="equal", interpolation="nearest")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.85, zorder=3)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_color("#C8C4BC")
        spine.set_linewidth(0.7)
    ax.set_xlabel("Destination AOI", fontsize=8.0, color=C["muted"], labelpad=18)
    ax.set_ylabel("Origin AOI", fontsize=8.0, color=C["muted"], labelpad=2)
    ax.set_title(
        "(a)  Observed voyage connectivity  ·  directed and time-varying",
        loc="left", fontsize=9.5, color=C["ink"], pad=10)
    ax.text(
        0.0, 1.02,
        f"Cell colour shows log total voyages over {n_weeks} weeks.  "
        "Weekly edge sets are substantially sparser and vary over time.",
        transform=ax.transAxes, fontsize=7.2, color=C["muted"],
        ha="left", va="bottom", clip_on=False)

    for i, sid in enumerate(AOI_ORDER):
        name, stype = AOI[sid]
        mk = dict(marker=TYPE_MARKER[stype], s=26, color=C["aoi"],
                  edgecolor="white", linewidth=0.4, clip_on=False, zorder=5)
        ax.scatter(-0.92, i, **mk)
        ax.text(-1.18, i, name, ha="right", va="center", fontsize=7.0,
                color="#1F3E5F", clip_on=False)
        ax.scatter(i, n - 0.5 + 0.42, **mk)
        ax.text(i, n - 0.5 + 0.72, name, ha="right", va="top", fontsize=7.0,
                color="#1F3E5F", rotation=45, rotation_mode="anchor",
                clip_on=False)

    bands = [("port", 0, 3), ("terminal", 3, 8), ("refinery", 8, 11)]
    band_col = {"port": "#1F4E79", "terminal": "#2E5A88", "refinery": "#7A93AE"}
    for stype, i0, i1 in bands:
        ax.add_patch(Rectangle(
            (-0.72, i0 - 0.5), 0.18, i1 - i0,
            facecolor=band_col[stype], edgecolor="none",
            clip_on=False, zorder=4))

    cbar = plt.colorbar(ax.images[0], cax=cax)
    ticks = np.log1p(np.array([1, 10, 100, 1000, 10000], dtype=float))
    ticks = ticks[ticks <= vmax + 1e-6]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["1", "10", "100", "1 000", "10 000"][:len(ticks)])
    cbar.ax.tick_params(labelsize=6.6, length=2.5, color="#BBBBBB")
    cbar.set_label("Total voyages  (log scale)", fontsize=7.2, color=C["muted"])
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#C8C4BC")


def fig36_edge_classes() -> None:
    voy, n_weeks, _t0, _t1 = _load_voyage_all()
    n_lanes = len(voy)
    n_voyages = float(voy["n_voyages"].sum())
    mat = _voyage_log_matrix(voy, AOI_ORDER)

    fig = plt.figure(figsize=(11.2, 6.90))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.0, 0.09, 0.24],
        width_ratios=[1.18, 0.055, 1.12],
        hspace=0.06, wspace=0.10,
        left=0.11, right=0.99, top=0.86, bottom=0.045)
    ax_l = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])
    ax_leg = fig.add_subplot(gs[1, :])
    ax_n = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    ax_n.axis("off")

    fig.suptitle("Two edge classes in the 17-node shipping graph",
                 x=0.11, ha="left", fontsize=11, color=C["ink"])

    _draw_voyage_heatmap(ax_l, cax, mat, n_weeks)

    _style_network(
        ax_r,
        "(b)  Specified corridor connectivity  ·  undirected and fixed",
        "13 AOI–chokepoint links, specified ex ante.  Identical every week.",
    )
    for cp, sites in CORRIDOR.items():
        cx, cy = CHOKE_XY[cp]
        for sid in sites:
            x, y = AOI_XY[sid]
            ax_r.plot([cx, x], [cy, y], color=C["corridor"], linewidth=1.45,
                      alpha=0.95, zorder=2, solid_capstyle="round")
    _draw_aoi_nodes(ax_r)
    _draw_choke_nodes(ax_r)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=C["aoi"],
               markeredgecolor="white", markersize=7, label="AOI  ·  port"),
        Line2D([], [], marker="s", linestyle="", color=C["aoi"],
               markeredgecolor="white", markersize=7, label="AOI  ·  terminal"),
        Line2D([], [], marker="^", linestyle="", color=C["aoi"],
               markeredgecolor="white", markersize=7, label="AOI  ·  refinery"),
        Line2D([], [], marker="D", linestyle="", color=C["choke"],
               markeredgecolor="white", markersize=7, label="Chokepoint"),
        Line2D([], [], color=C["corridor"], linewidth=1.8,
               label="Undirected corridor edge"),
    ]
    ax_leg.legend(handles=handles, loc="center", ncol=5, columnspacing=1.35,
                  handletextpad=0.45, fontsize=7.4)

    box(ax_n, 0.015, 0.12, 0.985, 0.90, face=C["band"], edge="none",
        radius=0.02, z=1)
    ax_n.set_xlim(0, 1)
    ax_n.set_ylim(0, 1)
    ax_n.text(
        0.50, 0.66,
        "Entering the GAT: the two classes are stacked, then symmetrised "
        "and self-looped.  Direction and edge type are not retained.",
        ha="center", va="center", fontsize=8.4, color=C["ink"],
        fontweight="bold",
    )
    ax_n.text(
        0.50, 0.32,
        f"(a) {n_lanes} directed lanes; {n_voyages:,.0f} voyages.  "
        "Symmetrised voyage counts enter only as an attention prior  "
        "(Appendix A.4.3).",
        ha="center", va="center", fontsize=7.5, color=C["muted"],
    )

    save(fig, "fig_3_6_edge_classes")


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
    "3.6": fig36_edge_classes,
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
