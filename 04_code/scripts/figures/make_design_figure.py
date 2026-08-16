"""
Figure 3.1 - research design flowchart (Section 3.1).

Layered rounded-box layout in the style of a methodology framework:
data sources → information sets → S1–S4 → model families → shared
evaluation → research questions. Colours match the rest of Chapter 3
(finance blue, remote sensing green, shipping orange). Keep in sync with
Table 3.1 and Section 3.5.

    python 04_code/scripts/figures/make_design_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "05_outputs" / "figures"

C = {
    "fin": "#2E5A88",
    "rs": "#2E7D5B",
    "ship": "#D1622B",
    "fin_bg": "#D4E4F2",
    "rs_bg": "#D0EBDA",
    "ship_bg": "#F5D8C8",
    "set_bg": "#EEF3F7",
    "flat_bg": "#D4E4F2",
    "deep_bg": "#F5D8C8",
    "eval_bg": "#F3E4C8",
    "rq_bg": "#EDE0D6",
    "m0_bg": "#F4F2EE",
    "line": "#3F3F3F",
    "ink": "#22303C",
    "muted": "#5C5C5C",
    "flat": "#2E5A88",
    "deep": "#D1622B",
    "eval": "#6A5A3A",
    "rq": "#7A6A55",
}

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
})


def box(ax, x0, y0, x1, y1, *, face="#FFFFFF", edge=C["line"], lw=1.2,
        ls="solid", radius=1.5, z=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z))


def arrow(ax, xy_from, xy_to, *, color=C["line"], lw=1.15, ls="solid",
          style="-|>", z=4):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=10,
        color=color, linewidth=lw, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0))


def hline(ax, x0, x1, y, *, color=C["line"], lw=1.15, ls="solid", z=3):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, linestyle=ls,
            zorder=z, solid_capstyle="butt")


def vline(ax, x, y0, y1, *, color=C["line"], lw=1.15, ls="solid", z=3):
    ax.plot([x, x], [y0, y1], color=color, linewidth=lw, linestyle=ls,
            zorder=z, solid_capstyle="butt")


def rail(ax, xs_from, y_from, xs_to, y_to, *, y_bar=None, color=C["line"],
         lw=1.15):
    """Drop from several x's onto a horizontal rail, then arrow into targets."""
    if y_bar is None:
        y_bar = (y_from + y_to) / 2.0
    span = list(xs_from) + list(xs_to)
    for x in xs_from:
        vline(ax, x, y_from, y_bar, color=color, lw=lw)
    hline(ax, min(span), max(span), y_bar, color=color, lw=lw)
    for x in xs_to:
        arrow(ax, (x, y_bar), (x, y_to), color=color, lw=lw)


def text(ax, x, y, s, *, size=8, color=C["ink"], weight="normal",
         ha="center", va="center", z=5, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, linespacing=1.32, style=style)


def build() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 12.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Layer 1 — data sources
    # ------------------------------------------------------------------
    src_y0, src_y1 = 89.2, 98.4
    sources = [
        (4.0, 32.0, C["fin_bg"], C["fin"], "Financial time series",
         "Brent spot and futures,\nmacro and oil-market series"),
        (36.0, 64.0, C["rs_bg"], C["rs"], "Remote sensing",
         "Sentinel-2 patches, optical indices,\n"
         "night-time lights  ·  11 AOIs"),
        (68.0, 96.0, C["ship_bg"], C["ship"], "Shipping",
         "AIS port calls and vessel traffic\n"
         "17-node graph  ·  6 chokepoints"),
    ]
    src_cx = []
    for x0, x1, face, edge, head, body in sources:
        box(ax, x0, src_y0, x1, src_y1, face=face, edge=edge, lw=1.35)
        cx = (x0 + x1) / 2
        src_cx.append(cx)
        text(ax, cx, src_y1 - 2.0, head, size=9.4, weight="bold",
             color=edge, va="top")
        text(ax, cx, src_y1 - 4.45, body, size=7.2, color=C["muted"], va="top")

    # ------------------------------------------------------------------
    # Layer 2 — information-set construction (wide)
    # ------------------------------------------------------------------
    info_y0, info_y1 = 82.0, 86.6
    info_cx = 50.0
    box(ax, 4.0, info_y0, 96.0, info_y1, face=C["set_bg"], edge=C["line"],
        lw=1.2)
    text(ax, info_cx, 85.35, "Information sets", size=9.6, weight="bold")
    text(ax, info_cx, 83.15,
         "S2, S3 and S4 extend S1 in parallel  ·  S4 combines both additions",
         size=7.2, color=C["muted"])

    rail(ax, src_cx, src_y0, [info_cx], info_y1, y_bar=87.9)

    # ------------------------------------------------------------------
    # Layer 3 — S1–S4
    # ------------------------------------------------------------------
    set_y0, set_y1 = 68.6, 79.4
    sets = [
        (4.0, 25.5, "S1", "finance only", [C["fin"]]),
        (27.5, 49.0, "S2", "+ remote sensing", [C["fin"], C["rs"]]),
        (51.0, 72.5, "S3", "+ shipping", [C["fin"], C["ship"]]),
        (74.5, 96.0, "S4", "+ both", [C["fin"], C["rs"], C["ship"]]),
    ]
    set_cx = []
    for x0, x1, name, sub, cols in sets:
        box(ax, x0, set_y0, x1, set_y1, face="#FFFFFF", edge=C["line"], lw=1.15)
        cx = (x0 + x1) / 2
        set_cx.append(cx)
        text(ax, cx, 77.55, name, size=11.2, weight="bold")
        text(ax, cx, 74.15, sub, size=7.35, color=C["muted"])
        step = 2.55
        start = cx - step * (len(cols) - 1) / 2
        for i, col in enumerate(cols):
            ax.scatter(start + i * step, 70.55, s=30, color=col, zorder=5,
                       edgecolor="white", linewidth=0.45)

    rail(ax, [info_cx], info_y0, set_cx, set_y1, y_bar=80.7)

    # ------------------------------------------------------------------
    # Layer 4 — model families
    # ------------------------------------------------------------------
    fam_y0, fam_y1 = 45.4, 65.8
    fam = [
        (4.0, 45.8, C["flat_bg"], C["flat"], "Flat family",
         "early feature fusion\n"
         "all predictors stacked into one weekly table\n"
         "(4-week lookback)   ·   Ridge   ·   XGBoost\n"
         "RS input: site optical indices + night-time lights"),
        (54.2, 96.0, C["deep_bg"], C["deep"], "Deep family",
         "representation-level fusion\n"
         "separate encoders: finance · remote sensing · shipping\n"
         "fusion: concatenation · gated (main) · cross-attention\n"
         "RS input: learned Sentinel-2 patch embeddings"),
    ]
    fam_cx = []
    for x0, x1, face, edge, head, body in fam:
        box(ax, x0, fam_y0, x1, fam_y1, face=face, edge=edge, lw=1.35)
        cx = (x0 + x1) / 2
        fam_cx.append(cx)
        text(ax, cx, fam_y1 - 2.2, head, size=10.2, weight="bold",
             color=edge, va="top")
        text(ax, cx, fam_y1 - 5.25, body, size=7.15, color=C["muted"], va="top")

    rail(ax, set_cx, set_y0, fam_cx, fam_y1, y_bar=67.2)

    arrow(ax, (46.2, 55.6), (53.8, 55.6), color=C["muted"], lw=1.05,
          style="<|-|>")
    text(ax, 50.0, 58.4, "paired comparison", size=6.4, color=C["muted"],
         style="italic")
    text(ax, 50.0, 52.7, "same information set", size=6.4, color=C["muted"],
         style="italic")

    # ------------------------------------------------------------------
    # Layer 5 — M0 + shared evaluation
    # ------------------------------------------------------------------
    ev_y0, ev_y1 = 29.4, 42.4
    eval_x0, eval_x1 = 25.8, 96.0
    eval_cx = (eval_x0 + eval_x1) / 2

    box(ax, 4.0, ev_y0, 22.4, ev_y1, face=C["m0_bg"], edge=C["eval"],
        lw=1.15, ls=(0, (4, 2.2)))
    text(ax, 13.2, 40.95, "M0", size=10.2, weight="bold", color=C["eval"])
    text(ax, 13.2, 38.35, "no-change benchmark", size=6.75, color=C["muted"])
    text(ax, 13.2, 35.5, r"$\hat{P}_{t+1|t}=P_t$", size=8.6, color=C["eval"])
    text(ax, 13.2, 32.35, "no predictors, not estimated", size=6.15,
         color=C["muted"], style="italic")

    arrow(ax, (22.4, 35.9), (25.8, 35.9), color=C["eval"], lw=1.05,
          ls=(0, (4, 2.2)))

    box(ax, eval_x0, ev_y0, eval_x1, ev_y1, face=C["eval_bg"], edge=C["eval"],
        lw=1.25)
    text(ax, eval_cx, ev_y1 - 2.05, "Shared expanding-window evaluation",
         size=9.5, weight="bold", color=C["eval"], va="top")
    text(ax, eval_cx, ev_y1 - 4.9,
         "365 Friday weeks (Jan 2019 – Dec 2025)  ·  104-week warm-up  ·  "
         "refit every 13 weeks\n"
         "257 forecast origins  ·  one week ahead  ·  no look-ahead\n"
         "RMSE  ·  MAE  ·  skill vs M0  ·  Diebold–Mariano and Clark–West tests",
         size=7.1, color=C["muted"], va="top")

    rail(ax, fam_cx, fam_y0, [eval_cx], ev_y1, y_bar=43.9)

    # ------------------------------------------------------------------
    # Layer 6 — research questions
    # ------------------------------------------------------------------
    rq_y0, rq_y1 = 6.8, 25.8
    rqs = [
        (4.0, 32.0, "RQ1",
         "Do remote sensing and shipping\nadd forecast value?\n\n"
         "S1–S4 within each family;\nevery model against M0"),
        (36.0, 64.0, "RQ2",
         "Does representation-level\nmodelling beat one flat table?\n\n"
         "paired Flat vs Deep;\nfusion variants inside Deep"),
        (68.0, 96.0, "RQ3",
         "What does the model rely on?\n\n"
         "modality gates and node attention,\nrestricted to Deep models\n"
         "that improve on M0"),
    ]
    rq_cx = []
    for x0, x1, name, body in rqs:
        box(ax, x0, rq_y0, x1, rq_y1, face=C["rq_bg"], edge=C["rq"], lw=1.2)
        cx = (x0 + x1) / 2
        rq_cx.append(cx)
        text(ax, cx, rq_y1 - 2.2, name, size=10.6, weight="bold",
             color=C["rq"], va="top")
        text(ax, cx, rq_y1 - 5.7, body, size=7.35, color=C["ink"], va="top")

    rail(ax, [eval_cx], ev_y0, rq_cx, rq_y1, y_bar=27.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_3_1_research_design.{ext}",
                    pad_inches=0.18)
    plt.close(fig)
    print(f"  saved fig_3_1_research_design.png / .pdf\n\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    build()
