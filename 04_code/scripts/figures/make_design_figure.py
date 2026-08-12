"""
Figure 3.1 - research design flowchart (Section 3.1).

Layout is hand-specified on a 0-100 canvas; nothing is read from data. Keep in
sync with Table 3.1 (information sets) and Section 3.5 (evaluation settings).

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
    "band": "#F4F2EE",
    "box": "#FFFFFF",
    "line": "#8A8A8A",
    "ink": "#22303C",
    "muted": "#5C5C5C",
    "flat": "#2E5A88",
    "deep": "#D1622B",
    "eval": "#4A4A4A",
    "rq": "#7A6A55",
}

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
})


def box(ax, x0, y0, x1, y1, *, face=C["box"], edge=C["line"], lw=1.0,
        ls="solid", radius=1.2, z=2):
    p = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, xy_from, xy_to, *, color=C["line"], lw=1.1, ls="solid",
          style="-|>", rad=0.0, z=4):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=11,
        color=color, linewidth=lw, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}"))


def text(ax, x, y, s, *, size=8, color=C["ink"], weight="normal",
         ha="center", va="center", z=5, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, linespacing=1.45, style=style)


def build() -> None:
    fig, ax = plt.subplots(figsize=(9.8, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------------- Layer A: data sources ----------------
    sources = [
        (4, 30, C["fin"], "Financial time series",
         "Brent spot and futures,\nmacro and oil-market series"),
        (36, 62, C["rs"], "Remote sensing",
         "Sentinel-2 patches, optical indices,\nnight-time lights  ·  11 AOIs"),
        (68, 94, C["ship"], "Shipping",
         "AIS port calls and vessel traffic\n17-node graph  ·  6 chokepoints"),
    ]
    for x0, x1, col, head, body in sources:
        box(ax, x0, 84.5, x1, 95.5, face="white", edge=col, lw=1.4)
        cx = (x0 + x1) / 2
        text(ax, cx, 92.4, head, size=9.5, weight="bold", color=col)
        text(ax, cx, 88.2, body, size=7.6, color=C["muted"])
        # The finance box sits left of the information-set band, so its arrow
        # angles in rather than dropping straight down.
        x_end, rad = (31.0, -0.16) if x0 == 4 else (cx, 0.0)
        arrow(ax, (cx, 84.5), (x_end, 78.7), color=col, lw=1.2, rad=rad)

    # ---------------- Layer B: information sets ----------------
    box(ax, 26, 62.5, 96, 78.5, face=C["band"], edge="none", z=1)
    text(ax, 28, 76.4, "Information sets", size=8.5, weight="bold",
         color=C["muted"], ha="left")

    for i, (col, lab) in enumerate([(C["fin"], "financial"), (C["rs"], "remote sensing"),
                                    (C["ship"], "shipping")]):
        x = 55 + i * 13.5
        ax.scatter(x, 76.4, s=24, color=col, zorder=5, edgecolor="white",
                   linewidth=0.5)
        text(ax, x + 1.4, 76.4, lab, size=6.9, color=C["muted"], ha="left")
    text(ax, 94, 63.6, "S2, S3 and S4 extend S1 in parallel", size=6.9,
         color=C["muted"], style="italic", ha="right")

    sets = [
        (28.5, 44.0, "S1", "finance only", [C["fin"]]),
        (45.0, 60.5, "S2", "+ remote sensing", [C["fin"], C["rs"]]),
        (61.5, 77.0, "S3", "+ shipping", [C["fin"], C["ship"]]),
        (78.0, 93.5, "S4", "+ both", [C["fin"], C["rs"], C["ship"]]),
    ]
    for x0, x1, name, sub, cols in sets:
        box(ax, x0, 64.5, x1, 74.5, face="white", edge=C["line"], lw=1.0)
        cx = (x0 + x1) / 2
        text(ax, cx, 71.6, name, size=10, weight="bold")
        text(ax, cx, 68.9, sub, size=7.6, color=C["muted"])
        step = 2.4
        start = cx - step * (len(cols) - 1) / 2
        for i, col in enumerate(cols):
            ax.scatter(start + i * step, 66.4, s=26, color=col, zorder=5,
                       edgecolor="white", linewidth=0.5)

    # ---------------- M0 benchmark ----------------
    box(ax, 3, 63.5, 22, 75.5, face="white", edge=C["eval"], lw=1.1, ls=(0, (4, 2)))
    text(ax, 12.5, 72.6, "M0", size=10, weight="bold", color=C["eval"])
    text(ax, 12.5, 69.6, "no-change benchmark", size=7.6, color=C["muted"])
    text(ax, 12.5, 67.0, "$\\hat{P}_{t+1|t}=P_t$", size=9, color=C["eval"])
    text(ax, 12.5, 64.8, "no predictors, not estimated", size=6.8,
         color=C["muted"], style="italic")

    # ---------------- Layer C: model families ----------------
    arrow(ax, (61, 62.5), (32, 55.2), color=C["line"], rad=0.12)
    arrow(ax, (61, 62.5), (76, 55.2), color=C["line"], rad=-0.12)

    box(ax, 14, 33.5, 51, 55.0, face="white", edge=C["flat"], lw=1.4)
    text(ax, 32.5, 51.6, "Flat family", size=9.5, weight="bold", color=C["flat"])
    text(ax, 32.5, 48.6, "early feature fusion", size=7.6, color=C["muted"],
         style="italic")
    text(ax, 32.5, 44.6,
         "all predictors stacked into one weekly\ntable (4-week lookback)",
         size=7.4)
    text(ax, 32.5, 40.2, "learners:  Ridge  ·  XGBoost", size=7.4)
    text(ax, 32.5, 35.9,
         "RS input: site optical indices\n+ night-time lights",
         size=7.2, color=C["rs"])

    box(ax, 57, 33.5, 94, 55.0, face="white", edge=C["deep"], lw=1.4)
    text(ax, 75.5, 51.6, "Deep family", size=9.5, weight="bold", color=C["deep"])
    text(ax, 75.5, 48.6, "representation-level fusion", size=7.6,
         color=C["muted"], style="italic")
    text(ax, 75.5, 44.6,
         "separate encoders per data type:\n"
         "finance  ·  remote sensing  ·  shipping graph", size=7.4)
    text(ax, 75.5, 40.2,
         "fusion:  concatenation  ·  gated (main)  ·  cross-attention",
         size=6.8)
    text(ax, 75.5, 35.9,
         "RS input: learned Sentinel-2\npatch embeddings (no night-time lights)",
         size=7.2, color=C["rs"])

    arrow(ax, (51.4, 44.2), (56.6, 44.2), color=C["muted"], lw=1.0,
          style="<|-|>")
    ax.plot([54, 54], [32.6, 42.9], color="#C4C0BA", linewidth=0.8,
            linestyle=(0, (2, 2)), zorder=1)
    text(ax, 54, 30.6, "paired comparison\nsame information set", size=7.0,
         color=C["muted"], style="italic")

    # ---------------- Layer D: shared evaluation ----------------
    arrow(ax, (32.5, 33.5), (32.5, 27.2), color=C["flat"], lw=1.2)
    arrow(ax, (75.5, 33.5), (75.5, 27.2), color=C["deep"], lw=1.2)
    # M0 enters the evaluation directly, bypassing estimation.
    arrow(ax, (12.5, 63.5), (12.5, 27.2), color=C["eval"], lw=1.0,
          ls=(0, (4, 2)))

    box(ax, 4, 14.0, 96, 27.0, face=C["band"], edge=C["eval"], lw=1.1)
    text(ax, 50, 24.4, "Shared expanding-window evaluation", size=9.5,
         weight="bold", color=C["eval"])
    text(ax, 50, 20.2,
         "365 Friday weeks (4 Jan 2019 – 26 Dec 2025)  ·  104-week warm-up  ·  "
         "refit every 13 weeks\n"
         "257 forecast origins (22 Jan 2021 – 19 Dec 2025)  ·  one week ahead  ·  "
         "no look-ahead", size=7.8)
    text(ax, 50, 15.9,
         "RMSE  ·  MAE  ·  skill vs M0  ·  "
         "Diebold–Mariano and Clark–West tests", size=7.6, color=C["muted"])

    # ---------------- Layer E: research questions ----------------
    rqs = [
        (4, 32, "RQ1",
         "Do remote sensing and shipping\nadd forecast value?\n"
         "S1–S4 within each family;\nevery model against M0"),
        (35, 65, "RQ2",
         "Does representation-level\nmodelling beat one flat table?\n"
         "paired Flat vs Deep;\nfusion variants inside Deep"),
        (68, 96, "RQ3",
         "What does the model rely on?\nmodality gates and node attention,\n"
         "restricted to Deep models\nthat improve on M0"),
    ]
    for x0, x1, name, body in rqs:
        arrow(ax, ((x0 + x1) / 2, 14.0), ((x0 + x1) / 2, 11.2), color=C["rq"],
              lw=1.0)
        box(ax, x0, 0.5, x1, 11.0, face="white", edge=C["rq"], lw=1.1)
        text(ax, (x0 + x1) / 2, 9.0, name, size=9, weight="bold", color=C["rq"])
        text(ax, (x0 + x1) / 2, 4.8, body, size=7.2, color=C["ink"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_3_1_research_design.{ext}")
    plt.close(fig)
    print(f"  saved fig_3_1_research_design.png / .pdf\n\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    build()
