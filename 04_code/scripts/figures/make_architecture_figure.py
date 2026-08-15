"""
Figure 3.5 — Deep model architecture (Section 3.5.2).

Hand-specified canvas; keep in sync with the three encoders and fusion
options described in Section 3.5.2. Colours match Figure 3.1.

    python 04_code/scripts/figures/make_architecture_figure.py
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
    "fin_bg": "#EEF3F7",
    "rs_bg": "#EAF4EE",
    "ship_bg": "#F8EEE8",
    "band": "#F4F2EE",
    "box": "#FFFFFF",
    "line": "#8A8A8A",
    "ink": "#22303C",
    "muted": "#5C5C5C",
    "deep": "#D1622B",
    "eval": "#4A4A4A",
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
            ha=ha, va=va, zorder=z, linespacing=1.35, style=style)


def column(ax, x0, x1, *, color, bg, title, input_title, input_body,
           encoder_title, encoder_body, z_label):
    """One modality column: header, input, encoder, representation."""
    cx = (x0 + x1) / 2
    box(ax, x0, 46.0, x1, 97.2, face=bg, edge=color, lw=1.15, z=1)
    box(ax, x0, 91.4, x1, 97.2, face=color, edge=color, lw=0.0, radius=1.0, z=2)
    text(ax, cx, 94.3, title, size=9.2, color="white", weight="bold")

    box(ax, x0 + 1.2, 77.6, x1 - 1.2, 89.8, face="white", edge=color, lw=1.0)
    text(ax, cx, 87.8, input_title, size=7.6, color=color, weight="bold")
    text(ax, cx, 82.6, input_body, size=7.2, color=C["muted"])

    arrow(ax, (cx, 77.6), (cx, 74.6), color=color, lw=1.05)

    box(ax, x0 + 1.2, 55.0, x1 - 1.2, 74.6, face="white", edge=color, lw=1.0)
    text(ax, cx, 71.8, encoder_title, size=7.6, color=color, weight="bold")
    text(ax, cx, 63.6, encoder_body, size=7.2)

    arrow(ax, (cx, 55.0), (cx, 52.0), color=color, lw=1.05)

    box(ax, x0 + 5.5, 46.8, x1 - 5.5, 52.0, face="white", edge=color, lw=1.25)
    text(ax, cx, 49.4, z_label, size=8.6, color=color, weight="bold")
    return cx


def build() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    ax.set_xlim(-0.4, 100)
    ax.set_ylim(0, 100.6)
    ax.axis("off")

    text(ax, 50, 99.2,
         "S4 shown. Unused encoders are omitted at S2 and S3. "
         "S1 passes $z_{\\mathrm{fin}}$ directly to the head.",
         size=7.3, color=C["muted"], style="italic")

    cx_fin = column(
        ax, 2.5, 32.0,
        color=C["fin"], bg=C["fin_bg"],
        title="Finance",
        input_title="Input",
        input_body="4-week financial sequence\n"
                   "prices, inventories, macro\nand oil-market series",
        encoder_title="Causal TCN",
        encoder_body="Each convolutional layer\nuses only the current and\n"
                     "earlier positions.\nOutput: one vector per origin.",
        z_label=r"$z_{\mathrm{fin}}$",
    )
    cx_rs = column(
        ax, 35.0, 64.5,
        color=C["rs"], bg=C["rs_bg"],
        title="Remote sensing",
        input_title="Input  ·  frozen",
        input_body="Prithvi-EO-2.0 embeddings\n"
                   "from Sentinel-2 patches\n11 AOIs  ·  4-week window",
        encoder_title="Temporal then site attention",
        encoder_body="Temporal attention pools\nfour weeks at each AOI.\n"
                     "Site attention then pools\nthe 11 AOIs into one vector.",
        z_label=r"$z_{\mathrm{rs}}$",
    )
    cx_ship = column(
        ax, 67.5, 97.5,
        color=C["ship"], bg=C["ship_bg"],
        title="Shipping",
        input_title="Input",
        input_body="Weekly 17-node graph\n"
                   "11 AOIs + 6 chokepoints\n4-week lookback",
        encoder_title="GAT with temporal encoding",
        encoder_body="Voyage and corridor links\nare symmetrised for message\n"
                     "passing. Output: one vector\nper forecast origin.",
        z_label=r"$z_{\mathrm{ship}}$",
    )

    # Three representations feed fusion.
    for cx, col in ((cx_fin, C["fin"]), (cx_rs, C["rs"]), (cx_ship, C["ship"])):
        arrow(ax, (cx, 46.8), (50, 41.4), color=col, lw=1.05)

    # S1 bypass: z_fin goes around fusion, along the left margin, into the head.
    z_fin_left = 2.5 + 5.5
    ax.plot([z_fin_left, 1.2], [49.4, 49.4], color=C["fin"], linewidth=1.0,
            linestyle=(0, (4, 2.2)), zorder=4)
    ax.plot([1.2, 1.2], [49.4, 10.9], color=C["fin"], linewidth=1.0,
            linestyle=(0, (4, 2.2)), zorder=4)
    arrow(ax, (1.2, 10.9), (22.0, 10.9), color=C["fin"], lw=1.0,
          ls=(0, (4, 2.2)))
    text(ax, 3.2, 43.6, r"S1: $z_{\mathrm{fin}}$ to head",
         size=6.6, color=C["fin"], style="italic", ha="left")

    box(ax, 2.5, 20.6, 97.5, 41.4, face=C["band"], edge="none", z=1)
    text(ax, 50, 39.6, "Fusion  (S2–S4)", size=8.4, weight="bold",
         color=C["muted"])

    box(ax, 3.5, 22.2, 27.5, 37.6, face="white", edge=C["line"], lw=1.0,
        ls=(0, (4, 2)))
    text(ax, 15.5, 34.8, "Concatenation", size=7.6, weight="bold",
         color=C["muted"])
    text(ax, 15.5, 30.4, "join encoder outputs\nthen a shared MLP",
         size=7.1, color=C["muted"])
    text(ax, 15.5, 24.4, "alternative", size=6.6, color=C["muted"],
         style="italic")

    box(ax, 30.5, 22.2, 69.5, 37.6, face="white", edge=C["deep"], lw=1.55)
    text(ax, 50, 35.4, "Gated fusion  (main)", size=8.4, weight="bold",
         color=C["deep"])
    text(ax, 50, 30.6,
         r"$\alpha=\mathrm{softmax}(\mathrm{MLP}([z_{\mathrm{fin}},\,z_{\mathrm{rs}},\,z_{\mathrm{ship}}]))$",
         size=7.6)
    text(ax, 50, 26.8,
         r"$z=\sum_i \alpha_i\,z_i$",
         size=8.2, color=C["ink"], weight="bold")
    text(ax, 50, 23.8,
         r"$\alpha_i \geq 0$, sum to 1 at each forecast origin",
         size=6.8, color=C["muted"], style="italic")

    box(ax, 72.5, 22.2, 96.5, 37.6, face="white", edge=C["line"], lw=1.0,
        ls=(0, (4, 2)))
    text(ax, 84.5, 34.8, "Cross-attention", size=7.6, weight="bold",
         color=C["muted"])
    text(ax, 84.5, 30.4, "finance as query over\nRS / shipping tokens",
         size=7.1, color=C["muted"])
    text(ax, 84.5, 24.4, "alternative", size=6.6, color=C["muted"],
         style="italic")

    arrow(ax, (50, 22.2), (50, 18.6), color=C["deep"], lw=1.2)

    box(ax, 22.0, 3.2, 50.5, 18.6, face="white", edge=C["eval"], lw=1.15)
    text(ax, 36.25, 15.4, "Regression head", size=8.4, weight="bold",
         color=C["eval"])
    text(ax, 36.25, 11.2, "MLP trained by MSE\non the one-week log return",
         size=7.2)
    text(ax, 36.25, 6.0, r"$\hat{r}_{t+1|t}$", size=9.0, color=C["eval"],
         weight="bold")

    arrow(ax, (50.5, 10.9), (55.0, 10.9), color=C["eval"], lw=1.15)

    box(ax, 55.0, 3.2, 90.0, 18.6, face="white", edge=C["eval"], lw=1.15)
    text(ax, 72.5, 15.4, "Price reconstruction", size=8.4, weight="bold",
         color=C["eval"])
    text(ax, 72.5, 10.6,
         r"$\hat{P}_{t+1|t}=P_t\exp(\hat{r}_{t+1|t})$",
         size=8.6, color=C["eval"])
    text(ax, 72.5, 6.0, "evaluated in USD per barrel",
         size=7.0, color=C["muted"], style="italic")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_3_5_deep_architecture.{ext}")
    plt.close(fig)
    print(f"  saved fig_3_5_deep_architecture.png / .pdf\n\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    build()
