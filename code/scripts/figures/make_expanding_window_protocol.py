"""
Expanding-window walk-forward protocol figure.

Main panel: shared outer calendar only. Lower panel: Fit-1 zoom as one
continuous flowchart (Flat vs Deep inner validation, merge, weekly
forecast, M0 parallel path, 257-week scoring).

    python code/scripts/figures/make_expanding_window_protocol.py
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "figures"

LOOKBACK = 4
MIN_TRAIN = 104
RETRAIN_EVERY = 13
N_RAW = 365
N_ELIGIBLE = 361
N_TEST = 257
N_FITS = 20

C = {
    "train": "#8FA8C8",
    "val": "#E7A33E",
    "refit": "#2E5A88",
    "frozen": "#E8B08A",
    "target": "#C44B2A",
    "lookback": "#3D6B9A",
    "deep": "#4F7D6E",
    "ink": "#22303C",
    "muted": "#5C5C5C",
    "line": "#C9C4BB",
    "paper": "#F4F2EE",
}

mpl.rcParams.update({
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _weeks():
    raw = pd.date_range("2019-01-04", "2025-12-26", freq="W-FRI")
    eligible = raw[LOOKBACK - 1:-1]
    test = eligible[MIN_TRAIN:]
    assert len(raw) == N_RAW and len(eligible) == N_ELIGIBLE and len(test) == N_TEST
    return raw, eligible, test


def _arrow(ax, x0, y0, x1, y1, color, lw=1.2):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8.5,
        linewidth=lw, color=color, zorder=6, shrinkA=0.5, shrinkB=0.5,
        clip_on=True))


def _flow(ax, pts, color, lw=1.25):
    """Orthogonal polyline; arrowhead only on the last segment."""
    if len(pts) > 2:
        ax.plot(
            [p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
            color=color, lw=lw, zorder=5, solid_capstyle="round",
            solid_joinstyle="miter", clip_on=True)
    ax.add_patch(FancyArrowPatch(
        pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, zorder=6, shrinkA=0, shrinkB=0.4,
        clip_on=True))


def _rbox(ax, x, y, w, h, fc, ec, lw=0.9):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    return p


def draw_calendar(ax, eligible, test) -> None:
    first_origin = eligible[0]
    week = pd.Timedelta(weeks=1)
    n_blocks = N_FITS

    for b in range(n_blocks):
        lo = b * RETRAIN_EVERY
        hi = min(lo + RETRAIN_EVERY, N_TEST)
        t_first, t_last = test[lo], test[hi - 1]
        y = n_blocks - 1 - b
        n_rest = hi - lo - 1

        ax.barh(y, (t_first - first_origin).days, left=first_origin, height=0.72,
                color=C["train"], edgecolor="white", linewidth=0.2, zorder=2)
        ax.barh(y, 7, left=t_first, height=0.72,
                color=C["refit"], edgecolor="white", linewidth=0.2, zorder=4)
        if n_rest > 0:
            ax.barh(y, n_rest * 7, left=t_first + week, height=0.72,
                    color=C["frozen"], edgecolor="white", linewidth=0.2, zorder=3)
        ax.plot(t_first, y, marker="v", markersize=5.0, color=C["refit"],
                markeredgecolor="white", markeredgewidth=0.4, zorder=6, clip_on=False)
        label = "Fit 1  (zoomed below)" if b == 0 else f"Fit {b + 1}"
        ax.text(first_origin - pd.Timedelta(days=22), y, label,
                ha="right", va="center",
                fontsize=6.6 if b else 7.0,
                color=C["refit"] if b == 0 else C["muted"],
                fontweight="bold" if b == 0 else "normal")

    t0 = test[0]
    ax.axvline(t0, color=C["ink"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=5)
    ax.text(t0, n_blocks + 0.12, "First scored origin  22 Jan 2021",
            ha="center", va="bottom", fontsize=7.5, color=C["ink"])

    y0 = n_blocks - 1
    ax.text(t0 + pd.Timedelta(weeks=14.5), y0,
            "13 forecast origins per full block:\n"
            "1 refit origin + 12 subsequent origins without refitting",
            ha="left", va="center", fontsize=7.2, color=C["frozen"], fontweight="bold")

    ax.text(test[-1] + pd.Timedelta(days=14), 0,
            "10 forecast origins:\n1 refit origin +\n9 subsequent origins",
            ha="left", va="center", fontsize=7.2, color=C["muted"])

    ax.set_ylim(-0.55, n_blocks + 1.05)
    ax.set_yticks([])
    ax.set_xlim(first_origin - pd.Timedelta(days=260),
                test[-1] + pd.Timedelta(days=210))
    ax.grid(axis="x", alpha=0.28, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C["line"])
    ax.tick_params(length=3, colors=C["muted"], labelsize=7.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.legend(handles=[
        Patch(facecolor=C["train"],
              label="Estimation fold (expanding; pinned at 25 Jan 2019)"),
        Line2D([], [], marker="v", linestyle="", color=C["refit"], markersize=7,
               label="Re-estimation origin (first forecast of the block)"),
        Patch(facecolor=C["frozen"],
              label="Subsequent origins, model held fixed"),
    ], ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.055),
        borderaxespad=0.0, frameon=False, fontsize=7.4)

    ax.set_title("Common outer calendar with expanding estimation folds "
                 "and periodic re-estimation",
                 loc="left", fontsize=10.5, fontweight="bold", color=C["ink"], pad=6)


def draw_flow(ax) -> None:
    """Continuous Fit-1 zoom: Flat/Deep steps, merge, weekly chain, M0, scoring."""
    ax.set_xlim(0, 20)
    ax.set_ylim(0.00, 10.55)
    ax.axis("off")

    ax.text(0.0, 10.38, "Zoomed detail of Fit 1",
            fontsize=11, fontweight="bold", color=C["ink"], va="center")
    ax.text(6.85, 10.38, "Different inner-validation rules for Flat and Deep",
            fontsize=8.2, color=C["muted"], va="center")

    # ----- Flat lane -----
    ax.text(0.15, 9.92, "Flat  —  inner validation, then full-fold refit",
            fontsize=8.8, fontweight="bold", color=C["refit"], va="center")
    ax.add_patch(Rectangle((0.15, 8.55), 4.55, 1.12, facecolor=C["train"],
                           edgecolor="white", zorder=2))
    ax.add_patch(Rectangle((4.70, 8.55), 4.55, 1.12, facecolor=C["val"],
                           edgecolor="white", zorder=2))
    ax.text(2.42, 9.11, "First 52 weeks\ncandidate fitting",
            ha="center", va="center", fontsize=7.8, color=C["ink"], fontweight="bold")
    ax.text(6.97, 9.11, "Last 52 weeks\nhyperparameter validation",
            ha="center", va="center", fontsize=7.8, color=C["ink"], fontweight="bold")
    ax.text(4.70, 8.40, "Ridge: 5 $\\alpha$ values;  XGBoost: 8 configurations"
            "   ·   Fit 2: first 65 weeks + last 52 weeks",
            ha="center", va="top", fontsize=6.8, color=C["muted"])

    _arrow(ax, 4.70, 8.55, 4.70, 7.88, C["refit"])
    _rbox(ax, 2.45, 7.08, 4.50, 0.76, C["paper"], C["refit"])
    ax.text(4.70, 7.46, "Select best hyperparameters",
            ha="center", va="center", fontsize=8.1, color=C["ink"], fontweight="bold")
    _arrow(ax, 4.70, 7.08, 4.70, 6.42, C["refit"])
    _rbox(ax, 2.45, 5.58, 4.50, 0.80, C["train"], "white")
    ax.text(4.70, 5.98, "Refit on all 104 weeks",
            ha="center", va="center", fontsize=8.3, color=C["ink"], fontweight="bold")

    # ----- Deep lane -----
    ax.text(10.55, 9.92, "Deep  —  dynamic trailing validation, early stopping",
            fontsize=8.8, fontweight="bold", color=C["deep"], va="center")
    w = 9.10
    w_tr, w_va = w * 84 / 104, w * 20 / 104
    ax.add_patch(Rectangle((10.55, 8.55), w_tr, 1.12, facecolor=C["train"],
                           edgecolor="white", zorder=2))
    ax.add_patch(Rectangle((10.55 + w_tr, 8.55), w_va, 1.12, facecolor=C["val"],
                           edgecolor="white", zorder=2))
    ax.text(10.55 + w_tr / 2, 9.11, "Approx. 84 weeks\ntraining",
            ha="center", va="center", fontsize=7.8, color=C["ink"], fontweight="bold")
    ax.text(10.55 + w_tr + w_va / 2, 9.11, "Approx.\n20 val.",
            ha="center", va="center", fontsize=7.5, color=C["ink"], fontweight="bold")
    ax.text(15.10, 8.40,
            r"Validation length $\approx$ min(52, 20% of the current estimation fold)"
            "   ·   Fit 2: approx. 94 train + 23 validation",
            ha="center", va="top", fontsize=6.8, color=C["muted"])

    _arrow(ax, 15.10, 8.55, 15.10, 7.88, C["deep"])
    _rbox(ax, 12.80, 7.08, 4.60, 0.76, C["paper"], C["deep"])
    ax.text(15.10, 7.46, "Early stopping, patience = 12",
            ha="center", va="center", fontsize=8.1, color=C["ink"], fontweight="bold")
    _arrow(ax, 15.10, 7.08, 15.10, 6.42, C["deep"])
    _rbox(ax, 12.80, 5.58, 4.60, 0.80, "#D7E4DE", C["deep"])
    ax.text(15.10, 5.98, "Retain best checkpoint\nNo full-fold refit",
            ha="center", va="center", fontsize=7.7, color=C["deep"], fontweight="bold")

    # ----- merge (right-angle) to fitted model -----
    _rbox(ax, 7.20, 4.42, 5.60, 0.78, C["refit"], C["refit"])
    ax.text(10.00, 4.81, "Fitted model at origin $t$",
            ha="center", va="center", fontsize=9.4, color="white", fontweight="bold")
    mid_y = 4.81
    _flow(ax, [(4.70, 5.58), (4.70, mid_y), (7.20, mid_y)], C["refit"], lw=1.3)
    _flow(ax, [(15.10, 5.58), (15.10, mid_y), (12.80, mid_y)], C["deep"], lw=1.3)

    # ----- weekly forecast chain immediately under the merge -----
    # rhat sits under the fitted-model node (x = 10); M0 branches from t only.
    y_chain, h_chain = 2.72, 1.16
    y_mid = y_chain + h_chain / 2
    names = [r"$t{-}3$" + "\n1 Jan", r"$t{-}2$" + "\n8 Jan",
             r"$t{-}1$" + "\n15 Jan", r"$t$" + "\n22 Jan"]
    gap, bw = 0.08, 1.08
    x0 = 0.10
    for i, name in enumerate(names):
        ax.add_patch(Rectangle((x0 + i * (bw + gap), y_chain), bw, h_chain,
                               facecolor=C["lookback"], edgecolor="white", zorder=2))
        ax.text(x0 + i * (bw + gap) + bw / 2, y_mid, name, ha="center", va="center",
                fontsize=7.6, color="white", fontweight="bold")
    t_cx = x0 + 3 * (bw + gap) + bw / 2
    t_right = x0 + 4 * (bw + gap) - gap
    ax.text(x0 + 2 * (bw + gap), y_chain - 0.10,
            r"4-week input  ·  $P_t$ known at $t$",
            ha="center", va="top", fontsize=7.5, color=C["lookback"])

    rhat_x, rhat_w = 8.85, 2.30
    _arrow(ax, 10.00, 4.42, 10.00, y_chain + h_chain + 0.04, C["refit"], lw=1.3)
    _arrow(ax, t_right, y_mid, rhat_x - 0.08, y_mid, C["ink"])
    _rbox(ax, rhat_x, y_chain, rhat_w, h_chain, C["paper"], C["refit"], lw=1.15)
    ax.text(rhat_x + rhat_w / 2, y_mid, r"$\hat{r}_{t+1}$",
            ha="center", va="center", fontsize=13.2, color=C["ink"])

    phat_x, phat_w = 11.50, 4.15
    _arrow(ax, rhat_x + rhat_w, y_mid, phat_x - 0.08, y_mid, C["ink"])
    _rbox(ax, phat_x, y_chain - 0.08, phat_w, h_chain + 0.16,
          C["paper"], C["refit"], lw=1.15)
    ax.text(phat_x + phat_w / 2, y_mid,
            r"$\hat{P}_{t+1\mid t}=P_t\exp(\hat{r}_{t+1})$",
            ha="center", va="center", fontsize=10.8, color=C["ink"])

    cmp_x, cmp_w = 16.00, 3.85
    _arrow(ax, phat_x + phat_w, y_mid, cmp_x - 0.08, y_mid, C["ink"])
    _rbox(ax, cmp_x, y_chain, cmp_w, h_chain, C["paper"], C["ink"], lw=1.1)
    ax.text(cmp_x + cmp_w / 2, y_mid, "compare with realised $P_{t+1}$",
            ha="center", va="center", fontsize=8.4, color=C["ink"])

    # M0: parallel branch from t / P_t only, then into compare
    m0_y, m0_h = 1.48, 0.72
    _flow(ax, [(t_cx, y_chain), (t_cx, m0_y + m0_h)], C["target"], lw=1.25)
    _rbox(ax, 0.10, m0_y, 7.70, m0_h, "white", C["target"], lw=1.2)
    ax.text(3.95, m0_y + m0_h / 2,
            r"M0:  $P_t \;\rightarrow\; \hat{P}_{t+1\mid t}=P_t$",
            ha="center", va="center", fontsize=9.4, color=C["target"],
            fontweight="bold")
    cmp_cx = cmp_x + cmp_w / 2
    m0_join_x = cmp_x + 0.45
    _flow(ax, [(7.80, m0_y + m0_h / 2), (m0_join_x, m0_y + m0_h / 2),
               (m0_join_x, y_chain)], C["target"], lw=1.2)

    # final scoring, right column under compare
    _arrow(ax, cmp_cx, y_chain, cmp_cx, 1.42, C["ink"])
    _rbox(ax, cmp_x, 0.10, cmp_w, 1.28, C["paper"], C["ink"], lw=1.1)
    ax.text(cmp_cx, 0.74,
            "Accumulate 257 price forecast errors\n"
            r"$\rightarrow$  Price RMSE" "\n"
            r"$\rightarrow$  RMSE improvement vs M0 (%)",
            ha="center", va="center", fontsize=8.3, color=C["ink"],
            linespacing=1.35)


def main() -> None:
    _, eligible, test = _weeks()
    fig = plt.figure(figsize=(12.8, 12.7))
    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.75, 5.05, 0.48],
        hspace=0.08,
        left=0.075, right=0.985, top=0.868, bottom=0.034,
    )
    fig.suptitle("Expanding-window walk-forward forecasting protocol",
                 fontsize=14.5, fontweight="bold", color=C["ink"],
                 x=0.075, ha="left", y=0.978)
    fig.text(
        0.075, 0.948,
        "365 Fridays  −  3 leading weeks  −  1 trailing week  =  361 eligible sequences"
        "  =  104 warm-up  +  257 OOS origins",
        fontsize=8.5, color=C["muted"], ha="left", va="top",
    )

    ax_g = fig.add_subplot(gs[0])
    ax_f = fig.add_subplot(gs[1])
    ax_cap = fig.add_subplot(gs[2])
    ax_cap.axis("off")

    draw_calendar(ax_g, eligible, test)
    draw_flow(ax_f)

    cap = (
        "The estimation fold expands from an initial 104 sequences and is re-estimated "
        "every 13 forecast origins. Flat models use the final 52 training sequences for "
        "hyperparameter validation and are subsequently refit on the full estimation fold. "
        "Deep models use a dynamically sized trailing validation segment for early stopping "
        "and retain the best checkpoint without full-fold retraining. Both model families "
        "are evaluated on the same 257 out-of-sample forecast origins."
    )
    ax_cap.text(0.0, 0.95, textwrap.fill(cap, width=128),
                fontsize=7.6, color=C["muted"], va="top", linespacing=1.45)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "fig_expanding_window_protocol"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"saved {stem}.png")
    print(f"saved {stem}.pdf")


if __name__ == "__main__":
    main()
