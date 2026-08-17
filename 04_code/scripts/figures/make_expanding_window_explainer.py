"""
Teaching figure: expanding-window protocol as one Gantt (panel B),
with the dropped weeks, 52-week tail, frozen 12-week block, 4-week
input, 1-week target, and Flat vs Deep inner-validation rules drawn
onto the same calendar.

    python 04_code/scripts/figures/make_expanding_window_explainer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "05_outputs" / "figures"

LOOKBACK = 4
MIN_TRAIN = 104
RETRAIN_EVERY = 13
VAL_WEEKS = 52
N_RAW = 365
N_ELIGIBLE = 361
N_TEST = 257
N_FITS = 20

C = {
    "drop": "#D9D4CB",
    "train": "#8FA8C8",
    "val": "#E7A33E",
    "refit": "#2E5A88",
    "frozen": "#E39A6A",
    "target": "#C44B2A",
    "lookback": "#3D6B9A",
    "deep": "#4F7D6E",
    "ink": "#22303C",
    "muted": "#5C5C5C",
    "line": "#C9C4BB",
}

mpl.rcParams.update({
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
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
    assert len(raw) == N_RAW
    eligible = raw[LOOKBACK - 1:-1]
    assert len(eligible) == N_ELIGIBLE
    test = eligible[MIN_TRAIN:]
    assert len(test) == N_TEST
    return raw, eligible, test


def _origin_inset(axi) -> None:
    """4-week lookback + 1-week target, using the first scored origin."""
    axi.set_xlim(-0.25, 10.4)
    axi.set_ylim(-0.12, 1.42)
    axi.axis("off")
    axi.add_patch(Rectangle((-0.25, -0.12), 10.65, 1.54, facecolor="white",
                            edgecolor=C["line"], linewidth=0.8, zorder=0))
    axi.text(0.0, 1.22, "Inside one forecast origin (22 Jan 2021)", fontsize=7.6,
             fontweight="bold", color=C["ink"], va="center")

    names = ["t-3\n01-01", "t-2\n01-08", "t-1\n01-15", "t\n01-22"]
    for i, name in enumerate(names):
        axi.add_patch(Rectangle((0.08 + i * 1.18, 0.38), 1.08, 0.52,
                                facecolor=C["lookback"], edgecolor="white",
                                linewidth=0.6, zorder=2))
        axi.text(0.62 + i * 1.18, 0.64, name, ha="center", va="center",
                 fontsize=6.6, color="white", fontweight="bold")
    axi.text(2.4, 0.14, "4-week input window (length fixed)", ha="center", va="center",
             fontsize=6.8, color=C["lookback"])

    axi.annotate("", xy=(5.55, 0.64), xytext=(4.88, 0.64),
                 arrowprops=dict(arrowstyle="-|>", color=C["target"], lw=1.3),
                 zorder=3)
    axi.plot(6.15, 0.64, "o", markersize=15, color=C["target"], zorder=4)
    axi.text(6.15, 0.64, "t+1\n01-29", ha="center", va="center",
             fontsize=6.4, color="white", fontweight="bold", zorder=5)
    axi.text(7.85, 0.64, "1-week target", ha="left", va="center",
             fontsize=7.2, color=C["target"], fontweight="bold")
    axi.text(5.5, 0.14, "Frozen 12 weeks: parameters held; only these 4 cells slide",
             ha="center", va="center", fontsize=6.6, color=C["muted"])


def _flat_deep_inset(axi) -> None:
    axi.set_xlim(0, 10)
    axi.set_ylim(0, 3.35)
    axi.axis("off")
    axi.add_patch(Rectangle((0, 0), 10, 3.35, facecolor="white",
                            edgecolor=C["line"], linewidth=0.8, zorder=0))

    axi.text(0.25, 3.05, "Same gold tail, different use in Flat and Deep", fontsize=7.6,
             fontweight="bold", color=C["ink"], va="center")

    axi.add_patch(Rectangle((0.25, 1.55), 2.35, 1.15, facecolor=C["train"],
                            edgecolor="white", zorder=2))
    axi.add_patch(Rectangle((2.60, 1.55), 2.20, 1.15, facecolor=C["val"],
                            edgecolor="white", zorder=2))
    axi.text(1.42, 2.12, "Front", ha="center", va="center", fontsize=7.2,
             color=C["ink"], fontweight="bold")
    axi.text(3.70, 2.12, "Last 52 weeks", ha="center", va="center", fontsize=7.2,
             color=C["ink"], fontweight="bold")

    axi.text(5.05, 2.55, "Flat", fontsize=7.4, fontweight="bold", color=C["refit"])
    axi.text(5.05, 1.85, "Tune on the gold tail, then\nrefit on the full fold (blue + gold)",
             fontsize=6.7, color=C["ink"], va="center")

    axi.text(0.25, 1.15, "Deep", fontsize=7.4, fontweight="bold", color=C["deep"])
    axi.text(0.25, 0.45, "Architecture locked (lookback = 4, d = 32). Gold tail used only for early\nstopping; keep the checkpoint, do not refit on blue + gold. First fold is short,\nso the tail is ~20 weeks before it reaches 52.",
             fontsize=6.7, color=C["ink"], va="center")


def draw_b(ax, raw, eligible, test) -> None:
    data_start = raw[0]
    first_origin = eligible[0]
    last_week = raw[-1]
    week = pd.Timedelta(weeks=1)
    n_blocks = N_FITS

    ax.axvspan(data_start, first_origin, facecolor=C["drop"], hatch="///",
               edgecolor="#B8B2A8", linewidth=0, zorder=0)
    ax.axvspan(test[-1] + week, last_week + week, facecolor=C["drop"], hatch="///",
               edgecolor="#B8B2A8", linewidth=0, zorder=0)

    for b in range(n_blocks):
        lo = b * RETRAIN_EVERY
        hi = min(lo + RETRAIN_EVERY, N_TEST)
        t_first, t_last = test[lo], test[hi - 1]
        y = n_blocks - 1 - b
        n_train = MIN_TRAIN + b * RETRAIN_EVERY
        n_val = min(VAL_WEEKS, n_train)
        val_start = t_first - pd.Timedelta(weeks=n_val)

        ax.barh(y, (val_start - first_origin).days, left=first_origin, height=0.74,
                color=C["train"], edgecolor="white", linewidth=0.2, zorder=2)
        ax.barh(y, (t_first - val_start).days, left=val_start, height=0.74,
                color=C["val"], edgecolor="white", linewidth=0.2, zorder=3)
        ax.barh(y, (t_last + week - t_first).days, left=t_first, height=0.74,
                color=C["frozen"], edgecolor="white", linewidth=0.2, zorder=3)
        ax.plot(t_first, y, marker="v", markersize=5.2, color=C["refit"],
                zorder=6, clip_on=False)

        if b in (0, 1, 4, 9, 14, 18, 19):
            ax.text(first_origin - pd.Timedelta(days=28), y, f"Fit {b + 1}",
                    ha="right", va="center", fontsize=7.1, color=C["muted"])

    t0 = test[0]
    ax.axvline(t0, color=C["ink"], linewidth=0.85, linestyle=(0, (3, 2)), zorder=5)
    ax.text(t0, n_blocks + 0.22, "First scored origin  22 Jan 2021",
            ha="center", va="bottom", fontsize=7.6, color=C["ink"])
    ax.text(test[-1] + pd.Timedelta(days=16), 0, "Final block\n10 weeks only",
            ha="left", va="center", fontsize=7.2, color=C["muted"])

    ax.set_ylim(-0.8, n_blocks + 1.45)
    ax.set_yticks([])
    ax.set_xlim(data_start - pd.Timedelta(days=210),
                last_week + pd.Timedelta(days=70))
    ax.grid(axis="x", alpha=0.28, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C["line"])
    ax.tick_params(length=3, colors=C["muted"], labelsize=7.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.legend(handles=[
        Patch(facecolor=C["drop"], hatch="///", edgecolor="#B8B2A8",
              label="Dropped weeks (first 3 + last 1)"),
        Patch(facecolor=C["train"], label="Training front (pinned at 25 Jan 2019)"),
        Patch(facecolor=C["val"], label="Last 52 weeks (inner validation; slides right)"),
        Line2D([], [], marker="v", linestyle="", color=C["refit"],
               markersize=7, label="Re-estimation origin (20 fits)"),
        Patch(facecolor=C["frozen"], label="Rest of block held fixed: no refit"),
        Patch(facecolor=C["lookback"], label="4-week input -> 1-week target (below)"),
    ], ncol=3, loc="lower left", bbox_to_anchor=(0.0, -0.12),
        frameon=False, fontsize=7.4)


def main() -> None:
    raw, eligible, test = _weeks()
    fig = plt.figure(figsize=(12.6, 10.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.55, 1.15],
                  hspace=0.36, wspace=0.08,
                  left=0.07, right=0.985, top=0.97, bottom=0.06)

    ax = fig.add_subplot(gs[0, :])
    ax_fd = fig.add_subplot(gs[1, 0])
    ax_or = fig.add_subplot(gs[1, 1])
    draw_b(ax, raw, eligible, test)
    _flat_deep_inset(ax_fd)
    _origin_inset(ax_or)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_expanding_window_explainer.{ext}")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'fig_expanding_window_explainer.png'}")
    print(f"saved {OUT_DIR / 'fig_expanding_window_explainer.pdf'}")


if __name__ == "__main__":
    main()
