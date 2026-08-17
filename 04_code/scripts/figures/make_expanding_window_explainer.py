"""
Expanding-window schedule (single panel): expanding training set,
52-week inner-validation tail, and the testing block after each fit.

    python 04_code/scripts/figures/make_expanding_window_explainer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

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
    "frozen": "#5B8A8A",
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

        if b in (0, 1, 4, 9, 14, 18, 19):
            ax.text(first_origin - pd.Timedelta(days=28), y, f"Fit {b + 1}",
                    ha="right", va="center", fontsize=7.1, color=C["muted"])

    ax.set_ylim(-0.8, n_blocks - 0.48)
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
              label="Dropped weeks (first 3 weeks + last 1 week)"),
        Patch(facecolor=C["train"],
              label="Training set (expanding every 13 weeks)"),
        Patch(facecolor=C["val"],
              label="Inner validation set (52 weeks)"),
        Patch(facecolor=C["frozen"], label="Testing"),
    ], ncol=4, loc="upper left", bbox_to_anchor=(0.0, -0.08),
        frameon=False, fontsize=8.0, columnspacing=1.35, handlelength=1.5)


def main() -> None:
    raw, eligible, test = _weeks()
    fig = plt.figure(figsize=(12.6, 7.2))
    ax = fig.add_axes([0.07, 0.14, 0.915, 0.84])
    draw_b(ax, raw, eligible, test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_expanding_window_explainer.{ext}")
    plt.close(fig)
    print(f"saved {OUT_DIR / 'fig_expanding_window_explainer.png'}")
    print(f"saved {OUT_DIR / 'fig_expanding_window_explainer.pdf'}")


if __name__ == "__main__":
    main()
