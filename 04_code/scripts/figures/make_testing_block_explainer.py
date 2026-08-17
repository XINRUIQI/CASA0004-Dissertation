"""
One Fit's testing origins: 13 weekly steps. Each origin uses a 4-week input
(t-3 … t) and scores the one-week-ahead target t+1; the fitted model is held
fixed while the five-week block slides forward.

    python 04_code/scripts/figures/make_testing_block_explainer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "05_outputs" / "figures"

LOOKBACK = 4
MIN_TRAIN = 104
RETRAIN_EVERY = 13
N_RAW = 365
N_ELIGIBLE = 361
N_TEST = 257

C = {
    "input": "#8FA8C8",
    "test": "#5B8A8A",
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
    eligible = raw[LOOKBACK - 1:-1]
    test = eligible[MIN_TRAIN:]
    assert len(raw) == N_RAW and len(eligible) == N_ELIGIBLE and len(test) == N_TEST
    return test


def main() -> None:
    test = _weeks()
    origins = test[:RETRAIN_EVERY]  # Fit 1: 13 scored origins
    week = pd.Timedelta(weeks=1)
    n = len(origins)

    fig = plt.figure(figsize=(12.0, 6.6))
    ax = fig.add_axes([0.10, 0.14, 0.88, 0.80])

    n_cells = LOOKBACK + 1  # t-3, t-2, t-1, t, t+1
    cell_w = week / pd.Timedelta(days=1)  # tile with no gap
    labels = [r"$t{-}3$", r"$t{-}2$", r"$t{-}1$", r"$t$", r"$t{+}1$"]

    for k, t in enumerate(origins):
        y = n - 1 - k
        t_start = t - pd.Timedelta(weeks=LOOKBACK - 1)
        for i in range(n_cells):
            left = t_start + i * week
            is_target = i == LOOKBACK
            color = C["test"] if is_target else C["input"]
            x0 = mdates.date2num(left)
            ax.add_patch(Rectangle(
                (x0, y - 0.36), cell_w, 0.72,
                facecolor=color, edgecolor="white", linewidth=0.9, zorder=3))
            if k == 0:
                ax.text(x0 + cell_w / 2, y, labels[i],
                        ha="center", va="center", fontsize=6.8,
                        color="white", fontweight="bold", zorder=4)
        ax.text(t_start - pd.Timedelta(days=6), y, f"Origin {k + 1}",
                ha="right", va="center", fontsize=8.0, color=C["muted"])

    ax.set_ylim(-0.7, n - 0.48)
    ax.set_yticks([])
    ax.set_xlim(origins[0] - pd.Timedelta(days=48),
                origins[-1] + pd.Timedelta(days=28))
    ax.grid(axis="x", alpha=0.28, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C["line"])
    ax.tick_params(length=3, colors=C["muted"], labelsize=7.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax.legend(handles=[
        Patch(facecolor=C["input"],
              label=r"Input weeks $t{-}3$, $t{-}2$, $t{-}1$, $t$"),
        Patch(facecolor=C["test"],
              label=r"Forecast target $t{+}1$"),
    ], ncol=2, loc="lower left", bbox_to_anchor=(0.0, -0.14),
        frameon=False, fontsize=8.2)

    ax.set_title("One Fit: testing origin advances one week at a time "
                 "(13 origins; model held fixed)",
                 loc="left", fontsize=10.5, fontweight="bold", color=C["ink"],
                 pad=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "fig_testing_block_origins"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"saved {stem}.png")
    print(f"saved {stem}.pdf")


if __name__ == "__main__":
    main()
