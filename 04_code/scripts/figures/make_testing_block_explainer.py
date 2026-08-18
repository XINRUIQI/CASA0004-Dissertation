"""
Testing sample within one estimation block (the first block shown): 13 weekly
steps. Each step uses a 4-week input (t-3 … t) and scores the one-week-ahead
target t+1; the fitted model is held fixed while the block slides forward.

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
    "input": "#C5DDB0",
    "target": "#2B5A40",
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

    fig = plt.figure(figsize=(10.4, 5.6))
    ax = fig.add_axes([0.09, 0.12, 0.88, 0.84])

    n_cells = LOOKBACK + 1  # t-3, t-2, t-1, t, t+1
    cell_w = week / pd.Timedelta(days=1)  # tile with no gap
    labels = [r"$t{-}3$", r"$t{-}2$", r"$t{-}1$", r"$t$", r"$t{+}1$"]

    for k, t in enumerate(origins):
        y = n - 1 - k
        t_start = t - pd.Timedelta(weeks=LOOKBACK - 1)
        x0_block = mdates.date2num(t_start)
        ax.add_patch(Rectangle(
            (x0_block, y - 0.36), n_cells * cell_w, 0.72,
            facecolor="none", edgecolor="#5A8A58", linewidth=0.9, zorder=5))
        for i in range(n_cells):
            left = t_start + i * week
            is_target = i == LOOKBACK
            color = C["target"] if is_target else C["input"]
            x0 = mdates.date2num(left)
            ax.add_patch(Rectangle(
                (x0, y - 0.36), cell_w, 0.72,
                facecolor=color, edgecolor="white", linewidth=0.9, zorder=3))
            if k == 0:
                ax.text(x0 + cell_w / 2, y, labels[i],
                        ha="center", va="center", fontsize=6.8,
                        color="white" if is_target else C["ink"],
                        fontweight="bold", zorder=4)

    ax.set_ylim(-0.52, n - 0.48)
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([f"Step {n - i}" for i in range(n)])
    ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-05-01"))
    ax.grid(axis="x", alpha=0.28, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C["line"])
    ax.tick_params(axis="x", length=3, pad=2, colors=C["muted"], labelsize=7.6)
    ax.tick_params(axis="y", length=0, colors=C["muted"], labelsize=8.0)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    y0s = []
    for tick in ax.get_xticklabels():
        if not tick.get_text():
            continue
        bb = tick.get_window_extent(renderer)
        y0s.append(inv.transform((bb.x0, bb.y0))[1])
    y_leg = (min(y0s) if y0s else -0.08) - 0.022
    ax.legend(handles=[
        Patch(facecolor=C["input"], label="Four-week input"),
        Patch(facecolor=C["target"], label="One-week-ahead target"),
    ], ncol=2, loc="upper center", bbox_to_anchor=(0.5, y_leg),
        borderaxespad=0.0, borderpad=0.15, frameon=False, fontsize=8.2,
        handlelength=1.0, handleheight=0.75, handletextpad=0.4,
        columnspacing=4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = [
        OUT_DIR / "fig_3_8_testing_block_origins",
        OUT_DIR / "fig_testing_block_origins",
    ]
    thesis_dir = ROOT / "06_writing" / "CASA-MSc-thesis-main" / "figures"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(stems[0].with_suffix(f".{ext}"),
                    bbox_inches="tight", pad_inches=0.04)
        stems[1].with_suffix(f".{ext}").write_bytes(
            stems[0].with_suffix(f".{ext}").read_bytes())
        thesis_dir.joinpath(f"fig_3_8_testing_block_origins.{ext}").write_bytes(
            stems[0].with_suffix(f".{ext}").read_bytes())
    plt.close(fig)
    print(f"saved {stems[0]}.png")
    print(f"saved {stems[0]}.pdf")


if __name__ == "__main__":
    main()
