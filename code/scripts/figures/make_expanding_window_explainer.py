"""
Expanding-window schedule (single panel): expanding training set,
52-week inner-validation tail, and the testing block after each fit.

    python code/scripts/figures/make_expanding_window_explainer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "figures"

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
    "figure.dpi": 150,
    "savefig.dpi": 600,
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

        n_blue = n_train - n_val
        n_test_block = hi - lo
        ax.barh(y, (val_start - first_origin).days, left=first_origin, height=0.74,
                color=C["train"], edgecolor="white", linewidth=0.2, zorder=2)
        ax.barh(y, (t_first - val_start).days, left=val_start, height=0.74,
                color=C["val"], edgecolor="white", linewidth=0.2, zorder=3)
        ax.barh(y, (t_last + week - t_first).days, left=t_first, height=0.74,
                color=C["frozen"], edgecolor="white", linewidth=0.2, zorder=3)

        mid_blue = first_origin + (val_start - first_origin) / 2
        ax.text(mid_blue, y, f"{n_blue} weeks", ha="center", va="center",
                fontsize=6.2, color=C["ink"], zorder=4, clip_on=True)

        if b in (0, 1, 4, 9, 14, 18, 19):
            ax.text(first_origin - pd.Timedelta(days=28), y, f"Fit {b + 1}",
                    ha="right", va="center", fontsize=7.1, color=C["muted"])

        if n_test_block != RETRAIN_EVERY:
            ax.annotate(
                "Final testing block\n(10 weeks only)",
                xy=(t_last + week, y),
                xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=6.5, color=C["muted"],
                annotation_clip=False, zorder=6, linespacing=1.15,
            )
    ax.set_ylim(-0.8, n_blocks - 0.48)
    ax.set_yticks([])
    ax.set_xlim(data_start - pd.Timedelta(days=210),
                last_week + pd.Timedelta(days=140))
    ax.grid(axis="x", alpha=0.28, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C["line"])
    ax.tick_params(length=3, colors=C["muted"], labelsize=7.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    items = [
        dict(fc=C["drop"], hatch="///", ec="#B8B2A8",
             lab="Dropped weeks (first 3 weeks + last 1 week)"),
        dict(fc=C["train"], lab="Training set (expanding every 13 weeks)"),
        dict(fc=C["val"], lab="Inner validation set (52 weeks)"),
        dict(fc=C["frozen"], lab="Testing (Fixed 13 weeks)"),
    ]
    xs = (0.11, 0.355, 0.565, 0.735)
    for x, item in zip(xs, items):
        ax.add_patch(Rectangle(
            (x, -0.068), 0.016, 0.028, transform=ax.transAxes,
            facecolor=item["fc"], edgecolor=item.get("ec", "none"),
            hatch=item.get("hatch", None), linewidth=0.4,
            clip_on=False, zorder=6))
        ax.text(x + 0.022, -0.054, item["lab"], transform=ax.transAxes,
                ha="left", va="center", fontsize=7.4, color=C["ink"],
                clip_on=False)


def main() -> None:
    raw, eligible, test = _weeks()
    fig = plt.figure(figsize=(12.6, 7.2))
    ax = fig.add_axes([0.07, 0.11, 0.915, 0.86])
    draw_b(ax, raw, eligible, test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thesis_dir = ROOT / "thesis" / "figures"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    primary = OUT_DIR / "fig_3_7_expanding_window"
    aliases = [
        OUT_DIR / "fig_expanding_window_explainer",
        thesis_dir / "fig_3_7_expanding_window",
    ]
    for ext in ("png", "pdf"):
        dest = primary.with_suffix(f".{ext}")
        fig.savefig(dest, dpi=600 if ext == "png" else None)
        data = dest.read_bytes()
        for alias in aliases:
            alias.with_suffix(f".{ext}").write_bytes(data)
        print(f"saved {dest}")
    plt.close(fig)


if __name__ == "__main__":
    main()
