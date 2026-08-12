"""
Animated version of Figure 3.3 - the shared expanding-window backtest.

One frame per refit: the training window grows, the 13-week test block moves
ahead of it, and the evaluated part of the Brent series fills in. Intended for
slides, not for the dissertation PDF.

Writes GIF (pillow) and a self-contained HTML player to 05_outputs/figures/.
ffmpeg is not required.

    python 04_code/scripts/figures/make_window_animation.py [--fps 1.4]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
FLAT_PRED = (ROOT / "05_outputs" / "baselines" / "Flat" / "M1_Flat"
             / "baseline_predictions.csv")
FEATURE_MATRIX = (ROOT / "03_data" / "processed" / "merge" / "outputs"
                  / "weekly_feature_matrix.csv")
OUT_DIR = ROOT / "05_outputs" / "figures"

MIN_TRAIN = 104
RETRAIN_EVERY = 13
HOLD_FIRST_MS = 1800  # linger on the warm-up frame
HOLD_LAST_MS = 2800  # linger on the completed backtest before looping

LOOKBACK = 4

C = {
    "price": "#C9C4BB",
    "done": "#22303C",
    "train": "#8FA8C8",
    "test": "#D1622B",
    "ink": "#22303C",
    "muted": "#5C5C5C",
}

mpl.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _hold_end_frames(gif: Path, step_ms: int) -> None:
    """Re-time the GIF so the first and last frames stay up longer.

    Duplicating frames in the animation itself does not work: the writer keeps
    only one copy of an unchanged frame.
    """
    from PIL import Image, ImageSequence

    with Image.open(gif) as im:
        frames = [f.copy() for f in ImageSequence.Iterator(im)]
    durations = [step_ms] * len(frames)
    durations[0] = HOLD_FIRST_MS
    durations[-1] = HOLD_LAST_MS
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)


def _draw_origin_inset(ax) -> None:
    """Static inset: what one forecast origin looks like.

    The test block is 13 origins, not one 13-week input. Without this panel the
    block reads as a single long input window.
    """
    axi = ax.inset_axes([0.575, 0.045, 0.405, 0.30])
    axi.set_xlim(-0.6, 8.4)
    axi.set_ylim(-0.15, 1.35)
    axi.axis("off")
    axi.add_patch(mpl.patches.Rectangle(
        (-0.6, -0.15), 9.0, 1.5, facecolor="white", edgecolor="#CCCCCC",
        linewidth=0.8, zorder=0))

    axi.text(-0.2, 1.16, f"Inside one forecast origin  (\u00d7{RETRAIN_EVERY} "
             "per test block)", fontsize=7.4, fontweight="bold",
             color=C["ink"], va="center", zorder=3)

    for i in range(LOOKBACK):
        axi.add_patch(mpl.patches.Rectangle(
            (i * 1.02, 0.38), 0.92, 0.34, facecolor=C["train"], alpha=0.55,
            edgecolor=C["train"], linewidth=0.8, zorder=2))
    axi.text(LOOKBACK * 1.02 / 2 - 0.05, 0.16,
             f"input: {LOOKBACK} weeks  (t\u2212{LOOKBACK - 1} \u2026 t)",
             fontsize=6.8, color=C["muted"], ha="center", va="center", zorder=3)

    axi.annotate("", xy=(5.45, 0.55), xytext=(4.25, 0.55),
                 arrowprops=dict(arrowstyle="-|>", color=C["test"],
                                 linewidth=1.2), zorder=3)
    # A Circle patch would be stretched by the inset's aspect ratio.
    axi.plot(5.95, 0.55, "o", markersize=8, color=C["test"],
             markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    axi.text(6.35, 0.55, "forecast\nweek t+1", fontsize=6.8, color=C["test"],
             ha="left", va="center", zorder=3)


def load():
    price = pd.read_csv(FEATURE_MATRIX, usecols=["week_ending_friday", "brent_price"],
                        parse_dates=["week_ending_friday"])
    price = price.set_index("week_ending_friday")["brent_price"].dropna()
    test_dates = pd.to_datetime(
        pd.read_csv(FLAT_PRED, usecols=[0]).iloc[:, 0]).sort_values()
    return price, test_dates


def build(fps: float) -> None:
    price, test_dates = load()
    data_start = price.index[0]
    n_test = len(test_dates)
    n_blocks = math.ceil(n_test / RETRAIN_EVERY)
    week = pd.Timedelta(weeks=1)

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.13)

    lo, hi = float(price.min()), float(price.max())
    pad = (hi - lo) * 0.12
    ax.set_xlim(data_start - pd.Timedelta(days=40),
                price.index[-1] + pd.Timedelta(days=40))
    ax.set_ylim(lo - pad, hi + pad * 1.5)
    ax.set_ylabel("Brent price (USD per barrel)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.plot(price.index, price.values, color=C["price"], linewidth=1.2, zorder=2)
    done_line, = ax.plot([], [], color=C["done"], linewidth=1.4, zorder=4)

    train_span = ax.axvspan(data_start, data_start, color=C["train"], alpha=0.30,
                            linewidth=0, zorder=1)
    test_span = ax.axvspan(data_start, data_start, color=C["test"], alpha=0.38,
                           linewidth=0, zorder=1)

    head = ax.text(0.0, 1.20, "", transform=ax.transAxes, fontsize=11,
                   fontweight="bold", color=C["ink"], va="top")
    sub = ax.text(0.0, 1.075, "", transform=ax.transAxes, fontsize=8.6,
                  color=C["muted"], va="top")
    counter = ax.text(0.995, 1.20, "", transform=ax.transAxes, fontsize=9.5,
                      color=C["test"], ha="right", va="top", fontweight="bold")

    ax.legend(handles=[
        mpl.patches.Patch(color=C["train"], alpha=0.30, label="training window (expanding)"),
        mpl.patches.Patch(color=C["test"], alpha=0.38,
                          label=f"test block ({RETRAIN_EVERY} forecast origins)"),
        Line2D([], [], color=C["done"], linewidth=1.6, label="weeks elapsed"),
    ], loc="upper left", ncol=3, fontsize=7.8, bbox_to_anchor=(0.0, 1.0))

    _draw_origin_inset(ax)

    def set_span(span, x0, x1):
        # axvspan returns a Rectangle on the x-axis transform: y stays 0-1.
        x0n, x1n = mpl.dates.date2num(x0), mpl.dates.date2num(x1)
        span.set_x(x0n)
        span.set_width(x1n - x0n)

    def draw(i: int):
        # Frame 0 shows the warm-up only; frames 1..n_blocks show each refit.
        b = min(max(i, 0), n_blocks)
        if b == 0:
            warm_end = test_dates.iloc[0]
            set_span(train_span, data_start, warm_end)
            set_span(test_span, warm_end, warm_end)
            warm = price.loc[:warm_end - week]
            done_line.set_data(warm.index, warm.values)
            head.set_text("Warm-up: no forecast is scored yet")
            sub.set_text(f"initial training window  {data_start.date()} → "
                         f"{(warm_end - week).date()}   "
                         f"({MIN_TRAIN} weekly targets)")
            counter.set_text("0 / 257 weeks scored")
            return train_span, test_span, done_line, head, sub, counter

        k = b - 1
        t_first = test_dates.iloc[k * RETRAIN_EVERY]
        t_last = test_dates.iloc[min((k + 1) * RETRAIN_EVERY, n_test) - 1]
        set_span(train_span, data_start, t_first)
        set_span(test_span, t_first, t_last + week)

        # Dark = weeks that have already happened (training history plus the
        # test weeks scored so far); grey = weeks still in the future.
        done = price.loc[:t_last]
        done_line.set_data(done.index, done.values)

        # Training targets available at this origin; the first 4 weeks of data
        # are consumed by the lookback window, hence MIN_TRAIN rather than a
        # raw week count.
        n_train = MIN_TRAIN + k * RETRAIN_EVERY
        scored = min((k + 1) * RETRAIN_EVERY, n_test)
        head.set_text(f"Refit {b} of {n_blocks}")
        n_origins = scored - k * RETRAIN_EVERY
        sub.set_text(
            f"train on {n_train} weekly targets to {(t_first - week).date()}"
            f"   ·   model then held fixed for {n_origins} forecast origins:  "
            f"{t_first.date()} → {t_last.date()}")
        counter.set_text(f"{scored} / {n_test} weeks scored")
        return train_span, test_span, done_line, head, sub, counter

    anim = FuncAnimation(fig, draw, frames=list(range(n_blocks + 1)),
                         interval=int(1000 / fps), blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gif = OUT_DIR / "anim_expanding_window.gif"
    anim.save(gif, writer=PillowWriter(fps=fps))
    _hold_end_frames(gif, int(1000 / fps))
    print(f"  saved {gif.name}")

    html = OUT_DIR / "anim_expanding_window.html"
    html.write_text(anim.to_jshtml(fps=fps), encoding="utf-8")
    print(f"  saved {html.name}")

    # Still frames for checking the animation renders as intended.
    for i, tag in [(0, "warmup"), (1, "refit01"), (n_blocks, "final")]:
        draw(i)
        fig.savefig(OUT_DIR / f"anim_frame_{tag}.png", dpi=150)
    plt.close(fig)
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.4)
    args = ap.parse_args()
    build(args.fps)
