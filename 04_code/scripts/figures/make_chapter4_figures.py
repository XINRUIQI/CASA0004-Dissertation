"""
Chapter 4 figures (current numbering, first mention in the results chapter).

Reads committed CSVs / npy under 05_outputs/ and the weekly feature matrix.
Does not retrain models.

    python 04_code/scripts/figures/make_chapter4_figures.py
    python 04_code/scripts/figures/make_chapter4_figures.py --only 4.1 4.2

    4.1  Flat ΔRMSE (grouped Cleveland dot plot)
    4.2  Deep ΔRMSE (gated vs cross-attention dumbbell)
    price  Brent price / returns (Figure 3.2; evaluation sample, no event windows)
    slope  Flat XGBoost → Deep gated paired slopes
    4.3  Shipping gate weight vs shipping |SHAP| share (dual-dot, no link)
    4.4  Two-panel node map (2022 | 2024; centre + proportional halo)
    4.5  Two-panel node × week shipping-internal SHAP-share heatmap
    B.1  Deep seed robustness (appendix; three seeds + mean, vs M0)

Legacy skill-bar / attention figures remain in make_result_figures.py --legacy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "05_outputs" / "baselines"
DEEP = BASE / "Deep"
M3 = DEEP / "M3_Deep"
OUT_DIR = ROOT / "05_outputs" / "figures"
NE_DIR = ROOT / "03_data" / "raw" / "00_spatial_anchors" / "naturalearth"
FEATURE_MATRIX = (ROOT / "03_data" / "processed" / "merge" / "outputs"
                  / "weekly_feature_matrix.csv")

SETS = ["S1", "S2", "S3", "S4"]
SET_NOTE = {
    "S1": "finance",
    "S2": "+ RS",
    "S3": "+ shipping",
    "S4": "+ RS + ship",
}

# EIA representative coordinates — same as Figure 3.3 / export_fig33_map_layers.py
AOI_XY = {
    "P001": ("Rotterdam", "port", 4.145, 51.950),
    "P002": ("Fujairah", "terminal", 56.356, 25.199),
    "P003": ("Ras Tanura", "terminal", 50.157, 26.643),
    "P004": ("Jurong Island", "refinery", 103.708, 1.274),
    "P005": ("Houston", "port", -95.100, 29.736),
    "P006": ("Ningbo-Zhoushan", "port", 121.982, 29.935),
    "P007": ("Jamnagar", "refinery", 69.860, 22.345),
    "P008": ("Basra", "terminal", 48.810, 29.681),
    "P009": ("Ulsan", "refinery", 129.343, 35.433),
    "P010": ("Kharg", "terminal", 50.324, 29.231),
    "P011": ("Yanbu", "terminal", 38.229, 23.961),
}
CHOKE_XY = {
    "hormuz": ("Hormuz", 56.25, 26.57),
    "suez": ("Suez", 32.35, 30.42),
    "malacca": ("Malacca", 100.40, 2.50),
    "mandeb": ("Bab el-Mandeb", 43.40, 12.60),
    "panama": ("Panama", -79.55, 9.08),
    "cape": ("Cape of Good Hope", 18.47, -34.36),
}
TYPE_MARKER = {"port": "o", "terminal": "s", "refinery": "^"}
CHOKE_IDS = list(CHOKE_XY)
NAMED_NODES = ["P004", "hormuz", "suez", "cape", "mandeb"]
NAMED_LABEL = {
    "P004": "Jurong Island",
    "hormuz": "Hormuz",
    "suez": "Suez",
    "cape": "Cape of Good Hope",
    "mandeb": "Bab el-Mandeb",
}
# Crop: Eurasia–Africa oil corridor. Houston and Panama fall outside.
MAP_EXTENT = (-20.0, -44.0, 150.0, 55.0)  # lon0, lat0, lon1, lat1
SHARE_MAX = 0.18          # lock halo size at ~18% of shipping |SHAP|
MAP_SIZE_AT_MAX = 1050.0  # matplotlib s (area) of the halo at SHARE_MAX
SIZE_GAMMA = 1.5          # area ∝ (share/SHARE_MAX)^γ; γ>1 exaggerates contrast
CENTER_SIZE = 28.0        # small centre dot for labelled nodes
OTHER_SIZE = 12.5         # unlabelled nodes; ~20% smaller, then slightly weaker
OTHER_COLOR = "#9A9A9A"
OTHER_ALPHA = 0.38
HALO_FACE_ALPHA = 0.30
HALO_EDGE_ALPHA = 0.78
HALO_EDGE_WIDTH = 0.75
SIZE_LEGEND_GREY = "#5E5E5E"
LABEL_STROKE = [
    patheffects.withStroke(linewidth=2.2, foreground="white"),
]
HEAT_ROLL = 6             # weeks; raw weekly field is speckled

EVENTS = [
    ("event_russia_ukraine", "2022-02-24", "Russia–Ukraine", "#B23A32"),
    ("event_eu_ru_oil_ban", "2022-06-01", "EU oil ban", "#6B5B95"),
    ("event_opec_plus", "2023-04-02", "OPEC+", "#2E7D5B"),
    ("event_red_sea", "2023-11-19", "Red Sea", "#D1622B"),
]
EVENT_WEEKS = 8

# Heatmap membership. Row order is computed from full-sample share_shipping
# (high to low) inside each panel — see fig45_node_heatmap.
HEAT_CHOKE = ["hormuz", "suez", "mandeb", "cape", "malacca", "panama"]
HEAT_AOI = ["P004", "P001", "P002", "P003", "P006", "P007",
            "P008", "P009", "P010", "P011", "P005"]
HEAT_LABEL = {
    "hormuz": "Hormuz", "suez": "Suez", "mandeb": "Bab el-Mandeb",
    "cape": "Cape of Good Hope", "malacca": "Malacca", "panama": "Panama",
    "P001": "Rotterdam", "P002": "Fujairah", "P003": "Ras Tanura",
    "P004": "Jurong Island", "P005": "Houston", "P006": "Ningbo-Zhoushan",
    "P007": "Jamnagar", "P008": "Basra", "P009": "Ulsan",
    "P010": "Kharg", "P011": "Yanbu",
}

C = {
    "ridge": "#8FA8C8",
    "xgb": "#2E5A88",
    "gated": "#D1622B",
    "xattn": "#E7A33E",
    "deep": "#D1622B",
    "finance": "#2E5A88",
    "shipping": "#D1622B",
    "aoi": "#2E5A88",
    "choke": "#D1622B",
    "land": "#EFEDE8",
    "coast": "#C9C4BB",
    "border": "#DCD7CE",
    "eval": "#D9D9D9",
    "pos": "#2E7D5B",
    "neg": "#B23A32",
    "grey": "#9A9A9A",
    "full": "#4A4A4A",
}

SHARE_CMAP = LinearSegmentedColormap.from_list(
    "ship_shap", ["#F7F1E8", "#F3D5A6", "#E08A4A", "#D1622B", "#8A3D16"]
)
# Heatmap: low end is sand, not near-white, so 0% is distinct from missing.
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "ship_shap_heat", ["#E4C496", "#E0B070", "#E08A4A", "#D1622B", "#8A3D16"]
)
HEAT_CMAP.set_bad("#D4D4D4")
EVENT_CODES = ["E1", "E2", "E3", "E4"]
WINDOW_FACE = "#8E8E8E"
RED_SEA_EDGE = "#C45C4A"

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

# Table 4.1 / 4.2 display values (percent). Plot these, not a second rounding of RMSE.
FLAT_IMP = [
    ("S1", -2.52, "\u22122.52%", -5.22, "\u22125.22%"),
    ("S2", -6.31, "\u22126.31%", -6.95, "\u22126.95%"),
    ("S3", -9.66, "\u22129.66%", -4.94, "\u22124.94%"),
    ("S4", -9.32, "\u22129.32%", -6.27, "\u22126.27%"),
]
DEEP_IMP = [
    ("S1", -2.36, "\u22122.36%", None, None),
    ("S2", -2.43, "\u22122.43%", -5.87, "\u22125.87%"),
    ("S3",  0.15, "+0.15%",       1.00, "+1.00%"),
    ("S4", -0.67, "\u22120.67%",  0.19, "+0.19%"),
]
DOT = {
    "ridge": "#1F4E79",
    "xgb": "#E07A2F",
    "gated": "#1F4E79",
    "xattn": "#6FA0C7",
    "link": "#C5C5C5",
    "muted": "#7A7A7A",
}
# Shared canvas for Figures 4.1 and 4.2. Identical size, crop, axis, grid, M0 rule.
IMP = {
    "figsize": (5.8, 3.90),
    "ylim": (-10.5, 1.5),
    "yticks": [-10, -8, -6, -4, -2, 0],
    "xlim": (-0.55, 3.80),
    "subplot": dict(left=0.155, right=0.975, top=0.965, bottom=0.185),
    "main_s": 80,
    "sec_s": 58,
    "link_w": 1.35,
    "label_fs": 7.5,
    "legend_fs": 8,
    "tick_fs": 8.5,
    "ylab_fs": 9,
    "m0_fs": 7.5,
}


def spread(values, min_gap: float):
    vals = np.asarray(values, float)
    order = np.argsort(vals)
    out = vals[order].copy()
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1] + min_gap)
    res = np.empty_like(out)
    res[order] = out
    return res


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def share_size(share: float) -> float:
    """Matplotlib scatter ``s`` (marker area in pt^2).

    Area scales as (share / SHARE_MAX) ** SIZE_GAMMA, locked at SHARE_MAX.
    SIZE_GAMMA > 1 makes large shares read larger and small shares smaller
    than strict area-linear encoding, so 2022 vs 2024 is easier to see.
    """
    frac = max(float(share), 0.0) / SHARE_MAX
    return (frac ** SIZE_GAMMA) * MAP_SIZE_AT_MAX


def event_bounds(centre: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    c = pd.Timestamp(centre)
    w = pd.Timedelta(weeks=EVENT_WEEKS)
    return c - w, c + w


def _save_imp(fig, name: str) -> None:
    """Fixed canvas; ignore the global tight-crop so 4.1 and 4.2 match."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None}):
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches=None,
                        pad_inches=0, facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def _open_improvement_fig():
    fig, ax = plt.subplots(figsize=IMP["figsize"])
    x = np.arange(4, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels(["S1", "S2", "S3", "S4"], fontsize=IMP["tick_fs"])
    ax.set_xlim(*IMP["xlim"])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", labelsize=IMP["tick_fs"])
    ax.axhline(0.0, color="#222222", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_ylim(*IMP["ylim"])
    ax.set_yticks(IMP["yticks"])
    ax.set_ylabel(r"$\Delta\mathrm{RMSE}$ (%)", fontsize=IMP["ylab_fs"])
    ax.grid(axis="y", visible=True, color="#D0D0D0", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.annotate(
        "M0", xy=(0.0, 0.0), xycoords=("axes fraction", "data"),
        xytext=(6.5, 2.4), textcoords="offset points",
        ha="left", va="bottom", fontsize=IMP["m0_fs"] + 1.0, fontweight="bold",
        color="#111111", zorder=6,
        path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")],
    )
    fig.subplots_adjust(**IMP["subplot"])
    return fig, ax, x


def _legend_below(ax, handles) -> None:
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.085),
              ncol=2, fontsize=IMP["legend_fs"], handletextpad=0.45,
              columnspacing=1.6, frameon=False, borderaxespad=0.0)


def _value_label(ax, x, y, text, color, *, dy=0.0, dy_pts=0.0) -> None:
    ax.annotate(
        text, xy=(x, y + dy), xytext=(6.5, dy_pts), textcoords="offset points",
        ha="left", va="center", fontsize=IMP["label_fs"], color=color, zorder=5,
        annotation_clip=False,
        path_effects=[patheffects.withStroke(linewidth=2.3, foreground="white")],
    )


# --------------------------------------------------------------------------
# 4.1  Flat: grouped Cleveland dot plot
# --------------------------------------------------------------------------
def fig41_flat_rmse() -> None:
    fig, ax, x = _open_improvement_fig()

    for xi, (_, ridge, ridge_lab, xgb, xgb_lab) in zip(x, FLAT_IMP):
        ax.plot([xi, xi], [ridge, xgb], color=DOT["link"],
                linewidth=IMP["link_w"], zorder=1, solid_capstyle="round")
        ax.scatter(xi, xgb, marker="s", s=IMP["sec_s"], color=DOT["xgb"],
                   zorder=4, edgecolors="white", linewidths=0.55)
        ax.scatter(xi, ridge, marker="o", s=IMP["main_s"], color=DOT["ridge"],
                   zorder=5, edgecolors="white", linewidths=0.55)
        close = abs(ridge - xgb) < 1.2
        _value_label(ax, xi, xgb, xgb_lab, DOT["xgb"],
                     dy=-0.18 if close else 0.0)
        _value_label(ax, xi, ridge, ridge_lab, DOT["ridge"],
                     dy=0.18 if close else 0.0)

    _legend_below(ax, [
        Line2D([], [], marker="o", linestyle="", color=DOT["ridge"],
               markersize=8, markeredgecolor="white", markeredgewidth=0.5,
               label="Ridge"),
        Line2D([], [], marker="s", linestyle="", color=DOT["xgb"],
               markersize=7, markeredgecolor="white", markeredgewidth=0.5,
               label="XGBoost"),
    ])
    _save_imp(fig, "fig_4_1_flat_rmse_improvement")


# --------------------------------------------------------------------------
# 4.2  Deep: dumbbell (gated main vs cross-attention)
# --------------------------------------------------------------------------
def fig42_deep_rmse() -> None:
    fig, ax, x = _open_improvement_fig()

    for xi, (_, gated, gated_lab, xattn, xattn_lab) in zip(x, DEEP_IMP):
        if xattn is not None:
            ax.plot([xi, xi], [gated, xattn], color=DOT["link"],
                    linewidth=IMP["link_w"], zorder=1, solid_capstyle="round")
            ax.scatter(xi, xattn, marker="D", s=IMP["sec_s"], facecolors="white",
                       edgecolors=DOT["xattn"], linewidths=1.4, zorder=4)
            # S4 +0.19% sits on the M0 rule; lift it a little above the dash.
            _value_label(ax, xi, xattn, xattn_lab, DOT["xattn"],
                         dy_pts=9.5 if xi == 3 else 0.0)
        ax.scatter(xi, gated, marker="o", s=IMP["main_s"], color=DOT["gated"],
                   zorder=5, edgecolors="white", linewidths=0.55)
        # S3 +0.15% up to roughly the height of S4 +0.19%, clearing the dash.
        _value_label(ax, xi, gated, gated_lab, DOT["gated"],
                     dy_pts=7.0 if xi == 2 else 0.0)

    _legend_below(ax, [
        Line2D([], [], marker="o", linestyle="", color=DOT["gated"],
               markersize=8, markeredgecolor="white", markeredgewidth=0.5,
               label="Gated (main)"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
               markeredgecolor=DOT["xattn"], markersize=7,
               markeredgewidth=1.35, label="Cross-attention"),
    ])
    _save_imp(fig, "fig_4_2_deep_rmse_improvement")


# --------------------------------------------------------------------------
# Figure 3.2 — Brent price and weekly log returns
# --------------------------------------------------------------------------
def fig32_price_returns() -> None:
    px = pd.read_csv(FEATURE_MATRIX,
                     usecols=["week_ending_friday", "brent_price",
                              "brent_log_return"],
                     parse_dates=["week_ending_friday"])
    px = px.set_index("week_ending_friday").sort_index()
    dates = pd.to_datetime(pd.read_csv(M3 / "deep_m3_shap_dates.csv")["date"])
    eval_lo, eval_hi = dates.min(), dates.max()

    fig, (axp, axr) = plt.subplots(
        2, 1, figsize=(9.6, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.08})

    for ax in (axp, axr):
        ax.fill_betweenx([0, 1], eval_lo, eval_hi,
                         transform=ax.get_xaxis_transform(),
                         color=C["eval"], alpha=0.50, zorder=0, linewidth=0)

    axp.tick_params(axis="x", which="both",
                    bottom=False, top=False, labelbottom=False)
    axr.tick_params(axis="x", which="both",
                    top=False, bottom=True, length=0, labelbottom=True)

    axp.plot(px.index, px["brent_price"], color="#333333", linewidth=1.15,
             zorder=3)
    axp.set_ylabel("Brent price (USD/bbl)")

    ret = px["brent_log_return"] * 100.0
    axr.axhline(0.0, color="#555555", linewidth=0.7, zorder=2)
    axr.plot(px.index, ret, color=C["finance"], linewidth=0.9, zorder=3)
    axr.set_ylabel("Weekly log return (%)")
    axr.set_xlabel("")

    axp.axvline(eval_lo, color="#555555", linewidth=0.9, linestyle="--", zorder=2)
    axp.annotate("evaluation sample",
                 xy=(eval_lo + pd.Timedelta(days=24), 0.92),
                 xycoords=("data", "axes fraction"),
                 fontsize=7.5, color="#444444", style="italic")

    handles = [Patch(facecolor=C["eval"], edgecolor="none",
                     label="Evaluation sample (from Jan 2021)")]
    axr.text(1.0, -0.18, "Week ending Friday", transform=axr.transAxes,
             ha="right", va="top", fontsize=9, clip_on=False)
    axr.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.0, -0.20), fontsize=7.5,
               handlelength=1.05, handleheight=0.65, handletextpad=0.35,
               borderpad=0.0, borderaxespad=0.0, frameon=False)

    axp.set_xlim(px.index.min() - pd.Timedelta(days=20),
                 px.index.max() + pd.Timedelta(days=20))
    axp.margins(y=0.10)
    axr.margins(y=0.14)
    for ax in (axp, axr):
        ax.grid(axis="x", visible=True)
        ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.22,
                        hspace=0.08)
    save(fig, "fig_3_2_price_returns")


# --------------------------------------------------------------------------
# 4.2  Flat → Deep paired slopes
# --------------------------------------------------------------------------
def _flat_xgb_rmse() -> dict[str, float]:
    """RMSE used in Tables 4.1 / 4.3 (not the stale subperiod file)."""
    files = {
        "S1": BASE / "Flat" / "M1_Flat" / "baseline_metrics.csv",
        "S2": BASE / "Flat" / "M2_Flat" / "baseline_metrics_anom.csv",
        "S3": BASE / "Flat" / "M3_Flat" / "baseline_metrics.csv",
        "S4": BASE / "Flat" / "M4_Flat" / "baseline_metrics_anom.csv",
    }
    out = {}
    for s, path in files.items():
        d = pd.read_csv(path).set_index("model")
        out[s] = float(d.loc[f"M{s[1]}_Flat_XGB", "RMSE"])
    return out


def fig42_slope() -> None:
    fusion = pd.read_csv(DEEP / "_cross" / "deep_fusion_matrix.csv")
    fusion = fusion.set_index(["combo", "fusion"])
    m0 = (pd.read_csv(DEEP / "_cross" / "deep_metrics.csv")
          .set_index("model").loc["M0_RW", "RMSE"])
    m1 = (pd.read_csv(DEEP / "_cross" / "deep_metrics.csv")
          .set_index("model").loc["M1_Deep", "RMSE"])
    flat_rmse = _flat_xgb_rmse()

    flat, deep = [], []
    for s, m in zip(SETS, ["M1", "M2", "M3", "M4"]):
        flat.append(flat_rmse[s])
        deep.append(m1 if s == "S1"
                    else float(fusion.loc[(f"{m}_Deep", "gated"), "RMSE"]))

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    flat_lab = spread(flat, 0.024)
    deep_lab = spread(deep, 0.024)
    s1 = SETS.index("S1")
    s3 = SETS.index("S3")
    # S1 Deep is 4.250; drop the label off the 4.25 grid line.
    deep_lab[s1] = float(deep_lab[s1]) - 0.024
    # S3 Deep sits 0.006 below M0; keep its right-hand label under the rule.
    deep_lab[s3] = min(float(deep_lab[s3]), m0 - 0.022)

    for i, s in enumerate(SETS):
        highlight = s == "S3"
        ax.plot([0, 1], [flat[i], deep[i]], "-o", markersize=5.5,
                color=C["deep"] if highlight else C["grey"],
                linewidth=2.2 if highlight else 1.3,
                zorder=3 if highlight else 2)
        ax.text(-0.05, flat_lab[i], f"{s}  {flat[i]:.3f}", ha="right",
                va="center", fontsize=8)
        ax.text(1.05, deep_lab[i], f"{deep[i]:.3f}  {s}", ha="left",
                va="center", fontsize=8,
                fontweight="bold" if highlight else "normal",
                color=C["deep"] if highlight else "black")

    ax.axhline(m0, color="black", linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate(
        f"M0 = {m0:.3f}",
        xy=(-0.05, m0),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right", va="bottom",
        fontsize=8, style="italic",
        zorder=4,
    )

    ax.set_xlim(-0.62, 1.48)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Flat\n(XGBoost)", "Deep\n(gated fusion)"])
    ax.set_ylabel("RMSE (USD/bbl)")
    y_lo = min(min(flat), min(deep), m0)
    y_hi = max(max(flat), max(deep), m0)
    pad = (y_hi - y_lo) * 0.16
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_4_3_flat_vs_deep_slope")


# --------------------------------------------------------------------------
# 4.3  Shipping only: gate weight vs |SHAP| share
# --------------------------------------------------------------------------
PERIOD_ORDER = [
    "full",
    "year_2021", "year_2022", "year_2023", "year_2024", "year_2025",
    "event_russia_ukraine", "event_eu_ru_oil_ban",
    "event_opec_plus", "event_red_sea",
]
PERIOD_YLABEL = {
    "full": "Full sample",
    "year_2021": "2021", "year_2022": "2022", "year_2023": "2023",
    "year_2024": "2024", "year_2025": "2025",
    "event_russia_ukraine": "Russia–Ukraine",
    "event_eu_ru_oil_ban": "EU oil ban",
    "event_opec_plus": "OPEC+",
    "event_red_sea": "Red Sea",
}


def fig43_shap_modality() -> None:
    """Dual-dot plot of shipping gate weight vs shipping |SHAP| share.

    Finance is omitted (always 100% minus shipping). The two series are not
    joined by a line: gate is a representation weight, SHAP is output
    attribution.
    """
    main = pd.read_csv(M3 / "deep_m3_rq3_period_main.csv")
    main = main.set_index("period_id").loc[PERIOD_ORDER]

    gate = main["gate_shipping"].to_numpy(float) * 100
    shap = main["shap_share_shipping"].to_numpy(float) * 100

    n = len(PERIOD_ORDER)
    y = np.arange(n)[::-1]
    # Offset so the two metrics sit as a pair, not on top of each other.
    y_gate = y + 0.16
    y_shap = y - 0.16

    fig, ax = plt.subplots(figsize=(6.8, 5.5))

    year_lo, year_hi = y[5] - 0.48, y[1] + 0.48
    event_lo, event_hi = y[9] - 0.48, y[6] + 0.48
    ax.axhspan(year_lo, year_hi, color="#F4F4F4", zorder=0)
    ax.axhline((y[5] + y[6]) / 2, color="#C8C8C8", linewidth=0.8, zorder=1)
    ax.hlines(y, 0, 60, color="#E8E8E8", linewidth=0.6, zorder=1)

    ax.scatter(gate, y_gate, s=62, marker="D",
               facecolors="white", edgecolors=C["shipping"],
               linewidths=1.35, zorder=4, label="Shipping gate weight")
    ax.scatter(shap, y_shap, s=50, marker="o",
               facecolors=C["shipping"], edgecolors=C["shipping"],
               linewidths=0.4, zorder=4,
               label="Shipping share of absolute SHAP")

    ax.set_yticks(y)
    ax.set_yticklabels([PERIOD_YLABEL[p] for p in PERIOD_ORDER], fontsize=8)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(-1.2, 60)
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_ylim(-0.72, n - 0.28)
    ax.set_xlabel("Percent")
    ax.set_title(
        "Shipping received 25–52% of the gate weight but only "
        "0.8–5.9% of absolute SHAP\n"
        "Gate is a representation weight; SHAP is output attribution "
        "— they are not interchangeable",
        loc="left",
    )
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    ax.text(61.6, (year_lo + year_hi) / 2, "calendar years",
            rotation=90, va="center", ha="left", fontsize=7.2,
            color="#666666", clip_on=False)
    ax.text(61.6, (event_lo + event_hi) / 2, "event windows (\u00b18 weeks)",
            rotation=90, va="center", ha="left", fontsize=7.2,
            color="#666666", clip_on=False)

    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.48, -0.11),
              fontsize=7.5, handletextpad=0.4)
    fig.subplots_adjust(left=0.20, right=0.90, top=0.88, bottom=0.16)
    save(fig, "fig_4_3_shap_modality")


# --------------------------------------------------------------------------
# 4.4  Centre-dot + proportional halo map (2022 | 2024)
# --------------------------------------------------------------------------
# Label offsets in degrees. Cape sits above the marker so the two-line
# label clears the southern frame and the longitude ticks.
MAP_OFF = {
    "P004": (7.2, -7.0, "left", "top"),
    "hormuz": (7.6, 5.0, "left", "bottom"),
    "suez": (-7.2, 5.4, "right", "bottom"),
    "cape": (0.0, 8.8, "center", "bottom"),
    "mandeb": (-7.8, -5.0, "right", "top"),
}


def _basemap(ax, bounds) -> None:
    import geopandas as gpd

    land = gpd.read_file(NE_DIR / "ne_110m_land.shp")
    land.plot(ax=ax, facecolor=C["land"], edgecolor=C["coast"], linewidth=0.4,
              zorder=0)
    borders = NE_DIR / "ne_110m_admin_0_boundary_lines_land.shp"
    if borders.exists():
        gpd.read_file(borders).plot(ax=ax, color=C["border"], linewidth=0.35,
                                    zorder=1)
    lon0, lat0, lon1, lat1 = bounds
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / math.cos(math.radians((lat0 + lat1) / 2)))


def _draw_halo(ax, lon: float, lat: float, share: float, color: str,
               z_halo: int = 4) -> None:
    """Semi-transparent disk; area scales with share (see share_size)."""
    ax.scatter(
        lon, lat, marker="o", s=share_size(share),
        facecolor=to_rgba(color, HALO_FACE_ALPHA),
        edgecolor=to_rgba(color, HALO_EDGE_ALPHA),
        linewidth=HALO_EDGE_WIDTH, zorder=z_halo,
    )
    ax.scatter(
        lon, lat, marker="o", s=CENTER_SIZE, color=color,
        edgecolor="white", linewidth=0.7, zorder=z_halo + 2,
    )


def _draw_size_ring(ax, lon: float, lat: float, share: float,
                    z: int = 4) -> None:
    """Neutral grey ring for the size legend (same area scale as the map)."""
    ax.scatter(
        lon, lat, marker="o", s=share_size(share),
        facecolor="none",
        edgecolor=to_rgba(SIZE_LEGEND_GREY, 0.95),
        linewidth=1.15, zorder=z,
    )


def _draw_node_label(ax, lon: float, lat: float, dx: float, dy: float,
                     ha: str, va: str, name: str, pct: str,
                     color: str) -> None:
    """Name in semi-bold, percentage one step lighter, same colour."""
    x, y = lon + dx, lat + dy
    name_kw = dict(
        ha=ha, color=color, fontsize=7.0, fontweight=600,
        zorder=7, path_effects=LABEL_STROKE, clip_on=False,
    )
    pct_kw = dict(
        ha=ha, color=color, fontsize=6.2, fontweight="normal",
        zorder=7, path_effects=LABEL_STROKE, alpha=0.70,
    )
    if va == "top":
        ax.text(x, y, name, va="top", **name_kw)
        ax.annotate(
            pct, xy=(x, y), xytext=(0, -9.2), textcoords="offset points",
            va="top", annotation_clip=False, **pct_kw,
        )
    else:
        ax.text(x, y, pct, va="bottom", clip_on=False, **pct_kw)
        ax.annotate(
            name, xy=(x, y), xytext=(0, 9.2), textcoords="offset points",
            va="bottom", annotation_clip=False, fontweight=600,
            ha=ha, color=color, fontsize=7.0, zorder=7,
            path_effects=LABEL_STROKE,
        )


def _draw_share_nodes(ax, shares: dict[str, float]) -> None:
    lon0, lat0, lon1, lat1 = MAP_EXTENT

    def on_map(lon, lat) -> bool:
        return lon0 <= lon <= lon1 and lat0 <= lat <= lat1

    # Unlabelled nodes: fixed grey dots, drawn above halos so Gulf/Malacca
    # neighbours remain visible inside a large neighbouring halo.
    for nid, (_name, _stype, lon, lat) in AOI_XY.items():
        if nid in NAMED_NODES or not on_map(lon, lat):
            continue
        ax.scatter(lon, lat, marker="o", s=OTHER_SIZE,
                   facecolor=to_rgba(OTHER_COLOR, OTHER_ALPHA),
                   edgecolor=to_rgba("#7A7A7A", OTHER_ALPHA * 0.85),
                   linewidth=0.3, zorder=5)
    for cid, (_name, lon, lat) in CHOKE_XY.items():
        if cid in NAMED_NODES or not on_map(lon, lat):
            continue
        ax.scatter(lon, lat, marker="o", s=OTHER_SIZE,
                   facecolor=to_rgba(OTHER_COLOR, OTHER_ALPHA),
                   edgecolor=to_rgba("#7A7A7A", OTHER_ALPHA * 0.85),
                   linewidth=0.3, zorder=5)

    for nid in NAMED_NODES:
        share = float(shares.get(nid, 0))
        if nid in AOI_XY:
            _name, _stype, lon, lat = AOI_XY[nid]
            color = C["aoi"]
            lab_color = "#1F3E5F"
        else:
            _name, lon, lat = CHOKE_XY[nid]
            color = C["choke"]
            lab_color = "#8A3D16"
        _draw_halo(ax, lon, lat, share, color)
        dx, dy, ha, va = MAP_OFF[nid]
        _draw_node_label(ax, lon, lat, dx, dy, ha, va,
                         NAMED_LABEL[nid], f"{share * 100:.1f}%", lab_color)


def fig44_node_map() -> None:
    if not (NE_DIR / "ne_110m_land.shp").exists():
        raise FileNotFoundError(
            f"Natural Earth 110m land not found under {NE_DIR}.")

    nodes = pd.read_csv(M3 / "deep_m3_shap_period_node.csv")
    # Calendar years with the largest L1 reallocation of shipping-internal
    # node shares (full sample excluded). 2022 is Jurong/Hormuz-led;
    # 2024 is Suez / Bab el-Mandeb / Cape-led.
    panels = [
        ("year_2022", "A  2022"),
        ("year_2024", "B  2024"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    for ax, (pid, title) in zip(axes, panels):
        sub = nodes[nodes["period_id"] == pid]
        shares = dict(zip(sub["node_id"], sub["share_shipping"].astype(float)))
        _basemap(ax, MAP_EXTENT)
        _draw_share_nodes(ax, shares)
        ax.set_title(title, loc="left", fontsize=10, pad=8)
        ax.set_xticks(range(0, 151, 30))
        ax.set_yticks(range(-30, 61, 30))
        ax.set_xticklabels(
            [f"{v}\u00b0E" if v else "0\u00b0" for v in range(0, 151, 30)],
            fontsize=7)
        ax.set_yticklabels(
            [f"{abs(v)}\u00b0{'S' if v < 0 else 'N' if v else ''}"
             for v in range(-30, 61, 30)],
            fontsize=7)
        ax.tick_params(length=2.2, pad=1.2)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#AAAAAA")
            sp.set_linewidth(0.7)

    fig.suptitle(
        "Shipping-internal node SHAP shares on the Eurasia–Africa "
        "oil corridor\n"
        "Halo size scales with share (locked at 18%). Shares are within shipping, "
        "not total attribution.\n"
        "Houston and Panama lie outside this frame.",
        x=0.02, ha="left", fontsize=10, linespacing=1.25,
    )
    fig.subplots_adjust(left=0.05, right=0.86, top=0.82, bottom=0.08,
                        wspace=0.08)

    # Same baseline as the panel titles: just above the map frame.
    title_off = ScaledTranslation(10 / 72, 8 / 72, fig.dpi_scale_trans)
    axes[1].text(
        1.0, 1.0, "Shipping SHAP share",
        transform=axes[1].transAxes + title_off,
        ha="left", va="bottom", fontsize=7.4, color="#333333",
        clip_on=False,
    )
    # Legend body starts below the header so the 15% ring cannot overlap it.
    leg = fig.add_axes([0.875, 0.08, 0.115, 0.68])
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.axis("off")
    y = 0.88
    for sh in (0.05, 0.10, 0.15):
        _draw_size_ring(leg, 0.30, y, sh)
        leg.text(0.54, y, f"{int(sh * 100)}%", va="center", fontsize=7.5,
                 color="#444444")
        y -= 0.22
    y = 0.24
    for size, color, lab, edge, alpha in (
        (CENTER_SIZE, C["choke"], "Chokepoint", "white", 1.0),
        (CENTER_SIZE, C["aoi"], "AOI", "white", 1.0),
        (OTHER_SIZE, OTHER_COLOR, "Other nodes", "#7A7A7A", OTHER_ALPHA),
    ):
        leg.scatter([0.22], [y], s=size, marker="o",
                    facecolor=to_rgba(color, alpha),
                    edgecolor=to_rgba(edge, min(alpha + 0.25, 1.0)),
                    linewidth=0.6, zorder=6)
        leg.text(0.42, y, lab, va="center", fontsize=7.2)
        y -= 0.10
    save(fig, "fig_4_4_node_shap_map")


# --------------------------------------------------------------------------
# 4.5  Node × time SHAP-share heatmap
# --------------------------------------------------------------------------
def _order_by_full_share(node_ids: list[str]) -> list[str]:
    """Full-sample within-shipping |SHAP| share, high to low."""
    nodes = pd.read_csv(M3 / "deep_m3_shap_period_node.csv")
    full = (nodes.loc[nodes["period_id"] == "full"]
            .set_index("node_id")["share_shipping"].astype(float))
    return sorted(node_ids, key=lambda n: -float(full.loc[n]))


def _draw_event_lane(ax, x0: float, x1: float) -> None:
    """Event names on one baseline; no grey header blocks."""
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.xaxis_date()
    ax.tick_params(bottom=False, labelbottom=False, length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("none")
    xs = [mdates.date2num(pd.Timestamp(c)) for _, c, _, _ in EVENTS]
    xs = spread(xs, min_gap=280)
    for x, (_, _, lab, _) in zip(xs, EVENTS):
        ax.text(x, 0.45, lab, ha="center", va="center", fontsize=7.4,
                color="#333333", fontweight="bold", zorder=2, clip_on=True)


def _draw_event_windows(ax) -> None:
    """±8-week windows as uniform light-grey bands."""
    for _, centre, _, _ in EVENTS:
        lo, hi = event_bounds(centre)
        ax.axvspan(mdates.date2num(lo), mdates.date2num(hi),
                   facecolor=WINDOW_FACE, alpha=0.26, zorder=3, linewidth=0)


def _heatmap_panel(ax, roll: pd.DataFrame, node_ids: list[str],
                   x0: float, x1: float):
    Z = np.ma.masked_invalid(roll[node_ids].to_numpy(float).T)
    im = ax.imshow(
        Z, aspect="auto", interpolation="nearest",
        cmap=HEAT_CMAP, vmin=0.0, vmax=SHARE_MAX,
        extent=[x0, x1, len(node_ids) - 0.5, -0.5],
        zorder=2,
    )
    ax.set_yticks(np.arange(len(node_ids)))
    ax.set_yticklabels([HEAT_LABEL[n] for n in node_ids], fontsize=8)
    for tick in ax.get_yticklabels():
        tick.set_color("black")
        tick.set_fontweight("normal")
        tick.set_style("normal")
    ax.set_xlim(x0, x1)
    ax.xaxis_date()
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(0.7)
    _draw_event_windows(ax)
    return im


def fig45_node_heatmap() -> None:
    weekly = pd.read_csv(M3 / "deep_m3_shap_weekly_node.csv", parse_dates=["date"])
    tot = weekly.groupby("date")["shap_abs"].transform("sum")
    weekly["share"] = weekly["shap_abs"] / tot
    pv = weekly.pivot(index="date", columns="node_id", values="share").sort_index()
    # Right-aligned trailing mean; first HEAT_ROLL-1 weeks are undefined.
    roll = pv.rolling(HEAT_ROLL, min_periods=HEAT_ROLL, center=False).mean()

    choke = _order_by_full_share(HEAT_CHOKE)
    aoi = _order_by_full_share(HEAT_AOI)

    red_lo, red_hi = event_bounds("2023-11-19")
    red = roll.loc[(roll.index >= red_lo) & (roll.index <= red_hi)]
    red_max = float(red.max().max())
    red_node = HEAT_LABEL[str(red.max().idxmax())]
    print(f"  Red Sea window max trailing-{HEAT_ROLL}w share: "
          f"{red_max * 100:.1f}% ({red_node})")

    fig = plt.figure(figsize=(10.8, 7.6))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.10, 1.0], hspace=0.04,
        left=0.16, right=0.90, top=0.88, bottom=0.08,
    )
    inner = outer[1].subgridspec(
        2, 2, width_ratios=[1.0, 0.028],
        height_ratios=[len(choke), len(aoi)],
        hspace=0.08, wspace=0.08,
    )
    axlab = fig.add_subplot(outer[0])
    ax0 = fig.add_subplot(inner[0, 0], sharex=axlab)
    ax1 = fig.add_subplot(inner[1, 0], sharex=axlab)
    cax = fig.add_subplot(inner[:, 1])

    x0 = mdates.date2num(roll.index[0])
    x1 = mdates.date2num(roll.index[-1])
    x_right = max(x1, mdates.date2num(pd.Timestamp("2026-01-01")))

    _draw_event_lane(axlab, x0, x_right)
    im = _heatmap_panel(ax0, roll, choke, x0, x1)
    _heatmap_panel(ax1, roll, aoi, x0, x1)
    for ax in (axlab, ax0, ax1):
        ax.set_xlim(x0, x_right)

    ax0.tick_params(labelbottom=False)

    year_ticks = [pd.Timestamp(f"{y}-01-01") for y in range(2021, 2027)]
    ax1.set_xticks([mdates.date2num(t) for t in year_ticks])
    ax1.set_xticklabels([str(t.year) for t in year_ticks], fontsize=8)
    ax1.set_xlabel("Forecast origin")

    fig.suptitle(
        "Temporal variation in node-level shipping attribution in Deep S3\n"
        "Six-week trailing mean of each node\u2019s share of absolute SHAP "
        "within the shipping modality",
        x=0.16, ha="left", fontsize=10,
    )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Node SHAP share (%)", fontsize=8)
    cbar.set_ticks([0.00, 0.06, 0.12, 0.18])
    cbar.set_ticklabels(["0", "6", "12", "18"])
    save(fig, "fig_4_5_node_shap_heatmap")


# --------------------------------------------------------------------------
# 4.6  Seed robustness
# --------------------------------------------------------------------------
SEED_SPECS = [
    ("m1_deep", "S1", "Deep (finance only)"),
    ("m2_deep_gated", "S2", "Gated"),
    ("m2_deep_concat", "S2", "Concat"),
    ("m2_deep_xattn", "S2", "Cross-attn"),
    ("m3_deep_gated", "S3", "Gated"),
    ("m3_deep_concat", "S3", "Concat"),
    ("m3_deep_xattn", "S3", "Cross-attn"),
    ("m4_deep_gated", "S4", "Gated"),
    ("m4_deep_concat", "S4", "Concat"),
    ("m4_deep_xattn", "S4", "Cross-attn"),
]
SEED_MARK = {42: ("D", 78, 1.15), 1: ("o", 42, 0.85), 2: ("s", 42, 0.85)}


def fig46_seed_robustness() -> None:
    pooled = pd.read_csv(DEEP / "_cross" / "deep_seed_pooled.csv")
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    y = np.arange(len(SEED_SPECS))[::-1]

    # Band by information set
    set_spans = {"S1": (9.5, 10.5), "S2": (6.5, 9.5),
                 "S3": (3.5, 6.5), "S4": (0.5, 3.5)}
    # y positions from bottom: last spec y=0 ... first spec y=9
    # Rebuild spans from actual y.
    by_set: dict[str, list[float]] = {}
    for yi, (cfg, s, _) in zip(y, SEED_SPECS):
        by_set.setdefault(s, []).append(yi)
    for s, ys in by_set.items():
        lo, hi = min(ys) - 0.45, max(ys) + 0.45
        ax.axhspan(lo, hi, color="#F4F4F4" if s in ("S1", "S3") else "white",
                   zorder=0)

    for yi, (cfg, s, lab) in zip(y, SEED_SPECS):
        d = pooled[pooled["config"] == cfg]
        skills = {int(r.seed): float(r.skill_vs_M0) * 100 for r in d.itertuples()}
        mean = float(np.mean(list(skills.values())))
        highlight = cfg == "m3_deep_gated"
        ax.plot([min(skills.values()), max(skills.values())], [yi, yi],
                color=C["gated"] if highlight else "#C8C8C8",
                linewidth=1.4 if highlight else 0.9, zorder=1)
        ax.plot(mean, yi, marker="|", markersize=14, markeredgewidth=2.0,
                color="#222222", zorder=4)
        for seed, (mk, sz, mew) in SEED_MARK.items():
            v = skills[seed]
            main = seed == 42
            ax.scatter(v, yi, marker=mk, s=sz, zorder=5 if main else 3,
                       facecolors=C["gated"] if main else "white",
                       edgecolors=C["gated"] if highlight else "#555555",
                       linewidths=mew)

    ax.axvline(0.0, color="black", linewidth=1.1, zorder=2)
    ax.set_yticks(y)
    labels = [f"{s}  {lab}" for _, s, lab in SEED_SPECS]
    ax.set_yticklabels(labels, fontsize=8)
    for tick, (cfg, _, _) in zip(ax.get_yticklabels(), SEED_SPECS):
        if cfg == "m3_deep_gated":
            tick.set_fontweight("bold")
    ax.set_xlabel("RMSE improvement vs M0 (%)")
    ax.set_title("Main-run +0.15% at gated S3 is the one positive seed; "
                 "the three-seed mean stays negative", loc="left")
    ax.set_xlim(-9.2, 2.4)
    ax.grid(axis="y", visible=False)

    handles = [
        Line2D([], [], marker="D", linestyle="", color=C["gated"],
               markersize=8, label="Main run (seed 42)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="#555555", markersize=7, label="Seed 1"),
        Line2D([], [], marker="s", linestyle="", markerfacecolor="white",
               markeredgecolor="#555555", markersize=7, label="Seed 2"),
        Line2D([], [], marker="|", linestyle="", color="#222222",
               markersize=10, markeredgewidth=2, label="Three-seed mean"),
    ]
    ax.legend(handles=handles, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_B_1_seed_robustness")


FIGURES = {
    "4.1": fig41_flat_rmse,
    "4.2": fig42_deep_rmse,
    "price": fig32_price_returns,
    "slope": fig42_slope,
    "4.3": fig43_shap_modality,
    "4.4": fig44_node_map,
    "4.5": fig45_node_heatmap,
    "B.1": fig46_seed_robustness,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()
    keys = args.only or list(FIGURES)
    for k in keys:
        print(f"[{k}] {FIGURES[k].__name__}")
        FIGURES[k]()
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
