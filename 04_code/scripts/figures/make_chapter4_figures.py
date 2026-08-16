"""
Chapter 4 figures (current numbering, first mention in the results chapter).

Reads committed CSVs / npy under 05_outputs/ and the weekly feature matrix.
Does not retrain models.

    python 04_code/scripts/figures/make_chapter4_figures.py
    python 04_code/scripts/figures/make_chapter4_figures.py --only 4.1 4.3 4.5

    4.1  Brent price, weekly log returns, evaluation sample, four event windows
    4.2  Flat XGBoost → Deep gated paired slopes (S3 highlighted; S1 as path reference)
    4.3  Shipping gate weight vs shipping |SHAP| share (dual-dot, no link)
    4.4  Two-panel proportional-symbol map (full sample | 2024)
    4.5  Node × week shipping-internal SHAP-share heatmap
    4.6  Deep seed robustness (three seeds + mean, vs M0)

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    "P004": ("Jurong", "refinery", 103.708, 1.274),
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
    "P004": "Jurong",
    "hormuz": "Hormuz",
    "suez": "Suez",
    "cape": "Cape of Good Hope",
    "mandeb": "Bab el-Mandeb",
}
# Crop: Eurasia–Africa oil corridor. Houston and Panama fall outside.
MAP_EXTENT = (-20.0, -40.0, 150.0, 55.0)  # lon0, lat0, lon1, lat1
SHARE_MAX = 0.18          # lock area / colour at ~18% of shipping |SHAP|
MAP_SIZE_AT_MAX = 520.0   # matplotlib s (area) for SHARE_MAX
HEAT_ROLL = 6             # weeks; raw weekly field is speckled

EVENTS = [
    ("event_russia_ukraine", "2022-02-24", "Russia–Ukraine", "#B23A32"),
    ("event_eu_ru_oil_ban", "2022-06-01", "EU oil ban", "#6B5B95"),
    ("event_opec_plus", "2023-04-02", "OPEC+", "#2E7D5B"),
    ("event_red_sea", "2023-11-19", "Red Sea", "#D1622B"),
]
EVENT_WEEKS = 8

# Heatmap rows: chokepoints first (named four, then Malacca / Panama), then AOIs
# with Jurong at the top of the AOI block.
HEAT_CHOKE = ["hormuz", "suez", "mandeb", "cape", "malacca", "panama"]
HEAT_AOI = ["P004", "P001", "P002", "P003", "P006", "P007",
            "P008", "P009", "P010", "P011", "P005"]
HEAT_LABEL = {
    "hormuz": "Hormuz", "suez": "Suez", "mandeb": "Bab el-Mandeb",
    "cape": "Cape of Good Hope", "malacca": "Malacca", "panama": "Panama",
    "P001": "Rotterdam", "P002": "Fujairah", "P003": "Ras Tanura",
    "P004": "Jurong", "P005": "Houston", "P006": "Ningbo-Zhoushan",
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

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
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
})


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
    return max(float(share), 0.0) / SHARE_MAX * MAP_SIZE_AT_MAX


def event_bounds(centre: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    c = pd.Timestamp(centre)
    w = pd.Timedelta(weeks=EVENT_WEEKS)
    return c - w, c + w


# --------------------------------------------------------------------------
# 4.1  Price, returns, event windows
# --------------------------------------------------------------------------
def fig41_price_returns() -> None:
    px = pd.read_csv(FEATURE_MATRIX,
                     usecols=["week_ending_friday", "brent_price",
                              "brent_log_return"],
                     parse_dates=["week_ending_friday"])
    px = px.set_index("week_ending_friday").sort_index()
    dates = pd.to_datetime(pd.read_csv(M3 / "deep_m3_shap_dates.csv")["date"])
    eval_lo, eval_hi = dates.min(), dates.max()

    fig, axes = plt.subplots(
        3, 1, figsize=(9.6, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [0.22, 1.45, 1.0], "hspace": 0.05})
    axlab, axp, axr = axes

    for ax in (axp, axr):
        ax.fill_betweenx([0, 1], eval_lo, eval_hi,
                         transform=ax.get_xaxis_transform(),
                         color=C["eval"], alpha=0.50, zorder=0, linewidth=0)

    for _, centre, lab, col in EVENTS:
        lo, hi = event_bounds(centre)
        axlab.axvspan(lo, hi, color=col, alpha=0.28, zorder=0, linewidth=0)
        axlab.text(pd.Timestamp(centre), 0.5, lab, ha="center", va="center",
                   fontsize=7.2, color=col, fontweight="bold", zorder=2)
        for ax in (axp, axr):
            ax.axvspan(lo, hi, color=col, alpha=0.20, zorder=1, linewidth=0)
            ax.axvline(pd.Timestamp(centre), color=col, linewidth=0.9,
                       linestyle=":", zorder=2)

    axlab.set_ylim(0, 1)
    axlab.set_yticks([])
    for sp in axlab.spines.values():
        sp.set_visible(False)
    axlab.grid(False)
    axlab.set_title("Brent price and weekly log returns, with the evaluation "
                    "sample and four \u00b18-week event windows", loc="left")

    axp.plot(px.index, px["brent_price"], color="#333333", linewidth=1.15,
             zorder=3)
    axp.set_ylabel("Brent price\n(USD per barrel)")

    ret = px["brent_log_return"] * 100.0
    axr.axhline(0.0, color="#555555", linewidth=0.7, zorder=2)
    axr.plot(px.index, ret, color=C["finance"], linewidth=0.9, zorder=3)
    axr.set_ylabel("Weekly log return (%)")
    axr.set_xlabel("Week ending Friday")
    axr.annotate("COVID crash", xy=(pd.Timestamp("2020-03-20"), ret.min()),
                 xytext=(pd.Timestamp("2019-06-01"), ret.min() + 6),
                 fontsize=7, color="#666666",
                 arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.6))

    axp.axvline(eval_lo, color="#555555", linewidth=0.9, linestyle="--", zorder=2)
    axp.annotate("evaluation sample",
                 xy=(eval_lo + pd.Timedelta(days=24), 0.92),
                 xycoords=("data", "axes fraction"),
                 fontsize=7.5, color="#444444", style="italic")

    handles = [Patch(facecolor=C["eval"], edgecolor="none",
                     label="Evaluation sample (from Jan 2021)")]
    handles += [Patch(facecolor=col, alpha=0.45, edgecolor="none", label=lab)
                for _, _, lab, col in EVENTS]
    axr.legend(handles=handles, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, -0.28), fontsize=7.5)

    axp.set_xlim(px.index.min() - pd.Timedelta(days=20),
                 px.index.max() + pd.Timedelta(days=20))
    axp.margins(y=0.10)
    axr.margins(y=0.14)
    for ax in (axp, axr):
        ax.grid(axis="x", visible=True)
        ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.16,
                        hspace=0.08)
    save(fig, "fig_4_1_price_returns")


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

    for i, s in enumerate(SETS):
        highlight = s == "S3"
        is_ref = s == "S1"
        ax.plot([0, 1], [flat[i], deep[i]], "-o", markersize=5.5,
                color=C["deep"] if highlight else C["grey"],
                linewidth=2.2 if highlight else 1.3,
                zorder=3 if highlight else 2)
        left = (f"{s} (ref.)  {flat[i]:.3f}" if is_ref
                else f"{s}  {flat[i]:.3f}")
        ax.text(-0.05, flat_lab[i], left, ha="right",
                va="center", fontsize=8)
        ax.text(1.05, deep_lab[i], f"{deep[i]:.3f}  {s}", ha="left",
                va="center", fontsize=8,
                fontweight="bold" if highlight else "normal",
                color=C["deep"] if highlight else "black")

    ax.axhline(m0, color="black", linewidth=1.0, linestyle="--")
    ax.text(0.04, m0, f"M0 = {m0:.3f}", ha="left", va="center",
            fontsize=8, style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    ax.set_xlim(-0.62, 1.48)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Flat\n(XGBoost)", "Deep\n(gated fusion)"])
    ax.set_ylabel("Out-of-sample RMSE (USD per barrel; reversed)")
    ax.set_title(
        "Within matched modality sets and the common evaluation sample,\n"
        "all Deep specifications reduce RMSE relative to XGBoost; only S3 also beats M0",
        loc="left",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", visible=False)
    save(fig, "fig_4_2_flat_vs_deep_slope")


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
# 4.4  Proportional-symbol map (full | 2024)
# --------------------------------------------------------------------------
# Label offsets in degrees for the cropped corridor frame.
MAP_OFF = {
    "P004": (3.2, -4.2, "left", "top"),
    "hormuz": (3.8, 2.4, "left", "bottom"),
    "suez": (-3.4, 3.0, "right", "bottom"),
    "cape": (3.4, -2.2, "left", "top"),
    "mandeb": (-3.6, -2.4, "right", "top"),
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


def _draw_share_nodes(ax, shares: dict[str, float]) -> None:
    lon0, lat0, lon1, lat1 = MAP_EXTENT

    def on_map(lon, lat) -> bool:
        return lon0 <= lon <= lon1 and lat0 <= lat <= lat1

    # Unnamed first (under), then named.
    for nid, (name, stype, lon, lat) in AOI_XY.items():
        if nid in NAMED_NODES or not on_map(lon, lat):
            continue
        ax.scatter(lon, lat, marker=TYPE_MARKER[stype],
                   s=max(share_size(shares.get(nid, 0)), 12),
                   color="#B8B8B8", edgecolor="white", linewidth=0.4,
                   alpha=0.85, zorder=4)
    for cid, (name, lon, lat) in CHOKE_XY.items():
        if cid in NAMED_NODES or not on_map(lon, lat):
            continue
        ax.scatter(lon, lat, marker="D",
                   s=max(share_size(shares.get(cid, 0)), 14),
                   color="#C4C4C4", edgecolor="white", linewidth=0.4,
                   alpha=0.9, zorder=4)

    for nid in NAMED_NODES:
        share = float(shares.get(nid, 0))
        s = share_size(share)
        if nid in AOI_XY:
            name, stype, lon, lat = AOI_XY[nid]
            marker, color = TYPE_MARKER[stype], C["aoi"]
        else:
            name, lon, lat = CHOKE_XY[nid]
            marker, color = "D", C["choke"]
        ax.scatter(lon, lat, marker=marker, s=s, color=color,
                   edgecolor="white", linewidth=0.8, zorder=6)
        dx, dy, ha, va = MAP_OFF[nid]
        ax.text(lon + dx, lat + dy, f"{NAMED_LABEL[nid]}\n{share * 100:.1f}%",
                ha=ha, va=va, fontsize=7.2, fontweight="bold", zorder=7,
                color="#1F3E5F" if nid in AOI_XY else "#8A3D16")


def fig44_node_map() -> None:
    if not (NE_DIR / "ne_110m_land.shp").exists():
        raise FileNotFoundError(
            f"Natural Earth 110m land not found under {NE_DIR}.")

    nodes = pd.read_csv(M3 / "deep_m3_shap_period_node.csv")
    panels = [
        ("full", "A  Full sample"),
        ("year_2024", "B  2024"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    for ax, (pid, title) in zip(axes, panels):
        sub = nodes[nodes["period_id"] == pid]
        shares = dict(zip(sub["node_id"], sub["share_shipping"].astype(float)))
        _basemap(ax, MAP_EXTENT)
        _draw_share_nodes(ax, shares)
        ax.set_title(title, loc="left", fontsize=10)
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

    # Shared legends to the right of panel B.
    ax = axes[1]
    # Area legend (area ∝ share).
    legend_shares = [0.05, 0.10, 0.15]
    x0, y0 = 152.5, 8.0
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[2])  # keep map crop; legend in fig
    # Draw area legend in figure coordinates via a dedicated axis.
    leg = fig.add_axes([0.88, 0.18, 0.11, 0.62])
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    leg.axis("off")
    leg.text(0.05, 0.98, "Share of\nshipping |SHAP|", fontsize=7.5,
             va="top", color="#333333")
    y = 0.72
    for sh in legend_shares:
        leg.scatter([0.22], [y], s=share_size(sh), marker="o",
                    color=C["choke"], edgecolor="white", linewidth=0.6)
        leg.text(0.48, y, f"{int(sh * 100)}%", va="center", fontsize=7.5)
        y -= 0.16
    y = 0.22
    for marker, color, lab in (
        ("D", C["choke"], "Chokepoint"),
        ("^", C["aoi"], "AOI (named)"),
        ("o", "#B8B8B8", "Other node"),
    ):
        leg.scatter([0.22], [y], s=42, marker=marker, color=color,
                    edgecolor="white", linewidth=0.6)
        leg.text(0.42, y, lab, va="center", fontsize=7.2)
        y -= 0.09

    fig.suptitle("Shipping-internal node SHAP shares on the Eurasia–Africa "
                 "oil corridor\nArea \u221d share (locked at 18%). Shares are "
                 "within shipping, not total attribution. "
                 "Houston and Panama lie outside this frame.",
                 x=0.02, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.86, top=0.84, bottom=0.08,
                        wspace=0.08)
    save(fig, "fig_4_4_node_shap_map")


# --------------------------------------------------------------------------
# 4.5  Node × time SHAP-share heatmap
# --------------------------------------------------------------------------
def fig45_node_heatmap() -> None:
    weekly = pd.read_csv(M3 / "deep_m3_shap_weekly_node.csv", parse_dates=["date"])
    tot = weekly.groupby("date")["shap_abs"].transform("sum")
    weekly["share"] = weekly["shap_abs"] / tot
    pv = weekly.pivot(index="date", columns="node_id", values="share").sort_index()
    roll = pv.rolling(HEAT_ROLL, min_periods=max(2, HEAT_ROLL // 2)).mean()

    rows = HEAT_CHOKE + [None] + HEAT_AOI
    mat = []
    ytick, ylab, ybold = [], [], []
    yi = 0
    for nid in rows:
        if nid is None:
            mat.append(np.full(len(roll), np.nan))
            ytick.append(yi)
            ylab.append("")
            ybold.append(False)
            yi += 1
            continue
        mat.append(roll[nid].to_numpy(float))
        ytick.append(yi)
        ylab.append(HEAT_LABEL[nid])
        ybold.append(nid in NAMED_NODES)
        yi += 1
    Z = np.vstack(mat)

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    x0 = mdates.date2num(roll.index[0])
    x1 = mdates.date2num(roll.index[-1])
    # imshow y: row 0 at the top
    im = ax.imshow(Z, aspect="auto", interpolation="nearest",
                   cmap=SHARE_CMAP, vmin=0.0, vmax=SHARE_MAX,
                   extent=[x0, x1, len(rows) - 0.5, -0.5],
                   zorder=2)
    ax.set_yticks(ytick)
    ax.set_yticklabels(ylab, fontsize=8)
    for tick, bold in zip(ax.get_yticklabels(), ybold):
        if bold:
            tick.set_fontweight("bold")
            tick.set_color("#8A3D16" if tick.get_text() in
                           {HEAT_LABEL[n] for n in HEAT_CHOKE} else "#1F3E5F")

    ax.xaxis_date()
    year_ticks = [pd.Timestamp(f"{y}-01-01") for y in range(2021, 2027)]
    ax.set_xticks([mdates.date2num(t) for t in year_ticks])
    ax.set_xticklabels([str(t.year) for t in year_ticks], fontsize=8)
    ax.set_xlim(x0, x1)

    top = ax.secondary_xaxis("top")
    top.set_xticks([mdates.date2num(pd.Timestamp(c)) for _, c, _, _ in EVENTS])
    top.set_xticklabels([lab for _, _, lab, _ in EVENTS], fontsize=6.8)
    for tick, (_, _, _, col) in zip(top.get_xticklabels(), EVENTS):
        tick.set_color(col)
        tick.set_fontweight("bold")
    top.tick_params(length=3.5, pad=2)

    for _, centre, lab, col in EVENTS:
        lo, hi = event_bounds(centre)
        ax.axvline(mdates.date2num(pd.Timestamp(centre)), color=col,
                   linewidth=1.0, linestyle="-", zorder=3, alpha=0.9)
        ax.axvspan(mdates.date2num(lo), mdates.date2num(hi),
                   color=col, alpha=0.07, zorder=1, linewidth=0)

    # Separator label
    gap_i = len(HEAT_CHOKE)
    ax.axhline(gap_i, color="white", linewidth=2.2, zorder=3)
    ax.text(x0 - (x1 - x0) * 0.01, len(HEAT_CHOKE) / 2 - 0.5, "Chokepoints",
            rotation=90, va="center", ha="right", fontsize=7.5,
            color="#8A3D16", clip_on=False)
    ax.text(x0 - (x1 - x0) * 0.01, len(HEAT_CHOKE) + 1 + len(HEAT_AOI) / 2,
            "AOI", rotation=90, va="center", ha="right", fontsize=7.5,
            color="#1F3E5F", clip_on=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Share of shipping |SHAP|", fontsize=8)
    cbar.set_ticks([0.00, 0.05, 0.10, 0.15, 0.18])
    cbar.set_ticklabels(["0%", "5%", "10%", "15%", "18%"])

    ax.set_title(f"Weekly shipping-internal node SHAP shares "
                 f"({HEAT_ROLL}-week rolling mean)\n"
                 "Same quantity and 18% colour lock as Figure 4.4. "
                 "No node exceeds 12% of shipping attribution in the Red Sea window.",
                 loc="left")
    ax.set_xlabel("Forecast origin")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#AAAAAA")
        sp.set_linewidth(0.7)
    fig.tight_layout()
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
    save(fig, "fig_4_6_seed_robustness")


FIGURES = {
    "4.1": fig41_price_returns,
    "4.2": fig42_slope,
    "4.3": fig43_shap_modality,
    "4.4": fig44_node_map,
    "4.5": fig45_node_heatmap,
    "4.6": fig46_seed_robustness,
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
