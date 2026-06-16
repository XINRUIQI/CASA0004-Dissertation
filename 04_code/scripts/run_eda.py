"""
Exploratory Data Analysis for the Brent oil price forecasting project.

Generates key EDA visualisations and saves to 05_outputs/figures/.

Usage:
    python run_eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"
FIG_DIR = PROJECT_ROOT / "05_outputs" / "figures"

CRISIS_EVENTS = {
    "2008 GFC": ("2008-09-01", "2009-03-31"),
    "2014 Oil Crash": ("2014-06-01", "2015-01-31"),
    "2020 COVID": ("2020-02-01", "2020-06-30"),
    "2022 Russia-Ukraine": ("2022-02-01", "2022-06-30"),
    "2023-24 Red Sea": ("2023-11-01", "2024-06-30"),
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
})


def load_data() -> pd.DataFrame:
    path = PROC_DIR / "weekly_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(PROC_DIR / "weekly_features.csv", index_col=0, parse_dates=True)


def plot_brent_price_timeline(df: pd.DataFrame):
    """Fig 1: Brent price with crisis periods shaded."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df.index, df["brent_price"], linewidth=0.8, color="#1f77b4")
    ax.set_ylabel("Brent Price (USD/bbl)")
    ax.set_title("Brent Crude Oil Weekly Price (2006–2025)")

    colors = plt.cm.Set2(np.linspace(0, 1, len(CRISIS_EVENTS)))
    for i, (name, (start, end)) in enumerate(CRISIS_EVENTS.items()):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.2, color=colors[i], label=name)

    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(FIG_DIR / "01_brent_price_timeline.png")
    plt.close(fig)
    print("  Saved: 01_brent_price_timeline.png")


def plot_return_distribution(df: pd.DataFrame):
    """Fig 2: Distribution of weekly Brent returns."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    returns = (df["brent_price"].pct_change() * 100).dropna()

    axes[0].hist(returns, bins=80, color="#2ca02c", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Weekly Return (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Distribution (mean={returns.mean():.2f}%, std={returns.std():.2f}%)")

    vol12 = np.log(df["brent_price"] / df["brent_price"].shift(1)).rolling(12).std()
    axes[1].plot(df.index, vol12, linewidth=0.7, color="#d62728")
    axes[1].set_ylabel("12-week Rolling Volatility")
    axes[1].set_title("Brent Realised Volatility")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_brent_return_distribution.png")
    plt.close(fig)
    print("  Saved: 02_brent_return_distribution.png")


def plot_eia_fundamentals(df: pd.DataFrame):
    """Fig 3: M1 supply / demand mechanism variables (kept set)."""
    cols = {
        "crude_stocks_change": "Crude Stocks Change (k bbl)",
        "global_econ_activity": "Global Economic Activity (Kilian REA)",
        "nonoil_industrial_commodity": "Non-oil Industrial Commodity Index",
        "futures_spread": "Brent Futures-Spot Spread",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (col, title) in zip(axes.flat, cols.items()):
        if col in df.columns:
            ax.plot(df.index, df[col], linewidth=0.7)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)

    fig.suptitle("M1 Supply / Demand Mechanism Variables (2006–2025)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_eia_fundamentals.png")
    plt.close(fig)
    print("  Saved: 03_eia_fundamentals.png")


def plot_macro_indicators(df: pd.DataFrame):
    """Fig 4: M1 macro / risk indicators (kept set)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    pairs = [
        ("ovx", "OVX (oil-specific uncertainty)"),
        ("gpr", "Geopolitical Risk (GPR)"),
        ("dgs10_change", "10Y Treasury Yield Change (ΔDGS10)"),
        ("gold_return", "Gold Return"),
        ("commodity_fx", "Commodity FX (CAD/AUD)"),
        ("global_econ_activity", "Global Economic Activity"),
    ]

    for ax, (col, title) in zip(axes.flat, pairs):
        if col in df.columns:
            ax.plot(df.index, df[col], linewidth=0.7)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)

    fig.suptitle("M1 Macro / Risk Indicators (2006–2025)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_macro_indicators.png")
    plt.close(fig)
    print("  Saved: 04_macro_indicators.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Fig 5: Cross-modality correlation heatmap (representative features vs target)."""
    rep_cols = [
        "brent_price", "crude_stocks_change", "futures_spread",
        "ovx", "gpr", "gold_return", "commodity_fx", "dgs10_change",
        "global_econ_activity", "nonoil_industrial_commodity",
        "gdelt_oil_geo_event_count", "gdelt_oil_geo_avg_tone",
        "gdelt_chokepoint_event_count", "gdelt_combined_event_count",
        "ntl_anomaly_ras_tanura", "ntl_anomaly_us_gulf",
        "s2_cloud_fraction_fujairah",
        "gfw_hormuz_total_hours", "gfw_suez_total_hours",
        "pw_hormuz_n_tanker", "pw_suez_n_tanker",
        "target_brent_price_next_1w",
    ]
    available = [c for c in rep_cols if c in df.columns]
    corr = df[available].corr()

    short_names = {
        "crude_stocks_change": "crude_stk_chg",
        "nonoil_industrial_commodity": "nonoil_ind",
        "global_econ_activity": "global_econ",
        "gdelt_oil_geo_event_count": "gdelt_geo_cnt",
        "gdelt_oil_geo_avg_tone": "gdelt_geo_tone",
        "gdelt_chokepoint_event_count": "gdelt_choke_cnt",
        "gdelt_combined_event_count": "gdelt_combined",
        "ntl_anomaly_ras_tanura": "NTLanom_RasTanura",
        "ntl_anomaly_us_gulf": "NTLanom_USGulf",
        "s2_cloud_fraction_fujairah": "S2cloud_Fujairah",
        "gfw_hormuz_total_hours": "gfw_hormuz_hrs",
        "gfw_suez_total_hours": "gfw_suez_hrs",
        "pw_hormuz_n_tanker": "pw_hormuz_tanker",
        "pw_suez_n_tanker": "pw_suez_tanker",
        "target_brent_price_next_1w": "TARGET",
    }
    corr = corr.rename(index=short_names, columns=short_names)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 7})
    ax.set_title("Cross-Modality Feature Correlation Matrix (M1–M4 vs Target)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_correlation_heatmap.png")
    plt.close(fig)
    print("  Saved: 05_correlation_heatmap.png")


def plot_remote_sensing_timeseries(df: pd.DataFrame):
    """Fig 6: Clean M2 NTL anomaly vs S2 cloud fraction for selected AOIs."""
    aois = ["rotterdam", "ras_tanura", "us_gulf", "ningbo"]
    names = {"rotterdam": "Rotterdam", "ras_tanura": "Ras Tanura",
             "us_gulf": "Houston / US Gulf", "ningbo": "Ningbo-Zhoushan"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, aoi in zip(axes.flat, aois):
        anom_col = f"ntl_anomaly_{aoi}"
        cloud_col = f"s2_cloud_fraction_{aoi}"
        if anom_col in df.columns:
            ax.plot(df.index, df[anom_col], linewidth=0.6, label="NTL anomaly (z)", alpha=0.8)
        if cloud_col in df.columns:
            ax.plot(df.index, df[cloud_col], linewidth=0.6, label="S2 cloud fraction", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.set_title(f"{names.get(aoi, aoi)}", fontsize=10)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)

    fig.suptitle("Clean M2 Remote Sensing (NTL anomaly / S2 cloud) at Oil AOIs", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_remote_sensing_timeseries.png")
    plt.close(fig)
    print("  Saved: 06_remote_sensing_timeseries.png")


def plot_nightlight_timeseries(df: pd.DataFrame):
    """Fig 7: VIIRS nightlight anomaly (z-score) by AOI."""
    aois = ["rotterdam", "ras_tanura", "jurong", "us_gulf", "ningbo", "jamnagar"]
    names = {"rotterdam": "Rotterdam", "ras_tanura": "Ras Tanura", "jurong": "Singapore",
             "us_gulf": "Houston / US Gulf", "ningbo": "Ningbo", "jamnagar": "Jamnagar"}

    fig, ax = plt.subplots(figsize=(14, 5))
    for aoi in aois:
        col = f"ntl_anomaly_{aoi}"
        if col in df.columns:
            ax.plot(df.index, df[col], linewidth=0.7, label=names.get(aoi, aoi), alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax.set_ylabel("NTL Anomaly (site z-score)")
    ax.set_title("VIIRS Nightlight Anomaly at Oil Infrastructure Sites")
    ax.legend(fontsize=8, ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)

    fig.savefig(FIG_DIR / "07_nightlight_timeseries.png")
    plt.close(fig)
    print("  Saved: 07_nightlight_timeseries.png")


def _shade_crises(ax):
    """Add crisis-period shading to an axis."""
    colors = plt.cm.Set2(np.linspace(0, 1, len(CRISIS_EVENTS)))
    for i, (name, (start, end)) in enumerate(CRISIS_EVENTS.items()):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color=colors[i])


def plot_gdelt_disruption(df: pd.DataFrame):
    """Fig 11: GDELT oil-disruption event count and tone with crisis shading."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    if "gdelt_oil_geo_event_count" in df.columns:
        ax = axes[0]
        ax.fill_between(df.index, df["gdelt_oil_geo_event_count"],
                        alpha=0.4, color="#1f77b4", linewidth=0)
        if "gdelt_oil_geo_event_count_4w_ma" in df.columns:
            ax.plot(df.index, df["gdelt_oil_geo_event_count_4w_ma"],
                    linewidth=1.2, color="#d62728", label="4-week MA")
            ax.legend(fontsize=8)
        ax.set_ylabel("Weekly Event Count")
        ax.set_title("GDELT Oil-Disruption Event Intensity")
        _shade_crises(ax)
        ax.grid(axis="y", alpha=0.3)

    if "gdelt_oil_geo_avg_tone" in df.columns:
        ax = axes[1]
        ax.plot(df.index, df["gdelt_oil_geo_avg_tone"],
                linewidth=0.5, color="#9467bd", alpha=0.5)
        if "gdelt_oil_geo_avg_tone_4w_ma" in df.columns:
            ax.plot(df.index, df["gdelt_oil_geo_avg_tone_4w_ma"],
                    linewidth=1.2, color="#9467bd", label="Avg Tone (4w MA)")
            ax.legend(fontsize=8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_ylabel("Average Media Tone")
        ax.set_title("GDELT Oil-Disruption Media Tone")
        _shade_crises(ax)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_gdelt_disruption_timeseries.png")
    plt.close(fig)
    print("  Saved: 11_gdelt_disruption_timeseries.png")


def plot_gdelt_transport(df: pd.DataFrame):
    """Fig 12: GDELT transport disruption events and chokepoint events."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    if "gdelt_transport_disruption_event_count" in df.columns:
        ax = axes[0]
        ax.fill_between(df.index, df["gdelt_transport_disruption_event_count"],
                        alpha=0.4, color="#ff7f0e", linewidth=0)
        if "gdelt_transport_event_count_4w_ma" in df.columns:
            ax.plot(df.index, df["gdelt_transport_event_count_4w_ma"],
                    linewidth=1.2, color="#d62728", label="4-week MA")
            ax.legend(fontsize=8)
        ax.set_ylabel("Weekly Event Count")
        ax.set_title("GDELT Transport Disruption Events")
        _shade_crises(ax)
        ax.grid(axis="y", alpha=0.3)

    if "gdelt_chokepoint_event_count" in df.columns:
        ax = axes[1]
        ax.fill_between(df.index, df["gdelt_chokepoint_event_count"],
                        alpha=0.4, color="#2ca02c", linewidth=0)
        ax.set_ylabel("Weekly Event Count")
        ax.set_title("GDELT Chokepoint-Related Events")
        _shade_crises(ax)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_gdelt_transport_timeseries.png")
    plt.close(fig)
    print("  Saved: 12_gdelt_transport_timeseries.png")


def plot_shipping_gfw(df: pd.DataFrame):
    """Fig 13: GFW vessel presence hours at 6 chokepoints."""
    chokepoints = {
        "hormuz": "Strait of Hormuz",
        "suez": "Suez Canal",
        "malacca": "Strait of Malacca",
        "mandeb": "Bab el-Mandeb",
        "panama": "Panama Canal",
        "cape": "Cape of Good Hope",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = plt.cm.tab10(np.linspace(0, 0.6, len(chokepoints)))
    for (key, name), color in zip(chokepoints.items(), palette):
        col = f"gfw_{key}_total_hours"
        if col in df.columns:
            series = df[col].dropna()
            ax.plot(series.index, series, linewidth=0.8, label=name, color=color, alpha=0.85)

    ax.set_ylabel("Monthly Vessel Presence Hours")
    ax.set_title("GFW Vessel Presence at Oil-Critical Chokepoints (2012–2018)")
    ax.legend(fontsize=8, ncol=2)
    _shade_crises(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_shipping_gfw_vessel_hours.png")
    plt.close(fig)
    print("  Saved: 13_shipping_gfw_vessel_hours.png")


def plot_shipping_portwatch(df: pd.DataFrame):
    """Fig 14: PortWatch weekly tanker transits at 6 chokepoints."""
    chokepoints = {
        "hormuz": "Strait of Hormuz",
        "suez": "Suez Canal",
        "malacca": "Strait of Malacca",
        "mandeb": "Bab el-Mandeb",
        "panama": "Panama Canal",
        "cape": "Cape of Good Hope",
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, (key, name) in zip(axes.flat, chokepoints.items()):
        tanker_col = f"pw_{key}_n_tanker"
        total_col = f"pw_{key}_n_total"
        if tanker_col in df.columns:
            series = df[tanker_col].dropna()
            ax.plot(series.index, series, linewidth=0.7, color="#d62728", label="Tanker")
        if total_col in df.columns:
            series = df[total_col].dropna()
            ax.plot(series.index, series, linewidth=0.7, color="#1f77b4", alpha=0.5, label="Total")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        _shade_crises(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("IMF PortWatch Weekly Vessel Transits at Chokepoints (2019–2025)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_shipping_portwatch_transits.png")
    plt.close(fig)
    print("  Saved: 14_shipping_portwatch_transits.png")


def plot_modality_coverage(df: pd.DataFrame):
    """Fig 8: Data availability heatmap by modality and year."""
    avail_cols = [c for c in df.columns if c.startswith("avail_")]
    if not avail_cols:
        return

    yearly = df[avail_cols].resample("YE").mean() * 100
    yearly.index = yearly.index.year

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(yearly.T, annot=True, fmt=".0f", cmap="YlGn",
                vmin=0, vmax=100, ax=ax, linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title("Modality Data Availability by Year (%)")
    ax.set_ylabel("")

    fig.savefig(FIG_DIR / "08_modality_coverage.png")
    plt.close(fig)
    print("  Saved: 08_modality_coverage.png")


def print_summary_stats(df: pd.DataFrame):
    """Print key summary statistics."""
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total weeks: {len(df)}")
    print(f"Period: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nBrent price:")
    print(f"  Mean:   ${df['brent_price'].mean():.2f}")
    print(f"  Median: ${df['brent_price'].median():.2f}")
    print(f"  Min:    ${df['brent_price'].min():.2f} ({df['brent_price'].idxmin().date()})")
    print(f"  Max:    ${df['brent_price'].max():.2f} ({df['brent_price'].idxmax().date()})")
    weekly_ret = df["brent_price"].pct_change() * 100
    print(f"\nWeekly return:")
    print(f"  Mean:  {weekly_ret.mean():.3f}%")
    print(f"  Std:   {weekly_ret.std():.3f}%")
    print(f"  Skew:  {weekly_ret.skew():.3f}")
    print(f"  Kurt:  {weekly_ret.kurtosis():.3f}")
    print(f"\nNext-week direction balance:")
    direction = df["target_brent_direction_next_1w"].value_counts(normalize=True) * 100
    print(f"  Up:   {direction.get(1, 0):.1f}%")
    print(f"  Flat: {direction.get(0, 0):.1f}%")
    print(f"  Down: {direction.get(-1, 0):.1f}%")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature matrix...\n")
    df = load_data()
    print(f"Shape: {df.shape}\n")

    print("Generating EDA figures...\n")

    plot_brent_price_timeline(df)
    plot_return_distribution(df)
    plot_eia_fundamentals(df)
    plot_macro_indicators(df)
    plot_correlation_heatmap(df)
    plot_remote_sensing_timeseries(df)
    plot_nightlight_timeseries(df)
    plot_modality_coverage(df)

    plot_gdelt_disruption(df)
    plot_gdelt_transport(df)
    plot_shipping_gfw(df)
    plot_shipping_portwatch(df)

    print_summary_stats(df)

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
