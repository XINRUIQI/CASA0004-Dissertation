"""
M3 (SHIPPING) — aggregate maritime activity to a leakage-safe weekly table.

Reads (raw layer, provider-organised):
  raw/03_shipping/
    IMF Portwatch/portwatch_chokepoints_daily.csv   (2019+, daily, 6 oil chokepoints)
    IMF Portwatch/portwatch_ports_daily.csv         (2019+, daily, export/import tanker volume)
    GFW/gfw_chokepoint_vessel_presence_monthly.csv  (2012+, monthly, AIS vessel presence)

Writes:
  processed/M3/outputs/m3_weekly_features.csv
  — One row per Friday-ending week (W-FRI), columns = source × chokepoint × metric.

Two design fixes vs. the previous version (see research_diary_phase3 P0 items):

  1) UNION-INDEX ALIGNMENT (fixes the 727 -> 362 sample drop).
     GFW is monthly 2012-2025 (~727 weeks once expanded); PortWatch is daily
     2019-2026 (~362 weeks). The old merge clipped to the PortWatch overlap and
     silently dropped every GFW-only week before 2019. Here EACH source is first
     reduced to its native frequency, then ALL sources are reindexed onto a single
     UNION W-FRI index (min start .. max end across sources). Monthly GFW is
     forward-filled ON the union index (month-end -> following Fridays). Nothing
     is dropped: GFW-only early weeks survive, PortWatch-only late weeks survive.

  2) PUBLICATION-TIMESTAMP LAG (no look-ahead; research plan 4.4 / 6.4).
     A value may only be used AFTER its real-world release:
       - GFW monthly presence: month-end aligned, then a conservative release lag
         (GFW_LAG_WEEKS). A full month is only complete at month end and the
         4Wings aggregate is published with a delay.
       - PortWatch daily -> weekly sum, then a short release lag (PW_LAG_WEEKS):
         the week-ending-Friday aggregate is not yet downloadable at that Friday's
         close. wow_pct / 4w_ma / pct_change are computed on the native series
         BEFORE the lag shift, so the whole block is simply shifted forward.
     Lags are module-level constants (also CLI-overridable) so the conservative
     monthly lag can be re-checked with the supervisor.

EMODnet monthly vessel-density rasters are NOT aggregated here (they need
rasterio + chokepoint/AOI polygon zonal stats); they are a later cross-validation
add-on. See external_sources.md (M3) for the plan.

Usage:
    python aggregate_shipping_to_weekly.py
    python aggregate_shipping_to_weekly.py --gfw-lag 4 --pw-lag 1
    python aggregate_shipping_to_weekly.py --no-lag        # diagnostic, leaky
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths  (this file lives at processed/M3/py/aggregate_shipping_to_weekly.py)
# ----------------------------------------------------------------------------
PY_DIR = Path(__file__).resolve().parent          # processed/M3/py
M3_DIR = PY_DIR.parent                             # processed/M3
DATA_DIR = M3_DIR.parents[1]                       # 03_data
RAW_SHIP = DATA_DIR / "raw" / "03_shipping"
PW_DIR = RAW_SHIP / "IMF Portwatch"
GFW_DIR = RAW_SHIP / "GFW"
OUT_DIR = M3_DIR / "outputs"
OUT_PATH = OUT_DIR / "m3_weekly_features.csv"

# ----------------------------------------------------------------------------
# Study window (M3 keeps full history from the GFW start so no early week is
# dropped; the modelling stage clips to the standardised 2019-2026 window).
# ----------------------------------------------------------------------------
STUDY_START = "2012-01-01"
STUDY_END = "2026-12-31"

# ----------------------------------------------------------------------------
# Conservative publication lags (weeks). A value is only usable AFTER release.
# These are the M3 "monthly conservative lag" items the research plan asks to
# re-check; override with --gfw-lag / --pw-lag, or disable with --no-lag.
# ----------------------------------------------------------------------------
GFW_LAG_WEEKS = 4     # GFW monthly presence: ~1 month publication delay
PW_LAG_WEEKS = 1      # PortWatch weekly aggregate not yet published at week close

# Six oil-relevant chokepoints (raw name -> short code).
CHOKEPOINT_SHORT = {
    "Strait of Hormuz": "hormuz",
    "Suez Canal": "suez",
    "Malacca Strait": "malacca",
    "Panama Canal": "panama",
    "Bab el-Mandeb Strait": "mandeb",
    "Bab el-Mandeb": "mandeb",
    "Cape of Good Hope": "cape",
}

# Port basket roles fallback (used only if the CSV lacks a `role` column).
# import_tanker / export_tanker are directional tanker VOLUME estimates (tonnes).
PORT_ROLE_FALLBACK = {
    "ras_tanura": "export", "juaymah": "export", "yanbu": "export",
    "ras_laffan": "export", "primorsk": "export", "novorossiysk": "export",
    "corpus_christi": "export", "sidi_kerir": "export", "bonny": "export",
    "rotterdam": "import", "singapore": "import", "ningbo": "import",
    "chiba": "import", "ulsan": "import",
}


# ============================================================================
# PortWatch chokepoints: daily -> weekly (W-FRI sum), native PW index, PRE-lag
# ============================================================================
def load_portwatch_chokepoints_weekly() -> pd.DataFrame:
    path = PW_DIR / "portwatch_chokepoints_daily.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"])
    df["cp"] = df["portname"].map(CHOKEPOINT_SHORT)
    df = df.dropna(subset=["cp"])

    keep = ["n_tanker", "n_total", "capacity_tanker", "capacity"]

    frames = []
    for cp, grp in df.groupby("cp"):
        ts = grp.set_index("date")[keep].sort_index()
        # daily -> weekly sum; a fully-empty week -> NaN (min_count=1), not 0.
        weekly = ts.resample("W-FRI").sum(min_count=1)

        weekly["tanker_share"] = _safe_div(weekly["n_tanker"], weekly["n_total"])
        weekly["tanker_cap_share"] = _safe_div(weekly["capacity_tanker"], weekly["capacity"])
        # Average tanker size (P070: capacity weighted by loading > tanker count).
        weekly["avg_tanker_size"] = _safe_div(weekly["capacity_tanker"], weekly["n_tanker"])
        weekly["n_tanker_wow_pct"] = weekly["n_tanker"].pct_change(fill_method=None) * 100
        weekly["capacity_tanker_4w_ma"] = weekly["capacity_tanker"].rolling(4, min_periods=1).mean()

        weekly.columns = [f"pw_{cp}_{c}" for c in weekly.columns]
        frames.append(weekly)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1).sort_index()

    # Cross-chokepoint aggregates (computed on the native weekly series).
    n_tanker_cols = [c for c in result.columns if c.endswith("_n_tanker")]
    n_total_cols = [c for c in result.columns if c.endswith("_n_total")]
    agg_tanker = result[n_tanker_cols].sum(axis=1, min_count=1)
    agg_total = result[n_total_cols].sum(axis=1, min_count=1)
    result["pw_all_n_tanker_sum"] = agg_tanker
    result["pw_all_n_total_sum"] = agg_total
    result["pw_all_tanker_share"] = _safe_div(agg_tanker, agg_total)

    print(f"  PortWatch chokepoints (pre-lag): {result.shape}, "
          f"{result.index.min().date()} ~ {result.index.max().date()}")
    return result


# ============================================================================
# PortWatch ports: export-vs-import tanker-volume asymmetry, PRE-lag
# ============================================================================
def load_portwatch_ports_weekly() -> pd.DataFrame:
    path = PW_DIR / "portwatch_ports_daily.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"])
    if "short" not in df.columns:
        print(f"  [skip] {path.name}: no 'short' column")
        return pd.DataFrame()

    # Prefer the CSV's own role column; fall back to the hard-coded basket.
    if "role" in df.columns and df["role"].notna().any():
        df["role"] = df["role"].where(df["role"].notna(), df["short"].map(PORT_ROLE_FALLBACK))
    else:
        df["role"] = df["short"].map(PORT_ROLE_FALLBACK)
    df = df.dropna(subset=["role"])

    exp = df[df["role"] == "export"].set_index("date")["export_tanker"]
    imp = df[df["role"] == "import"].set_index("date")["import_tanker"]
    exp_w = exp.resample("W-FRI").sum(min_count=1)
    imp_w = imp.resample("W-FRI").sum(min_count=1)

    out = pd.DataFrame({
        "pw_exp_hubs_export_vol": exp_w,
        "pw_imp_hubs_import_vol": imp_w,
    }).sort_index()
    out["pw_tanker_exp_imp_net"] = out["pw_exp_hubs_export_vol"] - out["pw_imp_hubs_import_vol"]
    denom = (out["pw_exp_hubs_export_vol"] + out["pw_imp_hubs_import_vol"]).replace(0, np.nan)
    out["pw_tanker_exp_imp_asym"] = out["pw_tanker_exp_imp_net"] / denom
    out["pw_tanker_exp_imp_log_ratio"] = np.log(
        (out["pw_exp_hubs_export_vol"] + 1.0) / (out["pw_imp_hubs_import_vol"] + 1.0)
    )
    out["pw_tanker_exp_imp_asym_4w_ma"] = out["pw_tanker_exp_imp_asym"].rolling(4, min_periods=1).mean()
    out["pw_exp_hubs_export_vol_wow_pct"] = out["pw_exp_hubs_export_vol"].pct_change(fill_method=None) * 100

    print(f"  PortWatch ports/directional (pre-lag): {out.shape}, "
          f"{out.index.min().date()} ~ {out.index.max().date()}")
    return out


# ============================================================================
# GFW: monthly -> month-end indexed derived block, PRE-lag, PRE-ffill
# ============================================================================
def load_gfw_monthly() -> pd.DataFrame:
    path = GFW_DIR / "gfw_chokepoint_vessel_presence_monthly.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df["cp"] = df["chokepoint"].map(CHOKEPOINT_SHORT)
    df = df.dropna(subset=["cp"])

    keep = ["total_hours", "total_vessels", "cargo_hours", "bunker_hours", "other_hours"]

    frames = []
    for cp, grp in df.groupby("cp"):
        ts = grp.set_index("date")[keep].sort_index()
        # Monthly-native derived metrics.
        ts["nontanker_hours"] = ts["cargo_hours"] + ts["bunker_hours"]
        ts["other_share"] = _safe_div(ts["other_hours"], ts["total_hours"])
        ts["total_hours_mom_pct"] = ts["total_hours"].pct_change(fill_method=None) * 100
        # Congestion / dwell proxy (P016): mean presence hours per unique vessel.
        ts["dwell_hours_per_vessel"] = _safe_div(ts["total_hours"], ts["total_vessels"])

        ts.columns = [f"gfw_{cp}_{c}" for c in ts.columns]
        frames.append(ts)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1).sort_index()
    hour_cols = [c for c in result.columns if c.endswith("_total_hours")]
    result["gfw_all_total_hours_sum"] = result[hour_cols].sum(axis=1, min_count=1)

    # Align month label to month end (a full month is only complete at month end).
    result.index = result.index + pd.offsets.MonthEnd(0)
    result = result[~result.index.duplicated(keep="last")]

    print(f"  GFW monthly (pre-lag, month-end): {result.shape}, "
          f"{result.index.min().date()} ~ {result.index.max().date()}")
    return result


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def build_union_index(*indexes: pd.Index) -> pd.DatetimeIndex:
    """Single W-FRI index spanning min(start)..max(end) across all sources."""
    bounds = [(ix.min(), ix.max()) for ix in indexes if ix is not None and len(ix)]
    if not bounds:
        return pd.DatetimeIndex([], name="week_ending_friday")
    start = min(b[0] for b in bounds)
    end = max(b[1] for b in bounds)
    idx = pd.date_range(start=start, end=end, freq="W-FRI")
    idx.name = "week_ending_friday"
    return idx


def align_weekly(df: pd.DataFrame, union: pd.DatetimeIndex, lag_weeks: int) -> pd.DataFrame:
    """Weekly source -> reindex onto union (no fill), then shift by release lag."""
    if df.empty:
        return df
    out = df.reindex(union)
    return out.shift(lag_weeks) if lag_weeks else out


def align_monthly(df: pd.DataFrame, union: pd.DatetimeIndex, lag_weeks: int) -> pd.DataFrame:
    """Monthly source -> ffill onto union (month-end -> following Fridays),
    then shift by release lag. This is the 'each ffill + union index' fix."""
    if df.empty:
        return df
    out = df.reindex(union, method="ffill")
    return out.shift(lag_weeks) if lag_weeks else out


def _check_lag_direction(name: str, pre: pd.DataFrame, post: pd.DataFrame,
                         lag: int) -> str | None:
    """Verify the lag moves the FIRST usable week forward by exactly `lag`
    weeks (a publication lag, never a look-ahead). Tail NaNs are not a valid
    check here: ffilled monthly data legitimately keeps its last released
    value, and sources end on different weeks."""
    if pre.empty:
        return None
    if lag == 0:
        return f"  [ -- ] {name}: no lag (diagnostic mode)"
    fv_pre = pre.dropna(how="all").index.min()
    post_valid = post.dropna(how="all")
    if post_valid.empty:
        return f"  [WARN] {name}: empty after shift"
    fv_post = post_valid.index.min()
    weeks = round((fv_post - fv_pre).days / 7)
    flag = "OK" if weeks == lag else "WARN"
    return (f"  [{flag}] {name}: first usable week {fv_pre.date()} -> "
            f"{fv_post.date()} (+{weeks}w, expected +{lag}w)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the M3 weekly shipping feature table.")
    ap.add_argument("--gfw-lag", type=int, default=GFW_LAG_WEEKS,
                    help=f"GFW monthly publication lag in weeks (default {GFW_LAG_WEEKS})")
    ap.add_argument("--pw-lag", type=int, default=PW_LAG_WEEKS,
                    help=f"PortWatch weekly publication lag in weeks (default {PW_LAG_WEEKS})")
    ap.add_argument("--no-lag", action="store_true",
                    help="disable all publication lags (DIAGNOSTIC ONLY; leaky)")
    args = ap.parse_args()
    gfw_lag = 0 if args.no_lag else args.gfw_lag
    pw_lag = 0 if args.no_lag else args.pw_lag

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"M3 shipping aggregation\nRaw dir: {RAW_SHIP}")
    print(f"Lags: GFW={gfw_lag}w  PortWatch={pw_lag}w"
          f"{'  (NO-LAG diagnostic)' if args.no_lag else ''}\n")

    # 1) Load each source at its native cadence (pre-lag, pre-union).
    pw_choke = load_portwatch_chokepoints_weekly()
    pw_ports = load_portwatch_ports_weekly()
    gfw = load_gfw_monthly()

    if pw_choke.empty and pw_ports.empty and gfw.empty:
        print("\nNo shipping data found. Nothing to output.")
        return

    # 2) Build ONE union W-FRI index across all sources (the 727->362 fix).
    union = build_union_index(
        pw_choke.index if not pw_choke.empty else None,
        pw_ports.index if not pw_ports.empty else None,
        gfw.index if not gfw.empty else None,
    )
    print(f"\n  Union W-FRI index: {len(union)} weeks, "
          f"{union.min().date()} ~ {union.max().date()}")

    # 3) Align each source onto the union (monthly ffill, weekly reindex), then
    #    shift by the release lag. Keep the pre-shift version to verify that the
    #    shift moves the first usable week FORWARD (a lag, never a look-ahead).
    gfw_pre = align_monthly(gfw, union, 0)
    pw_choke_pre = align_weekly(pw_choke, union, 0)
    pw_ports_pre = align_weekly(pw_ports, union, 0)
    gfw_a = gfw_pre.shift(gfw_lag) if (gfw_lag and not gfw_pre.empty) else gfw_pre
    pw_choke_a = pw_choke_pre.shift(pw_lag) if (pw_lag and not pw_choke_pre.empty) else pw_choke_pre
    pw_ports_a = pw_ports_pre.shift(pw_lag) if (pw_lag and not pw_ports_pre.empty) else pw_ports_pre

    lag_checks = [
        _check_lag_direction("GFW", gfw_pre, gfw_a, gfw_lag),
        _check_lag_direction("PortWatch chokepoints", pw_choke_pre, pw_choke_a, pw_lag),
        _check_lag_direction("PortWatch ports", pw_ports_pre, pw_ports_a, pw_lag),
    ]

    # 4) Concatenate column-wise on the shared union index.
    parts = [p for p in (gfw_a, pw_choke_a, pw_ports_a) if not p.empty]
    weekly = pd.concat(parts, axis=1)
    weekly = weekly.reindex(union)

    # 5) Modality-availability masks (post-lag = what is actually visible).
    gfw_cols = [c for c in weekly.columns if c.startswith("gfw_")]
    pw_choke_cols = [c for c in weekly.columns if c.startswith("pw_") and "_exp_" not in c and "_imp_" not in c]
    pw_ports_cols = [c for c in weekly.columns if c.startswith("pw_") and ("_exp_" in c or "_imp_" in c)]
    if gfw_cols:
        weekly["avail_gfw"] = weekly[gfw_cols].notna().any(axis=1).astype(int)
    if pw_choke_cols:
        weekly["avail_pw_chokepoints"] = weekly[pw_choke_cols].notna().any(axis=1).astype(int)
    if pw_ports_cols:
        weekly["avail_pw_ports"] = weekly[pw_ports_cols].notna().any(axis=1).astype(int)
    feat_cols = gfw_cols + pw_choke_cols + pw_ports_cols
    weekly["avail_shipping"] = weekly[feat_cols].notna().any(axis=1).astype(int)

    # 6) Clip to study window and write.
    weekly = weekly.loc[STUDY_START:STUDY_END]
    weekly.index.name = "week_ending_friday"
    weekly.to_csv(OUT_PATH)

    _report(weekly, lag_checks)


def _report(weekly: pd.DataFrame, lag_checks: list[str | None]) -> None:
    gfw_cols = [c for c in weekly.columns if c.startswith("gfw_")]
    pw_cols = [c for c in weekly.columns if c.startswith("pw_")]

    print(f"\n{'='*64}")
    print(f"Output: {OUT_PATH}")
    print(f"Shape:  {weekly.shape}   Period: "
          f"{weekly.index.min().date()} ~ {weekly.index.max().date()}")
    print(f"Weeks:  {len(weekly)}")

    print(f"\nFeature breakdown:")
    print(f"  GFW columns:       {len(gfw_cols)}")
    print(f"  PortWatch columns: {len(pw_cols)}")
    print(f"  Total columns:     {len(weekly.columns)}")

    # The headline fix: union weeks vs. the PortWatch-only overlap.
    if gfw_cols and pw_cols:
        gfw_w = weekly[gfw_cols].notna().any(axis=1).sum()
        pw_w = weekly[pw_cols].notna().any(axis=1).sum()
        both = (weekly[gfw_cols].notna().any(axis=1) & weekly[pw_cols].notna().any(axis=1)).sum()
        print(f"\nUnion-alignment check (727->362 fix):")
        print(f"  weeks with GFW data:        {gfw_w}")
        print(f"  weeks with PortWatch data:  {pw_w}")
        print(f"  weeks with BOTH (overlap):  {both}  <- old leaky inner-join size")
        print(f"  TOTAL union weeks kept:     {len(weekly)}  <- no early GFW week dropped")

    print(f"\nPer-chokepoint coverage (non-null weeks, post-lag):")
    for cp in dict.fromkeys(CHOKEPOINT_SHORT.values()):   # unique, insertion order
        gcol, pcol = f"gfw_{cp}_total_hours", f"pw_{cp}_n_tanker"
        gn = weekly[gcol].notna().sum() if gcol in weekly.columns else 0
        pn = weekly[pcol].notna().sum() if pcol in weekly.columns else 0
        print(f"  {cp:10s}: GFW={gn:4d}w  PW={pn:4d}w")

    # Leakage check: the publication lag must move each source's first usable
    # week FORWARD by exactly `lag` weeks (a lag, never a look-ahead).
    print(f"\nNo-look-ahead lag check (first usable week shifted forward):")
    for line in lag_checks:
        if line:
            print(line)
    print(f"{'='*64}\nDone.")


if __name__ == "__main__":
    main()
