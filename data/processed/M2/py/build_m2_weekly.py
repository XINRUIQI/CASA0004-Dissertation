"""
B1 - Channel B (M2) weekly mechanism-variable builder.

Turns the two raw monthly Channel B tables (Sentinel-2 optical indices + VIIRS
night-time lights) into a leakage-free weekly (W-FRI) feature table for the
11 oil-infrastructure AOIs, standardised comparison window 2019-2026.

Five indicators: NDVI / NDWI / NDBI / BSI (S2) + NTL (VIIRS).
Water-mask mode adds: MNDWI (S2) + s2_land_px_{site} (land pixel fraction).

Per (site, indicator) we build THREE forms:
  - level : raw monthly value (kept for interpretation)
  - anom  : within-site standardized anomaly = de-season + expanding z-score
            (expanding statistics include the current month but NO future ->
             past-only, leakage-free; see refs in the Channel B plan)
  - mom   : month-on-month first difference (Delta; robust to sign changes)

Monthly -> weekly alignment is an AS-OF JOIN on a conservative availability
date (month_end + PUB_LAG_DAYS), NOT a dumb forward-fill:
  - each Friday maps to the most recent ALREADY-RELEASED valid monthly obs
  - the value repeats within a month (remote sensing is monthly) BUT every week
    also carries an increasing `days_since_obs` (age) + masks, so staleness /
    missingness is explicit (research plan section 4.4).

Masks:
  - valid_mask    : an as-of valid monthly obs was found at all (1/0)
  - modality_mask : a *fresh enough* valid obs exists (age <= MAX_AGE_DAYS)

Outputs (-> data/processed/M2/outputs/):
  m2_weekly_features.csv          wide: {idx}_{aoi}, {idx}_anom_{aoi},
                                  {mod}_age/avail  (merge keeps 55 anom cols)
  m2_eda_weekly.csv          tidy EDA table: one row per (week, site, idx)

  --watermask (B4 robustness):
  m2_weekly_features_watermask.csv   same structure + MNDWI cols + s2_land_px_*
  m2_eda_weekly_watermask.csv   + land_px, low_land_coverage flag

Run:
  python3 data/processed/M2/py/build_m2_weekly.py
  python3 data/processed/M2/py/build_m2_weekly.py --watermask
  python3 data/processed/M2/py/build_m2_weekly.py --no-deseasonalize
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "data/raw/02_sentinel2/Channel B"
OUT = ROOT / "data/processed/M2/outputs"
S2_CSV    = RAW / "sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv"
S2_WM_CSV = RAW / "sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi.csv"
NTL_CSV   = RAW / "viirs_oil_sites_monthly_nightlights_201401_202512_11aoi.csv"

S2_INDICES    = ["NDVI", "NDWI", "NDBI", "BSI"]
S2_WM_INDICES = ["NDVI", "NDWI", "NDBI", "BSI", "MNDWI"]   # extra MNDWI in watermask mode
INDEX_ORDER   = S2_INDICES + ["NTL"]
INDEX_ORDER_WM = S2_WM_INDICES + ["NTL"]
TYPE_ORDER    = {"port": 0, "refinery": 1, "terminal": 2}

# Sentinel-2 water-masked CSV lacks short_name; use this static map (matches original).
SITE_SHORT = {
    "P001": "Rotterdam", "P002": "Fujairah",    "P003": "RasTanura",
    "P004": "Jurong",    "P005": "Houston",     "P006": "NingboZhoushan",
    "P007": "Jamnagar",  "P008": "Basra",       "P009": "Ulsan",
    "P010": "Kharg",     "P011": "Yanbu",
}

# --- alignment / anomaly parameters ---
WINDOW_START    = "2019-01"
WINDOW_END      = "2025-12"
PUB_LAG_DAYS    = 15    # availability = month_end + lag (conservative release delay)
OBS_DAY         = 15    # representative observation day within the month (for age)
MAX_AGE_DAYS    = 100   # modality flagged unavailable if newest valid obs older than this
MIN_HIST        = 12    # min months of history before an anomaly z-score is defined
LOW_LAND_THRESH = 0.05  # land_px < 5% → flag as low_land_coverage (B4 diagnostic)


# ----------------------------------------------------------------------------
# Load -> unified monthly long table
# ----------------------------------------------------------------------------
def load_monthly_long(watermask: bool = False) -> pd.DataFrame:
    """
    Load S2 + VIIRS monthly data into a unified long table.

    watermask=False (default): original CSV, 4 S2 indices (NDVI/NDWI/NDBI/BSI).
    watermask=True  (B4):      water-masked CSV, 5 indices (+MNDWI), plus land_px
                               and low_land_coverage columns carried through.
    """
    if watermask:
        s2 = pd.read_csv(S2_WM_CSV)
        s2["date"] = pd.to_datetime(s2["date_month"] + "-01")
        # water-masked CSV lacks short_name → derive from static map
        s2["short_name"] = s2["site_id"].map(SITE_SHORT)
        # land_px = land pixel fraction (land_pixel_fraction col is NaN in export)
        s2["_land_px"] = s2["land_px"]
        s2_indices = S2_WM_INDICES
    else:
        s2 = pd.read_csv(S2_CSV)
        s2["date"] = pd.to_datetime(s2["date_month"] + "-01")
        s2["_land_px"] = np.nan
        s2_indices = S2_INDICES

    sid2short = (s2[["site_id", "short_name"]].drop_duplicates()
                 .set_index("site_id")["short_name"].to_dict())
    s2_long = s2.melt(
        id_vars=["site_id", "site_name", "site_type", "date",
                 "valid_obs_count", "_land_px"],
        value_vars=s2_indices, var_name="index", value_name="level",
    )
    s2_long["modality"] = "S2"
    s2_long["sensor"] = "Sentinel-2"
    s2_long = s2_long.rename(columns={"valid_obs_count": "qual",
                                      "_land_px": "land_px"})

    ntl = pd.read_csv(NTL_CSV)
    ntl["date"] = pd.to_datetime(ntl["date_month"] + "-01")
    ntl_long = ntl[["site_id", "site_name", "site_type", "date",
                    "ntl_avg_rad_mean", "ntl_cf_cvg_mean"]].copy()
    ntl_long = ntl_long.rename(columns={"ntl_avg_rad_mean": "level",
                                        "ntl_cf_cvg_mean": "qual"})
    ntl_long["index"] = "NTL"
    ntl_long["modality"] = "VIIRS"
    ntl_long["sensor"] = "VIIRS"
    ntl_long["land_px"] = np.nan   # land_px is S2-only

    cols = ["site_id", "site_name", "site_type", "modality", "sensor",
            "index", "date", "level", "qual", "land_px"]
    long = pd.concat([s2_long[cols], ntl_long[cols]], ignore_index=True)
    long["short_name"] = long["site_id"].map(sid2short)
    return long.sort_values(["site_id", "index", "date"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Three forms: level / anom / mom  (all past-only, leakage-free)
# ----------------------------------------------------------------------------
def add_anomaly(df: pd.DataFrame, deseasonalize: bool) -> pd.DataFrame:
    df = df.copy()
    df["moy"] = df["date"].dt.month

    if deseasonalize:
        # expanding month-of-year climatology (past-only, includes current month)
        clim = (df.groupby(["site_id", "index", "moy"])["level"]
                .transform(lambda s: s.expanding(min_periods=1).mean()))
        resid = df["level"] - clim
    else:
        resid = df["level"]
    df["_resid"] = resid

    g = df.groupby(["site_id", "index"])["_resid"]
    mu = g.transform(lambda s: s.expanding(min_periods=MIN_HIST).mean())
    sd = g.transform(lambda s: s.expanding(min_periods=MIN_HIST).std())
    df["anom"] = (df["_resid"] - mu) / sd
    df["anom"] = df["anom"].replace([np.inf, -np.inf], np.nan)

    df["mom"] = df.groupby(["site_id", "index"])["level"].diff()
    return df.drop(columns=["_resid"])


# ----------------------------------------------------------------------------
# Monthly -> weekly as-of alignment (availability date + age + masks)
# ----------------------------------------------------------------------------
def align_weekly(monthly: pd.DataFrame, fridays: pd.DatetimeIndex) -> pd.DataFrame:
    m = monthly.copy()
    month_end = m["date"] + pd.offsets.MonthEnd(0)
    m["observation_date"] = m["date"] + pd.Timedelta(days=OBS_DAY - 1)
    m["availability_date"] = month_end + pd.Timedelta(days=PUB_LAG_DAYS)
    m["is_valid"] = m["level"].notna()

    valid = (m[m["is_valid"]]
             .sort_values("availability_date")
             .reset_index(drop=True))

    keys = m[["site_id", "index"]].drop_duplicates()
    left = keys.merge(pd.DataFrame({"week_fri": fridays}), how="cross")
    left = left.sort_values("week_fri").reset_index(drop=True)

    right_cols = ["site_id", "index", "availability_date", "observation_date",
                  "site_name", "site_type", "short_name", "modality", "sensor",
                  "level", "anom", "mom", "qual", "land_px"]
    res = pd.merge_asof(
        left, valid[right_cols],
        left_on="week_fri", right_on="availability_date",
        by=["site_id", "index"], direction="backward",
    )

    matched = res["observation_date"].notna()
    res["days_since_obs"] = (res["week_fri"] - res["observation_date"]).dt.days
    res["valid_mask"] = matched.astype(int)
    res["modality_mask"] = (matched &
                            (res["days_since_obs"] <= MAX_AGE_DAYS)).astype(int)
    return res


# ----------------------------------------------------------------------------
# Long -> wide feature matrix
# ----------------------------------------------------------------------------
def to_wide(long: pd.DataFrame, order_short: list[str],
            index_order: list[str]) -> pd.DataFrame:
    parts = []
    for idx in index_order:
        sub = long[long["index"] == idx]
        lvl = sub.pivot_table(index="week_fri", columns="short_name",
                              values="level", aggfunc="first").reindex(columns=order_short)
        lvl.columns = [f"{idx}_{c}" for c in lvl.columns]
        ano = sub.pivot_table(index="week_fri", columns="short_name",
                              values="anom", aggfunc="first").reindex(columns=order_short)
        ano.columns = [f"{idx}_anom_{c}" for c in ano.columns]
        parts.extend([lvl, ano])

    # per-modality metadata (S2 indices share one obs/age; NTL separate)
    for mod, tag in [("S2", "s2"), ("VIIRS", "ntl")]:
        sub = long[long["modality"] == mod]
        age = sub.pivot_table(index="week_fri", columns="short_name",
                              values="days_since_obs", aggfunc="first").reindex(columns=order_short)
        age.columns = [f"{tag}_age_days_{c}" for c in age.columns]
        avail = sub.pivot_table(index="week_fri", columns="short_name",
                                values="modality_mask", aggfunc="max").reindex(columns=order_short)
        avail.columns = [f"{tag}_avail_{c}" for c in avail.columns]
        parts.extend([age, avail])

    # land_px columns (S2 only; NaN for NTL rows → max picks S2 value)
    s2_sub = long[long["modality"] == "S2"]
    if s2_sub["land_px"].notna().any():
        lpx = (s2_sub.pivot_table(index="week_fri", columns="short_name",
                                  values="land_px", aggfunc="max")
                     .reindex(columns=order_short))
        lpx.columns = [f"s2_land_px_{c}" for c in lpx.columns]
        parts.append(lpx)

    wide = pd.concat(parts, axis=1).sort_index()
    wide.index.name = "week_fri"
    return wide.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=WINDOW_START)
    ap.add_argument("--end", default=WINDOW_END)
    ap.add_argument("--no-deseasonalize", action="store_true",
                    help="only de-scale (skip month-of-year de-seasonalisation)")
    ap.add_argument("--watermask", action="store_true",
                    help="B4: use MNDWI water-masked S2 CSV; adds MNDWI cols + "
                         "s2_land_px_* cols; outputs *_watermask.csv")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fridays = pd.date_range(f"{args.start}-01",
                            pd.Timestamp(f"{args.end}-01") + pd.offsets.MonthEnd(0),
                            freq="W-FRI")

    monthly = load_monthly_long(watermask=args.watermask)
    monthly = add_anomaly(monthly, deseasonalize=not args.no_deseasonalize)
    long = align_weekly(monthly, fridays)

    # site display order (port -> refinery -> terminal)
    meta = (monthly[["short_name", "site_type"]].drop_duplicates()
            .assign(_t=lambda d: d["site_type"].map(TYPE_ORDER))
            .sort_values(["_t", "short_name"]))
    order_short = meta["short_name"].tolist()

    # choose index order based on mode
    idx_order = INDEX_ORDER_WM if args.watermask else INDEX_ORDER

    # output filename suffix
    suffix = "_watermask" if args.watermask else ""

    # long output columns (land_px + low_land_coverage only in watermask mode)
    long_cols = [
        "week_fri", "site_id", "short_name", "site_name", "site_type",
        "modality", "sensor", "index", "level", "anom", "mom",
        "observation_date", "days_since_obs", "valid_mask", "modality_mask", "qual",
    ]
    if args.watermask:
        long_cols += ["land_px"]

    long_out = long[long_cols].rename(columns={"qual": "valid_obs_count"})

    # low_land_coverage flag: mark S2 rows where available land pixel fraction < threshold
    if args.watermask:
        long_out = long_out.copy()
        long_out["low_land_coverage"] = (
            (long_out["modality"] == "S2") &
            long_out["land_px"].notna() &
            (long_out["land_px"] < LOW_LAND_THRESH)
        ).astype(int)

    long_out = long_out.sort_values(["week_fri", "site_id", "index"])
    long_path = OUT / f"m2_eda_weekly{suffix}.csv"
    long_out.to_csv(long_path, index=False)

    wide = to_wide(long, order_short, idx_order)
    wide_path = OUT / f"m2_weekly_features{suffix}.csv"
    wide.to_csv(wide_path, index=False)

    # ---- console summary ----
    print(f"mode           : {'watermask' if args.watermask else 'standard'}")
    print(f"window         : {args.start} .. {args.end}  ({len(fridays)} Fridays)")
    print(f"de-seasonalise : {not args.no_deseasonalize}")
    print(f"long  -> {long_path}  shape={long_out.shape}")
    print(f"wide  -> {wide_path}  shape={wide.shape}  ({wide.shape[1]-1} features)")

    av = long.groupby("modality")["modality_mask"].mean().round(3)
    print(f"\n[modality availability (mean over window)]\n{av.to_string()}")
    anom_na = long.groupby("index")["anom"].apply(lambda s: round(s.isna().mean(), 3))
    print(f"\n[anomaly NaN fraction by index]\n{anom_na.to_string()}")

    if args.watermask:
        lc = (long_out[long_out["modality"] == "S2"]
              .groupby("short_name")["land_px"].median().round(3).sort_values())
        print(f"\n[land_px median by site (watermask mode)]\n{lc.to_string()}")
        low = long_out.query("low_land_coverage == 1")
        print(f"\nlow_land_coverage rows (land_px < {LOW_LAND_THRESH}): "
              f"{len(low)} / {(long_out['modality']=='S2').sum()} S2 rows "
              f"({100*len(low)/(long_out['modality']=='S2').sum():.1f}%)")

    demo = (long[(long["short_name"] == "Houston") & (long["index"] == "NDVI")]
            .sort_values("week_fri")
            .loc[:, ["week_fri", "level", "anom", "days_since_obs",
                     "valid_mask", "modality_mask"]]
            .head(10))
    print("\n[sanity: Houston NDVI first 10 weeks]")
    print(demo.to_string(index=False))


if __name__ == "__main__":
    main()
