"""
Data layer for the flat-feature-fusion baseline.

ONE source of truth: the merged leakage-safe weekly matrix
(processed/merge/outputs/weekly_feature_matrix.csv) + its feature dictionary.
Every modality config (M1..M4) reads the SAME matrix and differs only in which
columns are selected, so the rolling-origin backtest is identical across configs.

Modality column selection (by the dictionary `modality` field):
  M1 = finance/macro            (31 cols; the 4 avail_* masks are modality='mask',
                                 i.e. zero-variance in-window, so they are NOT M1
                                 features -- this is exactly why merged-M1 has 31
                                 cols vs the 35-col single table, which are
                                 equivalent because VarianceThreshold drops them.)
  M2 = M1 + remote sensing
  M3 = M1 + shipping        (main model = FULL tier, 164 cols, by default)
  M4 = M1 + remote sensing + shipping
Targets (target_*) and mask columns (avail_*) are never used as features.

M3 shipping tier (--m3-tier, see m3_data_dictionary.md §11):
  full (DEFAULT, 164 cols) every shipping column (PortWatch 64 + GFW 49 +
                          SAR dark-vessel 51); the
                          MAIN model, chosen because the hand-picked core tier
                          is not XGB-optimal (m3_data_dictionary.md §11 / robustness).
  core (38 cols)          GFW 6x4 (total_hours, total_vessels, cargo_hours,
                          total_hours_mom_pct) + PortWatch chokepoints 6x2 +
                          PortWatch ports 2; now a ROBUSTNESS arm. mean_presence
                          and the aggregate z-mean are EXCLUDED (separate exps).
Separate GFW experiments (not in the main model, see robustness_m3.py):
  GFW-Presence   6 x gfw_{cp}_mean_presence_hours_per_vessel.
  GFW-Aggregate  gfw_all_activity_zmean, DERIVED leak-free at build time
                 (past-only expanding z-score mean of the 6 chokepoint
                 total_hours), replacing the deprecated gfw_all_total_hours_sum.

M2 feature contract (--m2-features), per 2026-06-22_channelB_mechanism_plan.md §3/§4:
  anom        55 cols {NDVI,NDWI,NDBI,BSI,NTL}_anom_{aoi}   <- DEFAULT, main analysis
  level       55 cols {idx}_{aoi} raw level                 <- C3 robustness only
  all         110 cols anom + level                          <- robustness
  literature  4 cols NTL_anom of Fujairah/RasTanura/Rotterdam/Houston  <- C1 arm
  aoi4        20 cols {5 idx}_anom_{4 core AOI}  <- 2x2 sparsity: site-selected
  ntlall      11 cols NTL_anom_{11 AOI}          <- 2x2 sparsity: index-selected
The 55 level (scale-incomparable, seasonal, redundant with anom), 22 age
(days_since_obs; timeliness not signal) and 22 avail (near-constant in-window)
are excluded from the main analysis by design.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]          # casa0004 Dissertation
MERGE_DIR = ROOT / "data/processed/merge/outputs"
MERGE_CSV = MERGE_DIR / "weekly_feature_matrix.csv"
DICT_CSV  = MERGE_DIR / "weekly_feature_dictionary.csv"
# Long-history build of the same merge; the standard matrix is its 2019- slice.
# Only read to seed differences at the first in-sample week (see to_stationary).
FULL_MATRIX_CSV = MERGE_DIR / "weekly_feature_matrix_full.csv"

TARGET_PRICE = "brent_price"
RETURN_COL = "brent_log_return"                     # past r_t, for dir-persistence
WINDOW_START = "2019-01-01"
WINDOW_END = "2025-12-31"

MODALITY_SETS = {
    "M1": ["M1"],
    "M2": ["M1", "M2"],            # finance + remote sensing
    "M3": ["M1", "M3"],            # finance + shipping
    "M4": ["M1", "M2", "M3"],      # all
}

# ---------------------------------------------------------------------------
# M3 shipping CORE tier (m3_data_dictionary.md §11.1) -- now a ROBUSTNESS arm.
# The MAIN model uses the FULL tier (all shipping columns); this hand-picked
# core set is kept for the 'core' robustness arm and the GFW sub-experiments.
# ---------------------------------------------------------------------------
GFW_CHOKES = ["hormuz", "suez", "malacca", "mandeb", "panama", "cape"]
# Core-tier GFW = 4 metrics x 6 chokepoints = 24.
GFW_CORE_SUFFIXES = ["total_hours", "total_vessels", "cargo_hours", "total_hours_mom_pct"]
PW_CHOKE_CORE_SUFFIXES = ["n_tanker", "capacity_tanker"]
PW_PORTS_CORE = ["pw_exp_hubs_export_vol", "pw_imp_hubs_import_vol"]
# NOT in the main model (each drives its own separate experiment, §11):
#   mean_presence_hours_per_vessel -> GFW-Presence experiment
#   gfw_all_activity_zmean         -> GFW-Aggregate benchmark
GFW_PRESENCE_SUFFIXES = ["mean_presence_hours_per_vessel"]
GFW_ZMEAN_COL = "gfw_all_activity_zmean"   # derived leak-free at build time
GFW_ZMEAN_MIN_PERIODS = 12
M3_TIERS = ("core", "full")

RS_INDICES = ["NDVI", "NDWI", "NDBI", "BSI", "NTL"]
M2_FEATURE_MODES = ("anom", "level", "all", "literature", "aoi4", "ntlall")
# C1 literature arm: core night-time-light anomaly export hubs.
LITERATURE_AOIS = ["Fujairah", "RasTanura", "Rotterdam", "Houston"]

# feature-mode="returns": M1 trending level columns to stationarise.
LOGRET_COLS = ["brent_price", "wti_price"]
DIFF_COLS = [
    "crude_stocks_excl_spr", "cushing_stocks", "crude_production",
    "crude_imports", "crude_exports", "refinery_crude_input",
    "gasoline_supplied", "distillate_supplied", "jet_fuel_supplied",
    "dollar_index", "nonoil_industrial_commodity",
]


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
def load_matrix(path: "Path | str | None" = None) -> pd.DataFrame:
    """Load the merged feature matrix.

    path: override default MERGE_CSV (e.g. weekly_feature_matrix_watermask.csv
          for B4 water-mask sensitivity comparison).
    """
    p = Path(path) if path is not None else MERGE_CSV
    if not p.is_absolute():
        p = MERGE_DIR / p          # resolve relative to merge/outputs/
    df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    df.index.name = "week_ending_friday"
    return df


def load_dict(path: "Path | str | None" = None) -> pd.DataFrame:
    """Load the feature dictionary.

    path: override default DICT_CSV (auto-paired with the matrix override).
    """
    p = Path(path) if path is not None else DICT_CSV
    if not p.is_absolute():
        p = MERGE_DIR / p
    return pd.read_csv(p)


# ----------------------------------------------------------------------------
# Column selection
# ----------------------------------------------------------------------------
def aoi_of(col: str) -> str | None:
    """Extract the AOI/site suffix from an M2 remote-sensing column."""
    if "_anom_" in col:
        return col.split("_anom_", 1)[1]
    for p in RS_INDICES:
        if col.startswith(p + "_"):
            return col[len(p) + 1:]
    return None


def m2_columns(dico: pd.DataFrame, m2_features: str) -> list[str]:
    if m2_features not in M2_FEATURE_MODES:
        raise ValueError(f"m2_features must be one of {M2_FEATURE_MODES}")
    m2_all = dico.loc[dico["modality"] == "M2", "feature"].tolist()
    anom = [c for c in m2_all if "_anom_" in c]
    level = [c for c in m2_all
             if "_anom_" not in c and any(c.startswith(p + "_") for p in RS_INDICES)]
    if m2_features == "anom":
        return anom
    if m2_features == "level":
        return level
    if m2_features == "all":
        return anom + level
    if m2_features == "aoi4":
        # 2x2 sparsity cell: 4 core AOIs x 5 indices anomaly (site-selected) = 20
        want = [f"{idx}_anom_{a}" for a in LITERATURE_AOIS for idx in RS_INDICES]
        return [c for c in want if c in anom]
    if m2_features == "ntlall":
        # 2x2 sparsity cell: 11 AOIs x NTL anomaly (index-selected to NTL) = 11
        return [c for c in anom if c.startswith("NTL_anom_")]
    # literature: 4 core NTL_anom export hubs (intersect with what exists)
    want = [f"NTL_anom_{a}" for a in LITERATURE_AOIS]
    return [c for c in want if c in anom]


def list_aois(dico: pd.DataFrame) -> list[str]:
    m2_all = dico.loc[dico["modality"] == "M2", "feature"].tolist()
    return sorted({a for c in m2_all if (a := aoi_of(c)) is not None})


def gfw_core_columns(dico: pd.DataFrame) -> list[str]:
    """Core-tier GFW: 6 chokepoints x 4 metrics = 24 (no mean_presence)."""
    m3_all = set(dico.loc[dico["modality"] == "M3", "feature"])
    return [f"gfw_{cp}_{s}" for cp in GFW_CHOKES for s in GFW_CORE_SUFFIXES
            if f"gfw_{cp}_{s}" in m3_all]


def gfw_presence_columns(dico: pd.DataFrame) -> list[str]:
    """GFW-Presence experiment set: 6 x mean_presence_hours_per_vessel (NOT core)."""
    m3_all = set(dico.loc[dico["modality"] == "M3", "feature"])
    return [f"gfw_{cp}_{s}" for cp in GFW_CHOKES for s in GFW_PRESENCE_SUFFIXES
            if f"gfw_{cp}_{s}" in m3_all]


def m3_core_columns(dico: pd.DataFrame) -> list[str]:
    """§11.1 core-tier shipping columns present in the matrix (robustness arm).

    GFW 6x4 = 24 ; PortWatch chokepoints 6x2 = 12 ; PortWatch ports 2 = 38.
    mean_presence and gfw_all_activity_zmean are EXCLUDED (separate experiments).
    """
    m3_all = set(dico.loc[dico["modality"] == "M3", "feature"])
    cols: list[str] = gfw_core_columns(dico)
    cols += [f"pw_{cp}_{s}" for cp in GFW_CHOKES for s in PW_CHOKE_CORE_SUFFIXES
             if f"pw_{cp}_{s}" in m3_all]
    cols += [c for c in PW_PORTS_CORE if c in m3_all]
    return cols


def select_features(dico: pd.DataFrame, modality: str, m2_features: str = "anom",
                    drop_aoi: str | None = None, m3_tier: str = "full") -> list[str]:
    """Feature columns for a modality config (never targets/masks).

    drop_aoi (leave-one-AOI-out): remove every M2 column belonging to that AOI.
    m3_tier: 'full' (all 164 shipping cols, DEFAULT main model) or 'core'
             (§11.1 38-col hand-picked set, now a robustness arm).
    """
    if modality not in MODALITY_SETS:
        raise ValueError(f"modality must be one of {list(MODALITY_SETS)}")
    if m3_tier not in M3_TIERS:
        raise ValueError(f"m3_tier must be one of {M3_TIERS}")
    mods = MODALITY_SETS[modality]
    cols: list[str] = []
    if "M1" in mods:
        cols += dico.loc[dico["modality"] == "M1", "feature"].tolist()
    if "M2" in mods:
        m2 = m2_columns(dico, m2_features)
        if drop_aoi is not None:
            m2 = [c for c in m2 if aoi_of(c) != drop_aoi]
        cols += m2
    if "M3" in mods:
        if m3_tier == "core":
            cols += m3_core_columns(dico)
        else:
            cols += dico.loc[dico["modality"] == "M3", "feature"].tolist()
    return [c for c in cols if not c.startswith("target_")]


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _full_history() -> "pd.DataFrame | None":
    """Long-history merge output (2006-), or None if it has not been built.

    Same pipeline and the same source-side availability lags as the standard
    matrix, which is literally its 2019- slice, so a row read from here is
    as-of valid at its own week.
    """
    if not FULL_MATRIX_CSV.exists():
        return None
    full = pd.read_csv(FULL_MATRIX_CSV, index_col=0, parse_dates=True).sort_index()
    full.index.name = "week_ending_friday"
    return full


def presample_row(first_week: pd.Timestamp, cols: list[str]) -> "pd.DataFrame | None":
    """The last observation strictly before `first_week`, for `cols`.

    Used to seed differences at the first in-sample week without moving the
    sample start: the value is dated before the window, so it carries no
    information that was unavailable at `first_week`.
    """
    full = _full_history()
    if full is None:
        return None
    prior = full.index[full.index < first_week]
    have = [c for c in cols if c in full.columns]
    if len(prior) == 0 or not have:
        return None
    return full.loc[[prior.max()], have]


def to_stationary(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """feature_mode='returns': stationarise M1 trending level columns only.

    Differences are seeded from the pre-sample week so the first in-sample week
    carries a real change rather than a structural gap; the sample itself still
    starts at the window start. Where no pre-sample observation exists the
    first row falls back to zero, which is the neutral value for a change.

    Only the feature copy is touched; the price/target are always read from the
    original matrix, so P_t and r_{t+1} are never affected.
    """
    if mode != "returns":
        return df
    out = df.copy()
    cols = [c for c in LOGRET_COLS + DIFF_COLS if c in out.columns]
    prev = presample_row(df.index[0], cols)
    base = df[cols] if prev is None else pd.concat([prev, df[cols]])
    for c in LOGRET_COLS:
        if c in out:
            out[c] = np.log(base[c] / base[c].shift(1)).reindex(df.index)
    for c in DIFF_COLS:
        if c in out:
            out[c] = base[c].diff().reindex(df.index)
    first = df.index[0]
    out.loc[first, cols] = out.loc[first, cols].fillna(0.0)
    return out


def add_gfw_activity_zmean(df: pd.DataFrame,
                           min_periods: int = GFW_ZMEAN_MIN_PERIODS) -> pd.DataFrame:
    """Append gfw_all_activity_zmean = mean over the 6 chokepoints of each
    chokepoint's total_hours standardised by its OWN past-only expanding
    mean/std. Standardising before averaging stops large-polygon chokepoints
    from dominating a raw sum. Expanding (data up to t only) is leak-free and
    operates on the already +4w-lagged GFW columns. Replaces the deprecated raw
    sum gfw_all_total_hours_sum (m3_data_dictionary.md §9.2/§11).
    """
    cols = [f"gfw_{cp}_total_hours" for cp in GFW_CHOKES
            if f"gfw_{cp}_total_hours" in df.columns]
    if not cols:
        return df
    out = df.copy()
    z = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].astype(float)
        mu = s.expanding(min_periods=min_periods).mean()
        sd = s.expanding(min_periods=min_periods).std()
        z[c] = (s - mu) / sd
    out[GFW_ZMEAN_COL] = z.mean(axis=1, skipna=True)
    return out


def make_lagged(df: pd.DataFrame, feat_cols: list[str], lookback: int) -> pd.DataFrame:
    """Flatten the past `lookback` weeks: feature_lag0 (=t) .. feature_lag{L-1}."""
    parts = [df[feat_cols].shift(lag).add_suffix(f"_lag{lag}") for lag in range(lookback)]
    return pd.concat(parts, axis=1)


FILL_MODES = ("zero", "fold_median", "by_family")
DEFAULT_FILL_MODE = "by_family"

RS_ANOM_TAG = "_anom_"


def is_rs_anomaly(col: str) -> bool:
    """RS anomaly columns are observation minus own baseline, so they are
    centred on zero by construction and zero reads as 'no anomaly'."""
    return RS_ANOM_TAG in col


def fill_features(X: pd.DataFrame, mode: str = DEFAULT_FILL_MODE) -> pd.DataFrame:
    """Past-only gap filling (ffill), then a choice of leading-gap treatment.

    'by_family'   DEFAULT. ffill, then zero for RS anomalies only. Zero is the
                  neutral value for an anomaly but not for a shipping level
                  such as a tanker count, so those gaps go to the fold median.
    'zero'        ffill + residual leading NaN -> 0 on the raw scale.
    'fold_median' ffill only; leading NaN survive and are imputed inside each
                  training fold by the pipeline's median imputer, so the value
                  is re-estimated at every refit from past data alone.

    Only RS anomalies and PortWatch carry leading gaps, so 'by_family'
    reproduces 'zero' exactly for M1/M2 and 'fold_median' exactly for M3.
    Leading gaps are sparse and confined to the warm-up, and every mode keeps
    the EXACT same test weeks, so RMSE differences reflect feature content and
    the imputation rule, not sample changes.
    """
    assert mode in FILL_MODES, f"unknown fill mode {mode!r}"
    Xf = X.ffill()
    if mode == "zero":
        return Xf.fillna(0.0)
    if mode == "by_family":
        anom = [c for c in Xf.columns if is_rs_anomaly(c)]
        if anom:
            Xf[anom] = Xf[anom].fillna(0.0)
    return Xf


# ----------------------------------------------------------------------------
# Dataset assembly
# ----------------------------------------------------------------------------
def build_dataset(df: pd.DataFrame, feat_cols: list[str], lookback: int,
                  feature_mode: str = "all",
                  window_start: str = WINDOW_START,
                  window_end: str = WINDOW_END,
                  fill_mode: str = DEFAULT_FILL_MODE) -> dict:
    """Assemble the aligned arrays for the rolling-origin loop.

    Returns a dict with idx, X (n x p), P_t, P_next, r_next (=TARGET r_{t+1}),
    r_now (past r_t), and feat_names. No look-ahead: r_next is the only
    future-derived field and is the supervised target, never a feature.
    """
    assert not any(c.startswith("target_") for c in feat_cols), "target leaked into features"

    df_feat = to_stationary(df, feature_mode)
    if GFW_ZMEAN_COL in feat_cols and GFW_ZMEAN_COL not in df_feat.columns:
        df_feat = add_gfw_activity_zmean(df_feat)   # derived, leak-free
    Xfilled = fill_features(df_feat[feat_cols], mode=fill_mode)
    X_all = make_lagged(Xfilled, feat_cols, lookback)
    feat_names = X_all.columns.tolist()

    # Weeks that carry a complete `lookback`-week calendar window. Under
    # fill_mode='zero' this is implied by notna(); when leading NaN survive
    # they must not shrink the sample, so it is imposed here and every mode
    # scores the same test weeks.
    depth_ok = pd.Series(np.arange(len(df_feat)) >= lookback - 1, index=df_feat.index)

    P = df[TARGET_PRICE].astype(float)
    r_next = np.log(P.shift(-1) / P)                 # r_{t+1}, indexed at t (TARGET)
    P_next = P.shift(-1)
    r_now = df[RETURN_COL].astype(float)

    data = X_all.copy()
    data["__P_t"] = P
    data["__P_next"] = P_next
    data["__r_next"] = r_next
    data["__r_now"] = r_now

    in_win = (data.index >= window_start) & (data.index <= window_end)
    data = data[in_win]
    feat_ok = (data[feat_names].notna().all(axis=1) if fill_mode == "zero"
               else depth_ok.reindex(data.index).fillna(False))
    usable = (
        feat_ok
        & data["__P_t"].notna() & data["__P_next"].notna() & data["__r_next"].notna()
    )
    data = data[usable]

    return {
        "idx": data.index,
        "X": data[feat_names].to_numpy(dtype=float),
        "P_t": data["__P_t"].to_numpy(dtype=float),
        "P_next": data["__P_next"].to_numpy(dtype=float),
        "r_next": data["__r_next"].to_numpy(dtype=float),
        "r_now": data["__r_now"].to_numpy(dtype=float),
        "feat_names": feat_names,
        "n_raw": len(feat_cols),
    }
