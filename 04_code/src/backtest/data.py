"""
Data layer for the flat-feature-fusion baseline.

ONE source of truth: the merged leakage-safe weekly matrix
(processed/merge/outputs/weekly_feature_matrix.csv) + its feature dictionary.
Every modality config (M1..M4) reads the SAME matrix and differs only in which
columns are selected, so the rolling-origin backtest is identical across configs.

Modality column selection (by the dictionary `modality` field):
  M1 = finance/macro            (34 cols; the 4 avail_* masks are modality='mask',
                                 i.e. zero-variance in-window, so they are NOT M1
                                 features -- this is exactly why merged-M1 has 34
                                 cols vs the 38-col single table, which are
                                 equivalent because VarianceThreshold drops them.)
  M2 = M1 + remote sensing
  M3 = M1 + shipping
  M4 = M1 + remote sensing + shipping
Targets (target_*) and mask columns (avail_*) are never used as features.

M2 feature contract (--m2-features), per 2026-06-22_channelB_mechanism_plan.md §3/§4:
  anom        55 cols {NDVI,NDWI,NDBI,BSI,NTL}_anom_{aoi}   <- DEFAULT, main analysis
  level       55 cols {idx}_{aoi} raw level                 <- C3 robustness only
  all         110 cols anom + level                          <- robustness
  literature  4 cols NTL_anom of Fujairah/RasTanura/Rotterdam/Houston  <- C1 arm
The 55 level (scale-incomparable, seasonal, redundant with anom), 22 age
(days_since_obs; timeliness not signal) and 22 avail (near-constant in-window)
are excluded from the main analysis by design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]          # casa0004 Dissertation
MERGE_DIR = ROOT / "03_data/processed/merge/outputs"
MERGE_CSV = MERGE_DIR / "weekly_feature_matrix.csv"
DICT_CSV  = MERGE_DIR / "weekly_feature_dictionary.csv"

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

RS_INDICES = ["NDVI", "NDWI", "NDBI", "BSI", "NTL"]
M2_FEATURE_MODES = ("anom", "level", "all", "literature")
# C1 literature arm: core night-time-light anomaly export hubs.
LITERATURE_AOIS = ["Fujairah", "RasTanura", "Rotterdam", "Houston"]

# feature-mode="returns": M1 trending level columns to stationarise.
LOGRET_COLS = ["brent_price", "wti_price", "sp500"]
DIFF_COLS = [
    "crude_stocks_excl_spr", "cushing_stocks", "crude_production",
    "crude_imports", "crude_exports", "refinery_crude_input",
    "gasoline_supplied", "distillate_supplied", "jet_fuel_supplied",
    "net_crude_trade", "dollar_index", "nonoil_industrial_commodity",
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
    # literature: 4 core NTL_anom export hubs (intersect with what exists)
    want = [f"NTL_anom_{a}" for a in LITERATURE_AOIS]
    return [c for c in want if c in anom]


def list_aois(dico: pd.DataFrame) -> list[str]:
    m2_all = dico.loc[dico["modality"] == "M2", "feature"].tolist()
    return sorted({a for c in m2_all if (a := aoi_of(c)) is not None})


def select_features(dico: pd.DataFrame, modality: str, m2_features: str = "anom",
                    drop_aoi: str | None = None) -> list[str]:
    """Feature columns for a modality config (never targets/masks).

    drop_aoi (leave-one-AOI-out): remove every M2 column belonging to that AOI.
    """
    if modality not in MODALITY_SETS:
        raise ValueError(f"modality must be one of {list(MODALITY_SETS)}")
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
        cols += dico.loc[dico["modality"] == "M3", "feature"].tolist()
    return [c for c in cols if not c.startswith("target_")]


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
def to_stationary(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """feature_mode='returns': stationarise M1 trending level columns only.

    Only the feature copy is touched; the price/target are always read from the
    original matrix, so P_t and r_{t+1} are never affected.
    """
    if mode != "returns":
        return df
    out = df.copy()
    for c in LOGRET_COLS:
        if c in out:
            out[c] = np.log(out[c] / out[c].shift(1))
    for c in DIFF_COLS:
        if c in out:
            out[c] = out[c].diff()
    return out


def make_lagged(df: pd.DataFrame, feat_cols: list[str], lookback: int) -> pd.DataFrame:
    """Flatten the past `lookback` weeks: feature_lag0 (=t) .. feature_lag{L-1}."""
    parts = [df[feat_cols].shift(lag).add_suffix(f"_lag{lag}") for lag in range(lookback)]
    return pd.concat(parts, axis=1)


def fill_features(X: pd.DataFrame) -> pd.DataFrame:
    """ffill (past-only, no look-ahead) + residual leading NaN -> 0 (neutral).

    RS anomaly columns have sparse early gaps; this keeps every config on the
    EXACT same test weeks so RMSE differences reflect feature content, not
    sample changes. Residual NaNs only fall in the warm-up, never the test set.
    """
    return X.ffill().fillna(0.0)


# ----------------------------------------------------------------------------
# Dataset assembly
# ----------------------------------------------------------------------------
def build_dataset(df: pd.DataFrame, feat_cols: list[str], lookback: int,
                  feature_mode: str = "all",
                  window_start: str = WINDOW_START,
                  window_end: str = WINDOW_END) -> dict:
    """Assemble the aligned arrays for the rolling-origin loop.

    Returns a dict with idx, X (n x p), P_t, P_next, r_next (=TARGET r_{t+1}),
    r_now (past r_t), and feat_names. No look-ahead: r_next is the only
    future-derived field and is the supervised target, never a feature.
    """
    assert not any(c.startswith("target_") for c in feat_cols), "target leaked into features"

    df_feat = to_stationary(df, feature_mode)
    Xfilled = fill_features(df_feat[feat_cols])
    X_all = make_lagged(Xfilled, feat_cols, lookback)
    feat_names = X_all.columns.tolist()

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
    usable = (
        data[feat_names].notna().all(axis=1)
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
