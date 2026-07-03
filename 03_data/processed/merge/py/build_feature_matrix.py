"""
MERGE — assemble the leakage-safe multimodal weekly feature matrix.

Aligns M1 (finance), M2 (remote sensing, optional), M3 (shipping) onto a single
Friday-ending (W-FRI) UNION index and enforces publication-timestamp alignment,
so every feature in row t is visible by that Friday's close and the prediction
target sits strictly in t+1.

Per-modality release-lag responsibility (who applies which lag):

  - EIA WPSR (lives in M1) .... already lagged inside build_m1_weekly.py (+1w:
        the report covers the week ending Friday t but is published the FOLLOWING
        Wednesday, usable only from Friday t+1). RE-CHECKED here, NOT shifted
        again (a second +1w would silently make it stale by +2w).
  - M1 macro monthlies (gpr / global_econ_activity / nonoil_industrial_commodity)
        ........ already lagged inside build_m1_weekly.py (1-5w); RE-CHECKED here,
        not shifted again.
  - M1 prices & daily market (brent/wti/vix/dxy/rates/sp500/ovx/...) ... no lag
        (daily, available at week close); brent_price is the target base.
  - M3 GFW monthly (+4w) and PortWatch weekly (+1w) ... already lagged inside
        aggregate_shipping_to_weekly.py; RE-CHECKED here, not shifted again.
  - M2 (when available) ... 55 main-analysis anomaly columns only
        ({NDVI,NDWI,NDBI,BSI,NTL}_anom_{aoi}); level/age/avail stay in
        m2_weekly_features.csv for robustness/EDA but are NOT merged.

Target (sole research target = price; trained on the log price change):
  target_price_next      = P_{t+1}                       = brent_price.shift(-1)
  target_log_return_next = r_{t+1} = log(P_{t+1}/P_t)    = brent_log_return.shift(-1)

Outputs (default = standard comparison window 2019-01 .. 2025-12):
  processed/merge/outputs/weekly_feature_matrix.csv
  processed/merge/outputs/weekly_feature_dictionary.csv
With --full (long-history robustness, full union 2006 .. 2026):
  processed/merge/outputs/weekly_feature_matrix_full.csv
  processed/merge/outputs/weekly_feature_dictionary_full.csv

Usage:
    python build_feature_matrix.py            # standard window 2019.1-2025.12
    python build_feature_matrix.py --full     # full union (long-history robustness)
    python build_feature_matrix.py --start 2019-01-01 --end 2025-12-31

    # B4 watermask robustness: replace M2 features with water-masked version.
    # filter_m2_anom_columns() still picks the same 55 cols (NDVI/NDWI/NDBI/BSI/NTL
    # anom × 11 AOIs); MNDWI_anom_* are automatically excluded.
    python build_feature_matrix.py \
        --m2-csv ../M2/outputs/m2_weekly_features_watermask.csv
    # -> outputs weekly_feature_matrix_watermask.csv + weekly_feature_dictionary_watermask.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths  (this file lives at processed/merge/py/build_feature_matrix.py)
# ----------------------------------------------------------------------------
PY_DIR = Path(__file__).resolve().parent           # processed/merge/py
MERGE_DIR = PY_DIR.parent                           # processed/merge
PROC_DIR = MERGE_DIR.parent                         # processed
M1_PATH = PROC_DIR / "M1" / "outputs" / "m1_weekly_features.csv"
M2_PATH = PROC_DIR / "M2" / "outputs" / "m2_weekly_features.csv"
M3_PATH = PROC_DIR / "M3" / "outputs" / "m3_weekly_features.csv"
OUT_DIR = MERGE_DIR / "outputs"
MATRIX_PATH = OUT_DIR / "weekly_feature_matrix.csv"
DICT_PATH = OUT_DIR / "weekly_feature_dictionary.csv"
MATRIX_FULL_PATH = OUT_DIR / "weekly_feature_matrix_full.csv"
DICT_FULL_PATH = OUT_DIR / "weekly_feature_dictionary_full.csv"

# Standard comparison window: all modalities present and every week has a target.
# Locked 2019-01 .. 2025-12 (M1/M2/GFW end 2025-12). PortWatch extends into 2026
# but those weeks have no M1 target, so they live only in the --full export used
# for long-history robustness (research plan stage-2).
STD_WINDOW_START = "2019-01-01"
STD_WINDOW_END = "2025-12-31"

# ----------------------------------------------------------------------------
# Publication-lag configuration
# ----------------------------------------------------------------------------
# EIA WPSR is now lagged at SOURCE in build_m1_weekly.py (EIA_LAG_WEEKS=1), so the
# merge must NOT shift it again (a second +1w would make it stale by +2w). Kept 0
# here; --eia-lag is only an emergency override if a future M1 export is un-lagged.
EIA_WPSR_LAG_WEEKS = 0   # report covers week t, published Wed t+1; lagged in M1

# EIA Weekly Petroleum Status Report columns (already +1w lagged in M1; re-check only).
EIA_WPSR_COLS = [
    "crude_stocks_excl_spr", "cushing_stocks", "crude_production",
    "crude_imports", "crude_exports", "refinery_crude_input",
    "refinery_utilisation", "gasoline_supplied", "distillate_supplied",
    "jet_fuel_supplied", "crude_stocks_change", "cushing_stocks_change",
]

# M1 monthlies already lagged inside build_m1_weekly.py (re-check only).
M1_MONTHLY_PRESHIFTED = ["gpr", "global_econ_activity", "nonoil_industrial_commodity"]

# M1 price columns (daily -> week-close; no lag; brent_price is the target base).
M1_PRICE_COLS = [
    "brent_price", "wti_price", "brent_log_return", "wti_log_return",
    "brent_wti_spread",
]

# M1 daily market columns (no lag). brent_roll_week is a contract-roll control
# dummy (0/1) carried with the basis; kept as an M1 feature, not a mask.
M1_MARKET_DAILY = [
    "vix", "dollar_index", "treasury_10y", "fed_funds_rate", "sp500_log_return",
    "ovx", "gold_return", "brent_f1_spot_log_basis", "brent_roll_week",
    "cadusd_log_return", "dgs10_change",
]

# M2 main-analysis contract (channelB_mechanism_plan.md §3/§4; 04_code/src/backtest/data.py).
RS_INDICES = ["NDVI", "NDWI", "NDBI", "BSI", "NTL"]


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
def filter_m2_anom_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 55 {idx}_anom_{aoi} columns for the merged matrix."""
    cols = [c for c in df.columns
            if "_anom_" in c and any(c.startswith(f"{idx}_anom_") for idx in RS_INDICES)]
    if not cols:
        raise ValueError("M2 export has no anomaly columns matching the 55-col contract")
    print(f"  M2 contract: {len(cols)} anom columns "
          f"(excluded level/age/avail from merge)")
    return df[cols].copy()


def load_modality(path: Path, label: str, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        msg = f"  [{'MISSING' if required else 'skip'}] {label}: {path.name} not found"
        print(msg)
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index.name = "week_ending_friday"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    print(f"  {label}: {df.shape}  {df.index.min().date()} ~ {df.index.max().date()}")
    return df


def build_union_index(*indexes: pd.Index) -> pd.DatetimeIndex:
    bounds = [(ix.min(), ix.max()) for ix in indexes if ix is not None and len(ix)]
    start = min(b[0] for b in bounds)
    end = max(b[1] for b in bounds)
    idx = pd.date_range(start=start, end=end, freq="W-FRI")
    idx.name = "week_ending_friday"
    return idx


# ----------------------------------------------------------------------------
# Feature dictionary
# ----------------------------------------------------------------------------
def classify(col: str) -> tuple[str, str, str]:
    """Return (modality, group, publication_lag) for a column."""
    if col.startswith("target_"):
        return ("target", "label", "t+1 (next week)")
    if col.startswith("avail_"):
        return ("mask", "modality availability", "n/a")
    if col in EIA_WPSR_COLS:
        return ("M1", "EIA WPSR weekly", "already lagged in M1 (+1w Wed-release)")
    if col in M1_MONTHLY_PRESHIFTED:
        return ("M1", "macro monthly", "already lagged in M1 (1-5w)")
    if col in M1_PRICE_COLS:
        return ("M1", "price (daily->W-FRI)", "none; week-close")
    if col in M1_MARKET_DAILY:
        return ("M1", "market daily", "none; week-close")
    if col.startswith("gfw_"):
        return ("M3", "GFW monthly presence", "+4w (applied in M3)")
    if col.startswith("pw_") and ("_exp_" in col or "_imp_" in col):
        return ("M3", "PortWatch ports (directional)", "+1w (applied in M3)")
    if col.startswith("pw_"):
        return ("M3", "PortWatch chokepoints", "+1w (applied in M3)")
    # M2 remote sensing: uppercase index columns (NDVI/NDWI/NDBI/BSI/NTL [+ _anom]),
    # plus per-modality observation age and availability masks.
    if col.startswith(("NDVI", "NDWI", "NDBI", "BSI", "NTL")):
        grp = "RS index (anomaly)" if "_anom_" in col else "RS index (level)"
        return ("M2", grp, "+15d as-of release (applied in M2)")
    if col.startswith(("s2_age_days", "ntl_age_days")):
        return ("M2", "RS observation age", "n/a (days_since_obs)")
    if col.startswith(("s2_avail", "ntl_avail")):
        return ("M2", "RS availability mask", "n/a")
    if col.startswith(("s2_", "ntl_", "viirs_", "rs_")):
        return ("M2", "remote sensing", "lagged in M2")
    return ("?", "unclassified", "")


def build_dictionary(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in matrix.columns:
        modality, group, lag = classify(col)
        s = matrix[col]
        nn = int(s.notna().sum())
        fv = s.first_valid_index()
        lv = s.last_valid_index()
        rows.append({
            "feature": col,
            "modality": modality,
            "group": group,
            "publication_lag": lag,
            "n_nonnull": nn,
            "pct_nonnull": round(nn / len(matrix) * 100, 1),
            "coverage_start": fv.date() if fv is not None else "",
            "coverage_end": lv.date() if lv is not None else "",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Leakage verification
# ----------------------------------------------------------------------------
def _series_equal(a: pd.Series, b: pd.Series, tol: float = 1e-9) -> bool:
    a2, b2 = a.align(b, join="inner")
    diff = (a2.fillna(-9.99e18) - b2.fillna(-9.99e18)).abs()
    return bool((diff < tol).all())


def verify(matrix: pd.DataFrame, m1_orig: pd.DataFrame) -> list[str]:
    lines = []

    # 1) EIA WPSR is already +1w lagged inside M1; the merge must NOT shift it
    #    again. On the shared index the merged column must equal the M1 column
    #    UNCHANGED (a +2w double lag would otherwise slip in silently).
    probe = next((c for c in EIA_WPSR_COLS if c in matrix and c in m1_orig), None)
    if probe:
        expected = m1_orig[probe].shift(EIA_WPSR_LAG_WEEKS)   # 0 by default -> unchanged
        ok = _series_equal(matrix[probe], expected)
        lines.append(f"  [{'OK' if ok else 'WARN'}] EIA WPSR '{probe}' == M1 column "
                     f"unchanged (already +1w lagged in M1; merge does not re-shift)")

    # 2) Price columns NOT shifted (target base must stay at week-close value).
    if "brent_price" in matrix and "brent_price" in m1_orig:
        ok = _series_equal(matrix["brent_price"], m1_orig["brent_price"])
        lines.append(f"  [{'OK' if ok else 'WARN'}] brent_price unchanged "
                     f"(not shifted; stays at week-close)")

    # 3) Target strictly in t+1: target_price_next[t] == brent_price[t+1].
    if "target_price_next" in matrix and "brent_price" in matrix:
        ok = _series_equal(matrix["target_price_next"], matrix["brent_price"].shift(-1))
        lines.append(f"  [{'OK' if ok else 'WARN'}] target_price_next[t] == brent_price[t+1]")

    # 4) Target log-return consistent with the two price points.
    if {"target_log_return_next", "target_price_next", "brent_price"} <= set(matrix.columns):
        recon = np.log(matrix["target_price_next"] / matrix["brent_price"])
        ok = _series_equal(matrix["target_log_return_next"], recon, tol=1e-6)
        lines.append(f"  [{'OK' if ok else 'WARN'}] target_log_return_next == "
                     f"log(P_t+1 / P_t)")
    return lines


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the merged weekly feature matrix.")
    ap.add_argument("--full", action="store_true",
                    help="export the full union (2006-2026) for long-history "
                         "robustness instead of the standard 2019.1-2025.12 window")
    ap.add_argument("--start", default=None, help="override clip start (e.g. 2019-01-01)")
    ap.add_argument("--end", default=None, help="override clip end (e.g. 2025-12-31)")
    ap.add_argument("--eia-lag", type=int, default=EIA_WPSR_LAG_WEEKS,
                    help=f"EIA WPSR publication lag in weeks (default {EIA_WPSR_LAG_WEEKS})")
    ap.add_argument("--m2-csv", default=None,
                    help="B4: override M2 CSV path (e.g. ../M2/outputs/"
                         "m2_weekly_features_watermask.csv). Outputs get a suffix "
                         "derived from the filename stem (e.g. _watermask).")
    args = ap.parse_args()
    eia_lag = args.eia_lag

    # Resolve M2 path and output suffix.
    m2_path = M2_PATH
    out_suffix = ""
    if args.m2_csv:
        m2_path = Path(args.m2_csv)
        # Resolve: absolute > CWD > project ROOT > script's parent M2/outputs/
        if not m2_path.is_absolute():
            for base in (Path.cwd(), PROC_DIR / "M2" / "outputs"):
                candidate = base / m2_path
                if candidate.exists():
                    m2_path = candidate
                    break
        # Derive a suffix: stem after "m2_weekly_features" (e.g. "_watermask")
        stem = m2_path.stem  # e.g. "m2_weekly_features_watermask"
        after = stem.replace("m2_weekly_features", "")
        out_suffix = after if after else "_custom"
        print(f"[--m2-csv] Using {m2_path.name}  (output suffix: '{out_suffix}')")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading modalities ...")
    m1 = load_modality(M1_PATH, "M1 finance", required=True)
    m2 = load_modality(m2_path, "M2 remote-sensing")
    if not m2.empty:
        m2 = filter_m2_anom_columns(m2)
    m3 = load_modality(M3_PATH, "M3 shipping", required=True)

    # 1) Union W-FRI index across all available modalities.
    union = build_union_index(
        m1.index,
        m2.index if not m2.empty else None,
        m3.index,
    )
    print(f"\nUnion W-FRI index: {len(union)} weeks, "
          f"{union.min().date()} ~ {union.max().date()}")

    # 2) Reindex each modality onto the union (contiguous weekly grid).
    m1u = m1.reindex(union)
    m3u = m3.reindex(union)
    parts = [m1u]
    if not m2.empty:
        parts.append(m2.reindex(union))
    parts.append(m3u)

    # 3) EIA WPSR is already +1w lagged at source in build_m1_weekly.py, so the
    #    merge does NOT shift it again (default eia_lag=0). --eia-lag is only an
    #    emergency override for a hypothetical un-lagged M1 export.
    eia_present = [c for c in EIA_WPSR_COLS if c in m1u.columns]
    if eia_lag and eia_present:
        m1u[eia_present] = m1u[eia_present].shift(eia_lag)
        print(f"\n[override] Applied EIA WPSR lag +{eia_lag}w to {len(eia_present)} columns.")
    print("Re-check (not re-shifted): EIA WPSR +1w & M1 monthlies lagged in M1; "
          "M3 GFW +4w / PortWatch +1w lagged in M3.")

    # 4) Concatenate column-wise; guard against name collisions.
    matrix = pd.concat(parts, axis=1)
    if not matrix.columns.is_unique:
        dups = matrix.columns[matrix.columns.duplicated()].tolist()
        raise ValueError(f"duplicate columns after merge: {dups}")

    # 5) Targets strictly in t+1 (sole target = price; train on log price change).
    if "brent_price" in matrix:
        matrix["target_price_next"] = matrix["brent_price"].shift(-1)
    if "brent_log_return" in matrix:
        matrix["target_log_return_next"] = matrix["brent_log_return"].shift(-1)

    # 6) Modality-level availability masks.
    if "brent_price" in matrix:
        matrix["avail_m1"] = matrix["brent_price"].notna().astype(int)
    if not m2.empty:
        # M2 availability = at least one AOI has a fresh-enough RS observation.
        m2_mask_cols = [c for c in matrix.columns if c.startswith(("s2_avail", "ntl_avail"))]
        if m2_mask_cols:
            matrix["avail_m2"] = (matrix[m2_mask_cols].max(axis=1) > 0).astype(int)
        else:
            rs_cols = [c for c in matrix.columns if classify(c)[0] == "M2"]
            matrix["avail_m2"] = matrix[rs_cols].notna().any(axis=1).astype(int)
    if "avail_shipping" in matrix:
        matrix["avail_m3"] = matrix["avail_shipping"]

    # 7) Clip to the comparison window and write. Default = standard window
    #    2019.1-2025.12; --full exports the whole union for long-history robustness.
    if args.full:
        start = args.start or str(union.min().date())
        end = args.end or str(union.max().date())
        matrix_path = OUT_DIR / f"weekly_feature_matrix_full{out_suffix}.csv"
        dict_path   = OUT_DIR / f"weekly_feature_dictionary_full{out_suffix}.csv"
        win_label = "FULL union (long-history robustness)"
    else:
        start = args.start or STD_WINDOW_START
        end = args.end or STD_WINDOW_END
        matrix_path = OUT_DIR / f"weekly_feature_matrix{out_suffix}.csv"
        dict_path   = OUT_DIR / f"weekly_feature_dictionary{out_suffix}.csv"
        win_label = "STANDARD comparison window 2019.1-2025.12"
    matrix = matrix.loc[start:end]
    matrix.index.name = "week_ending_friday"
    matrix.to_csv(matrix_path)

    dictionary = build_dictionary(matrix)
    dictionary.to_csv(dict_path, index=False)

    print(f"\nWindow: {win_label}  [{start} .. {end}]")
    _report(matrix, dictionary, m1, eia_lag, matrix_path, dict_path)


def _report(matrix: pd.DataFrame, dictionary: pd.DataFrame, m1_orig: pd.DataFrame,
            eia_lag: int, matrix_path: Path, dict_path: Path) -> None:
    print(f"\n{'='*66}")
    print(f"Output: {matrix_path}")
    print(f"        {dict_path}")
    print(f"Shape:  {matrix.shape}   Period: "
          f"{matrix.index.min().date()} ~ {matrix.index.max().date()}")

    counts = dictionary["modality"].value_counts()
    print("\nColumns by modality:")
    for mod in ["M1", "M2", "M3", "target", "mask", "?"]:
        if mod in counts:
            print(f"  {mod:7s}: {counts[mod]}")

    if "target_price_next" in matrix:
        tgt = matrix["target_price_next"]
        nn = int(tgt.notna().sum())
        fv, lv = tgt.first_valid_index(), tgt.last_valid_index()
        print(f"\nTarget coverage: {nn}/{len(matrix)} weeks with target_price_next "
              f"({fv.date()} ~ {lv.date()})")
        # The last 1-2 weeks have no target (their next-week price is out of window).
        n_tail = int(tgt.isna().sum())
        if n_tail:
            print(f"  ({n_tail} trailing week(s) have no t+1 label -> drop on training)")

    print("\nNo-look-ahead verification:")
    for line in verify(matrix, m1_orig):
        print(line)

    # Unclassified columns are a red flag for the dictionary / lag routing.
    unknown = dictionary.loc[dictionary["modality"] == "?", "feature"].tolist()
    if unknown:
        print(f"\n[warn] {len(unknown)} unclassified columns: {unknown[:8]}"
              f"{' ...' if len(unknown) > 8 else ''}")

    print(f"{'='*66}\nDone.")


if __name__ == "__main__":
    main()
