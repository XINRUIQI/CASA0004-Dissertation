"""
Layer 2 (PROCESSED) — one script, one table.

Merges the former three-step M1 pipeline
    build_weekly_time_index.py  +  build_m1_to_build.py  +  merge_m1_to_build.py
into a single offline-first builder. It reads the provider-organised raw layer
and writes ONE weekly feature table (no intermediate CSV, no separate merge):

    raw/01_market_financial/
        EIA/    Brent/ , WTI/ , Weekly Petroleum Status Report/   (manual .xls)
        FRED/   DGS10, VIXCLS, DTWEXBGS, DFF, PINDUINDEXM         (+ gold/ovx fallback)
        Yahoo/  ^GSPC, ^OVX, BZ=F, CADUSD=X, AUDUSD=X, GC=F
        Other/  Dallas Fed Kilian REA (igrea), Caldara-Iacoviello GPR
                                   |
                                   v
    processed/M1/outputs/m1_weekly_features.csv

Every column is a peer M1 feature on a Friday-ending weekly index (W-FRI),
2006-01 ~ 2025-12. Weekly EIA reports (released the FOLLOWING Wednesday) and
monthly inputs both carry a conservative publication lag, so a value only
becomes usable AFTER its real-world release (no look-ahead).

Default is fully OFFLINE (local raw files only). Missing raw files are skipped
with a warning rather than crashing. Optional flags:
    python build_m1_weekly.py               # offline, from local raw (default)
    python build_m1_weekly.py --online      # download a missing source on the fly
    python build_m1_weekly.py --refresh-raw # run download_m1_raw.py first, then build
    python build_m1_weekly.py --base-only   # skip the 8 derived "to-build" columns

Output:
    processed/M1/outputs/m1_weekly_features.csv

Requirements: pandas, numpy, requests, yfinance (only if --online), openpyxl, xlrd
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PY_DIR = Path(__file__).resolve().parent                   # processed/M1/py
M1_DIR = PY_DIR.parent                                      # processed/M1
DATA_DIR = M1_DIR.parents[1]                                # 03_data
RAW_DIR = DATA_DIR / "raw" / "01_market_financial"
EIA_DIR = RAW_DIR / "EIA"
FRED_DIR = RAW_DIR / "FRED"
YAHOO_DIR = RAW_DIR / "Yahoo"
OTHER_DIR = RAW_DIR / "Other"
EIA_WPSR_DIR = EIA_DIR / "Weekly Petroleum Status Report"
OUT_DIR = M1_DIR / "outputs"
OUT_PATH = OUT_DIR / "m1_weekly_features.csv"

STUDY_START = "2006-01-01"
STUDY_END = "2025-12-31"

# Conservative publication lag (weeks): a value is only usable AFTER its
# real-world release date (no look-ahead).
# EIA WPSR covers the week ending Friday T but is released the FOLLOWING
# Wednesday (T+5 days); for a Friday-close forecast it is shifted +1 week so
# week T's report only becomes visible on the next Friday (T+7).
EIA_LAG_WEEKS = 1
GPR_LAG_WEEKS = 1
MONTHLY_LAG_WEEKS = 5

HEADERS = {"User-Agent": "Mozilla/5.0 (dissertation data build)"}
HTTP_TIMEOUT = 45
HTTP_RETRIES = 5
HTTP_BACKOFF = 4
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

KILIAN_URLS = [
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea_update.xlsx",
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea.xlsx",
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea_xls.xlsx",
]

ONLINE = False                       # set by --online; allow on-the-fly downloads
_WARN: list[str] = []

# EIA Weekly Petroleum Status Report: output column -> filename keyword.
WPSR_MAP = {
    "crude_stocks_excl_spr": "commercial_crude_stocks",
    "cushing_stocks": "cushing_crude_stocks",
    "crude_production": "crude_production",
    "crude_imports": "crude_imports",
    "crude_exports": "crude_exports",
    "refinery_crude_input": "refinery_crude_input",
    "refinery_utilisation": "refinery_utilisation",
    "gasoline_supplied": "gasoline_supplied",
    "distillate_supplied": "distillate_supplied",
    "jet_fuel_supplied": "jet_fuel_supplied",
}


# ----------------------------------------------------------------------------
# File discovery
# ----------------------------------------------------------------------------
def _find(folder: Path, *substrings: str, suffixes=(".csv", ".xls", ".xlsx", ".dta")) -> Path | None:
    """First data file in `folder` whose lowercase name contains all substrings."""
    if not folder.exists():
        return None
    subs = [s.lower() for s in substrings]
    hits = sorted(
        p for p in folder.glob("*")
        if p.suffix.lower() in suffixes and all(s in p.name.lower() for s in subs)
    )
    return hits[0] if hits else None


def _find_eia_daily(keyword: str) -> Path | None:
    cands = sorted(
        p for p in EIA_DIR.rglob("*.xls")
        if keyword in p.name.lower() and "daily" in p.name.lower()
    )
    return cands[0] if cands else None


# ----------------------------------------------------------------------------
# Raw-file parsers
# ----------------------------------------------------------------------------
def read_eia_xls(path: Path, value_name: str) -> pd.DataFrame:
    """EIA .xls: sheet 'Data 1', rows 3+ are date|value."""
    df = pd.read_excel(path, sheet_name="Data 1", header=None, skiprows=3)
    df.columns = ["date", value_name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df.dropna(subset=["date"]).set_index("date").sort_index()


def parse_two_col(path: Path) -> pd.Series:
    """Generic 2-column (date, value): FRED CSV / yfinance CSV."""
    df = pd.read_csv(path)
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(
        df[val_col].astype(str).replace(".", np.nan), errors="coerce"
    )
    return df.dropna(subset=[date_col]).set_index(date_col)[val_col].sort_index()


def parse_gpr(path: Path) -> pd.Series:
    """GPR: Stata .dta or .xls with a month/date column + 'GPR' column."""
    df = pd.read_stata(path) if path.suffix.lower() == ".dta" else pd.read_excel(path)
    date_col = next(c for c in df.columns if str(c).lower() in ("month", "date"))
    gpr_col = next(c for c in df.columns if str(c).strip().upper() == "GPR")
    return (
        df[[date_col, gpr_col]]
        .assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce")})
        .dropna(subset=[date_col]).set_index(date_col)[gpr_col].sort_index()
    )


def parse_kilian(path: Path) -> pd.Series:
    """Kilian REA: first col = date, second col = index value."""
    df = pd.read_excel(path)
    date_col, val_col = df.columns[0], df.columns[1]
    return (
        df[[date_col, val_col]]
        .assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce")})
        .dropna(subset=[date_col]).set_index(date_col)[val_col]
        .apply(pd.to_numeric, errors="coerce").sort_index()
    )


# ----------------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------------
def daily_to_weekly_last(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    return s.resample("W-FRI").last().reindex(idx)


def weekly_eia_to_friday(df: pd.DataFrame, idx: pd.DatetimeIndex,
                         lag_weeks: int = 0) -> pd.DataFrame:
    """Align EIA weekly data to its report Friday, reindex, then publication-lag.

    EIA WPSR covers the week ending Friday T but is only released the FOLLOWING
    Wednesday (T+5 days). For a Friday-close forecast, week T's value must be
    shifted forward by `lag_weeks` (=1) so it only becomes visible on the next
    Friday (T+7), strictly AFTER its real release (no look-ahead).
    """
    df = df.copy()
    df.index = df.index + pd.to_timedelta((4 - df.index.dayofweek) % 7, unit="D")
    df = df[~df.index.duplicated(keep="last")]
    out = df.reindex(idx)
    return out.shift(lag_weeks) if lag_weeks else out


def monthly_to_weekly(s: pd.Series, idx: pd.DatetimeIndex, lag_weeks: int) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    w = s.reindex(idx, method="ffill")
    return w.shift(lag_weeks) if lag_weeks else w


# ----------------------------------------------------------------------------
# Online fallbacks (only with --online)
# ----------------------------------------------------------------------------
def http_get(url: str):
    import requests
    last = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
            if r.status_code in RETRY_STATUS:
                last = requests.HTTPError(f"{r.status_code} for {url}")
                time.sleep(HTTP_BACKOFF * attempt)
                continue
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            time.sleep(HTTP_BACKOFF * attempt)
    raise last  # type: ignore[misc]


def fetch_fred(series_id: str) -> pd.Series:
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd={STUDY_START}&coed={STUDY_END}")
    df = pd.read_csv(io.StringIO(http_get(url).text))
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce")
    return df.dropna(subset=[date_col]).set_index(date_col)[val_col].sort_index()


def fetch_yf_close(ticker: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=STUDY_START, end=STUDY_END,
                     progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def _warn(msg: str) -> None:
    _WARN.append(msg)
    print(f"  [warn] {msg}")


# ----------------------------------------------------------------------------
# Base block: prices, EIA weekly, FRED macro, Yahoo S&P 500
# ----------------------------------------------------------------------------
def build_base(idx: pd.DatetimeIndex) -> pd.DataFrame:
    w = pd.DataFrame(index=idx)

    # --- prices (EIA Brent / WTI daily) -------------------------------------
    brent_path = _find_eia_daily("brent")
    wti_path = _find_eia_daily("wti")
    if brent_path is not None:
        w["brent_price"] = daily_to_weekly_last(
            read_eia_xls(brent_path, "brent_price")["brent_price"], idx)
    else:
        _warn("EIA Brent daily .xls not found; brent_price/target columns will be NaN")
    if wti_path is not None:
        w["wti_price"] = daily_to_weekly_last(
            read_eia_xls(wti_path, "wti_price")["wti_price"], idx)
    else:
        _warn("EIA WTI daily .xls not found; wti_price will be NaN")

    if "brent_price" in w:
        # Sole research target = next-week price P_{t+1}. The model is trained on
        # the weekly log price change r_{t+1}=log(P_{t+1}/P_t) (a learning-friendly
        # internal representation, NOT a second target) and the price is then
        # reconstructed as P_hat = P_t * exp(r_hat). Direction/return are AUXILIARY
        # metrics derived from the predicted price, not separate prediction tasks.
        # No volatility columns: volatility is neither predicted (no volatility
        # forecasting) nor needed here as a feature -- implied volatility is
        # already covered by ovx / vix (the literature-preferred vol features).
        assert (w["brent_price"].dropna() > 0).all(), "negative weekly Brent price: log-return undefined"
        w["brent_log_return"] = np.log(w["brent_price"] / w["brent_price"].shift(1))  # weekly log-return; its next-week value r_{t+1} is the training target
    if "wti_price" in w:
        assert (w["wti_price"].dropna() > 0).all(), "negative weekly WTI price: log-return undefined"
        w["wti_log_return"] = np.log(w["wti_price"] / w["wti_price"].shift(1))  # weekly log-return, same log convention as Brent (not ×100)
    if "brent_price" in w and "wti_price" in w:
        w["brent_wti_spread"] = w["brent_price"] - w["wti_price"]

    # --- EIA Weekly Petroleum Status Report ---------------------------------
    for col, kw in WPSR_MAP.items():
        path = _find(EIA_WPSR_DIR, kw) if EIA_WPSR_DIR.exists() else None
        if path is None:
            _warn(f"EIA WPSR file for '{col}' (*{kw}*) not found; column skipped")
            continue
        w[col] = weekly_eia_to_friday(read_eia_xls(path, col), idx, EIA_LAG_WEEKS)[col]

    if "crude_stocks_excl_spr" in w:
        w["crude_stocks_change"] = w["crude_stocks_excl_spr"].diff()
    if "cushing_stocks" in w:
        w["cushing_stocks_change"] = w["cushing_stocks"].diff()
    if "crude_imports" in w and "crude_exports" in w:
        w["net_crude_trade"] = w["crude_imports"] - w["crude_exports"]

    # --- FRED macro daily ---------------------------------------------------
    fred_daily = {
        "vix": ("VIXCLS", "VIXCLS"),
        "dollar_index": ("DTWEXBGS", "DTWEXBGS"),
        "treasury_10y": ("DGS10", "DGS10"),
        "fed_funds_rate": ("DFF", "DFF"),
    }
    for col, (token, series_id) in fred_daily.items():
        path = _find(FRED_DIR, token)
        if path is not None:
            w[col] = daily_to_weekly_last(parse_two_col(path), idx)
        elif ONLINE:
            try:
                w[col] = daily_to_weekly_last(fetch_fred(series_id), idx)
            except Exception as e:                   # noqa: BLE001
                _warn(f"FRED {series_id} online fetch failed ({type(e).__name__}); {col} NaN")
        else:
            _warn(f"FRED file for '{col}' (*{token}*) not found; column skipped")

    # --- Yahoo S&P 500 ------------------------------------------------------
    sp_path = _find(YAHOO_DIR, "sp500")
    if sp_path is not None:
        w["sp500"] = daily_to_weekly_last(parse_two_col(sp_path), idx)
    elif ONLINE:
        try:
            w["sp500"] = daily_to_weekly_last(fetch_yf_close("^GSPC"), idx)
        except Exception as e:                       # noqa: BLE001
            _warn(f"Yahoo ^GSPC online fetch failed ({type(e).__name__}); sp500 NaN")
    else:
        _warn("Yahoo sp500 file not found; sp500 skipped")
    if "sp500" in w:
        w["sp500_return_pct"] = w["sp500"].pct_change() * 100

    return w


# ----------------------------------------------------------------------------
# To-build block: the 8 derived M1 variables
# ----------------------------------------------------------------------------
def build_ovx(idx, base):
    path = _find(YAHOO_DIR, "OVX") or _find(FRED_DIR, "OVXCLS")
    if path is not None:
        return daily_to_weekly_last(parse_two_col(path), idx)
    if ONLINE:
        try:
            s = fetch_yf_close("^OVX")
        except Exception:                            # noqa: BLE001
            s = fetch_fred("OVXCLS")
        return daily_to_weekly_last(s, idx)
    raise FileNotFoundError("ovx (Yahoo ^OVX / FRED OVXCLS)")


def build_gold_return(idx, base):
    path = _find(FRED_DIR, "GOLDPMGBD") or _find(YAHOO_DIR, "GCF")
    if path is not None:
        s = parse_two_col(path)
    elif ONLINE:
        try:
            s = fetch_fred("GOLDPMGBD228NLBM")
        except Exception:                            # noqa: BLE001
            s = fetch_yf_close("GC=F")
    else:
        raise FileNotFoundError("gold (FRED GOLDPMGBD228NLBM / Yahoo GC=F)")
    gold_w = daily_to_weekly_last(s, idx)
    return np.log(gold_w / gold_w.shift(1))


def build_dgs10_change(idx, base):
    if "treasury_10y" in base:
        return base["treasury_10y"].diff()
    if ONLINE:
        return daily_to_weekly_last(fetch_fred("DGS10"), idx).diff()
    raise FileNotFoundError("dgs10_change (needs treasury_10y)")


def build_commodity_fx(idx, base):
    cad_p, aud_p = _find(YAHOO_DIR, "CADUSD"), _find(YAHOO_DIR, "AUDUSD")
    if cad_p is not None and aud_p is not None:
        cad = daily_to_weekly_last(parse_two_col(cad_p), idx).pct_change()
        aud = daily_to_weekly_last(parse_two_col(aud_p), idx).pct_change()
    elif ONLINE:
        try:
            cad = daily_to_weekly_last(fetch_yf_close("CADUSD=X"), idx).pct_change()
            aud = daily_to_weekly_last(fetch_yf_close("AUDUSD=X"), idx).pct_change()
        except Exception:                            # noqa: BLE001
            cad = -daily_to_weekly_last(fetch_fred("DEXCAUS"), idx).pct_change()
            aud = daily_to_weekly_last(fetch_fred("DEXUSAL"), idx).pct_change()
    else:
        raise FileNotFoundError("commodity_fx (Yahoo CADUSD=X / AUDUSD=X)")
    return pd.concat([cad, aud], axis=1).mean(axis=1)


def build_nonoil_industrial_commodity(idx, base):
    path = _find(FRED_DIR, "PINDUINDEXM")
    if path is not None:
        s = parse_two_col(path)
    elif ONLINE:
        s = fetch_fred("PINDUINDEXM")
    else:
        raise FileNotFoundError("nonoil_industrial_commodity (FRED PINDUINDEXM)")
    return monthly_to_weekly(s, idx, MONTHLY_LAG_WEEKS)


def build_futures_spread(idx, base):
    path = _find(YAHOO_DIR, "BZF")
    if path is not None:
        fut = daily_to_weekly_last(parse_two_col(path), idx)
    elif ONLINE:
        fut = daily_to_weekly_last(fetch_yf_close("BZ=F"), idx)
    else:
        raise FileNotFoundError("futures_spread (Yahoo BZ=F)")
    if "brent_price" in base:
        spot = base["brent_price"]
    elif ONLINE:
        spot = daily_to_weekly_last(fetch_fred("DCOILBRENTEU"), idx)
    else:
        raise FileNotFoundError("futures_spread spot leg (needs brent_price)")
    return np.log(fut) - np.log(spot)


def build_gpr(idx, base):
    path = _find(OTHER_DIR, "data_gpr_export")
    if path is not None:
        s = parse_gpr(path)
    elif ONLINE:
        url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
        df = pd.read_excel(io.BytesIO(http_get(url).content))
        date_col = next(c for c in df.columns if str(c).lower() in ("month", "date"))
        gpr_col = next(c for c in df.columns if str(c).strip().upper() == "GPR")
        s = (df[[date_col, gpr_col]]
             .assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce")})
             .dropna(subset=[date_col]).set_index(date_col)[gpr_col].sort_index())
    else:
        raise FileNotFoundError("gpr (Other/data_gpr_export.*)")
    return monthly_to_weekly(s, idx, GPR_LAG_WEEKS)


def build_global_econ_activity(idx, base):
    path = _find(OTHER_DIR, "igrea")
    if path is not None:
        s = parse_kilian(path)
    elif ONLINE:
        content = None
        for url in KILIAN_URLS:
            try:
                content = http_get(url).content
                break
            except Exception:                        # noqa: BLE001
                continue
        if content is None:
            raise RuntimeError("all Dallas Fed Kilian REA URLs failed")
        df = pd.read_excel(io.BytesIO(content))
        date_col, val_col = df.columns[0], df.columns[1]
        s = (df[[date_col, val_col]]
             .assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce")})
             .dropna(subset=[date_col]).set_index(date_col)[val_col]
             .apply(pd.to_numeric, errors="coerce").sort_index())
    else:
        raise FileNotFoundError("global_econ_activity (Other/igrea)")
    return monthly_to_weekly(s, idx, MONTHLY_LAG_WEEKS)


TO_BUILD = {
    "ovx": build_ovx,
    "gpr": build_gpr,
    "gold_return": build_gold_return,
    "global_econ_activity": build_global_econ_activity,
    "nonoil_industrial_commodity": build_nonoil_industrial_commodity,
    "futures_spread": build_futures_spread,
    "commodity_fx": build_commodity_fx,
    "dgs10_change": build_dgs10_change,
}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    global ONLINE
    ap = argparse.ArgumentParser(description="Build the single M1 weekly feature table.")
    ap.add_argument("--online", action="store_true",
                    help="download a missing source on the fly (FRED/Yahoo)")
    ap.add_argument("--refresh-raw", action="store_true",
                    help="run download_m1_raw.py before building")
    ap.add_argument("--base-only", action="store_true",
                    help="skip the 8 derived 'to-build' columns")
    args = ap.parse_args()
    ONLINE = args.online

    if args.refresh_raw:
        import subprocess
        print("Refreshing raw layer via download_m1_raw.py ...")
        subprocess.run([sys.executable, str(RAW_DIR / "download_m1_raw.py")], check=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(STUDY_START, STUDY_END, freq="W-FRI")
    idx.name = "week_ending_friday"
    mode = "ONLINE (local-first, download fallback)" if ONLINE else "OFFLINE (local raw only)"
    print(f"Mode: {mode}\nRaw dir: {RAW_DIR}\n")

    print("Building base block (EIA / FRED / Yahoo) ...")
    weekly = build_base(idx)

    if not args.base_only:
        print("\nBuilding to-build block (8 derived M1 variables) ...")
        for name, fn in TO_BUILD.items():
            print(f"  {name} ...", end=" ", flush=True)
            try:
                s = fn(idx, weekly).reindex(idx)
                weekly[name] = s
                print(f"ok  (non-null {s.notna().sum()}/{len(s)})")
            except Exception as e:                   # noqa: BLE001
                print(f"SKIPPED ({type(e).__name__}: {str(e)[:60]})")
                _WARN.append(f"{name}: {str(e)[:80]}")

    # --- modality availability flags ----------------------------------------
    if "brent_price" in weekly:
        weekly["avail_market"] = weekly["brent_price"].notna().astype(int)
    if "crude_stocks_excl_spr" in weekly:
        weekly["avail_eia_weekly"] = weekly["crude_stocks_excl_spr"].notna().astype(int)
    if "sp500" in weekly:
        weekly["avail_sp500"] = weekly["sp500"].notna().astype(int)
    if "dollar_index" in weekly:
        weekly["avail_dollar_index"] = weekly["dollar_index"].notna().astype(int)

    weekly = weekly.loc[STUDY_START:STUDY_END]
    weekly.index.name = "week_ending_friday"
    weekly.to_csv(OUT_PATH)

    # --- summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Output: {OUT_PATH}")
    print(f"Shape:  {weekly.shape}   Period: "
          f"{weekly.index.min().date()} ~ {weekly.index.max().date()}")
    print(f"\nColumns ({len(weekly.columns)}):")
    for c in weekly.columns:
        nn = weekly[c].notna().sum()
        print(f"  {c:32s}  {nn:5d} / {len(weekly):5d}  ({nn/len(weekly)*100:5.1f}%)")

    if _WARN:
        print(f"\n{'-'*60}\nWarnings ({len(_WARN)}):")
        for m in _WARN:
            print(f"  - {m}")

    print(f"\n{'='*60}\nDone.")


if __name__ == "__main__":
    main()
