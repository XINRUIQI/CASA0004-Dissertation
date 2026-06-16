"""
Download & build the M1 "to-build" variables (feature_groups M1.json -> M1_to_build).

Fills the gap between the literature-recommended M1 set and what is already in
weekly_time_index.csv. All series are aligned to the project's Friday-ending
weekly index (2006-01 ~ 2025-12) and merged-ready.

Variables built (canonical names match feature_groups M1.json -> "M1_to_build"):
    ovx                          OVX  (FRED OVXCLS, daily -> weekly last)
    gpr                          Geopolitical Risk (Caldara-Iacoviello, monthly)
    gold_return                  weekly log-return of LBMA gold PM (FRED)
    global_econ_activity         Kilian Real Economic Activity (Dallas Fed, monthly)
    nonoil_industrial_commodity  IMF Industrial Materials price index (FRED, monthly)
    futures_spread               Brent front-month futures vs spot (yfinance + FRED)
    commodity_fx                 avg weekly %-change of CAD/USD & AUD/USD (FRED)
    dgs10_change                 weekly change of US 10Y yield (FRED DGS10)

Data sources (no API key needed):
    - FRED public CSV endpoint  https://fred.stlouisfed.org/graph/fredgraph.csv?id=ID
    - yfinance                  Brent futures (BZ=F)
    - Caldara-Iacoviello GPR     https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls
    - Dallas Fed Kilian REA      https://www.dallasfed.org/-/media/documents/research/igrea/igrea_xls.xlsx

Each source is wrapped in try/except: if one download fails the rest still run,
and the failed column is reported at the end (with a manual-download hint).

Requirements: pandas, numpy, requests, yfinance, openpyxl, xlrd
Usage:
    python "build_m1_to_build.py"

Output:
    03_data/processed/m1_to_build_weekly.csv
"""

from __future__ import annotations

import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "03_data" / "processed"
WEEKLY_INDEX_CSV = OUT_DIR / "weekly_time_index.csv"

STUDY_START = "2006-01-01"
STUDY_END = "2025-12-31"

# Conservative publication lag (in weeks) applied to monthly series so that a
# month's value only becomes usable *after* its real-world release date.
GPR_LAG_WEEKS = 1          # GPR is published at the start of the next month
MONTHLY_LAG_WEEKS = 5      # IMF commodity index / Kilian REA ~1 month release lag

HEADERS = {"User-Agent": "Mozilla/5.0 (dissertation data build)"}

HTTP_TIMEOUT = 45         # seconds per attempt (fail fast, rely on retries)
HTTP_RETRIES = 5          # attempts before giving up
HTTP_BACKOFF = 4          # seconds, multiplied by attempt number

# Track failures for the final report
_FAILED: list[tuple[str, str]] = []


# ----------------------------------------------------------------------------
# Weekly index
# ----------------------------------------------------------------------------
def get_weekly_index() -> pd.DatetimeIndex:
    """Use the existing weekly anchor if present, else a W-FRI date range."""
    if WEEKLY_INDEX_CSV.exists():
        idx = pd.read_csv(WEEKLY_INDEX_CSV, usecols=[0], parse_dates=[0]).iloc[:, 0]
        idx = pd.DatetimeIndex(idx).sort_values()
        print(f"Weekly index from {WEEKLY_INDEX_CSV.name}: {len(idx)} weeks")
        return idx
    idx = pd.date_range(STUDY_START, STUDY_END, freq="W-FRI")
    print(f"Weekly index generated (W-FRI): {len(idx)} weeks")
    return idx


# ----------------------------------------------------------------------------
# Fetch helpers
# ----------------------------------------------------------------------------
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def http_get(url: str) -> requests.Response:
    """GET with retries + backoff for flaky timeouts and 429/5xx responses."""
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
    """Daily/monthly FRED series via the public fredgraph CSV (no API key).

    Restricting to the study window with cosd/coed shrinks the response (e.g.
    DGS10 from 1962 -> 2006), which avoids the gateway 504s on long histories.
    """
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={STUDY_START}&coed={STUDY_END}"
    )
    r = http_get(url)
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col].replace(".", np.nan), errors="coerce")
    s = df.dropna(subset=[date_col]).set_index(date_col)[val_col].sort_index()
    s.name = series_id
    return s


def read_weekly_col(name: str, weekly_index: pd.DatetimeIndex) -> pd.Series:
    """Read an already-built weekly column from weekly_time_index.csv."""
    if not WEEKLY_INDEX_CSV.exists():
        raise FileNotFoundError(WEEKLY_INDEX_CSV)
    df = pd.read_csv(WEEKLY_INDEX_CSV, index_col=0, parse_dates=[0])
    if name not in df.columns:
        raise KeyError(f"{name} not in {WEEKLY_INDEX_CSV.name}")
    return df[name].reindex(weekly_index)


def fetch_yf_close(ticker: str) -> pd.Series:
    """Daily close via yfinance."""
    import yfinance as yf

    df = yf.download(
        ticker, start=STUDY_START, end=STUDY_END,
        progress=False, auto_adjust=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):          # multiindex -> single column
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def daily_to_weekly_last(s: pd.Series, weekly_index: pd.DatetimeIndex) -> pd.Series:
    """Daily -> Friday-ending weekly (last obs), reindexed to the anchor."""
    w = s.resample("W-FRI").last()
    return w.reindex(weekly_index)


def monthly_to_weekly(s: pd.Series, weekly_index: pd.DatetimeIndex,
                      lag_weeks: int) -> pd.Series:
    """Monthly -> weekly via month-end ffill + a conservative publication lag.

    ffill ensures a Friday only sees the last *completed* month; the extra
    lag shift pushes the value to after its real release date (no look-ahead).
    """
    s = s.copy()
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    w = s.reindex(weekly_index, method="ffill")
    if lag_weeks:
        w = w.shift(lag_weeks)
    return w


# ----------------------------------------------------------------------------
# Individual variable builders
# ----------------------------------------------------------------------------
def build_ovx(idx):
    # yfinance ^OVX primary (FRED daily endpoint is flaky); FRED OVXCLS fallback.
    try:
        s = fetch_yf_close("^OVX")
    except Exception:                            # noqa: BLE001
        s = fetch_fred("OVXCLS")
    return daily_to_weekly_last(s, idx).rename("ovx")


def build_gold_return(idx):
    # Try FRED LBMA Gold PM; fall back to yfinance gold futures if unavailable.
    try:
        s = fetch_fred("GOLDPMGBD228NLBM")      # LBMA Gold Price PM (USD)
    except Exception:                            # noqa: BLE001
        s = fetch_yf_close("GC=F")               # COMEX gold futures
    gold_w = daily_to_weekly_last(s, idx)
    return np.log(gold_w / gold_w.shift(1)).rename("gold_return")


def build_dgs10_change(idx):
    # Derive from the local weekly treasury_10y (already in the anchor); no download.
    try:
        dgs10_w = read_weekly_col("treasury_10y", idx)
    except Exception:                            # noqa: BLE001
        dgs10_w = daily_to_weekly_last(fetch_fred("DGS10"), idx)
    return dgs10_w.diff().rename("dgs10_change")


def build_commodity_fx(idx):
    # Primary: yfinance CADUSD=X / AUDUSD=X (1 unit of commodity ccy in USD).
    # Both rise when the commodity currency strengthens -> use +pct_change.
    try:
        cad_str = daily_to_weekly_last(fetch_yf_close("CADUSD=X"), idx).pct_change()
        aud_str = daily_to_weekly_last(fetch_yf_close("AUDUSD=X"), idx).pct_change()
    except Exception:                            # noqa: BLE001
        # Fallback FRED: DEXCAUS = CAD per USD (invert), DEXUSAL = USD per AUD
        cad_str = -daily_to_weekly_last(fetch_fred("DEXCAUS"), idx).pct_change()
        aud_str = daily_to_weekly_last(fetch_fred("DEXUSAL"), idx).pct_change()
    out = pd.concat([cad_str, aud_str], axis=1).mean(axis=1)
    return out.rename("commodity_fx")


def build_nonoil_industrial_commodity(idx):
    # IMF Global price index of Industrial Materials (monthly)
    s = fetch_fred("PINDUINDEXM")
    return monthly_to_weekly(s, idx, MONTHLY_LAG_WEEKS).rename("nonoil_industrial_commodity")


def build_futures_spread(idx):
    fut = daily_to_weekly_last(fetch_yf_close("BZ=F"), idx)       # Brent front-month
    # Spot: prefer the local weekly brent_price anchor; fallback FRED DCOILBRENTEU.
    try:
        spot = read_weekly_col("brent_price", idx)
    except Exception:                            # noqa: BLE001
        spot = daily_to_weekly_last(fetch_fred("DCOILBRENTEU"), idx)
    # log futures-spot spread (proxy; true term spread needs multiple contracts)
    return (np.log(fut) - np.log(spot)).rename("futures_spread")


def build_gpr(idx):
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    r = http_get(url)
    df = pd.read_excel(io.BytesIO(r.content))
    date_col = next(c for c in df.columns if str(c).lower() in ("month", "date"))
    gpr_col = next(c for c in df.columns if str(c).strip().upper() == "GPR")
    s = (
        df[[date_col, gpr_col]]
        .assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce")})
        .dropna(subset=[date_col])
        .set_index(date_col)[gpr_col]
        .sort_index()
    )
    return monthly_to_weekly(s, idx, GPR_LAG_WEEKS).rename("gpr")


KILIAN_URLS = [
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea_update.xlsx",
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea.xlsx",
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea_xls.xlsx",
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea.csv",
]


def build_global_econ_activity(idx):
    # Kilian Real Economic Activity index (Dallas Fed, monthly)
    content = None
    for url in KILIAN_URLS:
        try:
            content = http_get(url).content
            break
        except Exception:                        # noqa: BLE001
            continue
    if content is None:
        raise RuntimeError("all Dallas Fed Kilian REA URLs failed")
    reader = pd.read_csv if content[:4] not in (b"PK\x03\x04",) else pd.read_excel
    df = reader(io.BytesIO(content))
    # first datetime-like column = date, first numeric column after it = index value
    date_col = df.columns[0]
    val_col = df.columns[1]
    s = (
        df[[date_col, val_col]]
        .assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce")})
        .dropna(subset=[date_col])
        .set_index(date_col)[val_col]
        .apply(pd.to_numeric, errors="coerce")
        .sort_index()
    )
    return monthly_to_weekly(s, idx, MONTHLY_LAG_WEEKS).rename("global_econ_activity")


BUILDERS = {
    "ovx": build_ovx,
    "gpr": build_gpr,
    "gold_return": build_gold_return,
    "global_econ_activity": build_global_econ_activity,
    "nonoil_industrial_commodity": build_nonoil_industrial_commodity,
    "futures_spread": build_futures_spread,
    "commodity_fx": build_commodity_fx,
    "dgs10_change": build_dgs10_change,
}

# Hint for manual download if an automatic source is unreachable
MANUAL_HINT = {
    "gpr": "https://www.matteoiacoviello.com/gpr.htm  (data_gpr_export.xls, col 'GPR')",
    "global_econ_activity": "https://www.dallasfed.org/research/igrea  (Kilian REA xlsx)",
    "futures_spread": "yfinance BZ=F may be rate-limited; retry later",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = get_weekly_index()
    idx.name = "week_ending_friday"

    out_path = OUT_DIR / "m1_to_build_weekly.csv"
    # Incremental mode: reuse already-complete columns from a previous run so a
    # rerun only re-downloads the ones that failed (avoids re-stressing FRED).
    existing = None
    if out_path.exists():
        existing = pd.read_csv(out_path, index_col=0, parse_dates=[0]).reindex(idx)

    cols = {}
    for name, fn in BUILDERS.items():
        if (existing is not None and name in existing.columns
                and existing[name].notna().sum() >= 0.9 * len(idx)):
            cols[name] = existing[name]
            print(f"Building {name} ... cached  (non-null {existing[name].notna().sum()}/{len(idx)})")
            continue
        print(f"Building {name} ...", end=" ", flush=True)
        try:
            s = fn(idx)
            s = s.reindex(idx)
            cols[name] = s
            print(f"ok  (non-null {s.notna().sum()}/{len(s)})")
        except Exception as e:  # noqa: BLE001
            _FAILED.append((name, str(e)))
            print(f"FAILED ({type(e).__name__})")
        time.sleep(2)  # be polite to FRED between requests

    if not cols:
        print("\nNo variables built. Check network connectivity.")
        return

    out = pd.DataFrame(cols, index=idx)
    out.to_csv(out_path)

    print(f"\n{'='*60}")
    print(f"Output: {out_path}")
    print(f"Shape:  {out.shape}   Period: {out.index.min().date()} ~ {out.index.max().date()}")
    print(f"\nColumns ({len(out.columns)}):")
    for c in out.columns:
        nn = out[c].notna().sum()
        print(f"  {c:30s}  {nn:5d} / {len(out):5d}  ({nn/len(out)*100:5.1f}%)")

    if _FAILED:
        print(f"\n{'-'*60}\nFAILED ({len(_FAILED)}) — build manually:")
        for name, err in _FAILED:
            print(f"  {name}: {err[:80]}")
            if name in MANUAL_HINT:
                print(f"      -> {MANUAL_HINT[name]}")

    print(f"\n{'='*60}\nDone. Merge into weekly_time_index.csv on 'week_ending_friday'.")


if __name__ == "__main__":
    main()
