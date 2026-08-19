"""
Layer 1 (RAW) — single entry point that assembles the M1 raw layer + manifest.

Raw files are organised BY PROVIDER (the host the file is actually downloaded
from), so the folder name always answers "where did this byte stream come from":

    01_market_financial/
    ├── EIA/      Brent/ , WTI/ , Weekly Petroleum Status Report/  (manual .xls — register only)
    ├── FRED/     DGS10, VIXCLS, DTWEXBGS, DFF, PINDUINDEXM, (gold/ovx fallbacks)
    ├── Yahoo/    ^GSPC, ^OVX, BZ=F, CADUSD=X, AUDUSD=X, GC=F
    ├── Other/    Dallas Fed Kilian REA (igrea), Caldara-Iacoviello GPR
    ├── download_m1_raw.py                                         (this file)
    └── manifest.csv                                               (one unified audit log)

Two task groups:
    task_eia_register()  register-only: the EIA .xls are exported manually from
                         the EIA website, so we DON'T download them — we just
                         record them in the manifest for auditing.
    task_auto()          download (or reuse cached) every automatable source and
                         drop each file into its PROVIDER folder. Variables with a
                         fallback (gold FRED->Yahoo, ovx Yahoo->FRED) land in the
                         folder of whichever host actually answered.

Everything is logged to ONE root `manifest.csv` (variable, raw_file, kind,
category=provider, source, identifier, URL, frequency, unit, download time,
rows, coverage, SHA-256, notes) so the processed builder reads local files and
stays offline-reproducible.

Requirements: pandas, numpy, requests, yfinance, openpyxl, xlrd
Usage:
    python "download_m1_raw.py"            # eia register + auto download (all providers)
    python "download_m1_raw.py" --eia      # only re-register EIA/ into the manifest
    python "download_m1_raw.py" --download  # only the auto-download providers
    python "download_m1_raw.py" --force    # re-download even if a raw file exists
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths — one folder per provider
# ----------------------------------------------------------------------------
RAW_BASE = Path(__file__).resolve().parent                 # .../01_market_financial
EIA_DIR = RAW_BASE / "EIA"
FRED_DIR = RAW_BASE / "FRED"
YAHOO_DIR = RAW_BASE / "Yahoo"
OTHER_DIR = RAW_BASE / "Other"
MANIFEST_CSV = RAW_BASE / "manifest.csv"
LEGACY_GPR = RAW_BASE / "data_gpr_export.dta"              # pre-migration GPR location

STUDY_START = "2006-01-01"
STUDY_END = "2025-12-31"
YF_END = "2026-01-01"                                       # yfinance end is exclusive

HEADERS = {"User-Agent": "Mozilla/5.0 (dissertation data build)"}
HTTP_TIMEOUT = 45
HTTP_RETRIES = 5
HTTP_BACKOFF = 4
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

# Canonical download filenames (provider prefix == destination folder).
F_DGS10 = "FRED_DGS10_daily.csv"
F_VIX = "FRED_VIXCLS_daily.csv"
F_DXY = "FRED_DTWEXBGS_daily.csv"
F_DFF = "FRED_DFF_daily.csv"
F_PINDU = "FRED_PINDUINDEXM_monthly.csv"
F_GOLD_FRED = "FRED_GOLDPMGBD228NLBM_daily.csv"
F_OVX_FRED = "FRED_OVXCLS_daily.csv"

F_SP500 = "Yahoo_sp500_daily.csv"
F_OVX = "Yahoo_OVX_daily.csv"
F_BZF = "Yahoo_BZF_daily.csv"
F_CAD = "Yahoo_CADUSD_daily.csv"
F_AUD = "Yahoo_AUDUSD_daily.csv"
F_GOLD_YF = "Yahoo_GCF_daily.csv"

F_KILIAN = "DallasFed_igrea_monthly.xlsx"
F_GPR = "data_gpr_export.dta"                               # monthly GPR (fallback)
F_GPRD = "data_gpr_daily_recent.xls"                        # daily GPRD (primary, 2026-07)
GPRD_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

KILIAN_URLS = [
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea_update.xlsx",
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea.xlsx",
    "https://www.dallasfed.org/-/media/documents/research/igrea/igrea_xls.xlsx",
]

# Plain single-source FRED series -> FRED/ (variable, id, find-glob, canonical,
#                                           frequency, unit, note).
FRED_SERIES = [
    ("treasury_10y", "DGS10", "*DGS10*.csv", F_DGS10, "daily", "percent",
     "10Y Treasury yield; dgs10_change = weekly first-difference in build"),
    ("vix", "VIXCLS", "*VIXCLS*.csv", F_VIX, "daily", "index points",
     "CBOE VIX close"),
    ("dollar_index", "DTWEXBGS", "*DTWEXBGS*.csv", F_DXY, "daily", "index",
     "broad trade-weighted USD index"),
    ("fed_funds_rate", "DFF", "*DFF*.csv", F_DFF, "daily", "percent",
     "effective federal funds rate"),
    ("nonoil_industrial_commodity", "PINDUINDEXM", "*PINDUINDEXM*.csv", F_PINDU,
     "monthly", "index", "IMF industrial materials price index (via FRED)"),
]

# EIA register-only: filename keyword -> variable. Sub-area inferred from subdir.
EIA_VAR_BY_KEY = {
    "brent_spot_price": "brent_price",
    "WTI_cushing": "wti_price",
    "commercial_crude_stocks": "crude_stocks_excl_spr",
    "cushing_crude_stocks": "cushing_stocks",
    "crude_production": "crude_production",
    "crude_imports": "crude_imports",
    "crude_exports": "crude_exports",
    "refinery_crude_input": "refinery_crude_input",
    "refinery_utilisation": "refinery_utilisation",
    "gasoline_supplied": "gasoline_supplied",
    "distillate_supplied": "distillate_supplied",
    "jet_fuel_supplied": "jet_fuel_supplied",
}

_MANIFEST_ROWS: list[dict] = []


# ----------------------------------------------------------------------------
# HTTP + fetch helpers
# ----------------------------------------------------------------------------
def http_get(url: str) -> requests.Response:
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


def fred_csv_text(series_id: str) -> str:
    """Original FRED fredgraph CSV (date col + value col), windowed to the study."""
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={STUDY_START}&coed={STUDY_END}"
    )
    return http_get(url).text


def yf_close_csv_text(ticker: str, auto_adjust: bool = False,
                      value_name: str = "close") -> str:
    """yfinance daily close serialised to a canonical 2-column CSV."""
    import yfinance as yf

    df = yf.download(ticker, start=STUDY_START, end=YF_END,
                     progress=False, auto_adjust=auto_adjust)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"observation_date": pd.to_datetime(close.index),
                        value_name: close.values})
    out = out.dropna().sort_values("observation_date")
    out["observation_date"] = out["observation_date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False)


# ----------------------------------------------------------------------------
# Manifest helpers
# ----------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_for_coverage(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".xls":
        # EIA convention: sheet 'Data 1', data starts on row 4 (date | value)
        try:
            df = pd.read_excel(path, sheet_name="Data 1", header=None, skiprows=3)
            df.columns = ["date"] + [f"v{i}" for i in range(1, df.shape[1])]
            return df
        except Exception:                            # noqa: BLE001
            df = pd.read_excel(path)
            # non-EIA xls (e.g. GPRD: integer 'DAY' first col + a real 'date'
            # col) -> prefer the explicit 'date' column so coverage is correct.
            if "date" in df.columns:
                df = df[["date"] + [c for c in df.columns if c != "date"]]
            return df
    return pd.read_csv(path)


def _coverage(path: Path) -> tuple[int, str, str]:
    """Best-effort (n_rows, start, end); never fatal."""
    try:
        df = _read_for_coverage(path)
        d = pd.to_datetime(df[df.columns[0]], errors="coerce").dropna()
        if len(d) == 0:
            return len(df), "", ""
        return len(df), str(d.min().date()), str(d.max().date())
    except Exception:                                # noqa: BLE001
        return -1, "", ""


def reg(variable: str, path: Path | None, kind: str, category: str, source: str,
        series_id: str, url: str, frequency: str, unit: str, notes: str) -> None:
    """Append one manifest row. `path` may be None for local-derived rows."""
    if path is not None and path.exists():
        rel = path.relative_to(RAW_BASE).as_posix()
        n, c0, c1 = _coverage(path)
        sha = _sha256(path)
    else:
        rel, n, c0, c1, sha = "", -1, "", "", ""
    _MANIFEST_ROWS.append({
        "variable": variable,
        "raw_file": rel,
        "kind": kind,                # downloaded | manual | local
        "category": category,        # eia | fred | yahoo | other | local
        "source": source,
        "series_id_or_ticker": series_id,
        "source_url": url,
        "frequency": frequency,
        "native_unit": unit,
        "download_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_rows": n,
        "coverage_start": c0,
        "coverage_end": c1,
        "sha256": sha,
        "notes": notes,
    })


def _find(folder: Path, glob_pat: str) -> Path | None:
    """First data file in `folder` matching `glob_pat` (ignores .DS_Store)."""
    hits = sorted(
        p for p in folder.glob(glob_pat)
        if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".dta")
    )
    return hits[0] if hits else None


# ----------------------------------------------------------------------------
# FRED/ provider downloads
# ----------------------------------------------------------------------------
def dl_fred_series(variable: str, series_id: str, glob_pat: str, canonical: str,
                   frequency: str, unit: str, note: str, force: bool) -> None:
    existing = _find(FRED_DIR, glob_pat)
    if existing is not None and not force:
        reg(variable, existing, "downloaded", "fred", "FRED", series_id,
            f"https://fred.stlouisfed.org/series/{series_id}", frequency, unit,
            f"cached ({existing.name})")
        print(f"  {variable}: cached ({existing.name})")
        return
    dest = FRED_DIR / canonical
    dest.write_text(fred_csv_text(series_id))
    reg(variable, dest, "downloaded", "fred", "FRED", series_id,
        f"https://fred.stlouisfed.org/series/{series_id}", frequency, unit, note)
    print(f"  {variable}: saved FRED/{dest.name}")


# ----------------------------------------------------------------------------
# Yahoo/ provider downloads
# ----------------------------------------------------------------------------
def dl_yahoo_sp500(force: bool) -> None:
    existing = _find(YAHOO_DIR, "*sp500*.csv")
    if existing is not None and not force:
        reg("sp500_log_return", existing, "downloaded", "yahoo", "Yahoo Finance", "^GSPC",
            "https://finance.yahoo.com/quote/%5EGSPC", "daily", "index points",
            f"cached ({existing.name}); build outputs sp500_log_return only (level dropped 2026-07)")
        print(f"  sp500_log_return: cached ({existing.name})")
        return
    dest = YAHOO_DIR / F_SP500
    dest.write_text(yf_close_csv_text("^GSPC", auto_adjust=True, value_name="sp500"))
    reg("sp500_log_return", dest, "downloaded", "yahoo", "Yahoo Finance", "^GSPC",
        "https://finance.yahoo.com/quote/%5EGSPC", "daily", "index points",
        "auto-adjusted close; build outputs sp500_log_return only (level dropped 2026-07)")
    print(f"  sp500_log_return: saved Yahoo/{dest.name}")


def dl_yahoo_futures(force: bool) -> None:
    existing = _find(YAHOO_DIR, "*BZF*.csv")
    if existing is not None and not force:
        reg("brent_f1_spot_log_basis", existing, "downloaded", "yahoo", "Yahoo Finance",
            "BZ=F", "https://finance.yahoo.com/quote/BZ%3DF", "daily", "USD/bbl",
            f"cached ({existing.name}); renamed futures_spread->brent_f1_spot_log_basis (2026-07); unadjusted front-month leg; spot leg = local brent_price")
        print(f"  brent_f1_spot_log_basis: cached ({existing.name})")
        return
    dest = YAHOO_DIR / F_BZF
    dest.write_text(yf_close_csv_text("BZ=F"))
    reg("brent_f1_spot_log_basis", dest, "downloaded", "yahoo", "Yahoo Finance", "BZ=F",
        "https://finance.yahoo.com/quote/BZ%3DF", "daily", "USD/bbl",
        "Brent front-month futures (unadjusted); basis = log(fut) - log(spot), spot = local brent_price")
    print(f"  brent_f1_spot_log_basis: saved Yahoo/{dest.name}")


def dl_yahoo_fx(force: bool) -> None:
    # CAD is the build feature (cadusd_log_return). AUD is still downloaded but NOT
    # used by build since 2026-07 (AUD leg dropped from the old commodity_fx); kept
    # as a commodity-currency robustness option -- see research diary 2026-07-03.
    fx = (
        ("CADUSD=X", F_CAD, "*CADUSD*.csv", "cadusd_log_return",
         "cadusd_log_return = weekly log-return of CAD/USD (2026-07; replaced commodity_fx mean(CAD,AUD))"),
        ("AUDUSD=X", F_AUD, "*AUDUSD*.csv", "audusd_log_return",
         "AUD/USD; NOT used by build since 2026-07 (AUD leg dropped); raw kept for robustness only"),
    )
    for ticker, fname, glob_pat, variable, note in fx:
        existing = _find(YAHOO_DIR, glob_pat)
        if existing is not None and not force:
            dest = existing
            print(f"  {variable}: cached ({existing.name})")
        else:
            dest = YAHOO_DIR / fname
            dest.write_text(yf_close_csv_text(ticker))
            print(f"  {variable}: saved Yahoo/{dest.name}")
        reg(variable, dest, "downloaded", "yahoo", "Yahoo Finance", ticker,
            f"https://finance.yahoo.com/quote/{ticker}", "daily", "USD per 1 ccy",
            note)


# ----------------------------------------------------------------------------
# Cross-provider downloads (primary host + fallback host)
# ----------------------------------------------------------------------------
def task_ovx(force: bool) -> None:
    """OVX: Yahoo ^OVX primary -> Yahoo/, FRED OVXCLS fallback -> FRED/."""
    y, f = _find(YAHOO_DIR, "*OVX*.csv"), _find(FRED_DIR, "*OVXCLS*.csv")
    if (y or f) and not force:
        used, cat, src = (y, "yahoo", "Yahoo Finance") if y else (f, "fred", "FRED")
        reg("ovx", used, "downloaded", cat, src, "^OVX / OVXCLS",
            "https://finance.yahoo.com/quote/%5EOVX", "daily", "index points",
            f"cached ({used.name}); oil implied volatility")
        print(f"  ovx: cached ({used.name})")
        return
    try:
        dest = YAHOO_DIR / F_OVX
        dest.write_text(yf_close_csv_text("^OVX"))
        reg("ovx", dest, "downloaded", "yahoo", "Yahoo Finance", "^OVX",
            "https://finance.yahoo.com/quote/%5EOVX", "daily", "index points",
            "primary source")
        print(f"  ovx: saved Yahoo/{dest.name}")
    except Exception as e:                            # noqa: BLE001
        dest = FRED_DIR / F_OVX_FRED
        dest.write_text(fred_csv_text("OVXCLS"))
        reg("ovx", dest, "downloaded", "fred", "FRED", "OVXCLS",
            "https://fred.stlouisfed.org/series/OVXCLS", "daily", "index points",
            f"yfinance failed ({type(e).__name__}); FRED fallback")
        print(f"  ovx: saved FRED/{dest.name} (FRED fallback)")


def task_gold(force: bool) -> None:
    """Gold: FRED LBMA primary -> FRED/, Yahoo GC=F fallback -> Yahoo/."""
    f, y = _find(FRED_DIR, "*GOLDPMGBD*.csv"), _find(YAHOO_DIR, "*GCF*.csv")
    if (f or y) and not force:
        used, cat, src = (f, "fred", "FRED (LBMA)") if f else (y, "yahoo", "Yahoo Finance")
        reg("gold_return", used, "downloaded", cat, src, "GOLDPMGBD228NLBM / GC=F",
            "https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM", "daily", "USD/oz",
            f"cached ({used.name}); gold_return = weekly log-return in build")
        print(f"  gold_return: cached ({used.name})")
        return
    try:
        dest = FRED_DIR / F_GOLD_FRED
        dest.write_text(fred_csv_text("GOLDPMGBD228NLBM"))
        reg("gold_return", dest, "downloaded", "fred", "FRED (LBMA)",
            "GOLDPMGBD228NLBM", "https://fred.stlouisfed.org/series/GOLDPMGBD228NLBM",
            "daily", "USD/oz", "level series; gold_return = weekly log-return in build")
        print(f"  gold_return: saved FRED/{dest.name}")
    except Exception as e:                            # noqa: BLE001
        dest = YAHOO_DIR / F_GOLD_YF
        dest.write_text(yf_close_csv_text("GC=F"))
        reg("gold_return", dest, "downloaded", "yahoo", "Yahoo Finance", "GC=F",
            "https://finance.yahoo.com/quote/GC%3DF", "daily", "USD/oz",
            f"FRED failed ({type(e).__name__}); COMEX gold futures fallback")
        print(f"  gold_return: saved Yahoo/{dest.name} (yfinance fallback)")


# ----------------------------------------------------------------------------
# Other/ provider downloads (Dallas Fed, Caldara-Iacoviello)
# ----------------------------------------------------------------------------
def task_kilian(force: bool) -> None:
    existing = _find(OTHER_DIR, "*igrea*")
    if existing is not None and not force:
        reg("global_econ_activity", existing, "downloaded", "other", "Dallas Fed",
            "igrea", "https://www.dallasfed.org/research/igrea", "monthly", "index",
            f"cached ({existing.name}); Kilian real economic activity")
        print(f"  global_econ_activity: cached ({existing.name})")
        return
    content, used_url = None, ""
    for url in KILIAN_URLS:
        try:
            content = http_get(url).content
            used_url = url
            break
        except Exception:                            # noqa: BLE001
            continue
    if content is None:
        raise RuntimeError("all Dallas Fed Kilian REA URLs failed")
    dest = OTHER_DIR / F_KILIAN
    dest.write_bytes(content)
    reg("global_econ_activity", dest, "downloaded", "other", "Dallas Fed", "igrea",
        used_url, "monthly", "index", "Kilian index of global real economic activity")
    print(f"  global_econ_activity: saved Other/{dest.name}")


def task_gpr(force: bool) -> None:
    # PRIMARY: daily GPRD ("Recent GPR"); build takes its weekly MEAN + 1w lag.
    existing = _find(OTHER_DIR, "data_gpr_daily*")
    if existing is not None and not force:
        reg("gpr", existing, "downloaded", "other", "Caldara-Iacoviello", "GPRD",
            GPRD_URL, "daily", "index",
            f"cached ({existing.name}); daily GPRD col 'GPRD'; weekly-mean + 1w lag in build; PRIMARY")
        print(f"  gpr: cached ({existing.name})")
    else:
        dest = OTHER_DIR / F_GPRD
        dest.write_bytes(http_get(GPRD_URL).content)
        reg("gpr", dest, "downloaded", "other", "Caldara-Iacoviello", "GPRD",
            GPRD_URL, "daily", "index",
            "daily GPRD col 'GPRD'; weekly-mean + 1w lag in build; PRIMARY")
        print(f"  gpr: downloaded Other/{dest.name}")
    # FALLBACK: register the monthly export if present (legacy .dta / .xls).
    monthly = _find(OTHER_DIR, "data_gpr_export.*")
    if monthly is None and LEGACY_GPR.exists():
        shutil.copy2(LEGACY_GPR, OTHER_DIR / F_GPR)
        monthly = OTHER_DIR / F_GPR
    if monthly is not None:
        reg("gpr", monthly, "downloaded", "other", "Caldara-Iacoviello", "GPR",
            "https://www.matteoiacoviello.com/gpr.htm", "monthly", "index",
            f"cached ({monthly.name}); col 'GPR'; monthly FALLBACK (superseded by daily GPRD)")
        print(f"  gpr: monthly fallback registered ({monthly.name})")


# ----------------------------------------------------------------------------
# Auto-download group (every provider)
# ----------------------------------------------------------------------------
def task_auto(force: bool) -> None:
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    YAHOO_DIR.mkdir(parents=True, exist_ok=True)
    OTHER_DIR.mkdir(parents=True, exist_ok=True)

    print("FRED/ (auto-download, cached):")
    for variable, sid, glob_pat, canonical, freq, unit, note in FRED_SERIES:
        try:
            dl_fred_series(variable, sid, glob_pat, canonical, freq, unit, note, force)
        except Exception as e:                       # noqa: BLE001
            print(f"  {variable}: FAILED ({type(e).__name__}: {str(e)[:60]})")
        time.sleep(1)

    print("\nYahoo/ (auto-download, cached):")
    for fn in (dl_yahoo_sp500, dl_yahoo_futures, dl_yahoo_fx):
        try:
            fn(force)
        except Exception as e:                       # noqa: BLE001
            print(f"  {fn.__name__}: FAILED ({type(e).__name__}: {str(e)[:60]})")
        time.sleep(1)

    print("\nCross-provider (primary host + fallback):")
    for fn in (task_ovx, task_gold):
        try:
            fn(force)
        except Exception as e:                       # noqa: BLE001
            print(f"  {fn.__name__}: FAILED ({type(e).__name__}: {str(e)[:60]})")
        time.sleep(1)

    print("\nOther/ (auto-download, cached):")
    for fn in (task_kilian, task_gpr):
        try:
            fn(force)
        except Exception as e:                       # noqa: BLE001
            print(f"  {fn.__name__}: FAILED ({type(e).__name__}: {str(e)[:60]})")
        time.sleep(1)


# ----------------------------------------------------------------------------
# EIA/ (register only — manual website exports, no download)
# ----------------------------------------------------------------------------
def _eia_area(parent_name: str) -> str:
    n = parent_name.lower()
    if "brent" in n:
        return "brent"
    if "wti" in n:
        return "wti"
    return "wpsr"


def _eia_variable(filename: str) -> str:
    for key, var in EIA_VAR_BY_KEY.items():
        if key.lower() in filename.lower():
            return var
    return Path(filename).stem


def task_eia_register() -> None:
    print("EIA/ (register only — manual exports, not downloaded):")
    if not EIA_DIR.exists():
        print("  [skip] EIA/ not found")
        return
    xls = sorted(EIA_DIR.rglob("*.xls")) + sorted(EIA_DIR.rglob("*.xlsx"))
    for path in xls:
        area = _eia_area(path.parent.name)
        variable = _eia_variable(path.name)
        freq = "weekly" if "weekly" in path.name.lower() else "daily"
        redundant = area in ("brent", "wti") and freq == "weekly"
        note = (f"{area}; redundant with daily, not used by build" if redundant
                else f"{area}; manual EIA website export")
        reg(variable, path, "manual", "eia", "EIA", "",
            "https://www.eia.gov/petroleum/", freq, "", note)
        print(f"  EIA/{path.parent.name}/{path.name}: registered ({variable})")


# ----------------------------------------------------------------------------
# Local-derived rows (built from already-local files; not stored in raw layer)
# ----------------------------------------------------------------------------
def record_local_only() -> None:
    reg("dgs10_change", None, "local", "local", "FRED (FRED/)", "DGS10",
        "https://fred.stlouisfed.org/series/DGS10", "daily", "percent",
        "first difference of treasury_10y (FRED/DGS10); built in build_m1_weekly")
    reg("brent_roll_week", None, "local", "local", "local", "", "", "weekly", "0/1",
        "contract-roll dummy: 1 = last W-FRI of each month (approx ICE Brent front-month roll); derived in build_m1_weekly (2026-07)")
    reg("brent_f1_spot_log_basis_spot_leg", None, "local", "local", "EIA (EIA/Brent)",
        "RBRTE/brent_price", "https://www.eia.gov/petroleum/", "weekly", "USD/bbl",
        "spot leg of brent_f1_spot_log_basis = local brent_price (EIA/Brent daily)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-download even if a raw file already exists")
    ap.add_argument("--eia", action="store_true", help="only re-register EIA/")
    ap.add_argument("--download", action="store_true",
                    help="only the auto-download providers (FRED/Yahoo/Other)")
    args = ap.parse_args()

    # No scope flag => run both groups.
    run_all = not (args.eia or args.download)
    print(f"Raw base: {RAW_BASE}\n")

    if run_all or args.eia:
        task_eia_register()
        print()
    if run_all or args.download:
        task_auto(args.force)
        print()

    record_local_only()

    man = pd.DataFrame(_MANIFEST_ROWS)
    man.to_csv(MANIFEST_CSV, index=False)
    print(f"Manifest: {MANIFEST_CSV}  ({len(man)} rows)")
    cols = ["variable", "raw_file", "kind", "category", "n_rows",
            "coverage_start", "coverage_end"]
    print(man[cols].to_string(index=False))


if __name__ == "__main__":
    main()
