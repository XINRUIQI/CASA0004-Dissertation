"""
Download IMF PortWatch port-level daily data for selected oil ports.

Unlike the chokepoint dataset (total transits, no direction), the port dataset
provides DIRECTIONAL tanker activity: `import_tanker` and `export_tanker`
port-call counts per port per day (from 2019-01-01 onward). This enables an
export-vs-import asymmetry feature for the M3 shipping module.

Port basket (oil-relevant, tanker-dominant):
  - Export hubs (crude exporters): Ras Tanura, Juaymah, Yanbu, Ras Laffan,
    Primorsk, Novorossiysk, Corpus Christi, Sidi Kerir, Bonny
  - Import/refining hubs (consumers): Rotterdam, Singapore, Ningbo, Chiba, Ulsan

Data source: https://portwatch.imf.org/
ArcGIS FeatureServer: services9.arcgis.com (org: weJ1QsnbMYJlCHdG)

Usage:
    python download_portwatch_ports.py

Output:
    data/raw/05_shipping/IMF Portwatch/portwatch_ports_daily.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

# portid -> (short label, role)
OIL_PORTS = {
    # export hubs
    "port1091": ("ras_tanura", "export"),
    "port526": ("juaymah", "export"),
    "port570": ("yanbu", "export"),
    "port1090": ("ras_laffan", "export"),
    "port1020": ("primorsk", "export"),
    "port833": ("novorossiysk", "export"),
    "port264": ("corpus_christi", "export"),
    "port2129": ("sidi_kerir", "export"),
    "port155": ("bonny", "export"),
    # import / refining hubs
    "port1114": ("rotterdam", "import"),
    "port1201": ("singapore", "import"),
    "port824": ("ningbo", "import"),
    "port239": ("chiba", "import"),
    "port1338": ("ulsan", "import"),
}

FEATURE_SERVICE_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)

PAGE_SIZE = 1000


def fetch_port(portid: str) -> list[dict]:
    """Fetch all daily records for a single port (paginated, robust)."""
    all_records: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": f"portid='{portid}'",
            "outFields": ("date,portid,portname,country,portcalls_tanker,"
                          "import_tanker,export_tanker"),
            "returnGeometry": "false",
            "f": "json",
            "resultRecordCount": PAGE_SIZE,
            "resultOffset": offset,
            "orderByFields": "date",
        }
        url = f"{FEATURE_SERVICE_URL}?{urlencode(params)}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        if "error" in data:
            raise RuntimeError(data["error"])
        features = data.get("features", [])
        if not features:
            break
        all_records.extend(f["attributes"] for f in features)
        offset += len(features)
        if len(features) < PAGE_SIZE:
            break
    return all_records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(OIL_PORTS)} oil ports from PortWatch (Daily_Ports_Data)...")

    records: list[dict] = []
    for pid, (short, role) in OIL_PORTS.items():
        try:
            recs = fetch_port(pid)
            records.extend(recs)
            print(f"  {short:16} ({role:6}) {pid:>9}: {len(recs)} days", flush=True)
        except Exception as e:
            print(f"  {short:16} ({role:6}) {pid:>9}: ERROR {e}")

    print(f"  Retrieved {len(records)} total records")
    if not records:
        print("\nNo data retrieved. Check network or API status.")
        return

    df = pd.DataFrame(records)
    # ArcGIS dateOnly returns epoch ms
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="ms")
    else:
        df["date"] = pd.to_datetime(df["date"])
    df["short"] = df["portid"].map(lambda p: OIL_PORTS.get(p, ("", ""))[0])
    df["role"] = df["portid"].map(lambda p: OIL_PORTS.get(p, ("", ""))[1])
    df = df.sort_values(["portid", "date"]).reset_index(drop=True)

    out_path = OUT_DIR / "portwatch_ports_daily.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Ports: {df['portid'].nunique()}")
    for pid, (short, role) in OIL_PORTS.items():
        sub = df[df["portid"] == pid]
        if len(sub):
            print(f"    {short:16} ({role:6}): {len(sub):5d} days, "
                  f"exp_tanker_sum={sub['export_tanker'].sum():.0f}, "
                  f"imp_tanker_sum={sub['import_tanker'].sum():.0f}")
        else:
            print(f"    {short:16} ({role:6}): NO DATA")


if __name__ == "__main__":
    main()
