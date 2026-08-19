"""
Download IMF PortWatch port-level daily data for the 11 oil-infrastructure AOIs
(the same 11 AOIs used by the remote-sensing module M2), to serve as NODE
features for the Stage-2 dynamic heterogeneous shipping graph G(t).

Difference vs. download_portwatch_ports.py
------------------------------------------
`download_portwatch_ports.py` fetched a 14-port *directional basket* (export vs.
import hubs) to build the chokepoint-level asymmetry family. This script instead
fetches the *11 graph nodes* one-to-one aligned with the M2 AOIs
(aoi_oil_infrastructure.csv), so every node in the shipping graph has a
PortWatch activity vector.

Fields available on Daily_Ports_Data (probed 2026-07-03): only port-call counts
and tonnage by ship type — there is NO per-port DWT / dwell / anchorage field.
So node-level capacity / vessel-size / dwell must still come from the chokepoint
transit data (capacity_*) and GFW (presence + port-visit durationHrs). This
script therefore captures per-node tanker/cargo port-call counts and import/
export tonnage only.

AOI -> PortWatch portid mapping (probed via ArcGIS FeatureServer, 2026-07-03)
----------------------------------------------------------------------------
  P001 Rotterdam   port1114  exact
  P002 Fujairah    port362   exact
  P003 RasTanura   port1091  exact
  P004 Jurong      port1201  PROXY -> Singapore (PortWatch has no standalone
                             Jurong Island oil port; "Estaleiro Jurong" is a
                             Brazilian shipyard and is excluded)
  P005 Houston     port481   exact
  P006 Ningbo      port824   exact (Zhoushan is a separate port1429, not merged)
  P007 Jamnagar    port1199  PROXY -> Sikka (Reliance Jamnagar's dedicated
                             crude port; PortWatch has no "Jamnagar")
  P008 Basra       port2479  Basrah Oil Terminal (port20 "Al Basrah" is the
                             general city port; the oil terminal is preferred)
  P009 Ulsan       port1338  exact
  P010 Kharg       port2164  exact
  P011 Yanbu       port570   Yanbu (King Fahd Port)

Data source: https://portwatch.imf.org/
ArcGIS FeatureServer: services9.arcgis.com (org: weJ1QsnbMYJlCHdG)

Usage:
    python download_portwatch_aoi_nodes.py

Output (same directory as this script):
    portwatch_aoi_nodes_daily.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

# site_id -> (short, portid, portwatch_name, match_quality)
AOI_PORTS: dict[str, tuple[str, str, str, str]] = {
    "P001": ("Rotterdam", "port1114", "Rotterdam", "exact"),
    "P002": ("Fujairah", "port362", "Fujairah", "exact"),
    "P003": ("RasTanura", "port1091", "Ras Tanura", "exact"),
    "P004": ("Jurong", "port1201", "Singapore", "proxy_singapore"),
    "P005": ("Houston", "port481", "Houston", "exact"),
    "P006": ("Ningbo", "port824", "Ningbo", "exact"),
    "P007": ("Jamnagar", "port1199", "Sikka", "proxy_sikka"),
    "P008": ("Basra", "port2479", "Basrah Oil Terminal", "exact_oil_terminal"),
    "P009": ("Ulsan", "port1338", "Ulsan", "exact"),
    "P010": ("Kharg", "port2164", "Kharg Island", "exact"),
    "P011": ("Yanbu", "port570", "Yanbu (King Fahd Port)", "exact"),
}

FEATURE_SERVICE_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)

OUT_FIELDS = (
    "date,portid,portname,country,"
    "portcalls_tanker,portcalls_cargo,portcalls,"
    "import_tanker,export_tanker,import,export"
)

# Server-side maxRecordCount for this FeatureServer is 1000; requesting more
# just returns 1000 and would wrongly look like the last page. Keep <= 1000 and
# rely on exceededTransferLimit / empty page to stop.
PAGE_SIZE = 1000


def fetch_port(portid: str) -> list[dict]:
    """Fetch all daily records for a single port (paginated, robust)."""
    all_records: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": f"portid='{portid}'",
            "outFields": OUT_FIELDS,
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
        # Stop only when the server says there is no more data.
        if not data.get("exceededTransferLimit") and len(features) < PAGE_SIZE:
            break
    return all_records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(AOI_PORTS)} AOI graph nodes from PortWatch "
          f"(Daily_Ports_Data)...")

    records: list[dict] = []
    for site_id, (short, pid, pw_name, quality) in AOI_PORTS.items():
        try:
            recs = fetch_port(pid)
            for r in recs:
                r["site_id"] = site_id
                r["site_short"] = short
                r["match_quality"] = quality
            records.extend(recs)
            print(f"  {site_id} {short:12} {pid:>9} ({quality:18}): "
                  f"{len(recs):5d} days", flush=True)
        except Exception as e:
            print(f"  {site_id} {short:12} {pid:>9}: ERROR {e}")

    if not records:
        print("\nNo data retrieved. Check network or API status.")
        return

    df = pd.DataFrame(records)
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="ms")
    else:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)

    lead = ["site_id", "site_short", "date", "portid", "portname", "country",
            "match_quality"]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]

    out_path = OUT_DIR / "portwatch_aoi_nodes_daily.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Nodes: {df['site_id'].nunique()}/11")
    for site_id, (short, pid, pw_name, quality) in AOI_PORTS.items():
        sub = df[df["site_id"] == site_id]
        if len(sub):
            print(f"    {site_id} {short:12}: {len(sub):5d} days, "
                  f"tanker_calls={sub['portcalls_tanker'].sum():.0f}, "
                  f"imp_t={sub['import_tanker'].sum():.0f}, "
                  f"exp_t={sub['export_tanker'].sum():.0f}")
        else:
            print(f"    {site_id} {short:12}: NO DATA")


if __name__ == "__main__":
    main()
