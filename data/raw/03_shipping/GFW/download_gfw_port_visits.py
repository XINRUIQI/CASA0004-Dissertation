"""
Download GFW AIS port-visit events for the 11 oil-infrastructure AOIs, then
derive the AOI-to-AOI origin-destination (O-D) voyage edges that form the
Stage-2 dynamic heterogeneous shipping graph G(t).

Why this script
---------------
PortWatch gives node-level port-call *counts* but no vessel identity, so it
cannot express *which node a tanker came from*. GFW's Events API returns
individual PORT_VISIT events carrying a stable `vessel.id` plus visit
`durationHrs`. By pulling port visits inside each AOI polygon and then chaining
each vessel's consecutive visits in time, we obtain:

  1. NODE features : per-AOI tanker/cargo port-visit count + dwell (durationHrs)
                     -> gfw_aoi_port_visits.csv (raw event level)
  2. EDGE features : directed AOI_i -> AOI_j voyages (one per consecutive pair
                     of AOI visits by the same vessel)
                     -> gfw_aoi_od_voyages.csv

Note on graph semantics: an edge AOI_i -> AOI_j means "the same vessel's next
observed AOI visit after AOI_i was AOI_j". The vessel may have called at
non-AOI ports in between (those are not observed here); this is therefore the
O-D graph *induced on the 11-node AOI subset*, which is exactly what the graph
module needs.

API (confirmed 2026-07-03 against api-doc.globalfishingwatch.org):
  POST https://gateway.api.globalfishingwatch.org/v3/events?limit=&offset=
  body: datasets=[public-global-port-visits-events:v3.0], types=[PORT_VISIT],
        geometry=<AOI polygon>, confidences=[3,4],
        vesselTypes=[BUNKER_OR_TANKER, CARGO], startDate, endDate

Prerequisites:
    GFW API token in env GFW_API_TOKEN or in ./.gfw_token

Usage:
    python download_gfw_port_visits.py                    # 2019-2025, buffer 0.25 deg
    python download_gfw_port_visits.py --start 2017 --end 2025 --buffer-deg 0.3
    python download_gfw_port_visits.py --resume           # skip already-saved (aoi, year)

Output (same directory as this script):
    gfw_aoi_port_visits.csv      raw port-visit events (node level, has dwell)
    gfw_aoi_od_voyages.csv       derived directed AOI->AOI voyage edges
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUT_DIR.parents[3]
TOKEN_FILE = OUT_DIR / ".gfw_token"
AOI_CSV = (PROJECT_ROOT / "data" / "raw" / "02_sentinel2"
           / "aoi_oil_infrastructure.csv")

EVENTS_URL = "https://gateway.api.globalfishingwatch.org/v3/events"
DATASET = "public-global-port-visits-events:v3.0"
# GFW Events API vessel-type enum (probed 2026-07-03; the API rejects
# BUNKER_OR_TANKER). CRITICAL: at pure crude-export ports oil tankers are
# classified as CARGO, not BUNKER (probe: Ras Tanura 2024 CARGO=1812 vs
# BUNKER=3; Kharg CARGO=89 vs BUNKER=0). So oil tankers live in CARGO; BUNKER
# is only refuelling vessels. The endpoint returns no per-event vesselType, so
# oil tankers cannot be isolated from other cargo here: at oil-only AOIs the
# CARGO class is effectively tankers, at multi-purpose ports it is mixed
# cargo. We keep both classes and document this as a graph-edge caveat.
VESSEL_TYPES = ["BUNKER", "CARGO"]
CONFIDENCES = ["3", "4"]
# Events API accepts large page sizes; big pages cut the (slow) server-side
# round-trips dramatically for busy ports (Rotterdam ~12k cargo visits/yr).
PAGE_LIMIT = 10000

RAW_OUT = OUT_DIR / "gfw_aoi_port_visits.csv"
OD_OUT = OUT_DIR / "gfw_aoi_od_voyages.csv"
PROGRESS = OUT_DIR / ".port_visits_progress.txt"


def get_token() -> str:
    token = os.environ.get("GFW_API_TOKEN", "").strip()
    if token:
        return token
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    raise RuntimeError(
        "GFW API token not found.\n"
        "  export GFW_API_TOKEN='your_token'\n"
        f"  or: echo 'your_token' > {TOKEN_FILE}"
    )


def load_aois() -> pd.DataFrame:
    df = pd.read_csv(AOI_CSV)
    return df[["site_id", "site_short", "site_type", "lon", "lat"]].copy()


def square_polygon(lon: float, lat: float, half_deg: float) -> list[list[float]]:
    """Axis-aligned square polygon (lon/lat) centred on the AOI."""
    return [
        [lon - half_deg, lat - half_deg],
        [lon + half_deg, lat - half_deg],
        [lon + half_deg, lat + half_deg],
        [lon - half_deg, lat + half_deg],
        [lon - half_deg, lat - half_deg],
    ]


def fetch_port_visits(
    token: str, coords: list[list[float]], date_start: str, date_end: str
) -> list[dict]:
    """Fetch all PORT_VISIT events for one polygon and date window (paginated)."""
    ctx = ssl.create_default_context()
    body = json.dumps({
        "datasets": [DATASET],
        "types": ["PORT_VISIT"],
        "startDate": date_start,
        "endDate": date_end,
        "confidences": CONFIDENCES,
        "vesselTypes": VESSEL_TYPES,
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }).encode("utf-8")

    all_entries: list[dict] = []
    offset = 0
    while True:
        url = f"{EVENTS_URL}?limit={PAGE_LIMIT}&offset={offset}"
        req = Request(
            url, data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            },
            method="POST",
        )
        for attempt in range(4):
            try:
                with urlopen(req, timeout=180, context=ctx) as resp:
                    data = json.loads(resp.read().decode())
                break
            except HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise
        else:
            raise RuntimeError("repeated HTTP errors")

        entries = data.get("entries", [])
        all_entries.extend(entries)
        total = data.get("total", len(all_entries))
        offset += len(entries)
        if not entries or offset >= total:
            break
        time.sleep(0.4)
    return all_entries


def parse_entry(e: dict, site_id: str, site_short: str) -> dict:
    v = e.get("vessel", {}) or {}
    pv = e.get("port_visit", {}) or {}
    ianch = pv.get("intermediateAnchorage") or pv.get("startAnchorage") or {}
    pos = e.get("position", {}) or {}
    return {
        "site_id": site_id,
        "site_short": site_short,
        "vessel_id": v.get("id"),
        "vessel_name": v.get("name"),
        "ssvid": v.get("ssvid"),
        "flag": v.get("flag"),
        "start": e.get("start"),
        "end": e.get("end"),
        "duration_hrs": pv.get("durationHrs"),
        "confidence": pv.get("confidence"),
        "anchorage_id": ianch.get("id"),
        "anchorage_name": ianch.get("name"),
        "lat": pos.get("lat"),
        "lon": pos.get("lon"),
        "event_id": e.get("id"),
    }


def build_od_edges(visits: pd.DataFrame) -> pd.DataFrame:
    """Chain each vessel's consecutive AOI visits into directed O-D edges."""
    v = visits.dropna(subset=["vessel_id", "start"]).copy()
    v["start_dt"] = pd.to_datetime(v["start"], errors="coerce", utc=True)
    v["end_dt"] = pd.to_datetime(v["end"], errors="coerce", utc=True)
    v = v.sort_values(["vessel_id", "start_dt"]).reset_index(drop=True)

    edges: list[dict] = []
    for vid, g in v.groupby("vessel_id", sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            depart = a["end_dt"] if pd.notna(a["end_dt"]) else a["start_dt"]
            arrive = b["start_dt"]
            transit_days = (arrive - depart).total_seconds() / 86400.0 \
                if pd.notna(depart) and pd.notna(arrive) else None
            edges.append({
                "vessel_id": vid,
                "vessel_name": a["vessel_name"],
                "ssvid": a["ssvid"],
                "flag": a["flag"],
                "from_site": a["site_id"],
                "to_site": b["site_id"],
                "depart_time": a["end"],
                "arrive_time": b["start"],
                "transit_days": transit_days,
                "is_self_loop": a["site_id"] == b["site_id"],
            })
    return pd.DataFrame(edges)


def load_progress() -> set[tuple[str, int]]:
    if not PROGRESS.exists():
        return set()
    done = set()
    for line in PROGRESS.read_text().splitlines():
        line = line.strip()
        if "," in line:
            sid, y = line.split(",", 1)
            try:
                done.add((sid, int(y)))
            except ValueError:
                pass
    return done


def mark_progress(site_id: str, year: int) -> None:
    with open(PROGRESS, "a") as f:
        f.write(f"{site_id},{year}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2019, help="start year (incl.)")
    ap.add_argument("--end", type=int, default=2025, help="end year (incl.)")
    ap.add_argument("--buffer-deg", type=float, default=0.25,
                    help="half-size of the AOI square polygon in degrees")
    ap.add_argument("--resume", action="store_true",
                    help="skip (aoi, year) windows already recorded as done")
    args = ap.parse_args()

    token = get_token()
    print(f"GFW token loaded (length={len(token)})")
    aois = load_aois()
    years = list(range(args.start, args.end + 1))

    # Fresh run wipes prior partial outputs; --resume keeps them and skips
    # already-completed (aoi, year) windows via the progress sidecar.
    if not args.resume:
        for p in (RAW_OUT, PROGRESS):
            if p.exists():
                p.unlink()
    done_keys = load_progress() if args.resume else set()
    if done_keys:
        print(f"  resume: {len(done_keys)} (aoi,year) windows already done")

    header_written = RAW_OUT.exists() and RAW_OUT.stat().st_size > 0
    cols = ["site_id", "site_short", "vessel_id", "vessel_name", "ssvid",
            "flag", "start", "end", "duration_hrs", "confidence",
            "anchorage_id", "anchorage_name", "lat", "lon", "event_id"]
    for _, r in aois.iterrows():
        coords = square_polygon(r["lon"], r["lat"], args.buffer_deg)
        for y in years:
            if (r["site_id"], y) in done_keys:
                continue
            ds, de = f"{y}-01-01", f"{y + 1}-01-01"
            tag = f"{r['site_id']} {r['site_short']:12} [{y}]"
            print(f"  {tag} ...", end=" ", flush=True)
            try:
                entries = fetch_port_visits(token, coords, ds, de)
                rows = [parse_entry(e, r["site_id"], r["site_short"])
                        for e in entries]
                if rows:
                    df = pd.DataFrame(rows)[cols]
                    df.to_csv(RAW_OUT, mode="a", header=not header_written,
                              index=False)
                    header_written = True
                mark_progress(r["site_id"], y)   # append-only sidecar
                print(f"{len(entries)} visits")
            except Exception as e:
                print(f"ERROR {e}")
            time.sleep(0.4)

    if not RAW_OUT.exists():
        print("\nNo port visits retrieved.")
        return

    visits = pd.read_csv(RAW_OUT)
    visits = visits.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    visits.to_csv(RAW_OUT, index=False)
    print(f"\nSaved raw port visits: {RAW_OUT}")
    print(f"  visits: {len(visits)}  vessels: {visits['vessel_id'].nunique()}")
    for sid in aois["site_id"]:
        sub = visits[visits["site_id"] == sid]
        if len(sub):
            print(f"    {sid}: {len(sub):6d} visits, "
                  f"mean_dwell_hrs={sub['duration_hrs'].astype(float).mean():.1f}")

    od = build_od_edges(visits)
    od.to_csv(OD_OUT, index=False)
    print(f"\nSaved O-D voyage edges: {OD_OUT}")
    print(f"  edges: {len(od)}  (self-loops: {int(od['is_self_loop'].sum())})")
    cross = od[~od["is_self_loop"]]
    print(f"  cross-node edges: {len(cross)}")
    if len(cross):
        top = (cross.groupby(["from_site", "to_site"]).size()
               .sort_values(ascending=False).head(10))
        print("  top AOI->AOI lanes:")
        for (fs, ts), n in top.items():
            print(f"    {fs} -> {ts}: {n}")


if __name__ == "__main__":
    main()
