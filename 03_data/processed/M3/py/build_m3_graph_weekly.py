"""
M3 Stage-2 — build the leakage-safe WEEKLY dynamic heterogeneous shipping graph
from the Stage-2 raw downloads (node table + O-D adjacency series + dark-vessel
weekly), aligned to the same W-FRI convention / publication-lag discipline as
aggregate_shipping_to_weekly.py.

Reads (raw layer):
  raw/03_shipping/IMF Portwatch/portwatch_aoi_nodes_daily.csv   (11 AOI nodes, daily)
  raw/03_shipping/GFW/gfw_aoi_port_visits.csv                   (port-visit events, dwell)
  raw/03_shipping/GFW/gfw_aoi_od_voyages.csv                    (AOI->AOI voyage edges)
  raw/03_shipping/GFW/gfw_sar_detections_monthly.csv            (SAR dark vessels, monthly)
  raw/02_sentinel2/aoi_oil_infrastructure.csv                   (11 AOI id/order)

Writes (processed/M3/outputs):
  m3_graph_nodes_weekly.csv        long: (week, site_id) x node features
  m3_graph_edges_weekly.csv        long: (week, from_site, to_site) x edge features
  m3_graph_darkvessel_weekly.csv   long: (week, region_type, region_id) x SAR
  m3_graph_tensors.npz             node_features (T,N,F) + adjacency (T,N,N) + axes

Design (mirrors the flat M3 builder):
  * One UNION W-FRI index across all Stage-2 sources; nothing dropped.
  * Publication lags (no look-ahead), module-level & CLI-overridable:
      - PortWatch node counts (daily->weekly sum)  : +PW_LAG_WEEKS  (=1)
      - GFW port-visit / voyage weekly aggregates  : +GFW_EVENT_LAG_WEEKS (=2, near-real-time but conservative)
      - GFW SAR monthly detections (month-end ffill): +SAR_LAG_WEEKS (=4)
  * Data-quality clips (raw carries anomalies, processed cleans them):
      - port-visit dwell capped at DWELL_CAP_HRS (=720h=30d); longer = AIS
        long-stay / stitching artefact -> NaN (see raw caveat, JESSICA B 13yr).
      - voyage transit capped at TRANSIT_CAP_DAYS (=90d); longer = an
        unobserved intermediate non-AOI call -> NaN for the mean, edge still counts.
  * O-D edges are AOI_i->AOI_j (cross-node only); self-loops become a node
    'turnover' feature (repeat visits to the same AOI).

Usage:
    python build_m3_graph_weekly.py
    python build_m3_graph_weekly.py --pw-lag 1 --gfw-event-lag 2 --sar-lag 4
    python build_m3_graph_weekly.py --no-lag        # diagnostic, leaky
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths  (this file lives at processed/M3/py/build_m3_graph_weekly.py)
# ----------------------------------------------------------------------------
PY_DIR = Path(__file__).resolve().parent
M3_DIR = PY_DIR.parent
DATA_DIR = M3_DIR.parents[1]                       # 03_data
RAW_SHIP = DATA_DIR / "raw" / "03_shipping"
PW_DIR = RAW_SHIP / "IMF Portwatch"
GFW_DIR = RAW_SHIP / "GFW"
AOI_CSV = DATA_DIR / "raw" / "02_sentinel2" / "aoi_oil_infrastructure.csv"
OUT_DIR = M3_DIR / "outputs"

NODES_OUT = OUT_DIR / "m3_graph_nodes_weekly.csv"
EDGES_OUT = OUT_DIR / "m3_graph_edges_weekly.csv"
DARK_OUT = OUT_DIR / "m3_graph_darkvessel_weekly.csv"
TENSOR_OUT = OUT_DIR / "m3_graph_tensors.npz"

# ----------------------------------------------------------------------------
# Study window (keep full history; modelling clips to 2019-2025 comparison win)
# ----------------------------------------------------------------------------
STUDY_START = "2019-01-01"
STUDY_END = "2026-12-31"

# Publication lags (weeks) — a value is only usable AFTER release.
PW_LAG_WEEKS = 1          # PortWatch weekly aggregate not yet published at week close
GFW_EVENT_LAG_WEEKS = 2   # GFW events are near-real-time (~96h); 2w is conservative
SAR_LAG_WEEKS = 4         # SAR monthly aggregate: ~1 month conservative delay

# Data-quality clips.
DWELL_CAP_HRS = 720.0     # 30 days: longer port-visit dwell = AIS long-stay artefact
TRANSIT_CAP_DAYS = 90.0   # 90 days: longer voyage gap = unobserved intermediate call

# SAR region name -> chokepoint short code (AOIs keep their P0xx id).
CHOKE_SHORT = {
    "Strait of Hormuz": "hormuz",
    "Suez Canal": "suez",
    "Malacca Strait": "malacca",
    "Bab el-Mandeb": "mandeb",
    "Cape of Good Hope": "cape",
    "Panama Canal": "panama",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _to_naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)


def load_sites() -> list[str]:
    df = pd.read_csv(AOI_CSV)
    return df.sort_values("site_id")["site_id"].tolist()


def build_union_index(*ranges: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DatetimeIndex:
    bounds = [r for r in ranges if r is not None]
    start = min(b[0] for b in bounds)
    end = max(b[1] for b in bounds)
    idx = pd.date_range(start=start, end=end, freq="W-FRI")
    idx.name = "week_ending_friday"
    return idx


def _lag(df: pd.DataFrame, weeks: int) -> pd.DataFrame:
    return df.shift(weeks) if weeks else df


# ----------------------------------------------------------------------------
# 1) NODE features
# ----------------------------------------------------------------------------
def load_portwatch_nodes() -> pd.DataFrame:
    """Daily 11-AOI PortWatch -> per (site, W-FRI) sums (pre-lag)."""
    path = PW_DIR / "portwatch_aoi_nodes_daily.csv"
    if not path.exists():
        print(f"  [skip] {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    keep = ["portcalls_tanker", "portcalls_cargo", "import_tanker", "export_tanker"]
    out = (df.set_index("date")
           .groupby("site_id")[keep]
           .resample("W-FRI").sum(min_count=1))
    out.columns = [f"pw_{c}" for c in out.columns]
    out.index = out.index.rename(["site_id", "week"])
    print(f"  PortWatch nodes (pre-lag): {out.shape}")
    return out


def load_portvisit_nodes() -> pd.DataFrame:
    """Port-visit events -> per (site, W-FRI): visit count + capped dwell (pre-lag)."""
    path = GFW_DIR / "gfw_aoi_port_visits.csv"
    if not path.exists():
        print(f"  [skip] {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=["site_id", "start", "duration_hrs", "event_id"])
    df["start_dt"] = _to_naive_utc(df["start"])
    df = df.dropna(subset=["start_dt"])
    df["dwell"] = df["duration_hrs"].where(df["duration_hrs"] <= DWELL_CAP_HRS)
    out = (df.set_index("start_dt")
           .groupby("site_id")
           .resample("W-FRI", include_groups=False)
           .agg(gfw_n_visits=("event_id", "size"),
                gfw_dwell_hrs_mean=("dwell", "mean"),
                gfw_dwell_hrs_median=("dwell", "median")))
    out.index = out.index.rename(["site_id", "week"])
    print(f"  Port-visit nodes (pre-lag): {out.shape}")
    return out


def load_selfloop_nodes() -> pd.DataFrame:
    """Self-loop voyages (repeat visits to same AOI) -> per (site, week) turnover."""
    path = GFW_DIR / "gfw_aoi_od_voyages.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=["from_site", "to_site", "arrive_time", "is_self_loop"])
    sl = df[df["is_self_loop"] == True].copy()  # noqa: E712
    if sl.empty:
        return pd.DataFrame()
    sl["arrive_dt"] = _to_naive_utc(sl["arrive_time"])
    sl = sl.dropna(subset=["arrive_dt"])
    out = (sl.set_index("arrive_dt")
           .groupby("from_site")
           .resample("W-FRI", include_groups=False).size()
           .to_frame("gfw_self_loops"))
    out.index = out.index.rename(["site_id", "week"])
    print(f"  Self-loop nodes (pre-lag): {out.shape}")
    return out


# ----------------------------------------------------------------------------
# 2) O-D edges
# ----------------------------------------------------------------------------
def load_edges() -> pd.DataFrame:
    """Cross-node voyages -> per (from, to, W-FRI): count + capped mean transit."""
    path = GFW_DIR / "gfw_aoi_od_voyages.csv"
    if not path.exists():
        print(f"  [skip] {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=["vessel_id", "from_site", "to_site",
                                    "arrive_time", "transit_days", "is_self_loop"])
    vo = df[df["is_self_loop"] == False].copy()  # noqa: E712
    vo["arrive_dt"] = _to_naive_utc(vo["arrive_time"])
    vo = vo.dropna(subset=["arrive_dt"])
    vo["transit"] = vo["transit_days"].where(
        (vo["transit_days"] > 0) & (vo["transit_days"] <= TRANSIT_CAP_DAYS))
    out = (vo.set_index("arrive_dt")
           .groupby(["from_site", "to_site"])
           .resample("W-FRI", include_groups=False)
           .agg(n_voyages=("vessel_id", "size"),
                mean_transit_days=("transit", "mean")))
    out.index = out.index.rename(["from_site", "to_site", "week"])
    print(f"  O-D edges (pre-lag): {out.shape}")
    return out


# ----------------------------------------------------------------------------
# 3) Dark vessels (SAR monthly)
# ----------------------------------------------------------------------------
def load_darkvessel_monthly() -> pd.DataFrame:
    path = GFW_DIR / "gfw_sar_detections_monthly.csv"
    if not path.exists():
        print(f"  [skip] {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["month_end"] = pd.to_datetime(df["month"], format="%Y-%m") + pd.offsets.MonthEnd(0)
    print(f"  SAR dark vessels (monthly): {df.shape}")
    return df


# ----------------------------------------------------------------------------
# Weekly matrix builder: (entity, week) long -> week x entity wide, union, lag
# ----------------------------------------------------------------------------
def to_weekly_matrix(long_df: pd.DataFrame, value: str, entity: str,
                     union: pd.DatetimeIndex, lag: int) -> pd.DataFrame:
    if long_df.empty or value not in long_df.columns:
        return pd.DataFrame(index=union)
    wide = (long_df[value].reset_index()
            .pivot_table(index="week", columns=entity, values=value))
    wide = wide.reindex(union)
    return _lag(wide, lag)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the M3 Stage-2 weekly graph tensors.")
    ap.add_argument("--pw-lag", type=int, default=PW_LAG_WEEKS)
    ap.add_argument("--gfw-event-lag", type=int, default=GFW_EVENT_LAG_WEEKS)
    ap.add_argument("--sar-lag", type=int, default=SAR_LAG_WEEKS)
    ap.add_argument("--no-lag", action="store_true", help="disable all lags (leaky)")
    args = ap.parse_args()
    pw_lag = 0 if args.no_lag else args.pw_lag
    ev_lag = 0 if args.no_lag else args.gfw_event_lag
    sar_lag = 0 if args.no_lag else args.sar_lag

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = load_sites()
    print(f"M3 Stage-2 graph builder\nSites ({len(sites)}): {sites}")
    print(f"Lags: PW={pw_lag}w  GFW-event={ev_lag}w  SAR={sar_lag}w"
          f"{'  (NO-LAG diagnostic)' if args.no_lag else ''}\n")

    pw_nodes = load_portwatch_nodes()
    pv_nodes = load_portvisit_nodes()
    sl_nodes = load_selfloop_nodes()
    edges = load_edges()
    dark = load_darkvessel_monthly()

    # Union W-FRI index across all Stage-2 sources.
    def _rng(idx_df, level="week"):
        if idx_df is None or idx_df.empty:
            return None
        wk = idx_df.index.get_level_values(level)
        return (wk.min(), wk.max())

    ranges = [_rng(pw_nodes), _rng(pv_nodes), _rng(edges)]
    if not dark.empty:
        ranges.append((dark["month_end"].min(), dark["month_end"].max()))
    union = build_union_index(*[r for r in ranges if r])
    print(f"\n  Union W-FRI index: {len(union)} weeks, "
          f"{union.min().date()} ~ {union.max().date()}")

    # ---- Node feature matrices (week x site), lagged --------------------
    node_feats: dict[str, pd.DataFrame] = {}
    for col in ["pw_portcalls_tanker", "pw_portcalls_cargo",
                "pw_import_tanker", "pw_export_tanker"]:
        node_feats[col] = to_weekly_matrix(pw_nodes, col, "site_id", union, pw_lag)
    for col in ["gfw_n_visits", "gfw_dwell_hrs_mean", "gfw_dwell_hrs_median"]:
        node_feats[col] = to_weekly_matrix(pv_nodes, col, "site_id", union, ev_lag)
    node_feats["gfw_self_loops"] = to_weekly_matrix(sl_nodes, "gfw_self_loops",
                                                    "site_id", union, ev_lag)

    # AOI-level dark vessels as node features (monthly ffill -> union -> lag).
    if not dark.empty:
        aoi_dark = dark[dark["region_type"] == "aoi"].copy()
        for src, dst in [("detections_total", "sar_detections_total"),
                         ("detections_dark", "sar_detections_dark"),
                         ("dark_share", "sar_dark_share")]:
            m = (aoi_dark.pivot_table(index="month_end", columns="region_id", values=src)
                 .reindex(union, method="ffill"))
            node_feats[dst] = _lag(m, sar_lag)

    # Reindex every node matrix to the full site set / union, assemble long table.
    for k in node_feats:
        node_feats[k] = node_feats[k].reindex(index=union, columns=sites)
    node_long = pd.concat(
        {k: v.stack(future_stack=True) for k, v in node_feats.items()}, axis=1)
    node_long.index = node_long.index.rename(["week_ending_friday", "site_id"])
    node_long = node_long.reset_index()
    node_long["avail_node"] = node_long[list(node_feats)].notna().any(axis=1).astype(int)
    node_long = node_long[(node_long["week_ending_friday"] >= STUDY_START) &
                          (node_long["week_ending_friday"] <= STUDY_END)]
    node_long.to_csv(NODES_OUT, index=False)

    # ---- Edge matrices (week x (from,to)), lagged, long + tensor -------
    if not edges.empty:
        nv = (edges["n_voyages"].reset_index()
              .pivot_table(index="week", columns=["from_site", "to_site"],
                           values="n_voyages").reindex(union))
        mt = (edges["mean_transit_days"].reset_index()
              .pivot_table(index="week", columns=["from_site", "to_site"],
                           values="mean_transit_days").reindex(union))
        nv, mt = _lag(nv, ev_lag), _lag(mt, ev_lag)
        edge_long = pd.concat({"n_voyages": nv.stack(["from_site", "to_site"], future_stack=True),
                               "mean_transit_days": mt.stack(["from_site", "to_site"], future_stack=True)},
                              axis=1)
        edge_long.index = edge_long.index.rename(["week_ending_friday", "from_site", "to_site"])
        edge_long = edge_long.reset_index().dropna(subset=["n_voyages"])
        edge_long = edge_long[(edge_long["week_ending_friday"] >= STUDY_START) &
                              (edge_long["week_ending_friday"] <= STUDY_END)]
        edge_long.to_csv(EDGES_OUT, index=False)
    else:
        nv = mt = None
        pd.DataFrame(columns=["week_ending_friday", "from_site", "to_site",
                              "n_voyages", "mean_transit_days"]).to_csv(EDGES_OUT, index=False)

    # ---- Dark vessel weekly long (all 17 regions) ----------------------
    if not dark.empty:
        frames = []
        for rid, grp in dark.groupby("region_id"):
            g = grp.set_index("month_end").sort_index()
            w = g[["detections_total", "detections_dark", "detections_matched",
                   "dark_share"]].reindex(union, method="ffill")
            w = _lag(w, sar_lag)
            w["region_type"] = grp["region_type"].iloc[0]
            w["region_id"] = rid
            w["region_short"] = CHOKE_SHORT.get(rid, rid)
            frames.append(w)
        dark_long = pd.concat(frames).reset_index(names="week_ending_friday")
        dark_long = dark_long[(dark_long["week_ending_friday"] >= STUDY_START) &
                              (dark_long["week_ending_friday"] <= STUDY_END)]
        lead = ["week_ending_friday", "region_type", "region_id", "region_short"]
        dark_long = dark_long[lead + [c for c in dark_long.columns if c not in lead]]
        dark_long.to_csv(DARK_OUT, index=False)
    else:
        dark_long = pd.DataFrame()

    # ---- Tensors (T, N, F) node + (T, N, N) adjacency ------------------
    weeks_study = union[(union >= STUDY_START) & (union <= STUDY_END)]
    feat_names = list(node_feats)
    node_tensor = np.full((len(weeks_study), len(sites), len(feat_names)), np.nan, np.float32)
    for fi, fn in enumerate(feat_names):
        m = node_feats[fn].reindex(index=weeks_study, columns=sites)
        node_tensor[:, :, fi] = m.to_numpy(dtype=np.float32)

    adj = np.zeros((len(weeks_study), len(sites), len(sites)), np.float32)
    if nv is not None:
        site_ix = {s: i for i, s in enumerate(sites)}
        nv_s = nv.reindex(weeks_study)
        for (fs, ts) in nv_s.columns:
            if fs in site_ix and ts in site_ix:
                col = nv_s[(fs, ts)].to_numpy(dtype=np.float32)
                adj[:, site_ix[fs], site_ix[ts]] = np.nan_to_num(col)

    np.savez_compressed(
        TENSOR_OUT,
        weeks=np.array([d.strftime("%Y-%m-%d") for d in weeks_study]),
        site_ids=np.array(sites),
        node_feature_names=np.array(feat_names),
        node_features=node_tensor,
        adjacency_n_voyages=adj,
    )

    _report(node_long, edge_long if not edges.empty else pd.DataFrame(),
            dark_long, weeks_study, sites, feat_names, node_tensor, adj,
            pw_lag, ev_lag, sar_lag)


def _report(node_long, edge_long, dark_long, weeks, sites, feats,
            node_tensor, adj, pw_lag, ev_lag, sar_lag) -> None:
    print(f"\n{'='*64}")
    print(f"Nodes : {NODES_OUT.name}  rows={len(node_long)}  "
          f"({len(weeks)}w x {len(sites)} sites x {len(feats)} feats)")
    print(f"Edges : {EDGES_OUT.name}  rows={len(edge_long)}")
    print(f"Dark  : {DARK_OUT.name}  rows={len(dark_long)}")
    print(f"Tensor: {TENSOR_OUT.name}  node{node_tensor.shape}  adj{adj.shape}")
    print(f"\nNode features ({len(feats)}): {feats}")

    # Node coverage per feature.
    print("\nNode feature coverage (non-null cells):")
    for f in feats:
        nn = node_long[f].notna().sum()
        print(f"  {f:24s}: {nn:5d} / {len(node_long)}")

    # Edge density.
    if len(edge_long):
        wk = edge_long["week_ending_friday"].nunique()
        pair = edge_long.groupby(["from_site", "to_site"]).size().shape[0]
        print(f"\nEdges: {wk} weeks, {pair} distinct AOI->AOI lanes, "
              f"total voyages={int(edge_long['n_voyages'].sum())}")
        top = (edge_long.groupby(["from_site", "to_site"])["n_voyages"].sum()
               .sort_values(ascending=False).head(6))
        for (fs, ts), n in top.items():
            print(f"    {fs}->{ts}: {int(n)}")

    # Dark share snapshot (last available value per region).
    if len(dark_long):
        print("\nDark-vessel share (mean over weeks, by region):")
        snap = (dark_long.groupby(["region_type", "region_id"])["dark_share"]
                .mean().sort_values(ascending=False))
        for (rt, rid), v in snap.items():
            if pd.notna(v):
                print(f"  {rt:10s} {rid:18s}: {v:.1%}")

    # Adjacency sanity: mean out-degree weight per week.
    nzw = (adj.sum(axis=(1, 2)) > 0).sum()
    print(f"\nAdjacency: {nzw}/{len(weeks)} weeks with >=1 edge; "
          f"total voyages in tensor={adj.sum():.0f}")
    print(f"\nLags applied: PW=+{pw_lag}w  GFW-event=+{ev_lag}w  SAR=+{sar_lag}w")
    print(f"{'='*64}\nDone.")


if __name__ == "__main__":
    main()
