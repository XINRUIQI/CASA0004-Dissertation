"""
M3 Stage-2 — assemble the FULL 17-node dynamic heterogeneous shipping graph:
  11 AOI nodes  (P001..P011)  +  6 chokepoint nodes (hormuz..cape)

Combines the already-lagged Stage-2 products into one graph tensor bundle:
  * AOI node features            <- m3_graph_tensors.npz  (11 x 11 feats)
  * chokepoint node features     <- m3_weekly_features.csv (gfw_/pw_ per cp)
                                    + m3_graph_darkvessel_weekly.csv (SAR cp rows)
  * dynamic O-D edges (AOI-AOI)  <- m3_graph_tensors.npz adjacency (T,11,11)
  * static AOI<->chokepoint edges<- geographic association (sites.md sec.4)

All inputs are ALREADY leakage-lagged upstream (AOI features +1/+2/+4w in
build_m3_graph_weekly.py; chokepoint gfw +4w / pw +1w in
aggregate_shipping_to_weekly.py; SAR +4w), so no extra shift is applied here.

Heterogeneous by design: AOI and chokepoint nodes keep DIFFERENT feature
spaces (F_aoi != F_choke); the encoder uses node-type-specific input
projections, then shared message passing over the 17-node graph. Node order is
fixed: AOI 0..10 then chokepoints 11..16.

Writes (processed/M3/outputs):
  m3_graph17_tensors.npz           full 17-node bundle (see keys below)
  m3_graph17_choke_nodes_weekly.csv  chokepoint node features, long (readable)

Usage:
    python build_m3_graph17.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PY_DIR = Path(__file__).resolve().parent
M3_DIR = PY_DIR.parent
OUT_DIR = M3_DIR / "outputs"
AOI_NPZ = OUT_DIR / "m3_graph_tensors.npz"
FLAT_CSV = OUT_DIR / "m3_weekly_features.csv"
DARK_CSV = OUT_DIR / "m3_graph_darkvessel_weekly.csv"
TENSOR_OUT = OUT_DIR / "m3_graph17_tensors.npz"
CHOKE_LONG_OUT = OUT_DIR / "m3_graph17_choke_nodes_weekly.csv"

# Fixed chokepoint node order (short codes as used in m3_weekly_features).
CHOKES = ["hormuz", "suez", "malacca", "mandeb", "panama", "cape"]

# Chokepoint feature blocks (per cp) taken from the flat weekly table.
GFW_STATS = ["total_hours", "total_vessels", "cargo_hours", "bunker_hours",
             "other_hours", "other_share", "total_hours_mom_pct",
             "mean_presence_hours_per_vessel"]
PW_STATS = ["n_tanker", "n_total", "capacity_tanker", "capacity",
            "tanker_share", "tanker_cap_share", "avg_tanker_size",
            "n_tanker_wow_pct", "capacity_tanker_4w_ma"]
SAR_STATS = ["detections_total", "detections_dark", "dark_share"]

# AOI <-> chokepoint static association (aoi_oil_infrastructure_sites.md sec.4).
# P007 (Jamnagar) is a demand-side refinery whose crude slate is dominated by
# Persian Gulf loadings, so it routes through Hormuz like the export terminals.
CHOKE_AOI = {
    "hormuz": ["P002", "P003", "P007", "P008", "P010"],
    "malacca": ["P004", "P006", "P009"],
    "suez": ["P011", "P001"],
    "mandeb": ["P011"],
    "cape": ["P001"],
    "panama": ["P005"],
}


def main() -> None:
    # ---- AOI side (already-built tensor bundle) ------------------------
    az = np.load(AOI_NPZ, allow_pickle=True)
    weeks = pd.to_datetime([str(w) for w in az["weeks"]])
    aoi_ids = [str(s) for s in az["site_ids"]]
    aoi_feat_names = [str(f) for f in az["node_feature_names"]]
    aoi_features = az["node_features"].astype(np.float32)      # (T,11,Faoi)
    od_adj = az["adjacency_n_voyages"].astype(np.float32)       # (T,11,11)
    T, N_aoi, F_aoi = aoi_features.shape
    print(f"AOI tensor: {aoi_features.shape}  weeks {weeks.min().date()}~{weeks.max().date()}")

    # ---- Chokepoint features from the flat weekly table ---------------
    flat = pd.read_csv(FLAT_CSV, parse_dates=["week_ending_friday"]).set_index("week_ending_friday")
    flat = flat.reindex(weeks)
    choke_feat_names = ([f"gfw_{s}" for s in GFW_STATS]
                        + [f"pw_{s}" for s in PW_STATS]
                        + [f"sar_{s}" for s in SAR_STATS])
    F_choke = len(choke_feat_names)

    # SAR chokepoint rows (region_short) from the dark-vessel long table.
    dark = pd.read_csv(DARK_CSV, parse_dates=["week_ending_friday"])
    dark_cp = dark[dark["region_type"] == "chokepoint"].copy()
    sar_wide = {
        s: (dark_cp.pivot_table(index="week_ending_friday", columns="region_short",
                                values=s).reindex(weeks))
        for s in SAR_STATS
    }

    choke_features = np.full((T, len(CHOKES), F_choke), np.nan, np.float32)
    for ci, cp in enumerate(CHOKES):
        fi = 0
        for s in GFW_STATS:
            col = f"gfw_{cp}_{s}"
            if col in flat.columns:
                choke_features[:, ci, fi] = flat[col].to_numpy(np.float32)
            fi += 1
        for s in PW_STATS:
            col = f"pw_{cp}_{s}"
            if col in flat.columns:
                choke_features[:, ci, fi] = flat[col].to_numpy(np.float32)
            fi += 1
        for s in SAR_STATS:
            w = sar_wide[s]
            if cp in w.columns:
                choke_features[:, ci, fi] = w[cp].to_numpy(np.float32)
            fi += 1
    print(f"Chokepoint tensor: {choke_features.shape}  ({F_choke} feats/cp)")

    # ---- Node axis (17) + static AOI<->chokepoint edges ---------------
    node_ids = aoi_ids + CHOKES
    node_types = ["aoi"] * N_aoi + ["chokepoint"] * len(CHOKES)
    idx = {n: i for i, n in enumerate(node_ids)}
    N = len(node_ids)

    static = np.zeros((N, N), np.float32)
    for cp, aois in CHOKE_AOI.items():
        ci = idx[cp]
        for a in aois:
            if a in idx:
                static[idx[a], ci] = 1.0
                static[ci, idx[a]] = 1.0
    print(f"Static AOI<->chokepoint edges: {int(static.sum() // 2)} undirected")

    # ---- Combined adjacency (T,17,17): dynamic O-D block + static -----
    adj = np.zeros((T, N, N), np.float32)
    adj[:, :N_aoi, :N_aoi] = od_adj                # directed AOI->AOI O-D
    adj[:, :, :] += static[None, :, :]             # broadcast static (undirected)

    # ---- Save bundle --------------------------------------------------
    np.savez_compressed(
        TENSOR_OUT,
        weeks=np.array([d.strftime("%Y-%m-%d") for d in weeks]),
        node_ids=np.array(node_ids),
        node_types=np.array(node_types),
        aoi_feature_names=np.array(aoi_feat_names),
        choke_feature_names=np.array(choke_feat_names),
        aoi_features=aoi_features,                  # (T,11,F_aoi)
        choke_features=choke_features,              # (T,6,F_choke)
        adjacency_od=od_adj,                        # (T,11,11) dynamic only
        static_edges=static,                        # (17,17) AOI<->cp
        adjacency=adj,                              # (T,17,17) combined
    )

    # ---- Chokepoint long CSV (readable audit) -------------------------
    rows = []
    for ci, cp in enumerate(CHOKES):
        for ti, wk in enumerate(weeks):
            rec = {"week_ending_friday": wk.strftime("%Y-%m-%d"), "chokepoint": cp}
            for fj, fn in enumerate(choke_feat_names):
                rec[fn] = choke_features[ti, ci, fj]
            rows.append(rec)
    pd.DataFrame(rows).to_csv(CHOKE_LONG_OUT, index=False)

    _report(weeks, node_ids, node_types, aoi_features, choke_features,
            adj, static, aoi_feat_names, choke_feat_names)


def _report(weeks, node_ids, node_types, aoi_f, choke_f, adj, static,
            aoi_names, choke_names) -> None:
    T = len(weeks)
    print(f"\n{'='*64}")
    print(f"Output: {TENSOR_OUT.name}")
    print(f"Nodes : {len(node_ids)} = {node_types.count('aoi')} AOI + "
          f"{node_types.count('chokepoint')} chokepoint")
    print(f"  order: {node_ids}")
    print(f"Weeks : {T}  ({weeks.min().date()} ~ {weeks.max().date()})")
    print(f"AOI   features ({len(aoi_names)}): {aoi_names}")
    print(f"Choke features ({len(choke_names)}): {choke_names}")
    print(f"Tensors: aoi{aoi_f.shape}  choke{choke_f.shape}  adj{adj.shape}")

    # Coverage.
    aoi_cov = np.isfinite(aoi_f).any(axis=2).mean()
    choke_cov = np.isfinite(choke_f).any(axis=2).mean()
    print(f"\nNode coverage (any feature non-NaN): AOI={aoi_cov:.1%}  "
          f"chokepoint={choke_cov:.1%}")

    # Static edge degrees.
    print(f"\nStatic AOI<->chokepoint degree (per chokepoint):")
    N_aoi = node_types.count("aoi")
    for i, n in enumerate(node_ids):
        if node_types[i] == "chokepoint":
            deg = int(static[i, :N_aoi].sum())
            linked = [node_ids[j] for j in range(N_aoi) if static[i, j] > 0]
            print(f"  {n:8s}: {deg}  {linked}")

    # Adjacency density per week (mean edges).
    wk_edges = (adj > 0).sum(axis=(1, 2))
    print(f"\nAdjacency: mean {wk_edges.mean():.1f} edges/week "
          f"(min {wk_edges.min()}, max {wk_edges.max()}); "
          f"static contributes {int((static > 0).sum())} every week")
    print(f"{'='*64}\nDone.")


if __name__ == "__main__":
    main()
