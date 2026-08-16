"""
Export QGIS-ready layers for Figure 3.3 (study-site world map).

Reads the canonical 11-AOI table and writes points, 6 chokepoints, 13 static
corridor edges, the Persian Gulf inset box, and Natural Earth 110m basemap
GeoJSON into 05_outputs/figures/fig_3_3_map_layers/.

    python 04_code/scripts/figures/export_fig33_map_layers.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[3]
AOI_CSV = ROOT / "03_data" / "raw" / "02_sentinel2" / "aoi_oil_infrastructure.csv"
NE_DIR = ROOT / "03_data" / "raw" / "00_spatial_anchors" / "naturalearth"
OUT = ROOT / "05_outputs" / "figures" / "fig_3_3_map_layers"

# Same representative coordinates as make_method_figures.py (EIA transit points).
CHOKE = {
    "hormuz": ("Strait of Hormuz", "霍尔木兹海峡", 56.25, 26.57, 11),
    "suez": ("Suez Canal", "苏伊士运河", 32.35, 30.42, 12),
    "malacca": ("Strait of Malacca", "马六甲海峡", 100.40, 2.50, 13),
    "mandeb": ("Bab el-Mandeb", "曼德海峡", 43.40, 12.60, 14),
    "panama": ("Panama Canal", "巴拿马运河", -79.55, 9.08, 15),
    "cape": ("Cape of Good Hope", "好望角", 18.47, -34.36, 16),
}

# Appendix A.4.2: 13 undirected AOI–chokepoint corridor edges.
EDGES = {
    "hormuz": ["P002", "P003", "P007", "P008", "P010"],
    "suez": ["P001", "P011"],
    "malacca": ["P004", "P006", "P009"],
    "mandeb": ["P011"],
    "cape": ["P001"],
    "panama": ["P005"],
}

MAP_LABEL = {
    "P001": "Rotterdam",
    "P002": "Fujairah",
    "P003": "Ras Tanura",
    "P004": "Jurong Island",
    "P005": "Houston",
    "P006": "Ningbo-Zhoushan",
    "P007": "Jamnagar",
    "P008": "Al Basrah Terminal",
    "P009": "Ulsan",
    "P010": "Kharg Island",
    "P011": "Yanbu",
}

ROLE = {
    "P001": "pricing / import",
    "P002": "transit / storage",
    "P003": "export",
    "P004": "transit / refining",
    "P005": "import / refining",
    "P006": "import",
    "P007": "refining",
    "P008": "export",
    "P009": "refining",
    "P010": "export",
    "P011": "export",
}

DEEP_PATCH_KM = {
    "P001": 6.4, "P002": 3.2, "P003": 2.56, "P004": 5.12, "P005": 6.4,
    "P006": 6.4, "P007": 5.12, "P008": 1.6, "P009": 5.12, "P010": 3.2,
    "P011": 3.2,
}

INSET_SITES = {"P002", "P003", "P008", "P010", "hormuz"}
# lon0, lat0, lon1, lat1 — same as make_method_figures.INSET_BOUNDS
INSET_BOUNDS = (46.5, 22.5, 60.0, 32.0)
# Main-map frame used by fig_3_3_study_sites_map
MAP_EXTENT = (-135.0, -48.0, 152.0, 68.0)

CRS = "EPSG:4326"


def _fc(rows: list[dict], geom_key: str = "geometry") -> dict:
    feats = []
    for r in rows:
        props = {k: v for k, v in r.items() if k != geom_key}
        feats.append({"type": "Feature", "properties": props, "geometry": r[geom_key]})
    return {"type": "FeatureCollection", "features": feats}


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _pt(lon: float, lat: float) -> dict:
    return {"type": "Point", "coordinates": [lon, lat]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aoi_src = list(csv.DictReader(AOI_CSV.open(encoding="utf-8-sig")))
    aoi_src.sort(key=lambda r: r["site_id"])

    aoi_rows = []
    aoi_xy = {}
    for i, r in enumerate(aoi_src):
        sid = r["site_id"]
        lon, lat = float(r["lon"]), float(r["lat"])
        aoi_xy[sid] = (lon, lat)
        choke_codes = [cp for cp, sites in EDGES.items() if sid in sites]
        row = {
            "site_id": sid,
            "graph_index": i,
            "map_label": MAP_LABEL[sid],
            "site_name": r["site_name"],
            "site_type": r["site_type"],
            "functional_role": ROLE[sid],
            "country": r["country"],
            "region": r["region"],
            "lon": lon,
            "lat": lat,
            "flat_buffer_km": float(r["buffer_km"]),
            "deep_patch_km": DEEP_PATCH_KM[sid],
            "chokepoints": " · ".join(choke_codes),
            "in_gulf_inset": int(sid in INSET_SITES),
            "node_class": "aoi",
            "geometry": _pt(lon, lat),
        }
        aoi_rows.append(row)

    choke_rows = []
    choke_xy = {}
    for code, (en, zh, lon, lat, idx) in CHOKE.items():
        choke_xy[code] = (lon, lat)
        choke_rows.append({
            "choke_id": code,
            "graph_index": idx,
            "map_label": en,
            "name_zh": zh,
            "lon": lon,
            "lat": lat,
            "n_corridor_edges": len(EDGES[code]),
            "linked_aois": " · ".join(EDGES[code]),
            "in_gulf_inset": int(code in INSET_SITES),
            "node_class": "chokepoint",
            "geometry": _pt(lon, lat),
        })

    edge_rows = []
    eid = 1
    for cp, sites in EDGES.items():
        clon, clat = choke_xy[cp]
        for sid in sites:
            slon, slat = aoi_xy[sid]
            edge_rows.append({
                "edge_id": f"E{eid:02d}",
                "aoi_id": sid,
                "aoi_label": MAP_LABEL[sid],
                "choke_id": cp,
                "choke_label": CHOKE[cp][0],
                "weight": 1,
                "edge_class": "static_corridor",
                "in_gulf_inset": int(sid in INSET_SITES or cp in INSET_SITES),
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[clon, clat], [slon, slat]],
                },
            })
            eid += 1

    lon0, lat0, lon1, lat1 = INSET_BOUNDS
    inset = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": "Persian Gulf inset",
                "name_zh": "波斯湾放大框",
                "lon_min": lon0, "lat_min": lat0,
                "lon_max": lon1, "lat_max": lat1,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon0, lat0], [lon1, lat0], [lon1, lat1],
                    [lon0, lat1], [lon0, lat0],
                ]],
            },
        }],
    }
    m0, n0, m1, n1 = MAP_EXTENT
    extent = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Figure 3.3 map frame"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [m0, n0], [m1, n0], [m1, n1], [m0, n1], [m0, n0],
                ]],
            },
        }],
    }

    _write_json(OUT / "aoi_points.geojson", _fc(aoi_rows))
    _write_json(OUT / "chokepoint_points.geojson", _fc(choke_rows))
    _write_json(OUT / "corridor_edges.geojson", _fc(edge_rows))
    _write_json(OUT / "persian_gulf_inset.geojson", inset)
    _write_json(OUT / "map_extent.geojson", extent)

    _write_csv(OUT / "aoi_points.csv", aoi_rows, [
        "site_id", "graph_index", "map_label", "site_name", "site_type",
        "functional_role", "country", "region", "lon", "lat",
        "flat_buffer_km", "deep_patch_km", "chokepoints", "in_gulf_inset",
    ])
    _write_csv(OUT / "chokepoint_points.csv", choke_rows, [
        "choke_id", "graph_index", "map_label", "name_zh", "lon", "lat",
        "n_corridor_edges", "linked_aois", "in_gulf_inset",
    ])
    _write_csv(OUT / "corridor_edges.csv", edge_rows, [
        "edge_id", "aoi_id", "aoi_label", "choke_id", "choke_label",
        "weight", "in_gulf_inset",
    ])

    # Combined 17-node layer (handy for a single labelled point layer).
    nodes = []
    for r in aoi_rows:
        nodes.append({
            "node_id": r["site_id"],
            "graph_index": r["graph_index"],
            "map_label": r["map_label"],
            "node_class": "aoi",
            "site_type": r["site_type"],
            "lon": r["lon"], "lat": r["lat"],
            "in_gulf_inset": r["in_gulf_inset"],
            "geometry": r["geometry"],
        })
    for r in choke_rows:
        nodes.append({
            "node_id": r["choke_id"],
            "graph_index": r["graph_index"],
            "map_label": r["map_label"],
            "node_class": "chokepoint",
            "site_type": "chokepoint",
            "lon": r["lon"], "lat": r["lat"],
            "in_gulf_inset": r["in_gulf_inset"],
            "geometry": r["geometry"],
        })
    _write_json(OUT / "nodes_17.geojson", _fc(nodes))

    for name in ("ne_110m_land", "ne_110m_admin_0_boundary_lines_land"):
        src = NE_DIR / f"{name}.shp"
        if not src.exists():
            print(f"  skip {name}: {src} missing")
            continue
        gdf = gpd.read_file(src).to_crs(CRS)
        gdf.to_file(OUT / f"{name}.geojson", driver="GeoJSON")
        print(f"  basemap {name}: {len(gdf)} features")

    print(f"AOI {len(aoi_rows)}  chokepoints {len(choke_rows)}  "
          f"corridor edges {len(edge_rows)}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
