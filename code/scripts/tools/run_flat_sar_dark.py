"""Experiment: add GFW SAR dark-vessel columns to Flat M3.

Does not modify the locked merge matrix, dictionary, or Flat/Deep runners.
Copies the main matrix/dict into results/experiments/flat_sar_dark/,
appends already-lagged SAR columns from m3_graph_darkvessel_weekly.csv, and
calls run_baseline.py with --matrix/--dict/--out-dir.

Arms
  cp   : main 113 shipping cols + 6 chokepoints x {total, dark, share}
  all  : main 113 + 17 regions x {total, dark, share}  (Deep geography)

Locked Flat metrics are copied, not rerun.

Run:
  python3 code/scripts/tools/run_flat_sar_dark.py
  python3 code/scripts/tools/run_flat_sar_dark.py --only m4
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "code" / "src"
sys.path.insert(0, str(SRC))

from backtest import data  # noqa: E402

OUT = ROOT / "results" / "_experiments" / "flat_sar_dark"
DARK_CSV = ROOT / "data/processed/M3/outputs/m3_graph_darkvessel_weekly.csv"
AOI_CSV = ROOT / "data/raw/02_sentinel2/aoi_oil_infrastructure.csv"
RUNNER = ROOT / "code" / "scripts" / "flat" / "run_baseline.py"
STATS = [
    ("detections_total", "total"),
    ("detections_dark", "dark"),
    ("dark_share", "share"),
]


def aoi_short_map() -> dict[str, str]:
    aoi = pd.read_csv(AOI_CSV)
    return {r.site_id: str(r.site_short).lower() for r in aoi.itertuples()}


def region_key(row, aoi_map: dict[str, str]) -> str:
    if row.region_type == "chokepoint":
        return f"cp_{row.region_short}"
    return f"aoi_{aoi_map.get(row.region_id, str(row.region_id).lower())}"


def pivot_sar(keep: str, aoi_map: dict[str, str]) -> pd.DataFrame:
    dark = pd.read_csv(DARK_CSV, parse_dates=["week_ending_friday"])
    if keep == "cp":
        dark = dark[dark["region_type"] == "chokepoint"].copy()
    dark["key"] = dark.apply(lambda r: region_key(r, aoi_map), axis=1)
    frames = []
    for src, short in STATS:
        wide = dark.pivot_table(
            index="week_ending_friday",
            columns="key",
            values=src,
            aggfunc="first",
        )
        wide.columns = [f"sar_{c}_{short}" for c in wide.columns]
        frames.append(wide)
    out = pd.concat(frames, axis=1).sort_index()
    return out


def dict_rows(cols: list[str], matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = matrix[c]
        nn = int(s.notna().sum())
        rows.append({
            "feature": c,
            "modality": "M3",
            "group": "GFW SAR dark-vessel (experiment)",
            "publication_lag": "already lagged in graph builder (+4w)",
            "n_nonnull": nn,
            "pct_nonnull": round(100.0 * nn / len(s), 1),
            "coverage_start": s.first_valid_index(),
            "coverage_end": s.last_valid_index(),
        })
    return pd.DataFrame(rows)


def write_bundle(df: pd.DataFrame, dico: pd.DataFrame, tag: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    mat_path = OUT / f"weekly_feature_matrix_{tag}.csv"
    dic_path = OUT / f"weekly_feature_dictionary_{tag}.csv"
    df.to_csv(mat_path)
    dico.to_csv(dic_path, index=False)
    return mat_path, dic_path


def run_flat(matrix: Path, dictionary: Path, tag: str,
             modality: str = "M3") -> Path:
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUNNER),
        "--modality", modality,
        "--matrix", str(matrix),
        "--dict", str(dictionary),
        "--out-dir", str(out_dir),
        "--no-plot",
        "--tag", tag,
    ]
    if modality == "M4":
        cmd += ["--m2-features", "anom"]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir / f"baseline_metrics_{tag}.csv"


def metrics_from(path: Path, model_key: str) -> dict:
    met = pd.read_csv(path, index_col=0)
    out = {}
    for mdl in ("Ridge", "XGB"):
        r = met.loc[f"{model_key}_{mdl}"]
        out[mdl] = {
            "RMSE": float(r["RMSE"]),
            "skill_vs_M0": float(r["RMSE_skill_vs_M0"]),
            "DM_p_vs_M0": float(r["DM_p_better_than_M0"]),
            "DM_p_vs_M1": (float(r["DM_p_vs_M1"])
                           if "DM_p_vs_M1" in r.index and pd.notna(r["DM_p_vs_M1"])
                           else None),
        }
    return out


def extra_n(keep: str, aoi_map: dict[str, str]) -> int:
    return pivot_sar(keep, aoi_map).shape[1]


def append_locked(records: list, modality: str, n_ship: int) -> None:
    if modality == "M3":
        path = ROOT / "results/baselines/Flat/M3_Flat/baseline_metrics.csv"
        key = "M3_Flat"
        arm = "locked_M3"
    else:
        path = ROOT / "results/baselines/Flat/M4_Flat/baseline_metrics_anom.csv"
        key = "M4_Flat"
        arm = "locked_M4"
    locked = metrics_from(path, key)
    for mdl, v in locked.items():
        records.append({
            "arm": arm,
            "n_shipping_cols": n_ship,
            "model": f"{key}_{mdl}",
            **v,
        })


def run_sar_arms(records: list, dico0: pd.DataFrame, n_m3: int,
                 aoi_map: dict[str, str], modality: str) -> None:
    prefix = "m3" if modality == "M3" else "m4"
    key = f"{modality}_Flat"
    for keep in ("cp", "all"):
        tag = f"{prefix}_sar_{keep}"
        mat_path = OUT / f"weekly_feature_matrix_m3_sar_{keep}.csv"
        dic_path = OUT / f"weekly_feature_dictionary_m3_sar_{keep}.csv"
        if not mat_path.exists() or not dic_path.exists():
            df0 = data.load_matrix()
            sar = pivot_sar(keep, aoi_map).reindex(df0.index)
            extra = [c for c in sar.columns if c not in df0.columns]
            df = df0.join(sar[extra])
            dico = pd.concat([dico0, dict_rows(extra, df)], ignore_index=True)
            mat_path, dic_path = write_bundle(df, dico, f"m3_sar_{keep}")
        n_extra = extra_n(keep, aoi_map)
        print(f"\narm {tag}: +{n_extra} SAR cols  modality={modality}",
              flush=True)
        met_path = run_flat(mat_path, dic_path, tag, modality=modality)
        got = metrics_from(met_path, key)
        for mdl, v in got.items():
            records.append({
                "arm": tag,
                "n_shipping_cols": n_m3 + n_extra,
                "model": f"{key}_{mdl}",
                **v,
            })


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("m3", "m4", "all"), default="all")
    args = ap.parse_args()

    dico0 = data.load_dict()
    aoi_map = aoi_short_map()
    n_m3 = int((dico0["modality"] == "M3").sum())
    print(f"locked dict {dico0.shape}  M3 cols={n_m3}", flush=True)

    records: list[dict] = []
    if args.only in ("m3", "all"):
        append_locked(records, "M3", n_m3)
        run_sar_arms(records, dico0, n_m3, aoi_map, "M3")
    if args.only in ("m4", "all"):
        append_locked(records, "M4", n_m3)
        run_sar_arms(records, dico0, n_m3, aoi_map, "M4")

    if args.only != "all" and (OUT / "flat_sar_dark_summary.csv").exists():
        prev = pd.read_csv(OUT / "flat_sar_dark_summary.csv")
        new = pd.DataFrame(records)
        keep = prev[~prev["arm"].isin(new["arm"])]
        summ = pd.concat([keep, new], ignore_index=True)
    else:
        summ = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "flat_sar_dark_summary.csv"
    summ.to_csv(out_csv, index=False)
    print("\n" + summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
