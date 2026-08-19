"""
B0 - Channel B (M2) coverage & quality audit.

Audits the two raw Channel B monthly tables for the 11 oil-infrastructure AOIs,
focused on the standardised 2019-2026 comparison window:

  - Sentinel-2 optical indices: NDVI / NDWI / NDBI / BSI
  - VIIRS night-time lights:     NTL (ntl_avg_rad_mean)

For each (site, index) it reports:
  - coverage  = non-null months / total months in window
  - std       = within-window temporal std  -> *information content*
                (a near-constant index, e.g. NDVI/NDWI over water-dominated
                 terminals, carries little signal even when "present")
  - cloud_probability and valid_obs_count as S2 data-quality context

Outputs (-> data/processed/M2/outputs/):
  m2_coverage_report.csv        tidy per-(site,index) report
  m2_s2_missing_heatmap.png     site x month validity (S2)
  m2_coverage_heatmap.png       site x index coverage
  m2_variability_heatmap.png    site x index std (per-index normalised)
  m2_cloud_validobs.png         per-site mean cloud% and valid_obs

Run:
  python3 data/processed/M2/py/audit_m2_coverage.py
  python3 data/processed/M2/py/audit_m2_coverage.py --start 2017-04
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "data/raw/02_sentinel2/Channel B"
OUT = ROOT / "data/processed/M2/outputs"
S2_CSV = RAW / "sentinel2_oil_sites_monthly_indices_201704_202512_11aoi.csv"
NTL_CSV = RAW / "viirs_oil_sites_monthly_nightlights_201401_202512_11aoi.csv"

S2_INDICES = ["NDVI", "NDWI", "NDBI", "BSI"]
ALL_INDICES = S2_INDICES + ["NTL"]
TYPE_ORDER = {"port": 0, "refinery": 1, "terminal": 2}


def load_s2() -> pd.DataFrame:
    df = pd.read_csv(S2_CSV)
    df["date"] = pd.to_datetime(df["date_month"] + "-01")
    return df


def load_ntl() -> pd.DataFrame:
    df = pd.read_csv(NTL_CSV)
    df["date"] = pd.to_datetime(df["date_month"] + "-01")
    return df


def site_order(df: pd.DataFrame) -> list[str]:
    meta = (
        df[["site_name", "site_type"]]
        .drop_duplicates()
        .assign(_t=lambda d: d["site_type"].map(TYPE_ORDER))
        .sort_values(["_t", "site_name"])
    )
    return meta["site_name"].tolist()


def build_report(s2: pd.DataFrame, ntl: pd.DataFrame,
                 start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    s2w = s2[(s2["date"] >= start) & (s2["date"] <= end)]
    for (sid, sname, stype), g in s2w.groupby(["site_id", "site_name", "site_type"]):
        n_total = g["date"].dt.to_period("M").nunique()
        for idx in S2_INDICES:
            vals = g[idx]
            n_valid = int(vals.notna().sum())
            rows.append(dict(
                modality="S2", site_id=sid, site_name=sname, site_type=stype,
                index=idx, n_total=int(n_total), n_valid=n_valid,
                coverage=round(n_valid / n_total, 3) if n_total else np.nan,
                mean=round(float(vals.mean()), 4) if n_valid else np.nan,
                std=round(float(vals.std()), 4) if n_valid else np.nan,
                mean_cloud_prob=round(float(g["cloud_probability"].mean()), 2)
                if g["cloud_probability"].notna().any() else np.nan,
                mean_valid_obs=round(float(g["valid_obs_count"].mean()), 2),
            ))

    ntlw = ntl[(ntl["date"] >= start) & (ntl["date"] <= end)]
    for (sid, sname, stype), g in ntlw.groupby(["site_id", "site_name", "site_type"]):
        n_total = g["date"].dt.to_period("M").nunique()
        vals = g["ntl_avg_rad_mean"]
        n_valid = int(vals.notna().sum())
        rows.append(dict(
            modality="VIIRS", site_id=sid, site_name=sname, site_type=stype,
            index="NTL", n_total=int(n_total), n_valid=n_valid,
            coverage=round(n_valid / n_total, 3) if n_total else np.nan,
            mean=round(float(vals.mean()), 3) if n_valid else np.nan,
            std=round(float(vals.std()), 3) if n_valid else np.nan,
            mean_cloud_prob=np.nan,
            mean_valid_obs=round(float(g["ntl_cf_cvg_mean"].mean()), 2),
        ))
    return pd.DataFrame(rows)


def plot_s2_missing(s2, order, start, end, path):
    g = s2[(s2["date"] >= start) & (s2["date"] <= end)].copy()
    g["valid"] = g[S2_INDICES].notna().any(axis=1).astype(int)
    g["ym"] = g["date"].dt.strftime("%Y-%m")
    piv = g.pivot_table(index="site_name", columns="ym", values="valid", aggfunc="max")
    piv = piv.reindex(order)
    plt.figure(figsize=(18, 5))
    sns.heatmap(piv, cmap=ListedColormap(["#d73027", "#1a9850"]),
                cbar=False, linewidths=0.3, linecolor="white")
    plt.title("Sentinel-2 monthly validity  (green = valid optical obs, red = missing)")
    plt.xlabel("month")
    plt.ylabel("")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_coverage(report, order, path):
    piv = report.pivot_table(index="site_name", columns="index", values="coverage")
    piv = piv.reindex(index=order, columns=ALL_INDICES)
    plt.figure(figsize=(7, 6))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, cbar_kws={"label": "coverage"})
    plt.title("Channel B coverage = non-null months / total")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_variability(report, order, path):
    piv = report.pivot_table(index="site_name", columns="index", values="std")
    piv = piv.reindex(index=order, columns=ALL_INDICES)
    norm = piv / piv.max(axis=0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="viridis",
                linewidths=0.5, cbar_kws={"label": "std (per-index max-normalised)"})
    plt.title("Information content = temporal std (per-index normalised)\n"
              "low value = near-constant = little signal")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cloud_validobs(report, order, path):
    s2r = (report[report["modality"] == "S2"]
           .drop_duplicates("site_name")
           .set_index("site_name")
           .reindex(order))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(s2r.index, s2r["mean_cloud_prob"], color="#4575b4")
    axes[0].set_title("Mean S2 cloud_probability")
    axes[0].invert_yaxis()
    axes[1].barh(s2r.index, s2r["mean_valid_obs"], color="#91bfdb")
    axes[1].set_title("Mean S2 valid_obs_count / month")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01", help="window start YYYY-MM")
    ap.add_argument("--end", default="2025-12", help="window end YYYY-MM")
    args = ap.parse_args()

    start = pd.Timestamp(args.start + "-01")
    end = pd.Timestamp(args.end + "-01") + pd.offsets.MonthEnd(0)

    OUT.mkdir(parents=True, exist_ok=True)
    s2, ntl = load_s2(), load_ntl()
    order = site_order(s2)

    report = build_report(s2, ntl, start, end)
    report = report.sort_values(["modality", "site_type", "site_name", "index"])
    rep_path = OUT / "m2_coverage_report.csv"
    report.to_csv(rep_path, index=False)

    plot_s2_missing(s2, order, start, end, OUT / "m2_s2_missing_heatmap.png")
    plot_coverage(report, order, OUT / "m2_coverage_heatmap.png")
    plot_variability(report, order, OUT / "m2_variability_heatmap.png")
    plot_cloud_validobs(report, order, OUT / "m2_cloud_validobs.png")

    print(f"window      : {args.start} .. {args.end}")
    print(f"report      : {rep_path}")
    print(f"figures     : {OUT}/m2_*.png")

    cov = report.pivot_table(index=["site_type", "site_name"],
                             columns="index", values="coverage")[ALL_INDICES]
    print("\n[coverage by site x index]")
    print(cov.round(2).to_string())

    std = report.pivot_table(index=["site_type", "site_name"],
                             columns="index", values="std")[ALL_INDICES]
    print("\n[temporal std by site x index]  (low = near-constant = little signal)")
    print(std.round(3).to_string())

    typ = (report[report["index"].isin(["NDVI", "NDWI", "NDBI", "BSI"])]
           .groupby(["site_type", "index"])["std"].mean().round(4)
           .unstack("index")[S2_INDICES])
    print("\n[mean temporal std by site_type]  (port vs refinery vs terminal)")
    print(typ.to_string())


if __name__ == "__main__":
    main()
