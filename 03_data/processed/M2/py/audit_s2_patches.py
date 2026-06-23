"""
Channel A — Sentinel-2 patch download / validity audit.

Cross-checks ``S2_patches_manifest_ALL.csv`` against on-disk GeoTIFFs.
Applies ``s2_patch_exclusions.csv`` and all-zero pixel checks so empty
exports (e.g. P008 2019_01) count as missing (valid_mask=0).

Outputs (-> 03_data/processed/M2/outputs/):
  s2_patch_index.csv              per (site, month) validity index
  s2_patch_coverage_report.csv    per-site summary
  s2_patch_validity_heatmap.png   site x month heatmap

Run:
  python3 03_data/processed/M2/py/audit_s2_patches.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from s2_patch_utils import EXCLUSIONS_CSV, PATCH_DIR, build_patch_index, load_exclusions

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "03_data/processed/M2/outputs"
TYPE_ORDER = {"port": 0, "refinery": 1, "terminal": 2}


def site_order(df: pd.DataFrame) -> list[str]:
    meta = (
        df[["site_name", "site_type"]]
        .drop_duplicates()
        .assign(_t=lambda d: d["site_type"].map(TYPE_ORDER))
        .sort_values(["_t", "site_name"])
    )
    return meta["site_name"].tolist()


def plot_validity_heatmap(index: pd.DataFrame, order: list[str], path: Path) -> None:
    g = index.copy()
    g["ym"] = g["month"].str.replace("_", "-")
    piv = g.pivot_table(index="site_name", columns="ym", values="valid_mask", aggfunc="max")
    piv = piv.reindex(order)
    plt.figure(figsize=(18, 5))
    sns.heatmap(
        piv,
        cmap=ListedColormap(["#d73027", "#1a9850"]),
        cbar=False,
        linewidths=0.3,
        linecolor="white",
    )
    plt.title("Channel A S2 patch validity  (green = usable, red = missing/invalid/excluded)")
    plt.xlabel("month")
    plt.ylabel("")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_site_report(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, sname, stype), g in index.groupby(["site_id", "site_name", "site_type"]):
        n = len(g)
        n_usable = int(g["valid_mask"].sum())
        n_missing_file = int((g["file_exists"] == 0).sum())
        n_excluded = int(g["excluded"].sum())
        n_empty = int(((g["file_exists"] == 1) & (g["valid_mask"] == 0) & (g["excluded"] == 0)).sum())
        rows.append(
            dict(
                site_id=sid,
                site_name=sname,
                site_type=stype,
                n_expected=n,
                n_usable=n_usable,
                coverage=round(n_usable / n, 3) if n else None,
                n_missing_file=n_missing_file,
                n_excluded=n_excluded,
                n_empty_on_disk=n_empty,
            )
        )
    return pd.DataFrame(rows).sort_values(["site_type", "site_name"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", type=Path, default=PATCH_DIR)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    exclusions = load_exclusions()
    index_rows = build_patch_index(patch_dir=args.patch_dir, exclusions=exclusions)
    index = pd.DataFrame(index_rows)
    order = site_order(index)

    index_path = OUT / "s2_patch_index.csv"
    index.to_csv(index_path, index=False)

    report = build_site_report(index)
    report_path = OUT / "s2_patch_coverage_report.csv"
    report.to_csv(report_path, index=False)

    heatmap_path = OUT / "s2_patch_validity_heatmap.png"
    plot_validity_heatmap(index, order, heatmap_path)

    n_total = len(index)
    n_usable = int(index["valid_mask"].sum())
    print(f"patch dir    : {args.patch_dir}")
    print(f"exclusions   : {len(exclusions)} manual row(s) -> {EXCLUSIONS_CSV.name}")
    print(f"index        : {index_path}")
    print(f"report       : {report_path}")
    print(f"heatmap      : {heatmap_path}")
    print(f"usable       : {n_usable}/{n_total} ({100 * n_usable / n_total:.1f}%)")

    bad = index[index["valid_mask"] == 0][
        ["site_id", "site_name", "month", "filename", "file_exists", "excluded", "exclude_reason"]
    ]
    if not bad.empty:
        print("\n[missing or invalid patches]")
        print(bad.to_string(index=False))

    print("\n[per-site coverage]")
    print(report[["site_name", "n_usable", "n_expected", "coverage", "n_missing_file", "n_empty_on_disk"]].to_string(index=False))


if __name__ == "__main__":
    main()
