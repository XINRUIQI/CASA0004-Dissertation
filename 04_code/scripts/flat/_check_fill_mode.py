# -*- coding: utf-8 -*-
"""Sanity checks for the fold_median leading-gap treatment (branch scratch)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

from backtest import data  # noqa: E402

df = data.load_matrix()
dico = data.load_dict()

print(f"matrix {df.shape}  {df.index.min().date()} ~ {df.index.max().date()}\n")

for mod in ("M1", "M2", "M3", "M4"):
    cols = data.select_features(dico, mod, "anom")
    dz = data.build_dataset(df, cols, 4, "all", fill_mode="zero")
    dm = data.build_dataset(df, cols, 4, "all", fill_mode="fold_median")
    same_idx = dz["idx"].equals(dm["idx"])
    Xz, Xm = dz["X"], dm["X"]
    nan_m = np.isnan(Xm)
    # rows where the two matrices differ (ignoring NaN positions)
    diff_non_nan = np.nanmax(np.abs(np.where(nan_m, 0.0, Xm) - np.where(nan_m, 0.0, Xz)))
    rows_with_nan = nan_m.any(axis=1)
    first_test = 104
    print(f"{mod:3s} n={len(dz['idx'])} raw={dz['n_raw']:3d} feats={Xz.shape[1]:4d} "
          f"| idx identical={same_idx} | non-NaN cells identical={diff_non_nan == 0}")
    print(f"    NaN cells kept for fold imputation: {nan_m.sum()} "
          f"({nan_m.mean()*100:.3f}% of matrix), affected rows={rows_with_nan.sum()}")
    if rows_with_nan.any():
        pos = np.where(rows_with_nan)[0]
        print(f"    NaN rows span {dm['idx'][pos[0]].date()} ~ {dm['idx'][pos[-1]].date()} "
              f"(last NaN row position {pos[-1]}, first scored row position {first_test})")
        print(f"    any NaN in the SCORED period (pos>={first_test})? "
              f"{bool(rows_with_nan[first_test:].any())}")
        ncols = np.where(nan_m.any(axis=0))[0]
        names = [dm["feat_names"][c].rsplit("_lag", 1)[0] for c in ncols]
        print(f"    affected raw features ({len(set(names))}): {sorted(set(names))}")
    print()

# Deep pathway uses fill_features on M1 only -> confirm zero NaN there.
m1 = data.select_features(dico, "M1", "anom")
print(f"M1 raw NaN cells in merged matrix: {int(df[m1].isna().sum().sum())} "
      f"-> Deep finance input unchanged by this branch")
