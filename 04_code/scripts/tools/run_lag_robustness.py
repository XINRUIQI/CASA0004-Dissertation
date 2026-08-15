"""Publication-lag robustness (Appendix A.3.4 / B.3.8).

The locked merged matrix already applies GFW monthly presence +4w and
MONTHLY_LAG_WEEKS +5w as a pure calendar shift after native-frequency
alignment. Extra-shifting those columns by (target - locked) is therefore
equivalent to rebuilding with the target lag, and does not touch the main
matrix. GFW-aggregate z-mean is dropped and recomputed inside build_dataset
from the shifted hours.

  GFW monthly presence : {1, 4, 8} weeks  -> Flat M3
  MONTHLY_LAG_WEEKS    : {3, 5, 7} weeks  -> Flat M1

Locked cells (GFW=4, monthly=5) are copied from the main baselines, not rerun.

Outputs -> 05_outputs/_experiments/lag_robustness/
  lag_robustness_summary.csv
  matrix_*.csv / baseline_metrics_*.csv

Run:
  python3 04_code/scripts/tools/run_lag_robustness.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "04_code" / "src"
sys.path.insert(0, str(SRC))

from backtest import data  # noqa: E402

OUT = ROOT / "05_outputs" / "_experiments" / "lag_robustness"
LOCKED_GFW = 4
LOCKED_MONTHLY = 5
MONTHLY_COLS = ["global_econ_activity", "nonoil_industrial_commodity"]
RUNNER = ROOT / "04_code" / "scripts" / "flat" / "run_baseline.py"


def extra_shift(df: pd.DataFrame, cols: list[str], k: int) -> pd.DataFrame:
    out = df.copy()
    have = [c for c in cols if c in out.columns]
    if not have or k == 0:
        return out
    out[have] = out[have].shift(k)
    return out


def write_matrix(df: pd.DataFrame, stem: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.csv"
    df.to_csv(path)
    return path


def run_flat(modality: str, matrix: Path, tag: str) -> Path:
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(RUNNER),
        "--modality", modality,
        "--matrix", str(matrix),
        "--dict", str(data.DICT_CSV),
        "--out-dir", str(out_dir),
        "--no-plot",
        "--tag", tag,
    ]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    suffix = f"_{tag}"
    return out_dir / f"baseline_metrics{suffix}.csv"


def pull_main(modality: str) -> dict[str, dict]:
    if modality == "M1":
        met = pd.read_csv(
            ROOT / "05_outputs/baselines/Flat/M1_Flat/baseline_metrics.csv",
            index_col=0)
        key = "M1_Flat"
    else:
        met = pd.read_csv(
            ROOT / "05_outputs/baselines/Flat/M3_Flat/baseline_metrics.csv",
            index_col=0)
        key = "M3_Flat"
    rows = {}
    for mdl in ("Ridge", "XGB"):
        r = met.loc[f"{key}_{mdl}"]
        rows[mdl] = {
            "RMSE": float(r["RMSE"]),
            "skill_vs_M0": float(r["RMSE_skill_vs_M0"]),
            "DM_p_vs_M0": float(r["DM_p_better_than_M0"]),
            "DM_p_vs_M1": (float(r["DM_p_vs_M1"])
                           if "DM_p_vs_M1" in r.index and pd.notna(r["DM_p_vs_M1"])
                           else None),
        }
    return rows


def rows_from_metrics(path: Path, modality: str) -> dict[str, dict]:
    met = pd.read_csv(path, index_col=0)
    key = f"{modality}_Flat"
    out = {}
    for mdl in ("Ridge", "XGB"):
        r = met.loc[f"{key}_{mdl}"]
        out[mdl] = {
            "RMSE": float(r["RMSE"]),
            "skill_vs_M0": float(r["RMSE_skill_vs_M0"]),
            "DM_p_vs_M0": float(r["DM_p_better_than_M0"]),
            "DM_p_vs_M1": (float(r["DM_p_vs_M1"])
                           if "DM_p_vs_M1" in r.index and pd.notna(r["DM_p_vs_M1"])
                           else None),
        }
    return out


def main() -> None:
    df0 = data.load_matrix()
    gfw_cols = [c for c in df0.columns if c.startswith("gfw_")]
    missing_m = [c for c in MONTHLY_COLS if c not in df0.columns]
    if missing_m:
        raise SystemExit(f"monthly columns missing: {missing_m}")
    print(f"matrix {df0.shape}  GFW cols={len(gfw_cols)}  "
          f"monthly={MONTHLY_COLS}")

    records = []

    # --- GFW {1,4,8} on Flat M3 ---
    main_m3 = pull_main("M3")
    for lag in (1, 4, 8):
        if lag == LOCKED_GFW:
            got = main_m3
            n_test = 257
        else:
            shifted = extra_shift(df0, gfw_cols, lag - LOCKED_GFW)
            if data.GFW_ZMEAN_COL in shifted.columns:
                shifted = shifted.drop(columns=[data.GFW_ZMEAN_COL])
            mat = write_matrix(shifted, f"weekly_feature_matrix_gfwlag{lag}")
            met = run_flat("M3", mat, f"gfwlag{lag}")
            got = rows_from_metrics(met, "M3")
            n_test = 257
        for mdl, v in got.items():
            records.append({
                "axis": "GFW monthly presence",
                "lag_weeks": lag,
                "locked": lag == LOCKED_GFW,
                "model": f"M3_Flat_{mdl}",
                "n_test": n_test,
                **v,
            })

    # --- MONTHLY_LAG {3,5,7} on Flat M1 ---
    main_m1 = pull_main("M1")
    for lag in (3, 5, 7):
        if lag == LOCKED_MONTHLY:
            got = main_m1
        else:
            shifted = extra_shift(df0, MONTHLY_COLS, lag - LOCKED_MONTHLY)
            mat = write_matrix(shifted, f"weekly_feature_matrix_macrolag{lag}")
            met = run_flat("M1", mat, f"macrolag{lag}")
            got = rows_from_metrics(met, "M1")
        for mdl, v in got.items():
            records.append({
                "axis": "MONTHLY_LAG_WEEKS",
                "lag_weeks": lag,
                "locked": lag == LOCKED_MONTHLY,
                "model": f"M1_Flat_{mdl}",
                "n_test": 257,
                **v,
            })

    summ = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "lag_robustness_summary.csv"
    summ.to_csv(out_csv, index=False)
    print("\n" + summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
