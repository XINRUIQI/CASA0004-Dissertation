"""
Generic early/late sub-period scorer for Flat AND Deep predictions (Appendix B).

Why this file exists
--------------------
`run_deep_advanced.py` splits the common test window into early (<=2022) vs
late (>=2023) and re-checks skill / CW-vs-M0, but ONLY for the M4 deep fusion
arms and only by re-running the deep models. This script reuses the SAME split
and the SAME accuracy tests, but works OFFLINE on any saved predictions CSV, so
Flat M1-M4 and every Deep config get an apples-to-apples sub-period table
without retraining anything.

It does not overwrite or import `run_deep_advanced.py`; it only reuses the
shared `backtest.metrics` module so the numbers match the main pipeline.

Convention (every predictions CSV already follows it)
-----------------------------------------------------
  index            = date (weekly Friday)
  P_next_actual    = realised next-week price
  r_actual         = realised next-week log return
  P_hat_M0         = random-walk benchmark price
  P_hat_<MODEL>    = model price (paired with r_hat_<MODEL>)

For each MODEL column (except M0) and each period we report, on the common
weeks: n_test, RMSE, MAE, skill vs M0, DirAcc, Clark-West p vs M0 (nested ->
"beats the random walk?") and Diebold-Mariano p vs a non-nested reference
(default M1_Flat_Ridge, matching run_deep_advanced.py).

Run (default manifest = Flat M1-M4 + Deep):
  python3 04_code/scripts/tools/subperiod_eval.py

Score arbitrary files:
  python3 04_code/scripts/tools/subperiod_eval.py \
      --pred 05_outputs/baselines/Deep/_cross/deep_predictions.csv:Deep_all \
      --pred 05_outputs/baselines/Flat/M3_Flat/baseline_predictions.csv:M3_Flat \
      --split 2023-01-01
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics  # noqa: E402
from model_naming import deep_cross_predictions  # noqa: E402

OUT_DIR = data.ROOT / "05_outputs/baselines/subperiod"
DEFAULT_SPLIT = pd.Timestamp("2023-01-01")
DEFAULT_REF_COL = "P_hat_M1_Flat_Ridge"   # non-nested DM reference (see metrics)

# label -> (relative predictions path, regex the MODEL name must match).
# The regex keeps each file to its own family so M1_Flat is not duplicated
# across every flat file. `None` keeps all non-M0 models in the file.
DEFAULT_MANIFEST: list[tuple[str, str, "str | None"]] = [
    ("M1_Flat", "05_outputs/baselines/Flat/M1_Flat/baseline_predictions.csv", r"^M1_Flat"),
    ("M2_Flat", "05_outputs/baselines/Flat/M2_Flat/baseline_predictions_anom.csv", r"^M2_Flat"),
    ("M3_Flat", "05_outputs/baselines/Flat/M3_Flat/baseline_predictions.csv", r"^M3_Flat"),
    ("M4_Flat", "05_outputs/baselines/Flat/M4_Flat/baseline_predictions_anom.csv", r"^M4_Flat"),
    ("M1_Deep", "05_outputs/baselines/Deep/M1_Deep/baseline_predictions.csv", r"^M1_Deep"),
    ("M2_Deep", "05_outputs/baselines/Deep/M2_Deep/baseline_predictions.csv", r"(M2_Deep|M_rs_deep)"),
    ("M3_Deep", "05_outputs/baselines/Deep/M3_Deep/baseline_predictions.csv", r"(M3_Deep|M_ship)"),
    ("M4_Deep", "05_outputs/baselines/Deep/M4_Deep/baseline_predictions.csv", r"^M4_Deep"),
    ("Deep_all", str(deep_cross_predictions(data.ROOT).relative_to(data.ROOT)),
     r"(Deep|GNN|rs_deep)"),
]


def load_predictions(path: Path) -> pd.DataFrame:
    """Read a predictions CSV with a parsed weekly-date index."""
    res = pd.read_csv(path, index_col=0, parse_dates=True)
    res.index.name = "date"
    return res


def list_models(res: pd.DataFrame, keep: "str | None" = None) -> list[str]:
    """All `P_hat_<MODEL>` columns except M0, optionally filtered by `keep`."""
    models = [c[len("P_hat_"):] for c in res.columns
              if c.startswith("P_hat_") and c != "P_hat_M0"]
    if keep:
        pat = re.compile(keep)
        models = [m for m in models if pat.search(m)]
    return models


def score_period(res: pd.DataFrame, model: str, idx: pd.Index,
                 ref_res: "pd.DataFrame | None" = None,
                 ref_col: str = DEFAULT_REF_COL) -> dict:
    """Price-space metrics for one model on the weeks in `idx`.

    CW vs M0 treats M0 as the nested (restricted) forecast, so a small p means
    the model significantly beats the random walk. DM vs the reference is the
    non-nested equal-accuracy test (skipped if < 30 shared weeks)."""
    common = res.index.intersection(idx)
    y = res.loc[common, "P_next_actual"].to_numpy()
    ym0 = res.loc[common, "P_hat_M0"].to_numpy()
    yhat = res.loc[common, f"P_hat_{model}"].to_numpy()
    rhat = res.loc[common, f"r_hat_{model}"].to_numpy()
    ract = res.loc[common, "r_actual"].to_numpy()

    rmse_m0 = float(np.sqrt(np.mean((ym0 - y) ** 2)))
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    mae = float(np.mean(np.abs(yhat - y)))
    _, cw0 = metrics.clark_west(y, ym0, yhat)

    out = {
        "n_test": int(len(common)),
        "RMSE": rmse,
        "MAE": mae,
        "skill_vs_M0": float(1 - rmse / rmse_m0) if rmse_m0 > 0 else np.nan,
        "DirAcc": metrics.directional_acc(rhat, ract),
        "CW_p_vs_M0": cw0,
        "DM_p_vs_ref": np.nan,
    }

    ref = ref_res if ref_res is not None else res
    if ref_col in ref.columns and ref_col != f"P_hat_{model}":
        rc = ref.index.intersection(common)
        if len(rc) > 30:
            yv = res.loc[rc, "P_next_actual"].to_numpy()
            _, dmp = metrics.dm_test(
                res.loc[rc, f"P_hat_{model}"].to_numpy() - yv,
                ref.loc[rc, ref_col].to_numpy() - yv)
            out["DM_p_vs_ref"] = dmp
    return out


def score_file(label: str, path: Path, split: pd.Timestamp,
               keep: "str | None", ref_col: str) -> list[dict]:
    """full / early / late rows for every kept model in one predictions file."""
    res = load_predictions(path)
    periods = {
        "full": res.index,
        f"early(<{split.date()})": res.index[res.index < split],
        f"late(>={split.date()})": res.index[res.index >= split],
    }
    rows = []
    for model in list_models(res, keep):
        for period, idx in periods.items():
            if len(idx) == 0:
                continue
            rows.append({"source": label, "model": model, "period": period,
                         **score_period(res, model, idx, ref_res=res,
                                        ref_col=ref_col)})
    return rows


def _parse_pred_arg(item: str) -> tuple[str, str, "str | None"]:
    """`path[:label]` -> (label, path, keep=None). Label defaults to file stem."""
    # split only on the LAST colon so Windows-style paths are unaffected.
    if ":" in item and not item[1:3] == ":\\":
        path, label = item.rsplit(":", 1)
    else:
        path, label = item, ""
    label = label or Path(path).stem
    return label, path, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", default=[],
                    help="predictions CSV as path[:label]; repeatable. "
                         "If omitted, the default Flat+Deep manifest is used.")
    ap.add_argument("--split", default=str(DEFAULT_SPLIT.date()),
                    help="early/late cut date (default 2023-01-01)")
    ap.add_argument("--ref-col", default=DEFAULT_REF_COL,
                    help="non-nested DM reference column (default M1_Flat_Ridge)")
    ap.add_argument("--out", default=str(OUT_DIR / "subperiod_summary.csv"),
                    help="output CSV path")
    args = ap.parse_args()

    split = pd.Timestamp(args.split)
    if args.pred:
        manifest = [_parse_pred_arg(p) for p in args.pred]
    else:
        manifest = DEFAULT_MANIFEST

    rows: list[dict] = []
    for label, rel, keep in manifest:
        path = Path(rel)
        if not path.is_absolute():
            path = data.ROOT / rel
        if not path.exists():
            print(f"[warn] skip missing {rel}")
            continue
        rows.extend(score_file(label, path, split, keep, args.ref_col))
        print(f"[ok] {label:8s} <- {rel}")

    if not rows:
        print("No predictions scored; check --pred paths.")
        return

    summ = pd.DataFrame(rows)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = data.ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_path, index=False)

    # Console view: skill vs M0 (%) by model across periods.
    view = summ.copy()
    view["skill%"] = (view["skill_vs_M0"] * 100).round(2)
    pivot = view.pivot_table(index=["source", "model"], columns="period",
                             values="skill%", sort=False)
    pd.set_option("display.width", 160)
    print(f"\nSplit = {split.date()}   (skill vs M0, %)\n")
    print(pivot.to_string())
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
