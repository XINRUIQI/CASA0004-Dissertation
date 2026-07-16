"""
One-shot relocation: flat baselines/deep/* -> baselines/Deep/{M*_Deep,_cross}/.

Also splits _cross/deep_predictions.csv into per-tier baseline_predictions.csv
when tier files are missing. Safe to re-run (skips if destination exists).

  python3 04_code/scripts/tools/relocate_deep_outputs.py
"""

from __future__ import annotations

import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from model_naming import (  # noqa: E402
    DEEP_METRICS_ANCHORS,
    DEEP_TIER_MODELS,
    deep_base_dir,
    deep_out_dir,
)
from backtest import data  # noqa: E402

ROOT = data.ROOT
BASE = ROOT / "05_outputs" / "baselines"

# glob patterns -> tier folder (first match wins)
MOVE_RULES: list[tuple[str, str]] = [
    ("deep_metrics*.csv", "cross"),
    ("deep_cw*.csv", "cross"),
    ("deep_predictions.csv", "cross"),
    ("deep_backtest.png", "cross"),
    ("deep_sweep*", "cross"),
    ("deep_fusion_matrix*", "cross"),
    ("baseline_run.log", "cross"),
    ("sweep_run.log", "cross"),
    ("fusion_matrix_run.log", "cross"),
    ("rs_anom_*", "M2"),
    ("rs_diagnostic.log", "M2"),
    ("deep_m3_*", "M3"),
    ("interpret_m3_run.log", "M3"),
    ("deep_advanced*", "M4"),
    ("deep_gate_*", "M4"),
    ("deep_interpret*", "M4"),
    ("deep_xattn_*", "M4"),
    ("advanced_run.log", "M4"),
    ("interpret_run.log", "M4"),
    ("interpret_stability_run.log", "M4"),
    ("xattn_run.log", "M4"),
]

_BASE_PRED = ["P_t", "P_next_actual", "r_actual", "r_now", "r_hat_M0", "P_hat_M0"]


def _legacy_dirs() -> list[Path]:
    """Candidate flat deep output roots (pre-restructure)."""
    cands = [BASE / "deep", BASE / "Deep"]
    out: list[Path] = []
    seen: set[str] = set()
    for p in cands:
        if not p.exists():
            continue
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        # only treat as legacy if it has loose files (not already tiered)
        if any(p.glob("deep_*.csv")) or any(p.glob("deep_*.png")):
            out.append(p)
    return out


def _dest_for(name: str) -> Path | None:
    for pat, tier in MOVE_RULES:
        if fnmatch(name, pat):
            return deep_out_dir(ROOT, tier)
    return None


def move_loose_files() -> int:
  n = 0
  for legacy in _legacy_dirs():
      for f in list(legacy.iterdir()):
          if f.name.startswith(".") or f.is_dir():
              continue
          dest_dir = _dest_for(f.name)
          if dest_dir is None:
              continue
          dest_dir.mkdir(parents=True, exist_ok=True)
          target = dest_dir / f.name
          if target.exists():
              continue
          shutil.move(str(f), str(target))
          print(f"  {f.name} -> {target.relative_to(BASE)}")
          n += 1
  return n


def split_predictions() -> int:
    cross_pred = deep_out_dir(ROOT, "cross") / "deep_predictions.csv"
    if not cross_pred.exists():
        return 0
    merged = pd.read_csv(cross_pred, index_col=0, parse_dates=True)
    n = 0
    for tier, models in DEEP_TIER_MODELS.items():
        tdir = deep_out_dir(ROOT, tier)
        out = tdir / "baseline_predictions.csv"
        if out.exists():
            continue
        present = [m for m in models if f"P_hat_{m}" in merged.columns]
        if not present:
            continue
        tdir.mkdir(parents=True, exist_ok=True)
        cols = _BASE_PRED + [f"P_hat_{c}" for c in present] + [f"r_hat_{c}" for c in present]
        merged[cols].to_csv(out)
        print(f"  split -> {out.relative_to(BASE)}")
        n += 1

    cross_met = deep_out_dir(ROOT, "cross") / "deep_metrics.csv"
    if cross_met.exists():
        summ = pd.read_csv(cross_met, index_col=0)
        for tier, models in DEEP_TIER_MODELS.items():
            tdir = deep_out_dir(ROOT, tier)
            met_out = tdir / "baseline_metrics.csv"
            if met_out.exists():
                continue
            present = [m for m in models if m in summ.index]
            if not present:
                continue
            tdir.mkdir(parents=True, exist_ok=True)
            idx = [i for i in DEEP_METRICS_ANCHORS if i in summ.index] + present
            summ.loc[[i for i in idx if i in summ.index]].to_csv(met_out)
            print(f"  split -> {met_out.relative_to(BASE)}")
            n += 1
    return n


def ensure_layout() -> None:
    deep_base_dir(ROOT).mkdir(parents=True, exist_ok=True)
    for tier in ("M1", "M2", "M3", "M4", "cross"):
        deep_out_dir(ROOT, tier).mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"Deep layout under {deep_base_dir(ROOT)} …")
    ensure_layout()
    moved = move_loose_files()
    split = split_predictions()
    print(f"Done. moved {moved} files, wrote {split} tier CSVs.")


if __name__ == "__main__":
    main()
