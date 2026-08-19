"""Align legacy Deep result CSVs with the frozen test plan (Section 3.7.2).

Two column names predate the plan and are removed or renamed in place:

  CW_p_vs_M0   Clark-West against the no-change benchmark. Deleted. Clark-West
               presumes the smaller specification is a parameter restriction of
               the larger one, which no Deep configuration satisfies, so the
               column is not interpretable. No current script emits it.

  DM_p_vs_M1   In Flat files this is the S1 increment under the same learner and
               remains valid. In Deep files the same name means "Deep config
               against Flat Ridge S1", which changes both the pathway and the
               information set at once, so it answers neither RQ1 nor RQ2. The
               value is a valid one-sided DM p, so it is renamed to the
               self-describing DM_p_vs_Flat_S1 used by the current scripts.

Field text is copied verbatim, so numbers are never reformatted. Formal
p-values for the dissertation come from results/tests/test_table_main.csv.

    python3 code/scripts/tools/fix_deep_legacy_cols.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEEP = ROOT / "results" / "baselines" / "Deep"

DROP = ("CW_p_vs_M0", "CW_stat_vs_M0")
RENAME = {"DM_p_vs_M1": "DM_p_vs_Flat_S1"}

TARGETS = [
    "M1_Deep/baseline_metrics.csv",
    "M2_Deep/baseline_metrics.csv",
    "M3_Deep/baseline_metrics.csv",
    "M4_Deep/baseline_metrics.csv",
    "M2_Deep/rs_anom_compare.csv",
    "M2_Deep/rs_anom_multiseed.csv",
    "M4_Deep/deep_advanced_summary.csv",
    "_cross/deep_metrics.csv",
    "_cross/deep_metrics_arch_lb8.csv",
    "_cross/deep_sweep_summary.csv",
]


def fix(path: Path, dry_run: bool) -> str:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "empty"

    header = rows[0]
    keep = [i for i, c in enumerate(header) if c not in DROP]
    dropped = [c for c in header if c in DROP]
    renamed = [c for c in header if c in RENAME]
    if not dropped and not renamed:
        return "already aligned"

    new_header = [RENAME.get(header[i], header[i]) for i in keep]
    out = [new_header] + [[r[i] for i in keep if i < len(r)] for r in rows[1:]]
    if not dry_run:
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(out)

    notes = []
    if dropped:
        notes.append("dropped " + ", ".join(dropped))
    if renamed:
        notes.append("renamed " + ", ".join(f"{c} -> {RENAME[c]}" for c in renamed))
    return "; ".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for rel in TARGETS:
        path = DEEP / rel
        if not path.exists():
            print(f"  {rel:38s} MISSING")
            continue
        print(f"  {rel:38s} {fix(path, args.dry_run)}")
    if args.dry_run:
        print("\ndry run: nothing written")


if __name__ == "__main__":
    main()
