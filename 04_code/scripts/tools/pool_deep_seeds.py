"""
Pool every multi-seed Deep run into one de-duplicated table, then re-aggregate the
per-configuration mean and SD that Appendix B.4 quotes.

Multi-seed evidence accumulated across producers that ran at different times and,
in some cases, under a different epoch budget:

  deep_seed_*.csv           every standalone multi-seed run, discovered by glob so
                            that adding seeds means adding a file, not editing this
                            script (run_deep_multiseed.py writes these)
  deep_fusion_matrix.csv    seed 42 for all nine combo x fusion cells (epochs 80)
  deep_sweep_summary.csv    group=="seed" rows                       (epochs 60)

Appendix B.4 must not hand-copy those figures, and must not append single runs as
extra table rows: it quotes a mean and SD per configuration, so the pooling and the
aggregation have to be reproducible from the CSVs. This script owns both steps.

De-duplication key is (config, seed). Where a pair appears in more than one source
the higher epoch budget wins, so the pooled table stays on the protocol used for
the headline results (epochs 80) and only falls back to the sweep's 60 where no
80-epoch run exists. Ties are broken by _src_rank below. Because the same
(config, seed) should differ only in epochs, the grid columns are compared across
duplicates and any disagreement is reported rather than silently resolved.

Only group=="seed" rows are taken from the sweep: its hyper / reg / rs groups also
contain config m3_deep_gated at seed 42 but at other grid points, which would
collide on the key while not being replicates.

Outputs (-> 05_outputs/baselines/Deep/_cross/):
  deep_seed_pooled.csv    one row per (config, seed), with source and epochs
  deep_seed_summary.csv   one row per config: n_seeds, mean, sd, min, max, n_positive

Run:
  python3 04_code/scripts/tools/pool_deep_seeds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data                       # noqa: E402
from model_naming import deep_out_dir           # noqa: E402

OUT_DIR = deep_out_dir(data.ROOT, "cross")

# Epoch budgets that the producing scripts do not record in their CSV. Taken from
# the argparse defaults of run_deep_fusion_matrix.py (80) and run_deep_sweep.py (60).
MATRIX_EPOCHS = 80
SWEEP_EPOCHS = 60

# rolling_origin_deep defaults, which the matrix run uses by passing no overrides.
MATRIX_GRID = dict(lookback=4, d=32, gat=2, tcn=2, rs_kind="meanpool",
                   lr=1e-3, wd=1e-4, dropout=0.1)

GRID_COLS = ["lookback", "d", "gat", "tcn", "rs_kind", "lr", "wd", "dropout"]
KEEP = ["config", "seed", "epochs", "source", "RMSE", "skill_vs_M0", "DirAcc",
        "n_test"] + GRID_COLS

# Outputs of this script, which must never be re-ingested as inputs.
SELF_OUTPUTS = {"deep_seed_pooled.csv", "deep_seed_summary.csv"}


def _src_rank(source: str) -> int:
    """Tie-break when two sources offer the same (config, seed) at equal epochs.
    Prefer a dedicated multi-seed run over the matrix cell, and the matrix over the
    sweep, whose epoch budget is inferred rather than recorded."""
    return {"sweep": 2, "fusion_matrix": 1}.get(source, 0)


def _read(name: str) -> pd.DataFrame | None:
    p = OUT_DIR / name
    if not p.exists():
        print(f"  skip (absent): {name}")
        return None
    return pd.read_csv(p)


def _matrix_seed() -> int:
    """Recover the matrix run's seed from its prediction dump rather than assuming
    42: the summary CSV carries no seed column, but the per-origin dump does."""
    preds = _read("deep_fusion_predictions.csv")
    if preds is None or "seed" not in preds.columns:
        raise SystemExit("deep_fusion_predictions.csv missing or has no seed column; "
                         "cannot establish the seed of deep_fusion_matrix.csv.\n"
                         "Run: python3 04_code/scripts/deep/run_deep_fusion_matrix.py")
    seeds = sorted(preds["seed"].unique())
    if len(seeds) != 1:
        raise SystemExit(f"fusion matrix spans several seeds {seeds}; expected one.")
    return int(seeds[0])


def collect() -> pd.DataFrame:
    frames = []

    mat = _read("deep_fusion_matrix.csv")
    if mat is not None:
        m = mat[["config", "RMSE", "skill_vs_M0", "DirAcc", "n_test"]].copy()
        m["seed"] = _matrix_seed()
        m["epochs"] = MATRIX_EPOCHS
        m["source"] = "fusion_matrix"
        for k, v in MATRIX_GRID.items():
            m[k] = v
        frames.append(m)
        print(f"  fusion_matrix     : {len(m)} rows (seed {m['seed'].iloc[0]}, "
              f"epochs {MATRIX_EPOCHS})")

    for p in sorted(OUT_DIR.glob("deep_seed_*.csv")):
        if p.name in SELF_OUTPUTS:
            continue
        d = pd.read_csv(p)
        missing = [c for c in ("config", "seed", "epochs", "skill_vs_M0")
                   if c not in d.columns]
        if missing:
            raise SystemExit(f"{p.name} lacks required column(s) {missing}")
        d["source"] = p.stem.replace("deep_seed_", "")
        frames.append(d)
        print(f"  {d['source'].iloc[0]:18s}: {len(d)} rows "
              f"(epochs {sorted(d['epochs'].unique())}, "
              f"configs {sorted(d['config'].unique())})")

    sw = _read("deep_sweep_summary.csv")
    if sw is not None:
        s = sw[sw["group"] == "seed"].copy()
        if "epochs" in s.columns:
            note = f"epochs {sorted(s['epochs'].unique())} as recorded"
        else:
            # Pre-dates the epochs column in run_deep_sweep.py. The budget is not
            # recoverable from the file, so it is inferred and flagged: if that
            # sweep was run with --epochs, this assumption is wrong.
            s["epochs"] = SWEEP_EPOCHS
            note = (f"epochs {SWEEP_EPOCHS} INFERRED from run_deep_sweep.py's "
                    f"default — file records no epochs column")
            print(f"  WARNING: deep_sweep_summary.csv has no epochs column; "
                  f"assuming {SWEEP_EPOCHS}. Re-run run_deep_sweep.py to record it.")
        s["source"] = "sweep"
        frames.append(s)
        print(f"  sweep             : {len(s)} rows of group=='seed' "
              f"(of {len(sw)} total, {note})")

    if not frames:
        raise SystemExit(f"no multi-seed sources found under {OUT_DIR}")
    pool = pd.concat(frames, ignore_index=True)
    for c in KEEP:
        if c not in pool.columns:
            pool[c] = pd.NA
    return pool[KEEP]


def check_replicates(pool: pd.DataFrame) -> None:
    """A (config, seed) pair repeated across sources should be the same experiment
    at a different epoch budget. Report any grid disagreement, which would mean the
    duplicates are not replicates and must not be de-duplicated silently."""
    for (cfg, seed), g in pool.groupby(["config", "seed"]):
        if len(g) < 2:
            continue
        differing = [c for c in GRID_COLS if g[c].nunique(dropna=False) > 1]
        if differing:
            print(f"  WARNING {cfg} seed={seed}: sources disagree on {differing}")
            print(g[["source", "epochs"] + differing].to_string(index=False))


def dedupe(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pool.copy()
    pool["_src_rank"] = pool["source"].map(_src_rank)
    ordered = pool.sort_values(["config", "seed", "epochs", "_src_rank"],
                              ascending=[True, True, False, True])
    keep = ordered.drop_duplicates(["config", "seed"], keep="first")
    dropped = ordered.loc[~ordered.index.isin(keep.index)]
    return (keep.drop(columns="_src_rank").reset_index(drop=True),
            dropped.drop(columns="_src_rank").reset_index(drop=True))


def summarise(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cfg, g in pooled.groupby("config"):
        g = g.sort_values("seed")
        sk = g["skill_vs_M0"] * 100
        rows.append({
            "config": cfg,
            "n_seeds": len(g),
            "seeds": ",".join(str(int(s)) for s in sorted(g["seed"])),
            "epochs": ",".join(str(int(e)) for e in sorted(g["epochs"].unique())),
            "skill_mean_pct": sk.mean(),
            "skill_sd_pct": sk.std() if len(g) > 1 else pd.NA,
            "skill_min_pct": sk.min(),
            "skill_max_pct": sk.max(),
            "n_positive": int((sk > 0).sum()),
            "sources": ",".join(sorted(g["source"].unique())),
        })
    return (pd.DataFrame(rows)
            .sort_values("skill_mean_pct", ascending=False)
            .reset_index(drop=True))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting multi-seed sources:")
    pool = collect()

    print("\nReplicate check on (config, seed) duplicates:")
    check_replicates(pool)

    pooled, dropped = dedupe(pool)
    if len(dropped):
        print(f"\nDe-duplicated {len(dropped)} row(s) (kept the higher epoch budget):")
        print(dropped[["config", "seed", "epochs", "source", "skill_vs_M0"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.6f}"))

    pooled_csv = OUT_DIR / "deep_seed_pooled.csv"
    pooled.to_csv(pooled_csv, index=False)

    summ = summarise(pooled)
    summ_csv = OUT_DIR / "deep_seed_summary.csv"
    summ.to_csv(summ_csv, index=False)

    print("\n" + "=" * 92)
    print("Pooled per-configuration multi-seed skill vs M0 (%), best mean first:")
    print(summ[["config", "n_seeds", "seeds", "epochs", "skill_mean_pct",
                "skill_sd_pct", "skill_min_pct", "skill_max_pct", "n_positive"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    multi = summ[summ["n_seeds"] >= 3]
    if len(multi):
        print("\nConfigurations with a full three-seed set (quotable in B.4):")
        for _, r in multi.iterrows():
            sd = "n/a" if pd.isna(r["skill_sd_pct"]) else f"{r['skill_sd_pct']:.2f}"
            print(f"  {r['config']:16s} {r['skill_mean_pct']:+.2f}% ± {sd}"
                  f"   (min {r['skill_min_pct']:+.2f}, max {r['skill_max_pct']:+.2f}, "
                  f"positive {r['n_positive']}/{int(r['n_seeds'])}, epochs {r['epochs']})")
    thin = summ[summ["n_seeds"] < 3]
    if len(thin):
        print("\nSingle-seed / incomplete — NOT a multi-seed mean, do not quote as one:")
        for _, r in thin.iterrows():
            print(f"  {r['config']:16s} n_seeds={r['n_seeds']} "
                  f"(seeds {r['seeds']}) skill {r['skill_mean_pct']:+.2f}%")
    print("=" * 92)
    print(f"\nSaved: {pooled_csv}\n       {summ_csv}")


if __name__ == "__main__":
    main()
