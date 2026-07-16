"""
Flat-feature-fusion baseline — single entry point for M0..M4.

ONE fair rolling-origin engine (04_code/src/backtest/) shared by every modality;
M0 is the internal benchmark and M1-M4 differ only in which columns of the merged
matrix are selected. Model display labels are M1_Flat … M4_Flat.

What it does for a chosen --modality:
  1. runs that config (Ridge + XGB) under the locked protocol;
  2. if it nests M1 (M2/M3/M4), also runs M1 on the SAME weeks and reports the
     incremental value with BOTH Diebold-Mariano (vs M0) and Clark-West (the
     correct nested test, vs M1_Flat) -- this is gap B1;
  3. M0 random walk is always included as the benchmark;
  4. optional --leave-one-aoi-out (gap B3): for an M2-bearing config, drop each
     AOI's RS columns in turn and report the RMSE change.

Outputs (-> 05_outputs/baselines/Flat/<M*_Flat>/):
  baseline_metrics[_<suffix>].csv
  baseline_predictions[_<suffix>].csv
  backtest[_<suffix>].png

  M1 example: 05_outputs/baselines/Flat/M1_Flat/baseline_metrics.csv
  M2 example: 05_outputs/baselines/Flat/M2_Flat/baseline_metrics_anom.csv

Run:
  python3 04_code/scripts/flat/run_baseline.py --modality M1
  python3 04_code/scripts/flat/run_baseline.py --modality M2 --m2-features anom
  python3 04_code/scripts/flat/run_baseline.py --modality M4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics, rolling          # noqa: E402
from model_naming import (                          # noqa: E402
    NESTED_BASE_FEAT, NESTED_BASE_LABEL, flat_feat_key, flat_label, flat_out_dir,
)

COLORS = {"Ridge": "tab:blue", "XGB": "tab:red"}


def get_out_dir(modality: str) -> Path:
    """baselines/Flat/M1_Flat/, …"""
    return flat_out_dir(data.ROOT, flat_feat_key(modality))


def run_config(df, dico, modality: str, m2_features: str, lookback: int,
               min_train: int, retrain_every: int, seed: int, feature_mode: str,
               tune: bool, val_weeks: int, drop_aoi: str | None = None,
               m3_tier: str = "full") -> pd.DataFrame:
    feat = flat_feat_key(modality)
    label = flat_label(feat)
    cols = data.select_features(dico, feat, m2_features, drop_aoi=drop_aoi,
                                m3_tier=m3_tier)
    ds = data.build_dataset(df, cols, lookback, feature_mode)
    return rolling.rolling_origin(ds, label, min_train, retrain_every,
                                  seed, tune, val_weeks)


def build_summary(res_main: pd.DataFrame, modality: str,
                  res_base: pd.DataFrame | None) -> pd.DataFrame:
    label = flat_label(modality)
    main_models = [f"{label}_Ridge", f"{label}_XGB"]
    m_main = metrics.evaluate(res_main, main_models)

    if res_base is None:
        summ = m_main
        summ["DM_p_vs_M1"] = np.nan
        summ["CW_stat_vs_M1"] = np.nan
        summ["CW_p_vs_M1"] = np.nan
        return summ

    base_models = [f"{NESTED_BASE_LABEL}_Ridge", f"{NESTED_BASE_LABEL}_XGB"]
    m_base = metrics.evaluate(res_base, base_models).drop(
        index=["Naive_DirPersist"], errors="ignore")
    m_main_only = m_main.drop(index=["M0_RW"], errors="ignore")
    summ = pd.concat([m_base, m_main_only])

    order = ["M0_RW"] + base_models + main_models + ["Naive_DirPersist"]
    summ = summ.reindex([o for o in order if o in summ.index])

    summ["DM_p_vs_M1"] = np.nan
    summ["CW_stat_vs_M1"] = np.nan
    summ["CW_p_vs_M1"] = np.nan
    for model in ("Ridge", "XGB"):
        inc = metrics.incremental_tests(
            res_main, res_base, label, NESTED_BASE_LABEL, model)
        row = f"{label}_{model}"
        summ.loc[row, "DM_p_vs_M1"] = inc["DM_p_vs_base"]
        summ.loc[row, "CW_stat_vs_M1"] = inc["CW_stat_vs_base"]
        summ.loc[row, "CW_p_vs_M1"] = inc["CW_p_vs_base"]
    return summ


def merge_predictions(res_main: pd.DataFrame, modality: str,
                      res_base: pd.DataFrame | None) -> pd.DataFrame:
    common = ["P_t", "P_next_actual", "r_actual", "r_now", "r_hat_M0", "P_hat_M0"]
    pred = res_main[common].copy()
    if res_base is not None:
        extra = [c for c in res_base.columns if c not in common]
        pred = pred.join(res_base[extra])
    extra = [c for c in res_main.columns if c not in common and c not in pred.columns]
    pred = pred.join(res_main[extra])
    return pred


def leave_one_aoi_out(df, dico, modality, m2_features, res_main, lookback,
                      min_train, retrain_every, seed, feature_mode, tune,
                      val_weeks) -> pd.DataFrame:
    label = flat_label(modality)
    full = metrics.evaluate(res_main, [f"{label}_Ridge", f"{label}_XGB"])
    full_r = float(full.loc[f"{label}_Ridge", "RMSE"])
    full_x = float(full.loc[f"{label}_XGB", "RMSE"])
    rows = [{"dropped_aoi": "(none/full)", "Ridge_RMSE": full_r, "XGB_RMSE": full_x,
             "Ridge_dRMSE": 0.0, "XGB_dRMSE": 0.0}]
    for aoi in data.list_aois(dico):
        res = run_config(df, dico, modality, m2_features, lookback, min_train,
                         retrain_every, seed, feature_mode, tune, val_weeks,
                         drop_aoi=aoi)
        m = metrics.evaluate(res, [f"{label}_Ridge", f"{label}_XGB"])
        r = float(m.loc[f"{label}_Ridge", "RMSE"])
        x = float(m.loc[f"{label}_XGB", "RMSE"])
        rows.append({"dropped_aoi": aoi, "Ridge_RMSE": r, "XGB_RMSE": x,
                     "Ridge_dRMSE": r - full_r, "XGB_dRMSE": x - full_x})
        print(f"    LOAO drop {aoi:16s} Ridge={r:.3f} ({r-full_r:+.3f})  "
              f"XGB={x:.3f} ({x-full_x:+.3f})")
    return pd.DataFrame(rows)


def make_plot(pred, summ, modality, res_base_present, path, suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = flat_label(modality)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5),
                                        gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax1.plot(pred.index, pred["P_next_actual"], color="black", lw=1.6, label="Actual P(t+1)")
    ax1.plot(pred.index, pred["P_hat_M0"], color="grey", lw=1.0, ls="--", label="M0 RW")
    for model, c in COLORS.items():
        col = f"P_hat_{label}_{model}"
        if col in pred:
            ax1.plot(pred.index, pred[col], lw=1.0, alpha=0.85, color=c,
                     label=f"{label}_{model}")
    ax1.set_title(f"Next-week Brent price: actual vs predicted{suffix}")
    ax1.set_ylabel("USD / barrel")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    bars = summ["RMSE"].dropna()
    bcolors = ["grey" if i == "M0_RW" else COLORS["Ridge"] if i.endswith("Ridge")
               else COLORS["XGB"] for i in bars.index]
    ax2.bar(range(len(bars)), bars.values, color=bcolors)
    ax2.axhline(bars.get("M0_RW", np.nan), color="grey", ls="--", lw=0.8)
    ax2.set_xticks(range(len(bars)))
    ax2.set_xticklabels(bars.index, rotation=35, ha="right", fontsize=7)
    ax2.set_title("RMSE on price (lower=better)")
    ax2.set_ylabel("USD / barrel")
    ax2.grid(alpha=0.3, axis="y")

    skill = summ["RMSE_skill_vs_M0"].drop(
        index=["M0_RW", "Naive_DirPersist"], errors="ignore").dropna() * 100
    scolors = [COLORS["Ridge"] if i.endswith("Ridge") else COLORS["XGB"] for i in skill.index]
    ax3.bar(range(len(skill)), skill.values, color=scolors)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(range(len(skill)))
    ax3.set_xticklabels(skill.index, rotation=35, ha="right", fontsize=7)
    ax3.set_title("Skill vs M0 (%)  >0 beats RW")
    ax3.set_ylabel("1 - RMSE/RMSE_M0 (%)")
    ax3.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flat-fusion baseline M0..M4 (single entry).")
    ap.add_argument("--modality", default="M1", choices=list(data.MODALITY_SETS))
    ap.add_argument("--m2-features", default="anom", choices=list(data.M2_FEATURE_MODES),
                    help="RS feature contract for M2/M4 (default anom = 55 cols)")
    ap.add_argument("--m3-tier", default="full", choices=list(data.M3_TIERS),
                    help="M3/M4 shipping tier: full (default) or core (robustness)")
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=104)
    ap.add_argument("--retrain-every", type=int, default=13)
    ap.add_argument("--feature-mode", choices=["all", "returns"], default="all")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--val-weeks", type=int, default=52)
    ap.add_argument("--leave-one-aoi-out", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--dict", default=None)
    ap.add_argument("--replot-only", action="store_true",
                    help="Rebuild PNG from existing metrics/predictions CSV (no retrain)")
    args = ap.parse_args()

    tune = not args.no_tune
    feat = flat_feat_key(args.modality)
    label = flat_label(feat)
    has_m2 = "M2" in data.MODALITY_SETS[feat]

    if args.tag:
        tag = args.tag
    elif has_m2:
        tag = args.m2_features
    else:
        tag = ""

    dict_path = args.dict
    if args.matrix and dict_path is None:
        from pathlib import Path as _P
        mp = _P(args.matrix)
        stem = mp.stem
        dict_stem = stem.replace("weekly_feature_matrix", "weekly_feature_dictionary")
        dict_path = mp.parent / (dict_stem + mp.suffix) if mp.is_absolute() else dict_stem + mp.suffix

    OUT_DIR = get_out_dir(feat)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    met_path = OUT_DIR / f"baseline_metrics{suffix}.csv"
    pred_path = OUT_DIR / f"baseline_predictions{suffix}.csv"

    if args.replot_only:
        if not met_path.exists() or not pred_path.exists():
            raise SystemExit(f"Missing {met_path} or {pred_path}")
        summ = pd.read_csv(met_path, index_col=0)
        pred = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        plot_path = OUT_DIR / f"backtest{suffix}.png"
        make_plot(pred, summ, feat, True, plot_path,
                  f"  [{label}{suffix}, L={args.lookback}]")
        print(f"Replotted: {plot_path}")
        return

    df = data.load_matrix(path=args.matrix)
    dico = data.load_dict(path=dict_path)
    print(f"Merged matrix: {df.shape}  {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"Config: modality={feat} label={label} | m2_features="
          f"{args.m2_features if has_m2 else '-'} | lookback={args.lookback}w "
          f"| tune={tune} | retrain_every={args.retrain_every} | tag={tag}\n")

    t0 = time.time()
    res_main = run_config(df, dico, feat, args.m2_features, args.lookback,
                          args.min_train, args.retrain_every, args.seed,
                          args.feature_mode, tune, args.val_weeks,
                          m3_tier=args.m3_tier)
    print(f"  {label:8s}: {res_main.attrs['n_raw']:3d} raw x{args.lookback} "
          f"= {res_main.attrs['n_features']:4d} feats | fits={res_main.attrs['n_fits']} "
          f"| test={len(res_main)} ({res_main.index.min().date()}~{res_main.index.max().date()})")

    res_base = None
    if feat != NESTED_BASE_FEAT:
        res_base = run_config(df, dico, NESTED_BASE_FEAT, args.m2_features, args.lookback,
                              args.min_train, args.retrain_every, args.seed,
                              args.feature_mode, tune, args.val_weeks,
                              m3_tier=args.m3_tier)
        common = res_main.index.intersection(res_base.index)
        res_main, res_base = res_main.loc[common], res_base.loc[common]
        print(f"  {NESTED_BASE_LABEL:8s}: {res_base.attrs['n_raw']:3d} raw x{args.lookback} "
              f"= {res_base.attrs['n_features']:4d} feats | base for CW/DM | "
              f"common test weeks={len(common)}")

    summ = build_summary(res_main, feat, res_base)
    pred = merge_predictions(res_main, feat, res_base)

    summ.to_csv(met_path)
    pred.to_csv(pred_path)

    print("\n" + "=" * 108)
    print(summ.to_string(float_format=lambda x: f"{x:8.4f}"))
    print("=" * 108)
    print("skill>0 beats M0.  DM_p_better_than_M0<0.05: sig. better than M0.")
    if res_base is not None:
        print("CW_p_vs_M1<0.05: added modality gives SIGNIFICANT nested increment over M1_Flat "
              "(Clark-West). DM_p_vs_M1 shown for reference.")

    loao_path = None
    if args.leave_one_aoi_out and has_m2:
        print(f"\nLeave-one-AOI-out ({label}, {args.m2_features}):")
        loao = leave_one_aoi_out(df, dico, feat, args.m2_features, res_main,
                                 args.lookback, args.min_train, args.retrain_every,
                                 args.seed, args.feature_mode, tune, args.val_weeks)
        loao_path = OUT_DIR / f"baseline_loao{suffix}.csv"
        loao.to_csv(loao_path, index=False)

    if not args.no_plot:
        plot_path = OUT_DIR / f"backtest{suffix}.png"
        make_plot(pred, summ, feat, res_base is not None, plot_path,
                  f"  [{label}{suffix}, L={args.lookback}{', tuned' if tune else ''}]")
    else:
        plot_path = None

    print(f"\nElapsed {time.time()-t0:.0f}s")
    print(f"Saved: {met_path}\n       {pred_path}"
          + (f"\n       {loao_path}" if loao_path else "")
          + (f"\n       {plot_path}" if plot_path else ""))


if __name__ == "__main__":
    main()
