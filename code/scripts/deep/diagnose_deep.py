"""
Post-hoc verification / overfitting diagnostics for the deep representation-level
layer. Two independent checks that need NO retraining:

  A. Capacity vs sample size — parameter count of every deep config against the
     rolling-origin training sizes (min_train .. N). A tiny-sample / large-model
     ratio is the first overfitting red flag.

  B. Significance re-examined on the SAVED predictions (deep_predictions.csv):
     for each model, on the common test weeks, report
       RMSE / skill vs M0 / DirAcc,
       Clark-West vs M0 (random walk) AND vs M1_Flat_Ridge (the two nested bases),
       Diebold-Mariano vs M0,
       Pesaran-Timmermann + sign test on direction,
       early(<=2022) / late(>=2023) sub-period skill,
       a moving-block bootstrap 95% CI on skill vs M0.
     The point: a Clark-West "win" vs the *weak* flat M1_Flat (which is itself worse
     than the random walk) is not the same as beating the random walk. This
     table shows both bases and the practical (skill) picture side by side.

Run:
  python3 code/scripts/deep/diagnose_deep.py
  python3 code/scripts/deep/diagnose_deep.py --pred results/baselines/Deep/_cross/deep_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from backtest import data, metrics                      # noqa: E402
from model_naming import deep_cross_predictions         # noqa: E402

SPLIT = pd.Timestamp("2023-01-01")


# ---------------------------------------------------------------------------
# A. capacity vs sample size
# ---------------------------------------------------------------------------
def param_report(lookback: int = 4) -> None:
    import torch  # noqa: F401  (kept local; never import xgboost in this process)
    from models.deep_rolling import CONFIGS, _make_model

    npz = np.load(data.ROOT / "data/processed/M3/outputs/m3_graph17_tensors.npz",
                  allow_pickle=True)
    f_aoi = int(npz["aoi_features"].shape[-1])
    f_choke = int(npz["choke_features"].shape[-1])
    n_aoi = int(npz["aoi_features"].shape[1])
    dico = data.load_dict()
    f_fin = len(data.select_features(dico, "M1"))

    # minimal fake ds carrying only the shapes _make_model reads.
    ds = {
        "aoi": np.zeros((1, lookback, n_aoi, f_aoi), np.float32),
        "choke": np.zeros((1, lookback, 6, f_choke), np.float32),
        "fin": np.zeros((1, lookback, f_fin), np.float32),
        "rs": np.zeros((1, lookback, n_aoi, 1024), np.float32),
        "with_rs": True,
    }
    print("=" * 92)
    print(f"A. CAPACITY vs SAMPLE SIZE   (f_aoi={f_aoi} f_choke={f_choke} "
          f"f_fin={f_fin} n_sites={n_aoi} lookback={lookback})")
    print("-" * 92)
    print(f"{'config':10s} {'modalities':22s} {'fusion':7s} {'#params':>10s} "
          f"{'params/1st-fold(104)':>20s}")
    for cfg, (mods, mname, ftype) in CONFIGS.items():
        model = _make_model(ds, mods, ftype, {})
        p = sum(t.numel() for t in model.parameters())
        print(f"{cfg:10s} {'+'.join(mods):22s} {ftype:7s} {p:10,d} "
              f"{p/104:20.1f}")
    print("-" * 92)
    print("Rolling-origin train size grows 104 -> ~356; refit every 13w (20 fits).")
    print("Rule of thumb: >~1 param per training sample => rely on early-stopping/"
          "weight-decay/dropout, and read the train-vs-test gap (check B / run_deep "
          "with --log-fit).")
    print("=" * 92 + "\n")


# ---------------------------------------------------------------------------
# B. significance re-examined on saved predictions
# ---------------------------------------------------------------------------
def pesaran_timmermann(r_hat: np.ndarray, r_act: np.ndarray) -> tuple[float, float]:
    """PT (1992) market-timing test. Returns (stat, one-sided p)."""
    from scipy import stats
    x = np.sign(r_hat)
    y = np.sign(r_act)
    m = (y != 0)
    x, y = x[m], y[m]
    n = len(y)
    if n < 10:
        return np.nan, np.nan
    yu, xu = (y > 0).astype(float), (x > 0).astype(float)
    p = float(np.mean(xu == yu))
    py, px = float(yu.mean()), float(xu.mean())
    pstar = py * px + (1 - py) * (1 - px)
    var_p = pstar * (1 - pstar) / n
    var_ps = (((2 * py - 1) ** 2) * px * (1 - px) / n
              + ((2 * px - 1) ** 2) * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    denom = var_p - var_ps
    if denom <= 0:
        return np.nan, np.nan
    pt = (p - pstar) / np.sqrt(denom)
    return float(pt), float(1 - stats.norm.cdf(pt))


def sign_test(r_hat: np.ndarray, r_act: np.ndarray) -> float:
    """One-sided binomial p that directional hit-rate > 0.5."""
    from scipy import stats
    s_hat, s_act = np.sign(r_hat), np.sign(r_act)
    m = s_act != 0
    hits = int(np.sum(s_hat[m] == s_act[m]))
    n = int(m.sum())
    return float(stats.binomtest(hits, n, 0.5, alternative="greater").pvalue)


def block_bootstrap_skill(y: np.ndarray, e_m0: np.ndarray, e_mod: np.ndarray,
                          block: int = 8, B: int = 5000, seed: int = 0) -> tuple:
    """Moving-block bootstrap 95% CI for skill = 1 - RMSE_mod/RMSE_M0."""
    rng = np.random.default_rng(seed)
    n = len(y)
    nblocks = int(np.ceil(n / block))
    sk = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - block + 1, size=nblocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        rm0 = np.sqrt(np.mean(e_m0[idx] ** 2))
        rmm = np.sqrt(np.mean(e_mod[idx] ** 2))
        sk[b] = 1 - rmm / rm0
    return float(np.percentile(sk, 2.5)), float(np.percentile(sk, 97.5))


def pred_report(pred_csv: Path) -> None:
    pred = pd.read_csv(pred_csv, index_col=0, parse_dates=True)
    y = pred["P_next_actual"].to_numpy()
    r_act = pred["r_actual"].to_numpy()
    e_m0 = pred["P_hat_M0"].to_numpy() - y
    rmse_m0 = float(np.sqrt(np.mean(e_m0 ** 2)))

    # every model with both a price and return column, excluding M0.
    cols = [c[len("P_hat_"):] for c in pred.columns
            if c.startswith("P_hat_") and c != "P_hat_M0"
            and f"r_hat_{c[len('P_hat_'):]}" in pred.columns]
    base_m1 = "M1_Flat_Ridge" if "P_hat_M1_Flat_Ridge" in pred.columns else None

    print("=" * 132)
    print(f"B. SIGNIFICANCE ON SAVED PREDICTIONS  ({pred_csv.name}; "
          f"n={len(pred)}  {pred.index.min().date()}~{pred.index.max().date()}  "
          f"RMSE_M0={rmse_m0:.4f})")
    print("-" * 132)
    hdr = (f"{'model':16s} {'RMSE':>7s} {'skill%':>7s} {'boot95%CI':>16s} "
           f"{'DirAcc':>7s} {'PT_p':>7s} {'sign_p':>7s} "
           f"{'CWvsM0_p':>9s} {'CWvsM1_p':>9s} {'DMvsM0_p':>9s} "
           f"{'skEarly%':>9s} {'skLate%':>8s}")
    print(hdr)
    for c in cols:
        yhat = pred[f"P_hat_{c}"].to_numpy()
        rhat = pred[f"r_hat_{c}"].to_numpy()
        e = yhat - y
        rmse = float(np.sqrt(np.mean(e ** 2)))
        skill = 1 - rmse / rmse_m0
        lo, hi = block_bootstrap_skill(y, e_m0, e)
        diracc = metrics.directional_acc(rhat, r_act)
        _, pt_p = pesaran_timmermann(rhat, r_act)
        sgn_p = sign_test(rhat, r_act)
        _, cw0_p = metrics.clark_west(y, pred["P_hat_M0"].to_numpy(), yhat)
        cw1_p = np.nan
        if base_m1 and c != base_m1:
            _, cw1_p = metrics.clark_west(y, pred[f"P_hat_{base_m1}"].to_numpy(), yhat)
        # Diagnostic script: CW columns above are kept for debugging only. The
        # reported study tests are DM-HLN (candidate second) plus Holm.
        _, dm0_p = metrics.dm_test(e_m0, e)

        early = pred.index < SPLIT
        late = ~early
        def _sk(m):
            if m.sum() < 5:
                return np.nan
            return (1 - np.sqrt(np.mean((yhat[m] - y[m]) ** 2))
                    / np.sqrt(np.mean(e_m0[m] ** 2)))
        print(f"{c:16s} {rmse:7.3f} {skill*100:7.2f} "
              f"[{lo*100:6.2f},{hi*100:6.2f}] {diracc:7.3f} {pt_p:7.3f} {sgn_p:7.3f} "
              f"{cw0_p:9.4f} {cw1_p:9.4f} {dm0_p:9.4f} "
              f"{_sk(early)*100:9.2f} {_sk(late)*100:8.2f}")
    print("-" * 132)
    print("Read: skill>0 & CWvsM0_p<0.05 => genuinely beats the random walk.")
    print("      CWvsM1_p<0.05 only => adds info over the (weak) flat M1_Flat, NOT "
          "necessarily over M0.")
    print("      boot95%CI straddling 0 => skill is statistically indistinguishable "
          "from the random walk.")
    print("=" * 132)


# ---------------------------------------------------------------------------
# C. train-vs-validation gap (the direct overfitting probe; DOES retrain folds)
# ---------------------------------------------------------------------------
def overfit_probe(configs: list[str], n_folds: int, epochs: int,
                  lookback: int, seed: int) -> None:
    """Retrain a few rolling folds and report train vs inner-val RMSE (in the
    per-fold STANDARDISED target units, so 1.0 ~ 'no better than predicting the
    mean'). tr<<va and va~1.0 => the model memorises the train fold while adding
    nothing out-of-sample; early-stopping should keep va from exploding."""
    import torch
    from models.deep_dataset import (apply_scalers, build_deep_dataset,
                                      fit_scalers)
    from models.deep_rolling import (CONFIGS, _forward, _to_tensors,
                                      _train_fold)

    df = data.load_matrix()
    dico = data.load_dict()
    ds = build_deep_dataset(df, dico, lookback=lookback)
    n = len(ds["idx"])
    min_train, retrain_every = 104, 13
    fit_is = [i for i in range(min_train, n) if (i - min_train) % retrain_every == 0]
    pick = np.unique(np.linspace(0, len(fit_is) - 1, n_folds).astype(int))
    sel = [fit_is[k] for k in pick]

    print("=" * 100)
    print(f"C. TRAIN vs INNER-VAL GAP (retrains {len(sel)} folds x {len(configs)} "
          f"configs, epochs={epochs}; RMSE in standardised-target units)")
    print("-" * 100)
    print(f"{'config':9s} {'week':11s} {'n_train':>7s} {'tr_RMSE':>8s} "
          f"{'va_RMSE':>8s} {'va/tr':>6s} {'testwk|e|mod':>12s} {'testwk|e|M0':>11s}")
    for cfg in configs:
        mods, mname, ftype = CONFIGS[cfg]
        for i in sel:
            sc = fit_scalers(ds, train_n=i)
            model, r_mean, r_std = _train_fold(
                ds, sc, i, mods, seed, epochs, 1e-3, 1e-4, 32, 52, "cpu", {}, ftype)
            X = _to_tensors(apply_scalers(ds, sc, slice(0, i)), "cpu")
            r = ds["r_next"][:i].astype(np.float32)
            y = torch.from_numpy(((r - r_mean) / (r_std)).astype(np.float32))
            n_val = min(52, max(10, i // 5))
            tr, va = np.arange(0, i - n_val), np.arange(i - n_val, i)
            model.eval()
            with torch.no_grad():
                tr_rmse = float(torch.sqrt(torch.mean(
                    (_forward(model, X, tr) - y[tr]) ** 2)))
                va_rmse = float(torch.sqrt(torch.mean(
                    (_forward(model, X, va) - y[va]) ** 2)))
                Xte = _to_tensors(apply_scalers(ds, sc, slice(i, i + 1)), "cpu")
                rhat = float(_forward(model, Xte, slice(0, 1)).item()) * r_std + r_mean
            Pt, Pn = ds["P_t"][i], ds["P_next"][i]
            e_mod = abs(Pt * np.exp(rhat) - Pn)
            e_m0 = abs(Pt - Pn)
            print(f"{cfg:9s} {str(ds['idx'][i].date()):11s} {i:7d} "
                  f"{tr_rmse:8.3f} {va_rmse:8.3f} {va_rmse/max(tr_rmse,1e-6):6.2f} "
                  f"{e_mod:12.3f} {e_m0:11.3f}")
    print("-" * 100)
    print("va~1.0 => inner-val no better than the mean; va/tr large => memorising "
          "train. (single test-week |e| is noisy, shown only for scale.)")
    print("=" * 100 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(deep_cross_predictions(data.ROOT).relative_to(data.ROOT)))
    ap.add_argument("--lookback", type=int, default=4)
    ap.add_argument("--skip-params", action="store_true")
    ap.add_argument("--skip-pred", action="store_true")
    ap.add_argument("--overfit", default=None,
                    help="comma configs to retrain for the train/val gap probe "
                         "(e.g. fin,fusion,m4rep). Off by default (it retrains).")
    ap.add_argument("--overfit-folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.skip_params:
        param_report(args.lookback)
    if not args.skip_pred:
        p = Path(args.pred)
        if not p.is_absolute():
            p = data.ROOT / p
        pred_report(p)
    if args.overfit:
        overfit_probe([c.strip() for c in args.overfit.split(",") if c.strip()],
                      args.overfit_folds, args.epochs, args.lookback, args.seed)


if __name__ == "__main__":
    main()
