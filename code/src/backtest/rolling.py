"""
Rolling-origin (walk-forward) backtest loop, shared by every modality config.

For test point t the model is trained only on samples whose target r_{tau+1} is
already realised (tau <= t-1) -> strictly no look-ahead. M0 (random walk) is
emitted on every test week as the internal benchmark. Optional time-aware inner
validation tunes Ridge alpha + a small XGB grid on the tail of the training fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import models


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def tune_hyperparams(Xtr: np.ndarray, rtr: np.ndarray, val_weeks: int,
                     seed: int) -> tuple[float, dict]:
    """Time-aware inner validation: the last `val_weeks` of the (time-ordered)
    training set are the validation fold. Returns (best ridge alpha, best xgb params)."""
    if len(Xtr) < val_weeks + 30:
        return models.RIDGE_DEFAULT_ALPHA, models.XGB_DEFAULT
    Xf, rf = Xtr[:-val_weeks], rtr[:-val_weeks]
    Xv, rv = Xtr[-val_weeks:], rtr[-val_weeks:]

    best_a, best_e = models.RIDGE_DEFAULT_ALPHA, np.inf
    for a in models.RIDGE_ALPHA_GRID:
        m = models.ridge_pipe(a, seed).fit(Xf, rf)
        e = _rmse(m.predict(Xv), rv)
        if e < best_e:
            best_e, best_a = e, a

    best_p, best_ex = models.XGB_DEFAULT, np.inf
    for params in models.XGB_GRID:
        m = models.xgb_pipe(params, seed).fit(Xf, rf)
        e = _rmse(m.predict(Xv), rv)
        if e < best_ex:
            best_ex, best_p = e, params
    return best_a, best_p


def rolling_origin(ds: dict, label: str, min_train: int, retrain_every: int,
                   seed: int, tune: bool, val_weeks: int) -> pd.DataFrame:
    """Run the walk-forward backtest for one config.

    `ds` is a build_dataset() dict. Emits per test week: P_t, actual P_{t+1},
    r_actual, r_now, M0 (r_hat=0), and {label}_Ridge / {label}_XGB predictions
    (both the predicted log-return r_hat and the reconstructed price P_hat).
    """
    idx = ds["idx"]
    Xmat, Pt, Pn = ds["X"], ds["P_t"], ds["P_next"]
    rn, rnow = ds["r_next"], ds["r_now"]
    n = len(idx)

    rid, xgb = f"{label}_Ridge", f"{label}_XGB"
    fitted: dict = {}
    rows = []
    n_fits = 0

    for i in range(n):
        if i < min_train:                              # expanding-window warm-up
            continue
        Xtr, rtr = Xmat[:i], rn[:i]                     # tau<=i-1: realised by idx[i]
        Xte = Xmat[i:i + 1]

        refit = (not fitted) or ((i - min_train) % retrain_every == 0)
        if refit:
            if tune:
                ra, xp = tune_hyperparams(Xtr, rtr, val_weeks, seed)
            else:
                ra, xp = models.RIDGE_DEFAULT_ALPHA, models.XGB_DEFAULT
            fitted = {
                rid: models.ridge_pipe(ra, seed).fit(Xtr, rtr),
                xgb: models.xgb_pipe(xp, seed).fit(Xtr, rtr),
            }
            n_fits += 1

        rec = {"date": idx[i], "P_t": Pt[i], "P_next_actual": Pn[i],
               "r_actual": rn[i], "r_now": rnow[i],
               "r_hat_M0": 0.0, "P_hat_M0": Pt[i]}
        for name, mdl in fitted.items():
            rhat = float(mdl.predict(Xte)[0])
            rec[f"r_hat_{name}"] = rhat
            rec[f"P_hat_{name}"] = Pt[i] * np.exp(rhat)
        rows.append(rec)

    res = pd.DataFrame(rows).set_index("date")
    res.attrs.update(label=label, n_features=len(ds["feat_names"]),
                     n_raw=ds.get("n_raw", len(ds["feat_names"])), n_fits=n_fits,
                     train_start=idx[0].date().isoformat())
    return res
