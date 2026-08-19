"""
Evaluation metrics + forecast-accuracy tests.

All metrics are computed on the RECONSTRUCTED price P_hat_{t+1} = P_t * exp(r_hat):
  RMSE, MAE, DirAcc (sign of derived r_hat vs actual), RMSE skill vs M0.

Forecast-accuracy tests follow one scheme for every comparison in the study:

  - Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction is
    the primary test everywhere. The loss differential is always written
    d_t = L(reference) - L(candidate), so a positive statistic means the
    candidate is the more accurate forecast. The alternative is one-sided where
    the research question is directional and two-sided otherwise.
  - Clark-West is SUPPLEMENTARY and is applied only to Ridge specifications
    that are nested at the predictor-set level. It is not applied to XGBoost or
    Deep comparisons, where the larger specification re-selects hyperparameters,
    tree structure or encoders and so is not a parameter restriction of the
    smaller one. On this sample the MSPE adjustment can otherwise report a
    "significant improvement" for a specification whose RMSE is worse.
  - Because many pairwise comparisons are made, p-values are reported both raw
    and Holm-adjusted within a pre-defined comparison family; formal
    significance is judged on the adjusted values.

The frozen families and the reference/candidate pairs live in
scripts/tools/build_test_tables.py, which is the only place that decides which
test applies to which comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def directional_acc(r_hat: np.ndarray, r_act: np.ndarray) -> float:
    s_hat, s_act = np.sign(r_hat), np.sign(r_act)
    mask = s_act != 0
    return float(np.mean(s_hat[mask] == s_act[mask])) if mask.any() else np.nan


LOSSES = ("squared_error", "absolute_error")
ALTERNATIVES = ("greater", "two-sided")


def _loss(e: np.ndarray, loss: str) -> np.ndarray:
    if loss == "squared_error":
        return e ** 2
    if loss == "absolute_error":
        return np.abs(e)
    raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")


def dm_test(e_ref: np.ndarray, e_cand: np.ndarray, h: int = 1,
            alternative: str = "greater",
            loss: str = "squared_error") -> tuple[float, float]:
    """Diebold-Mariano test with the HLN small-sample correction.

    The loss differential is d_t = L(e_ref) - L(e_cand), so a POSITIVE statistic
    means the CANDIDATE forecast is more accurate than the REFERENCE. Note the
    argument order: the reference (benchmark, smaller information set, or Flat
    pathway) comes first.

    alternative
        "greater"    H1: E[d_t] > 0, the candidate is more accurate
        "two-sided"  H1: E[d_t] != 0, the two forecasts differ in accuracy
    loss
        "squared_error" for the primary tests, so that the loss matches the RMSE
        headline metric; "absolute_error" for the MAE-loss robustness check.

    Returns (HLN-corrected statistic, p-value) using t with T-1 degrees of
    freedom.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    d = (_loss(np.asarray(e_ref, dtype=float), loss)
         - _loss(np.asarray(e_cand, dtype=float), loss))
    T = len(d)
    dbar = d.mean()
    gamma0 = np.sum((d - dbar) ** 2) / T
    acov = sum(2 * np.sum((d[k:] - dbar) * (d[:-k] - dbar)) / T for k in range(1, h))
    var_d = (gamma0 + acov) / T
    if var_d <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var_d)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * corr
    if alternative == "greater":
        p = 1.0 - stats.t.cdf(dm_hln, df=T - 1)
    else:
        p = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=T - 1))
    return float(dm_hln), float(p)


def holm(pvals) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values within one comparison family.

    Controls the family-wise error rate under arbitrary dependence between the
    tests. Non-finite inputs are dropped from the family and returned as NaN.
    """
    p = np.asarray(pvals, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return adj
    order = np.flatnonzero(ok)[np.argsort(p[ok])]
    running = 0.0
    for i, ix in enumerate(order):
        running = max(running, (m - i) * float(p[ix]))
        adj[ix] = min(running, 1.0)
    return adj


def clark_west(y: np.ndarray, yhat_small: np.ndarray,
               yhat_large: np.ndarray) -> tuple[float, float]:
    """Clark-West MSPE-adjusted test, SUPPLEMENTARY use only.

    small = restricted predictor set (e.g. Ridge S1), large = the specification
    that adds predictors to it (e.g. Ridge S2).
        f_t = (y - yhat_small)^2 - (y - yhat_large)^2 + (yhat_small - yhat_large)^2
    Regress f_t on a constant; CW = mean(f) / (std(f)/sqrt(T)).
    Returns (CW statistic, one-sided p for H1: large model more accurate), using
    standard-normal critical values (1.282@10%, 1.645@5%).

    Apply this only to Ridge specifications, and only as supplementary evidence
    described as nested at the predictor-set level: even there the penalty is
    re-tuned at every re-estimation, so the larger specification is not a strict
    parameter restriction of the smaller one. The MSPE adjustment term can
    dominate the realised loss difference, in which case the test reports a
    significant "improvement" for a specification whose RMSE is worse.
    """
    e_s = y - yhat_small
    e_l = y - yhat_large
    f = e_s ** 2 - e_l ** 2 + (yhat_small - yhat_large) ** 2
    T = len(f)
    fbar = f.mean()
    se = f.std(ddof=1) / np.sqrt(T)
    if se <= 0:
        return np.nan, np.nan
    cw = fbar / se
    return float(cw), float(1.0 - stats.norm.cdf(cw))


def evaluate(res: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """M0 + each learned model + a naive directional benchmark, on price errors."""
    Pn = res["P_next_actual"].to_numpy()
    r_act = res["r_actual"].to_numpy()
    e_m0 = res["P_hat_M0"].to_numpy() - Pn
    rmse_m0 = np.sqrt(np.mean(e_m0 ** 2))

    out = []
    for name in ["M0"] + model_names:
        P_hat = res[f"P_hat_{name}"].to_numpy()
        e = P_hat - Pn
        rmse, mae = float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))
        if name == "M0":
            out.append({"model": "M0_RW", "RMSE": rmse, "MAE": mae, "DirAcc": np.nan,
                        "RMSE_skill_vs_M0": 0.0, "DM_stat": np.nan,
                        "DM_p_better_than_M0": np.nan})
        else:
            r_hat = res[f"r_hat_{name}"].to_numpy()
            dm_h, p = dm_test(e_m0, e)
            out.append({"model": name, "RMSE": rmse, "MAE": mae,
                        "DirAcc": directional_acc(r_hat, r_act),
                        "RMSE_skill_vs_M0": float(1 - rmse / rmse_m0),
                        "DM_stat": dm_h, "DM_p_better_than_M0": p})

    out.append({"model": "Naive_DirPersist", "RMSE": np.nan, "MAE": np.nan,
                "DirAcc": directional_acc(res["r_now"].to_numpy(), r_act),
                "RMSE_skill_vs_M0": np.nan, "DM_stat": np.nan,
                "DM_p_better_than_M0": np.nan})
    return pd.DataFrame(out).set_index("model")


def incremental_tests(res_full: pd.DataFrame, res_base: pd.DataFrame,
                      full_label: str, base_label: str,
                      model: str) -> dict:
    """Incremental value of `full` over the smaller `base` config (same weeks).

    DM is the primary test (one-sided, H1: the larger information set is more
    accurate). Clark-West is returned ONLY for Ridge, where the two predictor
    sets are genuinely nested; for any other learner it is NaN, because the
    larger specification re-selects hyperparameters or structure and so is not a
    parameter restriction of the smaller one. On this sample the MSPE adjustment
    otherwise reports p < 0.001 for an XGBoost specification whose RMSE is worse
    than both the smaller specification and the no-change benchmark.
    """
    y = res_full["P_next_actual"].to_numpy()
    yhat_full = res_full[f"P_hat_{full_label}_{model}"].to_numpy()
    yhat_base = res_base[f"P_hat_{base_label}_{model}"].to_numpy()
    dm_stat, dm_p = dm_test(yhat_base - y, yhat_full - y)
    cw_stat, cw_p = (clark_west(y, yhat_base, yhat_full) if model == "Ridge"
                     else (np.nan, np.nan))
    return {"DM_stat_vs_base": dm_stat, "DM_p_vs_base": dm_p,
            "CW_stat_vs_base": cw_stat, "CW_p_vs_base": cw_p}
