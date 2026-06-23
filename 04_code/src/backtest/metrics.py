"""
Evaluation metrics + forecast-accuracy tests for the flat baseline.

All metrics are computed on the RECONSTRUCTED price P_hat_{t+1} = P_t * exp(r_hat):
  RMSE, MAE, DirAcc (sign of derived r_hat vs actual), RMSE skill vs M0.

Forecast-accuracy tests:
  - Diebold-Mariano (HLN small-sample corrected): NON-nested comparison, used
    against M0 (random walk). One-sided p that the model is more accurate.
  - Clark-West (MSPE-adjusted): the correct test for NESTED models, used for the
    incremental value of an added modality (e.g. M1+M2 nests M1). DM is biased
    for nested models because the larger model estimates parameters whose true
    value is zero under H0; CW corrects that bias. Reported as DM_p (vs M0) and
    CW_p (vs the nested base config).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def directional_acc(r_hat: np.ndarray, r_act: np.ndarray) -> float:
    s_hat, s_act = np.sign(r_hat), np.sign(r_act)
    mask = s_act != 0
    return float(np.mean(s_hat[mask] == s_act[mask])) if mask.any() else np.nan


def dm_test(e_model: np.ndarray, e_bench: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano on squared-error loss (HLN small-sample correction).

    Returns (DM statistic, one-sided p for H1: model more accurate than bench).
    For NESTED models prefer clark_west().
    """
    d = e_model ** 2 - e_bench ** 2
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
    return float(dm_hln), float(stats.t.cdf(dm_hln, df=T - 1))


def clark_west(y: np.ndarray, yhat_small: np.ndarray,
               yhat_large: np.ndarray) -> tuple[float, float]:
    """Clark-West MSPE-adjusted test for NESTED models.

    small = restricted (e.g. M1), large = unrestricted (e.g. M1+M2, nests M1).
        f_t = (y - yhat_small)^2 - (y - yhat_large)^2 + (yhat_small - yhat_large)^2
    Regress f_t on a constant; CW = mean(f) / (std(f)/sqrt(T)).
    Returns (CW statistic, one-sided p for H1: large model more accurate).
    Clark-West uses standard-normal critical values (1.282@10%, 1.645@5%).
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
            dm_h, p = dm_test(e, e_m0)
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
    """Incremental value of `full` over the nested `base` config (same weeks).

    Returns DM (non-nested-style, kept for reference) and Clark-West (the correct
    nested test) one-sided p-values for 'full more accurate than base'.
    """
    y = res_full["P_next_actual"].to_numpy()
    yhat_full = res_full[f"P_hat_{full_label}_{model}"].to_numpy()
    yhat_base = res_base[f"P_hat_{base_label}_{model}"].to_numpy()
    dm_stat, dm_p = dm_test(yhat_full - y, yhat_base - y)
    cw_stat, cw_p = clark_west(y, yhat_base, yhat_full)
    return {"DM_stat_vs_base": dm_stat, "DM_p_vs_base": dm_p,
            "CW_stat_vs_base": cw_stat, "CW_p_vs_base": cw_p}
