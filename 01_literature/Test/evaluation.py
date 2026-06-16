"""
Evaluation metrics and Diebold-Mariano test.
Covers Section 11 of project_plan_20260601.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score)


# ── Direction (3-class classification) ───────────────────────────
def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask].astype(int), y_pred[mask].astype(int)
    acc = accuracy_score(yt, yp)
    f1_macro = f1_score(yt, yp, average="macro", zero_division=0)

    non_flat = (yt != 0)
    da = accuracy_score(yt[non_flat], yp[non_flat]) if non_flat.sum() > 0 else np.nan

    cm = confusion_matrix(yt, yp, labels=[-1, 0, 1])
    prec = precision_score(yt, yp, average=None, labels=[-1, 0, 1], zero_division=0)
    rec = recall_score(yt, yp, average=None, labels=[-1, 0, 1], zero_division=0)

    return {
        "accuracy": acc, "macro_f1": f1_macro, "directional_acc": da,
        "precision_down": prec[0], "precision_flat": prec[1], "precision_up": prec[2],
        "recall_down": rec[0], "recall_flat": rec[1], "recall_up": rec[2],
        "confusion_matrix": cm,
    }


# ── Regression (volatility / price) ─────────────────────────────
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ── Diebold-Mariano test (Section 11.3) ─────────────────────────
def dm_test(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    pred_new: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> dict:
    """
    One-sided DM test: H0: new model is no better than base.
    Returns DM statistic and p-value.
    """
    mask = ~(np.isnan(y_true) | np.isnan(pred_base) | np.isnan(pred_new))
    yt, pb, pn = y_true[mask], pred_base[mask], pred_new[mask]

    if loss == "squared":
        e_base = (yt - pb) ** 2
        e_new = (yt - pn) ** 2
    else:
        e_base = np.abs(yt - pb)
        e_new = np.abs(yt - pn)

    d = e_base - e_new  # positive ⟹ new is better
    n = len(d)
    d_mean = d.mean()
    d_var = np.sum((d - d_mean) ** 2) / n

    for k in range(1, h):
        gamma_k = np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
        d_var += 2 * gamma_k

    d_var = max(d_var / n, 1e-12)
    dm_stat = d_mean / np.sqrt(d_var)
    p_value = 1 - stats.norm.cdf(dm_stat)

    return {"dm_stat": dm_stat, "p_value": p_value, "n": n,
            "mean_loss_diff": d_mean, "significant_5pct": p_value < 0.05}


# ── Convenience: format results row ─────────────────────────────
def results_row(model: str, layer: str, target: str, metrics: dict) -> dict:
    row = {"model": model, "layer": layer, "target": target}
    row.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
    return row


def save_results(rows: list[dict], path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[saved] {path}  ({len(df)} rows)")
    return df
