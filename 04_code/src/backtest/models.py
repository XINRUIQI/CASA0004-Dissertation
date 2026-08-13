"""
Model factories for the flat baseline.

M0 (random walk, r_hat=0 -> P_hat=P_t) is NOT a model object: it is an internal
benchmark computed inside the rolling loop / metrics, so it can never drift from
the M1-M4 protocol. Here we only define the learned tabular models and the
tuning grids.

Each pipeline starts with a median SimpleImputer then VarianceThreshold (drops
constant columns inside the training fold, no target leakage) and, for the
linear model, StandardScaler (fit on the training fold only). Every step is fit
on the training fold alone, so the imputation constant is re-estimated at each
refit from past data only. Under fill_mode='zero' the input carries no NaN and
the imputer is a no-op.
"""

from __future__ import annotations

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
XGB_GRID = [
    dict(n_estimators=n, max_depth=d, learning_rate=lr,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
    for d in (2, 3) for lr in (0.03, 0.05) for n in (200, 400)
]
XGB_DEFAULT = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)

RIDGE_DEFAULT_ALPHA = 10.0


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def ridge_pipe(alpha: float, seed: int) -> Pipeline:
    return Pipeline([
        ("im", _imputer()),
        ("vt", VarianceThreshold(0.0)),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=alpha, random_state=seed)),
    ])


def xgb_pipe(params: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("im", _imputer()),
        ("vt", VarianceThreshold(0.0)),
        ("m", XGBRegressor(random_state=seed, n_jobs=4,
                           objective="reg:squarederror", **params)),
    ])
