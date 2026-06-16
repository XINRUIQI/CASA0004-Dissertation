"""
Data loading, M1–M4 feature selection, and temporal train/val/test split.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (DATA_CSV, LAYER_FEATURES, TARGETS,
                    TRAIN_RATIO, VAL_RATIO, LOOKBACK, SEED,
                    LAG_MA_SPECS)


def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
    df.index.name = "week_ending_friday"
    return df


def get_features_for_layer(layer: str) -> list[str]:
    return LAYER_FEATURES[layer]


def build_lag_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamically create lag, moving-average, and momentum columns."""
    df = df.copy()
    for base_col, specs in LAG_MA_SPECS.items():
        if base_col not in df.columns:
            continue
        for spec in specs:
            t, w = spec["type"], spec.get("window") or spec.get("periods")
            if t == "lag":
                name = f"{base_col}_lag{w}w"
                df[name] = df[base_col].shift(w)
            elif t == "ma":
                name = f"{base_col}_ma{w}w"
                df[name] = df[base_col].rolling(w, min_periods=1).mean()
            elif t == "mom":
                name = f"{base_col}_mom{w}w"
                lagged = df[base_col].shift(w)
                df[name] = (df[base_col] - lagged) / lagged.abs().clip(lower=1e-9)
    return df


def get_lag_ma_col_names() -> list[str]:
    """Return the names of all lag/MA/momentum columns that will be created."""
    names = []
    for base_col, specs in LAG_MA_SPECS.items():
        for spec in specs:
            t, w = spec["type"], spec.get("window") or spec.get("periods")
            if t == "lag":
                names.append(f"{base_col}_lag{w}w")
            elif t == "ma":
                names.append(f"{base_col}_ma{w}w")
            elif t == "mom":
                names.append(f"{base_col}_mom{w}w")
    return names


def prepare_tabular(
    layer: str,
    target_key: str,
    df: pd.DataFrame | None = None,
    use_lag_ma: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           StandardScaler, list[str], pd.DatetimeIndex]:
    """
    Prepare tabular (non-sequence) data for baselines / XGBoost.
    When use_lag_ma=True, dynamically builds lag/MA/momentum features.
    Returns X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, test_index.
    """
    if df is None:
        df = load_weekly()

    if use_lag_ma:
        df = build_lag_ma_features(df)

    feat_cols = get_features_for_layer(layer)
    if use_lag_ma:
        lag_ma_cols = get_lag_ma_col_names()
        feat_cols = feat_cols + [c for c in lag_ma_cols if c in df.columns]
    target_col = TARGETS[target_key]

    present = [c for c in feat_cols if c in df.columns]
    sub = df[present + [target_col]].dropna()

    n = len(sub)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = sub.iloc[:n_train]
    val = sub.iloc[n_train:n_train + n_val]
    test = sub.iloc[n_train + n_val:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[present])
    X_val = scaler.transform(val[present])
    X_test = scaler.transform(test[present])

    y_train = train[target_col].values
    y_val = val[target_col].values
    y_test = test[target_col].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, present, test.index


def prepare_sequences(
    layer: str,
    target_key: str,
    lookback: int = LOOKBACK,
    df: pd.DataFrame | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           StandardScaler, list[str], pd.DatetimeIndex]:
    """
    Prepare sliding-window sequences for LSTM / TFT / ST-GNN.
    X shape: (samples, lookback, n_features), y shape: (samples,).
    """
    if df is None:
        df = load_weekly()

    feat_cols = get_features_for_layer(layer)
    target_col = TARGETS[target_key]

    present = [c for c in feat_cols if c in df.columns]
    sub = df[present + [target_col]].dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(sub[present])
    targets = sub[target_col].values
    dates = sub.index

    X_seq, y_seq, idx_seq = [], [], []
    for i in range(lookback, len(scaled)):
        X_seq.append(scaled[i - lookback:i])
        y_seq.append(targets[i])
        idx_seq.append(dates[i])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    idx_seq = pd.DatetimeIndex(idx_seq)

    n = len(X_seq)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    X_train = X_seq[:n_train]
    X_val = X_seq[n_train:n_train + n_val]
    X_test = X_seq[n_train + n_val:]
    y_train = y_seq[:n_train]
    y_val = y_seq[n_train:n_train + n_val]
    y_test = y_seq[n_train + n_val:]
    test_idx = idx_seq[n_train + n_val:]

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, present, test_idx
