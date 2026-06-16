"""
XGBoost model with SHAP analysis.
Ablation across M1–M4 layers × 3 prediction targets.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import xgboost as xgb
import shap

from config import SEED, OUT_DIR
from data_loader import load_weekly, prepare_tabular
from evaluation import direction_metrics, regression_metrics, results_row, save_results

LAYERS = ["M1", "M2", "M3", "M4", "M5"]
TARGET_KEYS = ["direction", "volatility", "price"]

COMMON_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
)

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def _map_labels(y: np.ndarray) -> np.ndarray:
    return np.vectorize(LABEL_MAP.get)(y.astype(int))


def _inv_labels(y: np.ndarray) -> np.ndarray:
    return np.vectorize(LABEL_INV.get)(y.astype(int))


def _save_shap_plot(shap_vals, X_test: np.ndarray, feat_names: list[str],
                    target_key: str, layer: str = "M4"):
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_test, feature_names=feat_names, show=False)
    path = OUT_DIR / f"shap_{layer}_{target_key}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [SHAP plot saved] {path}")


def run_xgboost():
    rows: list[dict] = []
    df = load_weekly()

    for layer in LAYERS:
        for target_key in TARGET_KEYS:
            print(f"\n{'='*60}")
            print(f"  XGBoost  Layer={layer}  Target={target_key}")
            print(f"{'='*60}")

            (X_train, X_val, X_test,
             y_train, y_val, y_test,
             scaler, feat_names, test_idx) = prepare_tabular(layer, target_key, df=df)

            if target_key == "direction":
                model = xgb.XGBClassifier(
                    **COMMON_PARAMS,
                    objective="multi:softmax",
                    num_class=3,
                    eval_metric="mlogloss",
                    early_stopping_rounds=30,
                )
                model.fit(
                    X_train, _map_labels(y_train),
                    eval_set=[(X_val, _map_labels(y_val))],
                    verbose=False,
                )
                pred_mapped = model.predict(X_test)
                pred = _inv_labels(pred_mapped)
                metrics = direction_metrics(y_test.astype(int), pred.astype(int))
            else:
                model = xgb.XGBRegressor(
                    **COMMON_PARAMS,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    early_stopping_rounds=30,
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                pred = model.predict(X_test)
                metrics = regression_metrics(y_test, pred)

            print(f"  metrics -> {metrics}")
            rows.append(results_row("XGBoost", layer, target_key, metrics))

            if layer in ("M4", "M5"):
                print(f"  Computing SHAP values for {layer} …")
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_test)
                _save_shap_plot(shap_vals, X_test, feat_names, target_key,
                                layer=layer)

    save_results(rows, OUT_DIR / "xgboost_results.csv")
    print("\nDone — XGBoost results saved.")


if __name__ == "__main__":
    run_xgboost()
