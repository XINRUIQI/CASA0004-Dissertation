"""
Baseline models: LogisticRegression, Ridge, RandomForest, SVM/SVR.
Ablation across M1–M4 layers × 3 prediction targets.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from config import SEED, OUT_DIR
from data_loader import load_weekly, prepare_tabular
from evaluation import direction_metrics, regression_metrics, results_row, save_results

LAYERS = ["M1", "M2", "M3", "M4", "M5"]
TARGET_KEYS = ["direction", "volatility", "price"]

CLF_MODELS = {
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=2000, multi_class="multinomial", solver="lbfgs", random_state=SEED,
    ),
    "RandomForest_clf": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=SEED, n_jobs=-1,
    ),
    "SVC": lambda: SVC(kernel="rbf", random_state=SEED),
}

REG_MODELS = {
    "Ridge": lambda: Ridge(alpha=1.0, random_state=SEED),
    "RandomForest_reg": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=10, random_state=SEED, n_jobs=-1,
    ),
    "SVR": lambda: SVR(kernel="rbf"),
}


def _display_name(key: str) -> str:
    return {
        "LogisticRegression": "LogisticRegression",
        "RandomForest_clf": "RandomForest",
        "SVC": "SVM",
        "Ridge": "Ridge",
        "RandomForest_reg": "RandomForest",
        "SVR": "SVM",
    }[key]


def run_baselines():
    rows: list[dict] = []
    df = load_weekly()

    for layer in LAYERS:
        for target_key in TARGET_KEYS:
            print(f"\n{'='*60}")
            print(f"  Layer={layer}  Target={target_key}")
            print(f"{'='*60}")

            X_train, X_val, X_test, y_train, y_val, y_test, *_ = prepare_tabular(
                layer, target_key, df=df,
            )
            X_fit = np.vstack([X_train, X_val])
            y_fit = np.concatenate([y_train, y_val])

            models = CLF_MODELS if target_key == "direction" else REG_MODELS

            for key, factory in models.items():
                model = factory()
                y_fit_cur = y_fit.astype(int) if target_key == "direction" else y_fit
                model.fit(X_fit, y_fit_cur)
                pred = model.predict(X_test)

                if target_key == "direction":
                    metrics = direction_metrics(y_test.astype(int), pred.astype(int))
                else:
                    metrics = regression_metrics(y_test, pred)

                name = _display_name(key)
                print(f"  {name:25s} -> {metrics}")
                rows.append(results_row(name, layer, target_key, metrics))

    save_results(rows, OUT_DIR / "baselines_results.csv")
    print("\nDone — baseline results saved.")


if __name__ == "__main__":
    run_baselines()
