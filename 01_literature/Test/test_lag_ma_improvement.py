"""
A/B test: compare XGBoost + Ridge performance WITH vs WITHOUT lag/MA features.
Runs on M1 only (pure financial) to isolate the effect of feature engineering.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from config import SEED, OUT_DIR, TARGETS
from data_loader import load_weekly, prepare_tabular, get_lag_ma_col_names
from evaluation import direction_metrics, regression_metrics

OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_comparison():
    df = load_weekly()
    results = []

    for use_lag in [False, True]:
        label = "WITH lag/MA" if use_lag else "WITHOUT lag/MA"
        for layer in ["M1", "M2", "M3", "M4"]:
            for target_key in ["direction", "price", "volatility"]:
                try:
                    X_tr, X_v, X_te, y_tr, y_v, y_te, scaler, feat_names, _ = \
                        prepare_tabular(layer, target_key, df=df, use_lag_ma=use_lag)
                except Exception:
                    continue

                X_fit = np.vstack([X_tr, X_v])
                y_fit = np.concatenate([y_tr, y_v])

                if len(X_te) < 10:
                    continue

                if target_key == "direction":
                    model = XGBClassifier(
                        n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="multi:softmax", num_class=3,
                        random_state=SEED, verbosity=0,
                        eval_metric="mlogloss",
                    )
                    y_fit_mapped = (y_fit + 1).astype(int)
                    y_te_mapped = (y_te + 1).astype(int)
                    model.fit(X_fit, y_fit_mapped,
                              eval_set=[(X_te, y_te_mapped)],
                              verbose=False)
                    pred = model.predict(X_te).astype(int) - 1
                    metrics = direction_metrics(y_te, pred.astype(float))
                    row = {"lag_ma": label, "layer": layer, "target": target_key,
                           "model": "XGBoost", "n_features": len(feat_names),
                           "n_test": len(y_te),
                           "accuracy": metrics["accuracy"],
                           "macro_f1": metrics["macro_f1"]}
                else:
                    for mname, mdl in [("XGBoost", XGBRegressor(
                            n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, verbosity=0, eval_metric="rmse")),
                                       ("Ridge", Ridge(alpha=1.0))]:
                        if mname == "XGBoost":
                            mdl.fit(X_fit, y_fit,
                                    eval_set=[(X_te, y_te)], verbose=False)
                        else:
                            mdl.fit(X_fit, y_fit)
                        pred = mdl.predict(X_te)
                        metrics = regression_metrics(y_te, pred)
                        row = {"lag_ma": label, "layer": layer, "target": target_key,
                               "model": mname, "n_features": len(feat_names),
                               "n_test": len(y_te),
                               "rmse": metrics["rmse"], "mae": metrics["mae"],
                               "r2": metrics["r2"]}
                        results.append(row)
                        continue
                    continue

                results.append(row)

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT_DIR / "lag_ma_comparison.csv", index=False)
    print(f"[saved] lag_ma_comparison.csv ({len(rdf)} rows)")

    print("\n" + "=" * 70)
    print("LAG/MA FEATURE ENGINEERING — A/B COMPARISON")
    print("=" * 70)

    print(f"\nLag/MA features created: {len(get_lag_ma_col_names())}")
    for name in get_lag_ma_col_names():
        print(f"  + {name}")

    for target_key in ["direction", "price", "volatility"]:
        print(f"\n--- {target_key.upper()} ---")
        sub = rdf[rdf["target"] == target_key]
        if sub.empty:
            continue

        metric_col = "macro_f1" if target_key == "direction" else "rmse"
        higher_better = target_key == "direction"

        for model in sub["model"].unique():
            print(f"\n  [{model}]")
            msub = sub[sub["model"] == model]
            for layer in ["M1", "M2", "M3", "M4"]:
                without = msub[(msub["lag_ma"] == "WITHOUT lag/MA") & (msub["layer"] == layer)]
                with_lag = msub[(msub["lag_ma"] == "WITH lag/MA") & (msub["layer"] == layer)]
                if without.empty or with_lag.empty:
                    continue
                v_without = without[metric_col].values[0]
                v_with = with_lag[metric_col].values[0]
                n_feat_before = without["n_features"].values[0]
                n_feat_after = with_lag["n_features"].values[0]

                if higher_better:
                    delta_pct = (v_with - v_without) / max(abs(v_without), 1e-9) * 100
                    better = "BETTER" if v_with > v_without else "WORSE"
                else:
                    delta_pct = (v_without - v_with) / max(abs(v_without), 1e-9) * 100
                    better = "BETTER" if v_with < v_without else "WORSE"

                print(f"    {layer}: {metric_col}  {v_without:.4f} → {v_with:.4f}  "
                      f"({delta_pct:+.1f}% {better})  "
                      f"[feats: {n_feat_before}→{n_feat_after}]")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    configs = [
        ("direction", "macro_f1", "Direction Macro-F1", True),
        ("price", "rmse", "Price RMSE", False),
        ("volatility", "rmse", "Volatility RMSE", False),
    ]
    for ax, (target_key, metric_col, title, higher_better) in zip(axes, configs):
        sub = rdf[rdf["target"] == target_key]
        if sub.empty:
            ax.set_visible(False)
            continue

        models = sorted(sub["model"].unique())
        layers = ["M1", "M2", "M3", "M4"]
        x_positions = []
        x_labels = []
        vals_without = []
        vals_with = []
        pos = 0

        for model in models:
            for layer in layers:
                wo = sub[(sub["model"] == model) & (sub["layer"] == layer) &
                         (sub["lag_ma"] == "WITHOUT lag/MA")]
                wi = sub[(sub["model"] == model) & (sub["layer"] == layer) &
                         (sub["lag_ma"] == "WITH lag/MA")]
                if wo.empty or wi.empty:
                    continue
                x_positions.append(pos)
                x_labels.append(f"{model}\n{layer}")
                vals_without.append(wo[metric_col].values[0])
                vals_with.append(wi[metric_col].values[0])
                pos += 1
            pos += 0.5

        x = np.array(x_positions)
        w = 0.35
        ax.bar(x - w / 2, vals_without, w, label="Without Lag/MA", color="#2196F3", alpha=0.8)
        ax.bar(x + w / 2, vals_with, w, label="With Lag/MA", color="#FF9800", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel(metric_col.upper())
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Effect of Lag/MA Feature Engineering on Model Performance",
                 fontsize=13, y=1.02)
    fig.savefig(OUT_DIR / "lag_ma_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] lag_ma_comparison.png")


if __name__ == "__main__":
    run_comparison()
