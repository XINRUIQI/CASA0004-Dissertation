"""
Test the effect of CV-enhanced features (EMODnet + FRT) on XGBoost.
Compares M2/M3/M4 with and without CV features.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import Ridge

from config import (SEED, OUT_DIR, TARGETS, M1_VARS, M2_RS_ADD,
                    M3_SHIP_GFW, M3_SHIP_EMODNET, LAG_MA_SPECS)
from data_loader import load_weekly, build_lag_ma_features, get_lag_ma_col_names
from evaluation import direction_metrics, regression_metrics

OUT_DIR.mkdir(parents=True, exist_ok=True)

M2_NO_FRT = [c for c in M2_RS_ADD if c != "frt_fill_level_fujairah"]
M3_NO_EMODNET = M3_SHIP_GFW

CONFIGS = {
    "M1": M1_VARS,
    "M2_no_FRT": M1_VARS + M2_NO_FRT,
    "M2_with_FRT": M1_VARS + M2_RS_ADD,
    "M3_no_EMODnet": M1_VARS + M3_NO_EMODNET,
    "M3_with_EMODnet": M1_VARS + M3_SHIP_GFW + M3_SHIP_EMODNET,
    "M4_no_CV": M1_VARS + M2_NO_FRT + M3_NO_EMODNET,
    "M4_with_CV": M1_VARS + M2_RS_ADD + M3_SHIP_GFW + M3_SHIP_EMODNET,
}

TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


def run_xgb(df, feat_list, target_key):
    from sklearn.preprocessing import StandardScaler
    target_col = TARGETS[target_key]
    lag_cols = get_lag_ma_col_names()
    all_feats = list(dict.fromkeys(
        [c for c in feat_list if c in df.columns] +
        [c for c in lag_cols if c in df.columns]
    ))
    sub = df[all_feats + [target_col]].dropna()
    if len(sub) < 50:
        return None

    n = len(sub)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(sub.iloc[:n_train + n_val][all_feats])
    X_test = scaler.transform(sub.iloc[n_train + n_val:][all_feats])
    y_fit = sub.iloc[:n_train + n_val][target_col].values
    y_test = sub.iloc[n_train + n_val:][target_col].values

    if len(X_test) < 10:
        return None

    if target_key == "direction":
        mdl = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            objective="multi:softmax", num_class=3,
                            random_state=SEED, verbosity=0, eval_metric="mlogloss")
        y_fit_m = (y_fit + 1).astype(int)
        y_test_m = (y_test + 1).astype(int)
        mdl.fit(X_fit, y_fit_m, eval_set=[(X_test, y_test_m)], verbose=False)
        pred = mdl.predict(X_test).astype(int) - 1
        m = direction_metrics(y_test, pred.astype(float))
        return {"n_total": len(sub), "n_test": len(y_test), "n_feats": len(all_feats),
                "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    else:
        mdl = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=SEED, verbosity=0, eval_metric="rmse")
        mdl.fit(X_fit, y_fit, eval_set=[(X_test, y_test)], verbose=False)
        pred = mdl.predict(X_test)
        m = regression_metrics(y_test, pred)
        return {"n_total": len(sub), "n_test": len(y_test), "n_feats": len(all_feats),
                "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]}


def main():
    df = build_lag_ma_features(load_weekly())
    rows = []

    for cfg_name, feat_list in CONFIGS.items():
        for target_key in ["direction", "price", "volatility"]:
            print(f"  {cfg_name:20s} | {target_key}")
            result = run_xgb(df, feat_list, target_key)
            if result:
                result["config"] = cfg_name
                result["target"] = target_key
                rows.append(result)

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT_DIR / "cv_enhancement_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("CV ENHANCEMENT (EMODnet + FRT) — XGBOOST COMPARISON")
    print("=" * 70)

    for target_key in ["direction", "price", "volatility"]:
        metric = "macro_f1" if target_key == "direction" else "rmse"
        print(f"\n--- {target_key.upper()} ({metric}) ---")
        sub = rdf[rdf["target"] == target_key].sort_values("config")
        for _, row in sub.iterrows():
            val = row.get(metric, "N/A")
            print(f"  {row['config']:20s}  {metric}={val:.4f}  "
                  f"n={row['n_total']:.0f}  test={row['n_test']:.0f}  "
                  f"feats={row['n_feats']:.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (target_key, metric, title) in zip(axes, [
        ("direction", "macro_f1", "Direction Macro-F1"),
        ("price", "rmse", "Price RMSE"),
        ("volatility", "rmse", "Volatility RMSE"),
    ]):
        sub = rdf[rdf["target"] == target_key].sort_values("config")
        if sub.empty:
            ax.set_visible(False)
            continue
        colors = []
        for cfg in sub["config"]:
            if "no_" in cfg or cfg == "M1":
                colors.append("#90CAF9")
            else:
                colors.append("#FF8A65")

        bars = ax.bar(range(len(sub)), sub[metric].values, color=colors, alpha=0.85)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["config"].values, fontsize=7, rotation=35, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"XGBoost: {title}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars, sub[metric].values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("CV Enhancement: Blue=Without, Orange=With (EMODnet + FRT)",
                 fontsize=12, y=1.02)
    fig.savefig(OUT_DIR / "cv_enhancement_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] cv_enhancement_comparison.png")


if __name__ == "__main__":
    main()
