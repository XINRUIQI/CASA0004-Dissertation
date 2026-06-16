"""
A/B test: M3 with GFW (727 weeks, 2012+) vs PortWatch (364 weeks, 2019+).
Also tests M3_combined = GFW + PW (362 weeks overlap).
Runs XGBoost + Ridge with lag/MA features enabled.
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

from config import (SEED, OUT_DIR, TARGETS, M1_VARS, M2_RS_ADD,
                    M3_SHIP_GFW, M3_SHIP_PW, LAG_MA_SPECS)
from data_loader import load_weekly, build_lag_ma_features, get_lag_ma_col_names
from evaluation import direction_metrics, regression_metrics

OUT_DIR.mkdir(parents=True, exist_ok=True)

SHIP_CONFIGS = {
    "M3_PW_only":   {"feats": M1_VARS + M3_SHIP_PW,  "label": "M3 PortWatch\n(364 wks)"},
    "M3_GFW_only":  {"feats": M1_VARS + M3_SHIP_GFW, "label": "M3 GFW\n(727 wks)"},
    "M3_GFW+PW":    {"feats": M1_VARS + M3_SHIP_GFW + M3_SHIP_PW, "label": "M3 GFW+PW\n(362 wks)"},
    "M4_PW_only":   {"feats": M1_VARS + M2_RS_ADD + M3_SHIP_PW,  "label": "M4 w/ PW\n(364 wks)"},
    "M4_GFW_only":  {"feats": M1_VARS + M2_RS_ADD + M3_SHIP_GFW, "label": "M4 w/ GFW\n(626 wks)"},
    "M4_GFW+PW":    {"feats": M1_VARS + M2_RS_ADD + M3_SHIP_GFW + M3_SHIP_PW, "label": "M4 GFW+PW\n(362 wks)"},
    "M1_baseline":  {"feats": M1_VARS, "label": "M1 baseline\n(1042 wks)"},
}

TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


def run_config(df, config_name, feat_list, target_key):
    target_col = TARGETS[target_key]
    present = [c for c in feat_list if c in df.columns]
    lag_ma_cols = get_lag_ma_col_names()
    all_feats = present + [c for c in lag_ma_cols if c in df.columns]
    all_feats = list(dict.fromkeys(all_feats))

    sub = df[all_feats + [target_col]].dropna()
    if len(sub) < 50:
        return None

    n = len(sub)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train = sub.iloc[:n_train]
    val = sub.iloc[n_train:n_train + n_val]
    test = sub.iloc[n_train + n_val:]

    X_fit = scaler.fit_transform(pd.concat([train, val])[all_feats])
    X_test = scaler.transform(test[all_feats])
    y_fit = pd.concat([train, val])[target_col].values
    y_test = test[target_col].values

    rows = []
    if target_key == "direction":
        for mname, mdl in [
            ("XGBoost", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8,
                                      objective="multi:softmax", num_class=3,
                                      random_state=SEED, verbosity=0, eval_metric="mlogloss")),
            ("LogReg", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]:
            y_fit_m = (y_fit + 1).astype(int)
            y_test_m = (y_test + 1).astype(int)
            if mname == "XGBoost":
                mdl.fit(X_fit, y_fit_m, eval_set=[(X_test, y_test_m)], verbose=False)
            else:
                mdl.fit(X_fit, y_fit_m)
            pred = mdl.predict(X_test).astype(int) - 1
            m = direction_metrics(y_test, pred.astype(float))
            rows.append({"config": config_name, "model": mname, "target": target_key,
                         "n_total": n, "n_test": len(y_test), "n_feats": len(all_feats),
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    else:
        for mname, mdl in [
            ("XGBoost", XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=SEED, verbosity=0, eval_metric="rmse")),
            ("Ridge", Ridge(alpha=1.0)),
        ]:
            if mname == "XGBoost":
                mdl.fit(X_fit, y_fit, eval_set=[(X_test, y_test)], verbose=False)
            else:
                mdl.fit(X_fit, y_fit)
            pred = mdl.predict(X_test)
            m = regression_metrics(y_test, pred)
            rows.append({"config": config_name, "model": mname, "target": target_key,
                         "n_total": n, "n_test": len(y_test), "n_feats": len(all_feats),
                         "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]})
    return rows


def main():
    df = build_lag_ma_features(load_weekly())
    all_rows = []

    for config_name, cfg in SHIP_CONFIGS.items():
        for target_key in ["direction", "price", "volatility"]:
            print(f"  {config_name:20s} | {target_key}")
            rows = run_config(df, config_name, cfg["feats"], target_key)
            if rows:
                all_rows.extend(rows)

    rdf = pd.DataFrame(all_rows)
    rdf.to_csv(OUT_DIR / "gfw_vs_pw_comparison.csv", index=False)
    print(f"\n[saved] gfw_vs_pw_comparison.csv ({len(rdf)} rows)")

    print("\n" + "=" * 80)
    print("GFW vs PORTWATCH SHIPPING DATA — COMPARISON")
    print("=" * 80)

    for target_key in ["direction", "price", "volatility"]:
        metric = "macro_f1" if target_key == "direction" else "rmse"
        higher_better = target_key == "direction"
        print(f"\n--- {target_key.upper()} ({metric}) ---")

        sub = rdf[rdf["target"] == target_key]
        for model in sorted(sub["model"].unique()):
            print(f"\n  [{model}]")
            msub = sub[sub["model"] == model].sort_values("config")
            for _, row in msub.iterrows():
                val = row.get(metric, "N/A")
                print(f"    {row['config']:20s}  {metric}={val:.4f}  "
                      f"n_total={row['n_total']:4.0f}  n_test={row['n_test']:3.0f}  "
                      f"feats={row['n_feats']:2.0f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    configs_order = ["M1_baseline", "M3_PW_only", "M3_GFW_only", "M3_GFW+PW",
                     "M4_PW_only", "M4_GFW_only", "M4_GFW+PW"]
    labels = [SHIP_CONFIGS[c]["label"] for c in configs_order]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#FF9800",
              "#1565C0", "#2E7D32", "#E65100"]

    for ax, (target_key, metric, title, hb) in zip(axes, [
        ("direction", "macro_f1", "Direction Macro-F1", True),
        ("price", "rmse", "Price RMSE", False),
        ("volatility", "rmse", "Volatility RMSE", False),
    ]):
        sub = rdf[(rdf["target"] == target_key) & (rdf["model"] == "XGBoost")]
        vals = []
        for cfg in configs_order:
            v = sub[sub["config"] == cfg][metric]
            vals.append(v.values[0] if len(v) > 0 else 0)

        bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"XGBoost: {title}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Shipping Data Source Comparison: GFW (727 wks) vs PortWatch (364 wks)",
                 fontsize=13, y=1.02)
    fig.savefig(OUT_DIR / "gfw_vs_pw_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] gfw_vs_pw_comparison.png")


if __name__ == "__main__":
    main()
