"""
Controlled M3 before/after comparison (variable-set effect, leakage-safe).

Compares the M3 shipping feature set BEFORE the update (effective old set =
GFW total_hours/total_vessels + all_sum; the old emodnet_* columns never existed
in weekly_features.csv) vs AFTER the update (config.M3_SHIP_ADD: PortWatch
count/capacity/avg_size + global sum + export-import directional asymmetry +
GFW dwell-time congestion proxy).

To ISOLATE the variable effect from the sample-period confound, both M3_old and
M3_new (and an M1 reference) are evaluated on the SAME common window — the rows
where every feature in all sets is available (= 2019+, driven by PortWatch).
Same temporal 70/15/15 split, same XGBoost params as 02_xgboost_model.py.

Output: results/m3_before_after.csv  (+ console table)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config import (M1_VARS, M3_SHIP_ADD, TARGETS, TRAIN_RATIO, VAL_RATIO,
                    SEED, OUT_DIR)
from data_loader import load_weekly, build_lag_ma_features, get_lag_ma_col_names
from evaluation import direction_metrics, regression_metrics, dm_test

# Effective OLD M3 shipping set (pre-update; emodnet_* excluded — never in data)
M3_SHIP_OLD = [
    "gfw_hormuz_total_hours", "gfw_hormuz_total_vessels",
    "gfw_suez_total_hours", "gfw_suez_total_vessels",
    "gfw_malacca_total_hours", "gfw_malacca_total_vessels",
    "gfw_all_total_hours_sum",
]
M3_SHIP_NEW = M3_SHIP_ADD

COMMON_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
)
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}
TARGET_KEYS = ["direction", "volatility", "price"]


def feature_set(name: str, lag_ma_cols: list[str]) -> list[str]:
    base = {
        "M1": M1_VARS,
        "M3_old": M1_VARS + M3_SHIP_OLD,
        "M3_new": M1_VARS + M3_SHIP_NEW,
    }[name]
    return base + lag_ma_cols


def run():
    df = build_lag_ma_features(load_weekly())
    lag_ma_cols = [c for c in get_lag_ma_col_names() if c in df.columns]

    sets = {n: [c for c in feature_set(n, lag_ma_cols) if c in df.columns]
            for n in ["M1", "M3_old", "M3_new"]}

    rows = []
    for target_key in TARGET_KEYS:
        target_col = TARGETS[target_key]
        # COMMON sample: rows where ALL features across all sets + target exist
        all_feats = sorted(set(sum(sets.values(), [])))
        sub = df[all_feats + [target_col]].dropna()
        n = len(sub)
        n_tr = int(n * TRAIN_RATIO)
        n_va = int(n * VAL_RATIO)
        tr, va, te = sub.iloc[:n_tr], sub.iloc[n_tr:n_tr+n_va], sub.iloc[n_tr+n_va:]
        win = f"{te.index.min().date()}~{te.index.max().date()}"

        preds = {}
        for set_name, feats in sets.items():
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(tr[feats]); Xva = scaler.transform(va[feats]); Xte = scaler.transform(te[feats])
            ytr, yva, yte = tr[target_col].values, va[target_col].values, te[target_col].values

            if target_key == "direction":
                ytr_m = np.vectorize(LABEL_MAP.get)(ytr.astype(int))
                yva_m = np.vectorize(LABEL_MAP.get)(yva.astype(int))
                model = xgb.XGBClassifier(**COMMON_PARAMS, objective="multi:softmax",
                                          num_class=3, eval_metric="mlogloss",
                                          early_stopping_rounds=30)
                model.fit(Xtr, ytr_m, eval_set=[(Xva, yva_m)], verbose=False)
                pred = np.vectorize(LABEL_INV.get)(model.predict(Xte).astype(int))
                m = direction_metrics(yte.astype(int), pred.astype(int))
            else:
                model = xgb.XGBRegressor(**COMMON_PARAMS, objective="reg:squarederror",
                                         eval_metric="rmse", early_stopping_rounds=30)
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                pred = model.predict(Xte)
                m = regression_metrics(yte, pred)

            preds[set_name] = (yte, np.asarray(pred, dtype=float))
            m = {k: v for k, v in m.items() if k != "confusion_matrix"}
            rows.append({"target": target_key, "set": set_name, "n_features": len(feats),
                         "n_total": n, "test_window": win, "test_n": len(te), **m})

        # DM test (M3_new vs M3_old) on regression targets, squared loss
        if target_key in ("price", "volatility") and {"M3_old", "M3_new"}.issubset(preds):
            yt = preds["M3_new"][0]
            dm = dm_test(yt.astype(float), preds["M3_old"][1], preds["M3_new"][1], loss="squared")
            print(f"  [DM] {target_key}: M3_new vs M3_old  stat={dm['dm_stat']:+.3f} "
                  f"p={dm['p_value']:.3f} (sig5%={dm['significant_5pct']}, +⇒new better)")

    res = pd.DataFrame(rows)
    out = OUT_DIR / "m3_before_after.csv"
    res.to_csv(out, index=False)

    # Pretty print + deltas (M3_new vs M3_old)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Controlled M3 before/after (same common window, XGBoost) ===")
    for tk in TARGET_KEYS:
        sub = res[res.target == tk].set_index("set")
        print(f"\n--- target = {tk}  | common N={sub['n_total'].iloc[0]}  "
              f"test={sub['test_window'].iloc[0]} (N={sub['test_n'].iloc[0]}) ---")
        keep = [c for c in ["accuracy","macro_f1","rmse","r2","mae"] if c in sub.columns]
        print(sub[["n_features"] + keep].to_string())
        if {"M3_old","M3_new"}.issubset(sub.index):
            for c in keep:
                d = sub.loc["M3_new", c] - sub.loc["M3_old", c]
                print(f"    Δ({c}) M3_new−M3_old = {d:+.4f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run()
