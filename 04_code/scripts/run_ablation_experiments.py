"""
M1→M4 ablation experiments for Brent oil price forecasting.

Progressively adds feature modalities to measure marginal contribution:
  M1: Market + Macro (baseline)
  M2: M1 + Text/NLP features
  M3: M2 + Remote Sensing
  M4: M3 + Shipping (full multimodal)

Models: Logistic Regression, Random Forest, XGBoost, LightGBM
Task: Binary classification (next-week Brent direction: up/down)
Split: Temporal — Train / Validation / Test

Usage:
    python run_ablation_experiments.py

Output:
    05_outputs/model_results/ablation_results.csv
    05_outputs/model_results/ablation_summary.csv
    05_outputs/figures/09_ablation_results.png
    05_outputs/figures/10_shap_feature_importance.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_ROOT / "03_data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "05_outputs" / "model_results"
FIG_DIR = PROJECT_ROOT / "05_outputs" / "figures"

TRAIN_END = "2022-12-31"
VAL_END = "2024-06-30"

TARGET_COL = "target_brent_direction_next_1w"

EXCLUDE_COLS = {
    "brent_direction",
    "target_brent_return_next_1w", "target_brent_return_next_2w",
    "target_brent_direction_next_1w", "target_brent_price_next_1w",
}


def load_data():
    path = PROC_DIR / "weekly_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(PROC_DIR / "weekly_features.csv", index_col=0, parse_dates=True)


def define_feature_groups(df: pd.DataFrame) -> dict:
    """Define M1-M4 feature groups based on available columns."""
    groups_path = PROC_DIR / "feature_groups.json"
    if groups_path.exists():
        with open(groups_path) as f:
            groups = json.load(f)
    else:
        groups = {"M1_market_macro": [], "M2_add_text": [], "M3_add_rs": [], "M4_add_shipping": []}

    valid_cols = set(df.columns) - EXCLUDE_COLS
    avail_cols = [c for c in df.columns if c.startswith("avail_")]

    m1_cols = [c for c in groups.get("M1_market_macro", []) if c in valid_cols and not c.startswith("avail_")]
    m2_text = [c for c in groups.get("M2_add_text", []) if c in valid_cols]
    m3_rs = [c for c in groups.get("M3_add_rs", []) if c in valid_cols]
    m4_ship = [c for c in groups.get("M4_add_shipping", []) if c in valid_cols]

    return {
        "M1": m1_cols,
        "M2": m1_cols + m2_text,
        "M3": m1_cols + m2_text + m3_rs,
        "M4": m1_cols + m2_text + m3_rs + m4_ship,
    }


def temporal_split(df, features, target):
    """Split data temporally into train/val/test."""
    mask_valid = df[target].notna()
    df_valid = df[mask_valid]

    train = df_valid.loc[:TRAIN_END]
    val = df_valid.loc[TRAIN_END:VAL_END].iloc[1:]
    test = df_valid.loc[VAL_END:].iloc[1:]

    available_feats = [f for f in features if f in df.columns]

    X_train = train[available_feats].copy()
    y_train = train[target].astype(int)
    X_val = val[available_feats].copy()
    y_val = val[target].astype(int)
    X_test = test[available_feats].copy()
    y_test = test[target].astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test, available_feats


def get_models():
    """Return dict of model name → (model, needs_scaling, handles_nan)."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=42),
            True, False
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
            False, False
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42,
                use_label_encoder=False,
            ),
            False, True
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            ),
            False, True
        ),
    }


def evaluate(model, X, y) -> dict:
    """Evaluate a fitted model on (X, y)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "auc": roc_auc_score(y, y_proba),
    }


def run_ablation(df: pd.DataFrame):
    """Run the full M1→M4 × 4 models ablation experiment."""
    feature_groups = define_feature_groups(df)
    models_dict = get_models()

    results = []

    for exp_name, features in feature_groups.items():
        if not features:
            print(f"\n{exp_name}: no features available, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} ({len(features)} features)")
        print(f"{'='*60}")

        X_train, y_train, X_val, y_val, X_test, y_test, used_feats = \
            temporal_split(df, features, TARGET_COL)

        print(f"  Train: {len(X_train)} weeks ({X_train.index.min().date()} ~ {X_train.index.max().date()})")
        print(f"  Val:   {len(X_val)} weeks ({X_val.index.min().date()} ~ {X_val.index.max().date()})")
        print(f"  Test:  {len(X_test)} weeks ({X_test.index.min().date()} ~ {X_test.index.max().date()})")
        print(f"  Features used: {len(used_feats)}")
        print(f"  Train class balance: {y_train.mean():.3f} (up)")

        for model_name, (model, needs_scaling, handles_nan) in models_dict.items():
            print(f"\n  Training {model_name}...")

            Xtr = X_train.copy()
            Xv = X_val.copy()
            Xte = X_test.copy()

            if not handles_nan:
                Xtr = Xtr.fillna(0)
                Xv = Xv.fillna(0)
                Xte = Xte.fillna(0)

            if needs_scaling:
                scaler = StandardScaler()
                Xtr = pd.DataFrame(scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index)
                Xv = pd.DataFrame(scaler.transform(Xv), columns=Xv.columns, index=Xv.index)
                Xte = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)

            model.fit(Xtr, y_train)

            val_metrics = evaluate(model, Xv, y_val)
            test_metrics = evaluate(model, Xte, y_test)

            results.append({
                "experiment": exp_name,
                "model": model_name,
                "n_features": len(used_feats),
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "test_auc": test_metrics["auc"],
            })

            print(f"    Val  — Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            print(f"    Test — Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")

    return pd.DataFrame(results)


def plot_ablation_results(results: pd.DataFrame):
    """Plot grouped bar chart of ablation results."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["test_accuracy", "test_f1", "test_auc"]
    titles = ["Accuracy", "F1 Score", "AUC-ROC"]

    for ax, metric, title in zip(axes, metrics, titles):
        pivot = results.pivot(index="experiment", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"Test {title}")
        ax.set_ylabel(title)
        ax.set_ylim(0.3, 0.8)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("M1→M4 Ablation Results: Progressive Modality Addition", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_ablation_results.png", dpi=200)
    plt.close(fig)
    print("\nSaved: 09_ablation_results.png")


def run_shap_analysis(df: pd.DataFrame):
    """Run SHAP feature importance on best model (M3, LightGBM)."""
    print(f"\n{'='*60}")
    print("SHAP Feature Importance Analysis (M3 + LightGBM)")
    print(f"{'='*60}")

    feature_groups = define_feature_groups(df)
    best_exp = "M3" if feature_groups.get("M3") else "M1"
    features = feature_groups[best_exp]

    X_train, y_train, X_val, y_val, X_test, y_test, used_feats = \
        temporal_split(df, features, TARGET_COL)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    fig, ax = plt.subplots(figsize=(10, 12))
    shap.summary_plot(shap_vals, X_test, max_display=25, show=False)
    plt.title(f"SHAP Feature Importance ({best_exp}, LightGBM)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_shap_feature_importance.png", dpi=200)
    plt.close()
    print("Saved: 10_shap_feature_importance.png")

    importance = pd.DataFrame({
        "feature": used_feats,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(RESULTS_DIR / "shap_feature_importance.csv", index=False)
    print(f"\nTop 15 features by SHAP importance:")
    print(importance.head(15).to_string(index=False))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature matrix...\n")
    df = load_data()
    print(f"Shape: {df.shape}")

    results = run_ablation(df)

    results.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    summary = results.groupby("experiment")[["test_accuracy", "test_f1", "test_auc"]].max()
    summary.to_csv(RESULTS_DIR / "ablation_summary.csv")

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY (best model per experiment)")
    print(f"{'='*60}")
    print(summary.to_string())

    plot_ablation_results(results)
    run_shap_analysis(df)

    print(f"\n{'='*60}")
    print(f"All results saved to: {RESULTS_DIR}/")
    print(f"All figures saved to: {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
