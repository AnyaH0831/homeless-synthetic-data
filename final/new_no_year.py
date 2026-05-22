"""
Youth Chronic Homelessness XGBoost
----------------------------------
VERSION WITHOUT YEAR COLUMN

Usage:
python final/new_no_year.py synthetic_data/synthetic_individuals.csv
"""

import argparse
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import shap

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

TARGET = "chronic_homeless"

# YEAR REMOVED
NUMERIC_COLS = [
    "age",
    "years_homeless",
]

BINARY_COLS = [
    "mental_health",
    "substance_use",
    "physical_health",
    "outdoor_sleeping",
    "lgbtq",
    "indigenous",
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
    "housing_loss_health",
    "youth",
]

# ============================================================
# CLEANING
# ============================================================

def clean_binary(df, cols):

    for col in cols:

        if col not in df.columns:
            df[col] = 0

        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype(int)
        )

    return df

# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(df):

    feature_cols = NUMERIC_COLS + BINARY_COLS

    # numeric
    for col in NUMERIC_COLS:

        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = df[col].fillna(
            df[col].median()
        )

    # binary
    df = clean_binary(df, BINARY_COLS)

    X = df[feature_cols]

    y = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    return X, y, feature_cols

# ============================================================
# TRAIN
# ============================================================

def train(path):

    print(f"\nLoading {path}")

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    # youth only
    if "youth" in df.columns:

        df = df[df["youth"] == 1]

        print("Youth only:", df.shape)

    X, y, feature_cols = build_features(df)

    print("\nFeatures:")
    print(feature_cols)

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    scale_pos = neg / max(pos, 1)

    print(f"\nscale_pos_weight = {scale_pos:.2f}")

    # ========================================================
    # XGBOOST
    # ========================================================

    model = XGBClassifier(

        n_estimators=400,
        max_depth=4,

        learning_rate=0.03,

        subsample=0.8,
        colsample_bytree=0.8,

        min_child_weight=3,

        gamma=0.5,

        reg_alpha=0.5,
        reg_lambda=1.5,

        scale_pos_weight=scale_pos,

        random_state=42,

        eval_metric="logloss",

        tree_method="hist",

        verbosity=0,
    )

    print("\nTraining model...\n")

    model.fit(X_train, y_train)

    # ========================================================
    # PREDICT
    # ========================================================

    probs = model.predict_proba(X_test)[:, 1]

    preds = (probs >= 0.35).astype(int)

    auc = roc_auc_score(y_test, probs)

    print("\nROC AUC:", round(auc, 4))

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            preds,
            target_names=[
                "Not Chronic",
                "Chronic",
            ]
        )
    )

    cm = confusion_matrix(
        y_test,
        preds,
    )

    print("\nConfusion Matrix:\n")
    print(cm)

    # ========================================================
    # SAVE MODEL
    # ========================================================

    bundle = {

        "xgb": model,

        "feature_cols": feature_cols,

        "target": TARGET,

        "numeric_cols": NUMERIC_COLS,

        "binary_cols": BINARY_COLS,

        "roc_auc": auc,
    }

    with open(
        "youth_xgboost_no_year.pkl",
        "wb",
    ) as f:

        pickle.dump(bundle, f)

    print("\nSaved: youth_xgboost_no_year.pkl")

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    imp_df = pd.DataFrame({

        "feature": feature_cols,
        "importance": model.feature_importances_,

    }).sort_values(
        "importance",
        ascending=False,
    )

    print("\nFeature Importance:\n")
    print(imp_df)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Not Chronic",
            "Chronic",
        ],
        yticklabels=[
            "Not Chronic",
            "Chronic",
        ]
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "youth_no_year_confusion_matrix.png",
        dpi=150,
    )

    plt.close()

    # ========================================================
    # ROC CURVE
    # ========================================================

    fpr, tpr, _ = roc_curve(
        y_test,
        probs,
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "youth_no_year_roc_curve.png",
        dpi=150,
    )

    plt.close()

    # ========================================================
    # FEATURE IMPORTANCE PLOT
    # ========================================================

    imp_plot = imp_df.sort_values(
        "importance"
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        imp_plot["feature"],
        imp_plot["importance"],
    )

    plt.xlabel("Importance")

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "youth_no_year_feature_importance.png",
        dpi=150,
    )

    plt.close()

    # ========================================================
    # SHAP
    # ========================================================

    print("\nComputing SHAP values...")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_test
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_cols,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        "youth_no_year_shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("\nDone.")

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv"
    )

    args = parser.parse_args()

    train(args.csv)

if __name__ == "__main__":
    main()