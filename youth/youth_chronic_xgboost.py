"""
Chronic Homeless XGBoost Classifier — Ottawa SASM Fields
=========================================================
Trains an XGBoost binary classifier to predict chronic homelessness
using Ottawa SASM intake fields. Handles mixed numeric/categorical data
via label encoding and median imputation.

NOTE: This dataset does not have a youth column — all records are used.
      The target is derived from duration_homeless_past_year and
      episodes_homeless_past_year per HUD chronic homelessness definition,
      OR read directly from a "chronic_homeless" column if present.

Usage
-----
  python youth_chronic_xgboost.py data.csv
  python youth_chronic_xgboost.py data.csv --out-model my_model.pkl
  python youth_chronic_xgboost.py data.csv --target chronic_homeless
  python youth_chronic_xgboost.py --list-cols

Output
------
  youth_chronic_xgboost.pkl  - trained model bundle

Load and predict on new data:
  import pickle, pandas as pd
  with open("youth_chronic_xgboost.pkl", "rb") as f:
      model = pickle.load(f)
  df = pd.read_csv("new_data.csv")
  df_enc = model["encode_fn"](df, model["cat_maps"])
  X = df_enc[model["feature_cols"]].fillna(0)
  predictions   = model["xgb"].predict(X)               # 0 or 1
  probabilities = model["xgb"].predict_proba(X)[:, 1]   # P(chronic)
"""


import argparse
import sys
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_TARGET = "chronic_homeless"

# Numeric columns — used as-is after median imputation
NUMERIC_COLS = [
    "year",
    "age",
    "years_homeless",
]

# Binary/categorical columns — used directly (already 0/1 in this dataset)
CATEGORICAL_COLS = [
    "mental_health",
    "substance_use",
    "physical_health",
    "outdoor_sleeping",
    "lgbtq",
    "indigenous",
    "indigenous_flag",   # alias — only one will be present
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
    "housing_loss_health",
    "youth",
]

# ── Encoding ──────────────────────────────────────────────────────────────────

def build_cat_maps(df, cat_cols):
    """Build a {col: {value: int}} mapping from training data."""
    cat_maps = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        uniques = df[col].dropna().astype(str).unique()
        cat_maps[col] = {v: i for i, v in enumerate(sorted(uniques))}
    return cat_maps


def encode_df(df, cat_maps):
    """Apply label encoding using a pre-built cat_maps dict."""
    df = df.copy()
    for col, mapping in cat_maps.items():
        if col not in df.columns:
            df[col] = np.nan
            continue
        df[col] = df[col].astype(str).map(mapping)  # unseen values → NaN
    return df


# ── Target derivation ─────────────────────────────────────────────────────────

def derive_target(df, target_col):
    """
    If target_col exists in df, use it directly.
    Otherwise derive from HUD chronic definition:
      chronic = (episodes >= 4) OR (duration >= 12 months in past year)
    """
    if target_col in df.columns:
        print(f"  Using existing column '{target_col}' as target.")
        return df[target_col].astype(int)

    if "episodes_homeless_past_year" in df.columns or "duration_homeless_past_year" in df.columns:
        print(f"  '{target_col}' not found — deriving from HUD chronic definition.")
        episodes = pd.to_numeric(df.get("episodes_homeless_past_year", 0), errors="coerce").fillna(0)
        duration = pd.to_numeric(df.get("duration_homeless_past_year", 0), errors="coerce").fillna(0)
        y = ((episodes >= 4) | (duration >= 12)).astype(int)
        print(f"  Derived chronic: {y.sum():,} chronic / {(~y.astype(bool)).sum():,} not chronic")
        return y

    sys.exit(
        f"ERROR: target column '{target_col}' not found and cannot be derived.\n"
        f"Pass --target <col> to specify a different target column, or ensure\n"
        f"'episodes_homeless_past_year' or 'duration_homeless_past_year' is present."
    )


# ── Build feature matrix ──────────────────────────────────────────────────────

def build_features(df_enc, target_col):
    all_candidates = NUMERIC_COLS + CATEGORICAL_COLS
    feature_cols = [c for c in all_candidates if c in df_enc.columns and c != target_col]

    X = df_enc[feature_cols].copy()

    num_present = [c for c in NUMERIC_COLS if c in X.columns]
    for col in num_present:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X[num_present] = X[num_present].fillna(X[num_present].median())

    cat_present = [c for c in CATEGORICAL_COLS if c in X.columns]
    X[cat_present] = X[cat_present].fillna(-1)

    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    return X, feature_cols


# ── Train ─────────────────────────────────────────────────────────────────────

def train(path, target_col, out_model, test_size, seed):
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    y = derive_target(df, target_col)

    valid = y.notna()
    df = df[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    cat_maps = build_cat_maps(df, CATEGORICAL_COLS)
    df_enc = encode_df(df, cat_maps)
    X, feature_cols = build_features(df_enc, target_col)

    print(f"\n  Target distribution:")
    vc = y.value_counts()
    print(f"    Not chronic (0): {vc.get(0, 0):,}")
    print(f"    Chronic     (1): {vc.get(1, 0):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    print(f"\nTraining XGBoost  ({len(feature_cols)} features, scale_pos_weight={scale_pos:.2f}) ...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
        enable_categorical=False,
    )
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)


    print(f"\n-- Evaluation (test set {test_size:.0%} holdout) --")
    print(f"  ROC-AUC: {auc:.4f}")
    report = classification_report(y_test, y_pred, target_names=["Not Chronic", "Chronic"], output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=["Not Chronic", "Chronic"])
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:\n{cm}")

    # Plot and save confusion matrix as PNG
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Chronic", "Chronic"], yticklabels=["Not Chronic", "Chronic"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('youth_chronic_xgboost_confusion_matrix.png')
    plt.close()
    print("  Saved confusion matrix plot as youth_chronic_xgboost_confusion_matrix.png")

    # Plot and save classification report as PNG
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    table_data = []
    for label in ["Not Chronic", "Chronic", "accuracy", "macro avg", "weighted avg"]:
        if label in report:
            row = [label]
            if label == "accuracy":
                row += ["", f"{report['accuracy']:.2f}", "", ""]
            else:
                row += [f"{report[label]['precision']:.2f}", f"{report[label]['recall']:.2f}", f"{report[label]['f1-score']:.2f}", f"{report[label]['support']:.0f}"]
            table_data.append(row)
    col_labels = ["", "Precision", "Recall", "F1-score", "Support"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig('youth_chronic_xgboost_classification_report.png')
    plt.close()
    print("  Saved classification report plot as youth_chronic_xgboost_classification_report.png")

    # Plot and save ROC-AUC as PNG (text)
    fig, ax = plt.subplots(figsize=(4, 1.5))
    ax.axis('off')
    ax.text(0.5, 0.5, f'ROC-AUC: {auc:.4f}', fontsize=14, ha='center', va='center')
    plt.tight_layout()
    plt.savefig('youth_chronic_xgboost_roc_auc.png')
    plt.close()
    print("  Saved ROC-AUC plot as youth_chronic_xgboost_roc_auc.png")

    importances = sorted(
        zip(feature_cols, xgb.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n  Feature importances (top 10):")
    for feat, imp in importances[:10]:
        print(f"    {feat:<35s} {imp:.4f}")

    bundle = {
        "xgb":              xgb,
        "feature_cols":     feature_cols,
        "cat_maps":         cat_maps,
        "encode_fn":        encode_df,
        "target":           target_col,
        "classes":          {0: "Not Chronic Homeless", 1: "Chronic Homeless"},
        "roc_auc":          round(auc, 4),
        "numeric_cols":     NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "note":             "Categorical columns are label-encoded. Use encode_fn(df, cat_maps) before predict.",
    }
    with open(out_model, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Saved model -> {out_model}")
    print(f"""
  To predict on new data:
    import pickle, pandas as pd
    with open("{out_model}", "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv("new_data.csv")
    df_enc = model["encode_fn"](df, model["cat_maps"])
    X = df_enc[model["feature_cols"]].fillna(0)
    predictions   = model["xgb"].predict(X)
    probabilities = model["xgb"].predict_proba(X)[:, 1].round(3)
    df["chronic_pred"] = predictions
    df["chronic_prob"] = probabilities
    df.to_csv("predictions.csv", index=False)
""")
    print("Done")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost to predict chronic homelessness — Ottawa SASM fields"
    )
    parser.add_argument("input", nargs="?", default="data.csv", help="Path to input CSV")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help=f"Target column (default: {DEFAULT_TARGET}). Derived from HUD definition if absent.")
    parser.add_argument("--out-model", default="youth_chronic_xgboost.pkl",
                        help="Output pkl path")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-cols", action="store_true",
                        help="Print expected columns and exit")
    args = parser.parse_args()

    if args.list_cols:
        print("Numeric columns:")
        for c in NUMERIC_COLS:
            print(f"  {c}")
        print("\nCategorical columns (auto label-encoded):")
        for c in CATEGORICAL_COLS:
            print(f"  {c}")
        sys.exit(0)

    train(args.input, args.target, args.out_model, args.test_size, args.seed)


if __name__ == "__main__":
    main()
