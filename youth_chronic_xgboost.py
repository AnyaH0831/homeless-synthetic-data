"""
Youth Chronic Homeless XGBoost Classifier
==========================================
Filters to youth=1 records only, then trains an XGBoost binary classifier
to predict chronic_homeless (0 = not chronic, 1 = chronic).

Usage
-----
  python youth_chronic_xgboost.py                                  # default path
  python youth_chronic_xgboost.py synthetic_data/synthetic_individuals.csv
  python youth_chronic_xgboost.py --out-model my_model.pkl

Output
------
  youth_chronic_xgboost.pkl  - trained model bundle (load with pickle)

Load and predict:
  import pickle, pandas as pd
  with open("youth_chronic_xgboost.pkl", "rb") as f:
      model = pickle.load(f)
  new_df = pd.read_csv("new_data.csv")
  new_df = new_df[new_df[model["youth_col"]] == 1]   # youth only
  X = new_df[model["feature_cols"]].fillna(new_df[model["feature_cols"]].median())
  X_scaled = model["scaler"].transform(X)
  predictions = model["xgb"].predict(X_scaled)        # 0 or 1
  probabilities = model["xgb"].predict_proba(X_scaled)[:, 1]  # P(chronic)
"""

import argparse
import sys
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

TARGET = "chronic_homeless"
YOUTH_COL = "youth"

# Features used for training — excludes target and youth (constant after filter)
FEATURE_CANDIDATES = [
    "age",
    "years_homeless",
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
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_youth_only(path: str) -> pd.DataFrame:
    print(f"Loading {path} …")
    df = pd.read_csv(path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns (full dataset)")

    if YOUTH_COL not in df.columns:
        sys.exit(f"ERROR: '{YOUTH_COL}' column not found.")
    if TARGET not in df.columns:
        sys.exit(f"ERROR: '{TARGET}' column not found.")

    df_youth = df[df[YOUTH_COL] == 1].copy().reset_index(drop=True)
    n_dropped = len(df) - len(df_youth)
    print(f"  Filtered to youth=1: {len(df_youth):,} kept, {n_dropped:,} adults dropped")
    return df_youth


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Keep only candidates that actually exist in this CSV (handles aliases)
    seen_base = set()
    features = []
    for c in FEATURE_CANDIDATES:
        if c not in df.columns:
            continue
        # Deduplicate indigenous / indigenous_flag aliases
        base = "indigenous" if "indigenous" in c else c
        if base in seen_base:
            continue
        seen_base.add(base)
        features.append(c)

    X = df[features].copy().fillna(df[features].median(numeric_only=True))
    print(f"  Features ({len(features)}): {features}")
    return X, features


def train(path: str, out_model: str, test_size: float, seed: int):
    df = load_youth_only(path)

    X, feature_cols = build_features(df)
    y = df[TARGET].astype(int)

    print(f"\n  Target distribution:")
    vc = y.value_counts()
    print(f"    Not chronic (0): {vc.get(0, 0):,}")
    print(f"    Chronic     (1): {vc.get(1, 0):,}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Class imbalance weight
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    print(f"\nTraining XGBoost (scale_pos_weight={scale_pos:.2f}) …")
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
    )
    xgb.fit(X_train, y_train)

    # Evaluate
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n── Evaluation (test set, {test_size:.0%} holdout) ──")
    print(f"  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Chronic", "Chronic"]))
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:\n{cm}")

    # Feature importance
    importances = sorted(
        zip(feature_cols, xgb.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n  Feature importances (top 10):")
    for feat, imp in importances[:10]:
        print(f"    {feat:<30s} {imp:.4f}")

    # Save bundle
    bundle = {
        "xgb":          xgb,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "target":       TARGET,
        "youth_col":    YOUTH_COL,
        "classes":      {0: "Not Chronic Homeless", 1: "Chronic Homeless"},
        "roc_auc":      round(auc, 4),
        "note":         "Trained on youth=1 records only. Filter new data to youth=1 before predicting.",
    }
    with open(out_model, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Saved model → {out_model}")
    print(f"""
  To predict on new data:
    import pickle, pandas as pd
    with open("{out_model}", "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv("new_data.csv")
    df = df[df[model["youth_col"]] == 1]
    X = df[model["feature_cols"]].fillna(df[model["feature_cols"]].median())
    X_scaled = model["scaler"].transform(X)
    predictions  = model["xgb"].predict(X_scaled)         # 0 or 1
    probabilities = model["xgb"].predict_proba(X_scaled)[:, 1]  # P(chronic)
    df["chronic_pred"]  = predictions
    df["chronic_prob"]  = probabilities.round(3)
""")
    print("Done ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost to predict chronic homelessness in youth"
    )
    parser.add_argument(
        "input", nargs="?",
        default="synthetic_data/synthetic_individuals.csv",
        help="Path to synthetic_individuals.csv",
    )
    parser.add_argument(
        "--out-model", default="youth_chronic_xgboost.pkl",
        help="Output path for the pkl model bundle",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data held out for evaluation (default: 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.input, args.out_model, args.test_size, args.seed)


if __name__ == "__main__":
    main()
