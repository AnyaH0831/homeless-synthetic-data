"""
xgb_homelessness_type.py
─────────────────────────
XGBoost model: predict what TYPE of homelessness (cluster) a person belongs to,
based on their individual-level features.

Reads:   synthetic_data/validation/synthetic_individuals.csv
         (or sasm_clusters.csv if you've already run sasm_analysis.py)

Outputs:
  xgb_type_model.pkl          — saved XGBoost model
  xgb_feature_importance.csv  — SHAP/gain-based feature importances
  xgb_confusion_matrix.png    — per-cluster confusion matrix
  xgb_shap_summary.png        — SHAP beeswarm plot (if shap installed)

Usage:
  python xgb_homelessness_type.py

  # To predict on new individuals:
  python xgb_homelessness_type.py --predict my_new_people.csv
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ─────────────────────────────────────────────────────────────────────

# Path to raw individuals CSV (columns match your schema)
RAW_CSV   = "synthetic_data/validation/synthetic_individuals.csv"
# If you already ran sasm_analysis.py, use the clustered file instead:
CLUST_CSV = "sasm_clusters.csv"

N_CLUSTERS = 6   # must match sasm_analysis.py

# Human-readable labels — edit to match your cluster inspection results
CLUSTER_NAMES = {
    0: "Indigenous / Sheltered",
    1: "Sheltered / Mental Health / Chronic",
    2: "Youth / Sheltered",
    3: "Sheltered / Dual-Diagnosis / Chronic",
    4: "Sheltered",
    5: "Unsheltered",
}

# Features used as model inputs
# These are the raw columns in synthetic_individuals.csv
FEATURE_COLS = [
    "age",
    "years_homeless",
    "mental_health",
    "substance_use",
    "physical_health",
    "outdoor_sleeping",
    "chronic_homeless",
    "lgbtq",
    "indigenous",
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
    "housing_loss_health",
    "youth",
    # Encoded versions added by encode_categoricals()
    "gender_encoded",
    "race_encoded",
    "shelter_encoded",
]

# XGBoost hyperparameters — tuned for tabular binary/multiclass
XGB_PARAMS = dict(
    n_estimators      = 400,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 5,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    use_label_encoder = False,
    eval_metric       = "mlogloss",
    random_state      = 42,
    n_jobs            = -1,
)


# ── 1. LOAD & ENCODE ───────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer-encoded columns for gender, race, shelter_type."""
    gender_map  = {"male": 0, "female": 1, "trans_nonbinary": 2}
    race_map    = {"black": 0, "white": 1, "indigenous": 2, "other": 3}
    shelter_map = {"emergency_shelter": 0, "respite": 1, "other": 2, "outdoor": 3}

    if "gender" in df.columns:
        df["gender_encoded"]  = df["gender"].map(gender_map).fillna(0).astype(int)
    else:
        df["gender_encoded"] = 0

    if "race" in df.columns:
        df["race_encoded"]    = df["race"].map(race_map).fillna(3).astype(int)
    else:
        df["race_encoded"] = 3

    if "shelter_type" in df.columns:
        df["shelter_encoded"] = df["shelter_type"].map(shelter_map).fillna(2).astype(int)
    else:
        df["shelter_encoded"] = 2

    return df


def load_data() -> pd.DataFrame:
    """
    Load data.  Priority order:
      1. sasm_clusters.csv (already has cluster labels — fastest)
      2. RAW_CSV            (need to cluster on the fly)
    """
    clust_path = Path(CLUST_CSV)
    raw_path   = Path(RAW_CSV)

    if clust_path.exists():
        print(f"Loading pre-clustered data from {CLUST_CSV} …")
        df = pd.read_csv(clust_path)
        df = encode_categoricals(df)
        print(f"  {len(df):,} records, cluster column: 'cluster'")
        return df

    if raw_path.exists():
        print(f"Loading raw data from {RAW_CSV} …")
        df = pd.read_csv(raw_path)
        df = encode_categoricals(df)
        print(f"  {len(df):,} records — running KMeans (k={N_CLUSTERS}) to assign clusters …")
        df = assign_clusters(df)
        return df

    raise FileNotFoundError(
        f"Neither '{CLUST_CSV}' nor '{RAW_CSV}' found.\n"
        "Run sasm_analysis.py first, or adjust the path constants at the top of this file."
    )


def assign_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run KMeans on the same features used by sasm_analysis.py.
    Only needed if sasm_clusters.csv doesn't exist yet.
    """
    cluster_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cluster_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    print(f"  Cluster distribution:\n{df['cluster'].value_counts().sort_index().to_string()}")
    return df


# ── 2. PREPARE X / y ──────────────────────────────────────────────────────────

def prepare_Xy(df: pd.DataFrame):
    """Return feature matrix X and label vector y (cluster id)."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Note: columns not found, will be zero-filled: {missing}")
        for c in missing:
            df[c] = 0

    X = df[FEATURE_COLS].fillna(0).astype(float)
    y = df["cluster"].astype(int)
    return X, y


# ── 3. TRAIN ───────────────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Train XGBoost multiclass classifier.
    Reports 5-fold cross-validated accuracy so you know how well features
    predict cluster membership (i.e., 'type' of homelessness).
    """
    print("\nTraining XGBoost …")
    model = XGBClassifier(
        num_class=N_CLUSTERS,
        objective="multi:softprob",
        **XGB_PARAMS,
    )

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_bal = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    print(f"  5-fold CV accuracy          : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  5-fold CV balanced accuracy : {cv_bal.mean():.3f} ± {cv_bal.std():.3f}")

    # Final fit on full data
    model.fit(X, y)
    print("  Full-data fit complete.")
    return model


# ── 4. EVALUATE ────────────────────────────────────────────────────────────────

def evaluate(model, X: pd.DataFrame, y: pd.Series) -> None:
    """Print classification report and save confusion matrix plot."""
    y_pred = model.predict(X)

    target_names = [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in sorted(y.unique())]
    print("\nClassification report (full training data):")
    print(classification_report(y, y_pred, target_names=target_names))

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=30)
    ax.set_title("XGBoost — Homelessness Type Prediction\nConfusion Matrix (training data)")
    plt.tight_layout()
    plt.savefig("xgb_confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: xgb_confusion_matrix.png")


# ── 5. FEATURE IMPORTANCE ─────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Plot and save feature importances two ways:
      - XGBoost gain (built-in)
      - SHAP values (if shap package is installed)

    Gain importance tells you which features reduce loss the most when split on.
    SHAP gives a more nuanced per-prediction breakdown.
    """
    # ── Gain importance ──────────────────────────────────────────────────────
    importance = pd.Series(
        model.feature_importances_, index=feature_names, name="importance"
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importance)))
    importance.plot.barh(ax=ax, color=colors)
    ax.set_title("XGBoost Feature Importance (Gain)\nPredicting Homelessness Type")
    ax.set_xlabel("Relative importance")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.savefig("xgb_feature_importance.png", dpi=150)
    plt.close()
    print("  Saved: xgb_feature_importance.png")

    imp_df = importance.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv("xgb_feature_importance.csv", index=False)
    print("  Saved: xgb_feature_importance.csv")

    return imp_df


def shap_analysis(model, X: pd.DataFrame) -> None:
    """
    Proper SHAP analysis using a random sample of rows.
    Shows which features push predictions toward each cluster.
    """
    try:
        import shap
        print("\nRunning SHAP analysis …")
        n_sample = min(2000, len(X))
        X_sample = X.sample(n=n_sample, random_state=42).reset_index(drop=True)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)  # shape: (n_classes, n_samples, n_features)

        # For multi-class, create a summary plot for each class
        n_classes = len(CLUSTER_NAMES)
        fig, axes = plt.subplots(
            n_classes, 1, figsize=(12, 3.5 * n_classes), constrained_layout=True
        )
        if n_classes == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            plt.sca(ax)
            try:
                # For XGBoost multiclass, use the i-th set of SHAP values
                shap.summary_plot(
                    shap_values[i],
                    X_sample,
                    show=False,
                    max_display=10,
                    plot_type="dot",
                )
            except Exception:
                # Fallback: manual bar plot
                mean_abs = np.abs(shap_values[i]).mean(axis=0)
                top_idx = np.argsort(mean_abs)[-10:]
                top_features = [X.columns[j] for j in top_idx]
                top_values = mean_abs[top_idx]
                ax.barh(top_features, top_values, color="steelblue")
                ax.set_xlabel("Mean |SHAP value|")
            
            ax.set_title(
                f"SHAP — {CLUSTER_NAMES.get(i, f'Cluster {i}')}",
                fontsize=11, fontweight="bold",
            )

        plt.savefig("xgb_shap_summary.png", dpi=130, bbox_inches="tight")
        plt.close()
        print("  Saved: xgb_shap_summary.png")

        # Also save mean |SHAP| per feature per cluster
        rows = []
        for i in range(n_classes):
            mean_abs = np.abs(shap_values[i]).mean(axis=0)
            for feat, val in zip(X.columns, mean_abs):
                rows.append({
                    "cluster":      i,
                    "cluster_name": CLUSTER_NAMES.get(i, f"Cluster {i}"),
                    "feature":      feat,
                    "mean_abs_shap": round(float(val), 5),
                })
        shap_df = pd.DataFrame(rows).sort_values(
            ["cluster", "mean_abs_shap"], ascending=[True, False]
        )
        shap_df.to_csv("xgb_shap_by_cluster.csv", index=False)
        print("  Saved: xgb_shap_by_cluster.csv")

    except ImportError:
        print("  SHAP not installed — skipping. Run: pip install shap")
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")


# ── 6. PREDICT ON NEW DATA ────────────────────────────────────────────────────

def predict_new(model_path: str, input_csv: str) -> None:
    """
    Load a saved model and predict homelessness type for new individuals.
    The input CSV must have the same columns as synthetic_individuals.csv.
    """
    print(f"\nPredicting on new data: {input_csv}")
    model = joblib.load(model_path)
    df    = pd.read_csv(input_csv)
    df    = encode_categoricals(df)

    available = [c for c in FEATURE_COLS if c in df.columns]
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    X = df[FEATURE_COLS].fillna(0).astype(float)

    pred_class = model.predict(X)
    pred_proba = model.predict_proba(X)

    df["predicted_cluster"]      = pred_class
    df["predicted_cluster_name"] = [CLUSTER_NAMES.get(c, str(c)) for c in pred_class]

    # Add probability per cluster as columns
    for i in range(N_CLUSTERS):
        df[f"prob_cluster_{i}_{CLUSTER_NAMES.get(i,'').split('/')[0].strip()}"] = pred_proba[:, i].round(3)

    out_path = Path(input_csv).stem + "_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"  Predictions saved to: {out_path}")
    print(f"  Distribution of predicted types:\n{df['predicted_cluster_name'].value_counts().to_string()}")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main(predict_csv: str = None) -> None:

    if predict_csv:
        predict_new("xgb_type_model.pkl", predict_csv)
        return

    # ── Load ─────────────────────────────────────────────────────────────────
    df = load_data()
    X, y = prepare_Xy(df)

    print(f"\nFeature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"Target classes : {sorted(y.unique())} ({y.nunique()} clusters)")
    print(f"Class balance  :\n{y.value_counts().sort_index().rename(CLUSTER_NAMES).to_string()}")

    # ── Train ────────────────────────────────────────────────────────────────
    model = train_model(X, y)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    evaluate(model, X, y)

    # ── Feature importance ───────────────────────────────────────────────────
    print("\nFeature importance (gain):")
    imp_df = plot_feature_importance(model, list(X.columns))
    print(imp_df.head(10).to_string(index=False))

    # ── SHAP (detailed, per-cluster) ─────────────────────────────────────────
    shap_analysis(model, X)

    # ── Save model ───────────────────────────────────────────────────────────
    joblib.dump(model, "xgb_type_model.pkl")
    print("\nSaved: xgb_type_model.pkl")

    print("\n" + "=" * 60)
    print("Done. Outputs:")
    print("  xgb_type_model.pkl           — trained model")
    print("  xgb_feature_importance.png   — gain importance bar chart")
    print("  xgb_feature_importance.csv   — importance table")
    print("  xgb_confusion_matrix.png     — per-cluster confusion matrix")
    print("  xgb_shap_by_cluster.csv      — SHAP importances per cluster")
    print("  xgb_shap_summary.png         — SHAP beeswarm (if shap installed)")
    print("=" * 60)

    print("""
To predict on new individuals:
  python xgb_homelessness_type.py --predict my_new_people.csv
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost homelessness type classifier")
    parser.add_argument(
        "--predict", metavar="CSV", default=None,
        help="Path to new individuals CSV to predict on (uses saved model)"
    )
    args = parser.parse_args()
    main(predict_csv=args.predict)