"""
predict_bnl_no_year.py
======================
Loads youth_xgboost_no_year.pkl and evaluates it on BNL data.

Accepts TWO input formats automatically:
  (a) clean_anonymized_bnl_v2.csv  — already has model feature columns
  (b) Raw Anonymized_BNL_Community_By_Name_List.csv — extracts features on the fly

Usage
-----
  python predict_bnl_no_year.py --bnl clean_anonymized_bnl_v2.csv --model youth_xgboost_no_year.pkl
  python predict_bnl_no_year.py --bnl Anonymized_BNL_Community_By_Name_List.csv --model youth_xgboost_no_year.pkl
  python predict_bnl_no_year.py --bnl clean_anonymized_bnl_v2.csv --model youth_xgboost_no_year.pkl --all-ages
  python merge/files/predict_bnl_no_year.py --bnl merge/clean_anonymized_bnl_v2.csv --model no_year/youth_xgboost_no_year.pkl
  python merge/files/predict_bnl_no_year.py --bnl merge/bnl_fused.csv --model no_year/youth_xgboost_no_year.pkl --all-ages
  python final/predict_bnl_no_year.py --bnl final/new_output.csv --model final/youth_xgboost_no_year.pkl --all-ages

Outputs (saved in current directory)
-------------------------------------
  bnl_predictions.csv
  bnl_confusion_matrix.png
  bnl_roc_curve.png
  bnl_feature_importance.png
  bnl_shap_summary.png
  bnl_shap_bar.png
"""

import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# Features the model was trained on (no year)
MODEL_FEATURES = [
    "age", "years_homeless", "mental_health", "substance_use",
    "physical_health", "outdoor_sleeping", "lgbtq", "indigenous",
    "immigrant", "foster_care_history", "incarceration_history",
    "no_income", "housing_loss_income", "housing_loss_health", "youth",
]

REF_DATE = pd.Timestamp("2025-01-01")


# ── Detect whether the CSV is already clean ───────────────────────────────────

def is_already_clean(df: pd.DataFrame) -> bool:
    """Returns True if the CSV already has the model feature columns."""
    return all(c in df.columns for c in MODEL_FEATURES)


# ── Feature extraction from raw BNL CSV ──────────────────────────────────────

def extract_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Extract model features from the raw multi-column BNL CSV."""
    out = pd.DataFrame(index=df.index)

    # age — from Date of birth (avoids '---' Excel formula placeholders)
    dob_col = next((c for c in df.columns if "date of birth" in str(c).lower()), None)
    if dob_col:
        dob = pd.to_datetime(df[dob_col], errors="coerce")
        out["age"] = ((REF_DATE - dob).dt.days / 365.25).round(1)
        out["age"] = out["age"].fillna(out["age"].median())
    else:
        age_col = next((c for c in df.columns if "age" in str(c).lower()), None)
        out["age"] = pd.to_numeric(df[age_col], errors="coerce").fillna(40) if age_col else 40.0

    # years_homeless
    mo_col = next(
        (c for c in df.columns if "12 months total" in str(c).lower()
         or "months_homeless_year" in str(c).lower()), None
    )
    out["years_homeless"] = (
        pd.to_numeric(df[mo_col], errors="coerce").fillna(0) / 12
    ) if mo_col else 0.0

    # outdoor_sleeping
    sleep_col = next((c for c in df.columns if "sleeping" in str(c).lower()), None)
    if sleep_col:
        sleep = df[sleep_col].fillna("").astype(str).str.lower()
        kw = ["unsheltered", "outside", "encampment", "tent", "vehicle", "street", "trailer"]
        out["outdoor_sleeping"] = sleep.apply(lambda x: int(any(k in x for k in kw)))
    else:
        out["outdoor_sleeping"] = 0

    # indigenous
    indig_col = next((c for c in df.columns if "indigenous identity" in str(c).lower()), None)
    flag_col  = next((c for c in df.columns if "3.1" in str(c) and "indigenous" in str(c).lower()), None)
    indig_text = df[indig_col].fillna("").astype(str).str.strip() if indig_col else pd.Series("", index=df.index)
    indig_flag = (df[flag_col] == "ü") if flag_col else pd.Series(False, index=df.index)
    indigenous_from_text = indig_text.apply(
        lambda x: 1 if x not in ["", "Non-Indigenous", "nan", "DD"] else 0
    )
    out["indigenous"] = np.where(indig_flag, 1, indigenous_from_text).astype(int)

    # mental_health / substance_use from assigned + referral agency
    assigned = next((c for c in df.columns if "assigned agency" in str(c).lower()), None)
    referral  = next((c for c in df.columns if "referral agency" in str(c).lower()), None)
    ag_text = (
        df[assigned].fillna("").astype(str) + " " +
        df[referral].fillna("").astype(str)
    ).str.lower() if assigned and referral else pd.Series("", index=df.index)
    mh_kw  = ["mental health", "cmha", "psychiatric", "psychiatry"]
    sub_kw = ["addiction", "substance", "harm reduction", "methadone"]
    out["mental_health"]  = ag_text.apply(lambda x: int(any(k in x for k in mh_kw)))
    out["substance_use"]  = ag_text.apply(lambda x: int(any(k in x for k in sub_kw)))

    # no_income
    income_col = next((c for c in df.columns if "income source" in str(c).lower()), None)
    if income_col:
        income = df[income_col].fillna("").astype(str).str.lower().str.strip()
        has_income_kw = ["ow", "odsp", "oas", "cpp", "ei", "employ", "vysa",
                         "pension", "support", "salary", "wage", "disability"]
        out["no_income"] = income.apply(
            lambda x: 0 if any(k in x for k in has_income_kw)
            else (1 if x in ["no income", "none", "nil"] else 0)
        )
    else:
        out["no_income"] = 0

    # youth
    youth_flag_col = next((c for c in df.columns if "3.1" in str(c) and "youth" in str(c).lower()), None)
    hoh_col = next((c for c in df.columns if "head of household" in str(c).lower()), None)
    youth_flag = (df[youth_flag_col] == "ü") if youth_flag_col else pd.Series(False, index=df.index)
    youth_hoh  = (df[hoh_col].astype(str).str.strip() == "Youth") if hoh_col else pd.Series(False, index=df.index)
    youth_age  = (out["age"] >= 16) & (out["age"] <= 25)
    out["youth"] = (youth_flag | youth_hoh | youth_age).astype(int)

    # incarceration_history
    inst_col  = next((c for c in df.columns if "provincial institution" in str(c).lower()), None)
    inst_flag = (df[inst_col] == "ü") if inst_col else pd.Series(False, index=df.index)
    sleep_col2 = next((c for c in df.columns if "sleeping" in str(c).lower()), None)
    corr = df[sleep_col2].fillna("").astype(str).str.lower().apply(
        lambda x: int("correctional" in x)
    ) if sleep_col2 else pd.Series(0, index=df.index)
    out["incarceration_history"] = (inst_flag | corr.astype(bool)).astype(int)

    # physical_health proxy
    out["physical_health"] = ((out["outdoor_sleeping"] == 1) | (out["age"] > 50)).astype(int)

    # housing_loss proxies
    out["housing_loss_income"] = out["no_income"]
    out["housing_loss_health"] = out["mental_health"]

    # not in BNL
    out["immigrant"]           = 0
    out["foster_care_history"] = 0
    out["lgbtq"]               = 0

    return out


def derive_true_labels_from_raw(df: pd.DataFrame) -> pd.Series:
    """Derive chronic labels from the raw BNL CSV columns."""
    months_col  = next((c for c in df.columns if "12 months total" in str(c).lower()), None)
    months_3col = next((c for c in df.columns if "36 months total" in str(c).lower()), None)
    chronic_data_col = next((c for c in df.columns if str(c).strip() == "Chronic data"), None)
    chronic_flag_col = next((c for c in df.columns if "3.1" in str(c) and "chronically homeless" in str(c).lower()), None)

    months_yr  = pd.to_numeric(df[months_col],  errors="coerce") if months_col  else pd.Series(np.nan, index=df.index)
    months_3yr = pd.to_numeric(df[months_3col], errors="coerce") if months_3col else pd.Series(np.nan, index=df.index)

    chronic_data = (df[chronic_data_col] == "Chronic") if chronic_data_col else pd.Series(False, index=df.index)
    chronic_flag = (df[chronic_flag_col] == "ü")       if chronic_flag_col else pd.Series(False, index=df.index)
    hud          = (months_yr >= 12) | (months_3yr >= 12)

    return (chronic_data | chronic_flag | hud.fillna(False)).astype(int)


# ── Load input CSV (auto-detect format) ──────────────────────────────────────

def load_input(path: str):
    """
    Returns (X_features_df, y_labels_series, df_original).
    Auto-detects clean vs raw format.
    """
    df = pd.read_csv(path, low_memory=False)

    if is_already_clean(df):
        print("  Detected: already-clean feature CSV (clean_bnl_v2 format)")
        X = df[MODEL_FEATURES].copy().fillna(0)
        if "chronic_homeless" in df.columns:
            y = pd.to_numeric(df["chronic_homeless"], errors="coerce").fillna(0).astype(int)
        else:
            print("  WARNING: no chronic_homeless column found — evaluation metrics will be skipped")
            y = None
        return X, y, df

    # Raw format — multi-header parse
    print("  Detected: raw BNL CSV — extracting features...")
    headers = pd.read_csv(path, low_memory=False, header=None, nrows=3).iloc[2]
    df_raw  = pd.read_csv(path, low_memory=False, header=None, skiprows=3)
    df_raw.columns = headers.values

    X = extract_features_from_raw(df_raw)
    y = derive_true_labels_from_raw(df_raw)
    return X[MODEL_FEATURES].fillna(0), y, df_raw


# ── Main ─────────────────────────────────────────────────────────────────────

def main(bnl_path: str, model_path: str, youth_only: bool):
    # Load model
    print(f"Loading model: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model        = bundle["xgb"]
    feature_cols = bundle["feature_cols"]
    print(f"  Features expected ({len(feature_cols)}): {feature_cols}")
    print(f"  Training ROC-AUC: {bundle.get('roc_auc', 'N/A')}")

    # Load BNL data
    print(f"\nLoading BNL data: {bnl_path}")
    X_all, y_all, df_orig = load_input(bnl_path)
    print(f"  Shape: {X_all.shape}")

    # Youth filter
    if youth_only:
        mask    = X_all["youth"] == 1
        X_all   = X_all[mask].reset_index(drop=True)
        y_all   = y_all[mask].reset_index(drop=True) if y_all is not None else None
        df_orig = df_orig[mask].reset_index(drop=True)
        print(f"  Youth-only filter: {len(X_all)} rows retained")

    # Align features to exact model order
    for col in feature_cols:
        if col not in X_all.columns:
            X_all[col] = 0
    X = X_all[feature_cols].fillna(0)

    if len(X) == 0:
        print("\nERROR: No rows remain after filtering.")
        return

    # Predict
    print("\nRunning predictions ...")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.30).astype(int)

    # ADDED
    print(pd.Series(probs).describe())
    print("\nTop probs:")
    print(np.sort(probs)[-20:])

    # Save predictions CSV
    out_df = df_orig.copy()
    out_df["chronic_pred"] = preds
    out_df["chronic_prob"] = probs.round(4)
    if y_all is not None:
        out_df["true_label"] = y_all.values
    out_df.to_csv("bnl_predictions.csv", index=False)
    print("  Saved: bnl_predictions.csv")

    # Evaluation
    if y_all is None:
        print("\nNo true labels — skipping evaluation metrics.")
        print(f"  Predicted chronic: {preds.sum()} / {len(preds)} ({preds.mean()*100:.1f}%)")
        return

    y   = y_all
    auc = roc_auc_score(y, probs)
    print(f"\n  ROC-AUC: {auc:.4f}")
    print(f"\n{classification_report(y, preds, target_names=['Not Chronic', 'Chronic'])}")
    cm = confusion_matrix(y, preds)
    print(f"Confusion matrix:\n{cm}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Chronic", "Chronic"],
                yticklabels=["Not Chronic", "Chronic"],
                annot_kws={"size": 14})
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — BNL Data (AUC = {auc:.3f})", fontsize=12)
    plt.tight_layout()
    plt.savefig("bnl_confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: bnl_confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1D9E75", lw=2.5, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — Chronic Homelessness (BNL Data)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("bnl_roc_curve.png", dpi=150)
    plt.close()
    print("  Saved: bnl_roc_curve.png")

    # Feature importance
    importances = model.feature_importances_
    imp_sorted  = sorted(zip(feature_cols, importances), key=lambda x: x[1])
    feats, vals = zip(*imp_sorted)
    colors = ["#1D9E75" if v > 0.1 else "#378ADD" if v > 0.03 else "#888780" for v in vals]
    fig, ax = plt.subplots(figsize=(8, len(feats) * 0.45 + 1.2))
    bars = ax.barh(feats, vals, color=colors, height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", ha="left", fontsize=9, color="#444")
    ax.set_xlabel("Feature importance (XGBoost gain)", fontsize=10)
    ax.set_title("Feature importance — BNL predictions", fontsize=11, pad=12)
    ax.set_xlim(0, max(vals) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    legend_items = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#1D9E75", "#378ADD", "#888780"]]
    ax.legend(legend_items, ["> 10%  (dominant)", "3–10%  (secondary)", "< 3%   (minor)"],
              fontsize=8, loc="lower right", framealpha=0.6)
    plt.tight_layout()
    plt.savefig("bnl_feature_importance.png", dpi=150)
    plt.close()
    print("  Saved: bnl_feature_importance.png")

    # SHAP
    print("\nComputing SHAP values ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.title("SHAP summary — BNL chronic homelessness", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig("bnl_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: bnl_shap_summary.png")

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, plot_type="bar", show=False)
    plt.title("SHAP mean absolute importance — BNL", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig("bnl_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: bnl_shap_bar.png")

    print("\nDone.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run youth_xgboost_no_year.pkl on BNL data and produce evaluation plots."
    )
    parser.add_argument(
        "--bnl",
        default="clean_anonymized_bnl_v2.csv",
        help="Path to BNL CSV — either clean_bnl_v2 format or raw BNL (auto-detected)",
    )
    parser.add_argument(
        "--model",
        default="youth_xgboost_no_year.pkl",
        help="Path to the trained model pkl",
    )
    parser.add_argument(
        "--all-ages",
        action="store_true",
        help="Run on all rows instead of youth-only (age 16-25). Default is youth-only.",
    )
    args = parser.parse_args()
    main(args.bnl, args.model, youth_only=not args.all_ages)