"""
generate_model_csv_limited_mean.py

KEY CHANGE:
Instead of averaging ALL matching synthetic rows during imputation,
this version randomly samples up to 5 matching rows and averages ONLY those.

This preserves:
- variance
- separability
- realistic chronic distributions

while still keeping some smoothing/stability.

Critical behavior:
- chronic rows ONLY impute from chronic synthetic rows
- non-chronic rows ONLY impute from non-chronic synthetic rows

This should substantially improve:
- probability spread
- confusion matrix balance
- chronic predictions

python final/generate_model_csv.py \
        --bnl  final/new_lanark.csv \
        --syn  synthetic_data/synthetic_individuals.csv \
        --out  final/new_output.csv
"""

import argparse
from datetime import date
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------

COL_DOB = "Date of birth"
COL_AGE_CALC = "Age calculator"
COL_SLEEPING = "Current sleeping arrangements"
COL_INDIGENOUS = "Indigenous identity"
COL_ASYLUM = "Asylum Seeker"
COL_CHRONIC = "3.1\nChronically homeless"
COL_INCOME = "Income Source"

REFERENCE_DATE = date.today()

# ---------------------------------------------------------------------
# Imputed columns
# ---------------------------------------------------------------------

IMPUTED_COLS = [
    "years_homeless",
    "mental_health",
    "substance_use",
    "physical_health",
    "outdoor_sleeping",
    "lgbtq",
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "housing_loss_income",
    "housing_loss_health",
]

MATCH_COLS = [
    "age",
    "youth",
    "indigenous",
    "no_income",
]

OUTPUT_COLS = [
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
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def calc_age(dob_val, age_calc_val):

    try:
        a = float(age_calc_val)

        if 0 < a < 120:
            return a

    except:
        pass

    if not pd.isna(dob_val):

        try:
            dob = pd.to_datetime(
                str(dob_val),
                dayfirst=True
            ).date()

            return round(
                (REFERENCE_DATE - dob).days / 365.25,
                2
            )

        except:
            pass

    return np.nan


def is_youth(age):

    if pd.isna(age):
        return np.nan

    return 1 if age <= 25 else 0


def parse_chronic(val):

    if pd.isna(val):
        return 0

    s = str(val).strip().lower()

    # positive values
    if s in (
        "ü",
        "yes",
        "y",
        "true",
        "1",
        "1.0",
    ):
        return 1

    # negative values
    if s in (
        "",
        "0",
        "0.0",
        "no",
        "n",
        "false",
    ):
        return 0

    try:
        return 1 if float(s) >= 1 else 0

    except:
        return 0


def parse_indigenous(val):

    if pd.isna(val):
        return 0

    return 1 if str(val).strip() in (
        "Métis",
        "First Nations: off-reserve",
        "Non-status",
    ) else 0


def parse_outdoor_sleeping(val):

    if pd.isna(val):
        return np.nan

    v = str(val).strip().lower()

    outdoor_terms = [
        "unsheltered",
        "trailer: unsafe",
    ]

    return 1 if any(
        t in v for t in outdoor_terms
    ) else 0


def parse_no_income(val):

    if pd.isna(val):
        return 0

    s = str(val).strip().lower()

    return 1 if s in (
        "?",
        "none",
        "no income",
        "noincome",
    ) else 0


# ---------------------------------------------------------------------
# Build features
# ---------------------------------------------------------------------

def build_bnl_features(df):

    out = pd.DataFrame(index=df.index)

    out["age"] = [
        calc_age(dob, age_calc)
        for dob, age_calc in zip(
            df[COL_DOB],
            df[COL_AGE_CALC]
        )
    ]

    out["youth"] = out["age"].apply(is_youth)

    out["chronic_homeless"] = df[
        COL_CHRONIC
    ].apply(parse_chronic)

    out["indigenous"] = df[
        COL_INDIGENOUS
    ].apply(parse_indigenous)

    out["outdoor_sleeping"] = df[
        COL_SLEEPING
    ].apply(parse_outdoor_sleeping)

    out["immigrant"] = df[
        COL_ASYLUM
    ].apply(
        lambda v: 1 if (
            not pd.isna(v)
            and str(v).strip() != ""
        ) else np.nan
    )

    out["no_income"] = df[
        COL_INCOME
    ].apply(parse_no_income)

    return out


# ---------------------------------------------------------------------
# LIMITED-MEAN IMPUTATION
# ---------------------------------------------------------------------

def impute_from_synthetic(
    bnl_features,
    syn
):

    syn_work = syn.copy()

    results = []

    for _, row in bnl_features.iterrows():

        chronic_val = row.get("chronic_homeless")

        # -------------------------------------------------------------
        # STRICT chronic separation
        # -------------------------------------------------------------

        if not pd.isna(chronic_val):

            syn_subset = syn_work[
                syn_work["chronic_homeless"] == chronic_val
            ]

        else:

            syn_subset = syn_work.copy()

        global_avgs = {
            c: syn_subset[c].mean()
            for c in IMPUTED_COLS
            if c in syn_subset.columns
        }

        age_val = row.get("age")

        criteria = {
            col: row[col]
            for col in MATCH_COLS
            if col != "age"
            and not pd.isna(row.get(col))
        }

        best_match = None
        keys = list(criteria.keys())

        for n_drop in range(len(keys) + 1):

            active = keys[:len(keys) - n_drop]

            filtered = syn_subset.copy()

            # ---------------------------------------------------------
            # age window
            # ---------------------------------------------------------

            if not pd.isna(age_val):

                filtered = filtered[
                    (filtered["age"] >= age_val - 5)
                    & (filtered["age"] <= age_val + 5)
                ]

            # ---------------------------------------------------------
            # exact matches
            # ---------------------------------------------------------

            for col in active:

                filtered = filtered[
                    filtered[col] == criteria[col]
                ]

            if len(filtered) > 0:

                # -----------------------------------------------------
                # KEY CHANGE:
                # only average up to 5 random matches
                # -----------------------------------------------------

                sample_n = min(5, len(filtered))

                sampled = filtered.sample(
                    n=sample_n,
                    replace=False
                )

                CONTINUOUS_COLS = [
                    "years_homeless",
                ]

                BINARY_COLS = [
                    "mental_health",
                    "substance_use",
                    "physical_health",
                    "outdoor_sleeping",
                    "lgbtq",
                    "immigrant",
                    "foster_care_history",
                    "incarceration_history",
                    "housing_loss_income",
                    "housing_loss_health",
                ]

                best_match = {}

                # continuous -> mean
                for c in CONTINUOUS_COLS:

                    best_match[c] = sampled[c].mean()

                # binary -> sample real value
                for c in BINARY_COLS:

                    best_match[c] = sampled.iloc[0][c]

                break

        if best_match is None:
            best_match = global_avgs.copy()

        results.append(best_match)

    return pd.DataFrame(
        results,
        index=bnl_features.index
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(df):

    n = len(df)

    print(f"\n{'=' * 60}")
    print(f"COHORT SUMMARY (n={n:,})")
    print(f"{'=' * 60}")

    def show(series, label, binary=False):

        vals = pd.to_numeric(
            series,
            errors="coerce"
        ).fillna(0)

        if binary:

            positives = int(vals.sum())

            pct = positives / len(vals) * 100

            print(
                f"  {label:<38} "
                f"{pct:5.1f}%  "
                f"({positives:,}/{len(vals):,})"
            )

        else:

            print(
                f"  {label:<38} "
                f"avg = {vals.mean():.2f} "
                f"(n={len(vals):,})"
            )

    show(df["age"], "Age (avg)")
    show(df["years_homeless"], "Years homeless (avg)")

    show(
        df["chronic_homeless"],
        "Chronically homeless",
        binary=True
    )

    show(
        df["youth"],
        "Youth (age ≤ 25)",
        binary=True
    )

    show(
        df["indigenous"],
        "Indigenous",
        binary=True
    )

    show(
        df["outdoor_sleeping"],
        "Outdoor / unsafe sleeping",
        binary=True
    )

    show(
        df["immigrant"],
        "Immigrant (imputed)",
        binary=True
    )

    show(
        df["no_income"],
        "No income",
        binary=True
    )

    show(
        df["mental_health"],
        "Mental health (imputed)",
        binary=True
    )

    show(
        df["substance_use"],
        "Substance use (imputed)",
        binary=True
    )

    show(
        df["physical_health"],
        "Physical health (imputed)",
        binary=True
    )

    show(
        df["lgbtq"],
        "LGBTQ+ (imputed)",
        binary=True
    )

    show(
        df["foster_care_history"],
        "Foster care history (imputed)",
        binary=True
    )

    show(
        df["incarceration_history"],
        "Incarceration history (imputed)",
        binary=True
    )

    show(
        df["housing_loss_income"],
        "Housing loss: income (imputed)",
        binary=True
    )

    show(
        df["housing_loss_health"],
        "Housing loss: health (imputed)",
        binary=True
    )

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(bnl_path, syn_path, out_path):

    print(f"Reading BNL: {bnl_path}")

    df = pd.read_csv(
        bnl_path,
        low_memory=False,
        skiprows=[1]
    )

    print(f"Loaded {len(df):,} BNL rows")

    print(f"Reading synthetic: {syn_path}")

    syn = pd.read_csv(syn_path)

    print(f"Loaded {len(syn):,} synthetic rows")

    print("\nBuilding features...")

    bnl_feat = build_bnl_features(df)

    before = len(bnl_feat)

    bnl_feat = bnl_feat.dropna(
        subset=["age"]
    )

    print(
        f"Dropped {before - len(bnl_feat):,} "
        f"rows missing age"
    )

    print("\nImputing...")

    imputed = impute_from_synthetic(
        bnl_feat,
        syn
    )

    result = bnl_feat.copy()

    # -----------------------------------------------------------------
    # Apply imputations correctly
    # -----------------------------------------------------------------

    for col in IMPUTED_COLS:

        # fully impute ALL imputed columns
        result[col] = imputed[col]

    # ensure binary columns are numeric
    binary_cols = [
        "chronic_homeless",
        "youth",
        "indigenous",
        "outdoor_sleeping",
        "mental_health",
        "substance_use",
        "physical_health",
        "lgbtq",
        "immigrant",
        "foster_care_history",
        "incarceration_history",
        "no_income",
        "housing_loss_income",
        "housing_loss_health",
    ]

    for col in binary_cols:

        result[col] = (
            pd.to_numeric(result[col], errors="coerce")
            .fillna(0)
            .clip(0, 1)
        )

    # years homeless must stay continuous
    result["years_homeless"] = pd.to_numeric(
        result["years_homeless"],
        errors="coerce"
    )

    # chronic must stay binary
    result["chronic_homeless"] = (
        pd.to_numeric(
            result["chronic_homeless"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    result = result[
        OUTPUT_COLS
    ].round(4)

    result.to_csv(
        out_path,
        index=False
    )

    print(
        f"\nSaved → {out_path}"
    )

    print_summary(result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bnl",
        default="new_lanark.csv"
    )

    parser.add_argument(
        "--syn",
        default="synthetic_individuals.csv"
    )

    parser.add_argument(
        "--out",
        default="output_model_ready.csv"
    )

    args = parser.parse_args()

    main(
        args.bnl,
        args.syn,
        args.out
    )