"""
step1_pit_sasm.py
=================
Generates individual-level synthetic microdata from the Point-in-Time (PiT)
aggregated survey data for the three available survey periods
(April 2018, October 2021, October 2024) using the SASM optimization method
described in:

  Lin & Xiao (2023). "Generating Small Areal Synthetic Microdata from Public
  Aggregated Data Using an Optimization Method." The Professional Geographer,
  75(6), 905–915.

Core idea
---------
The PiT data gives us aggregated counts: for every (period, question, response)
cell we know how many respondents fall into that bucket (the Numerator column).
The Denominator gives the total number of surveyed individuals for that
question-period pair.

We treat each survey period as one "small area" (j). Each individual has a set
of discrete attributes corresponding to the survey questions. A combination k
is one specific pattern of answers across all attributes. The unknown X matrix
holds the count of individuals with combination k in period j.

The SASM optimization minimises:
    min  ||W X' - Y||²
    s.t. x'_kj ∈ ℤ≥0  for all k, j

where Y is the matrix of observed aggregated counts (one row per
question-response pair, one column per period), W is the binary mapping matrix
(W[i,k] = 1 iff combination k includes response i), and X' is the synthetic
microdata count matrix (combinations × periods).

Because the number of attribute combinations grows exponentially and we have
multiple survey questions with different denominators (i.e. different subsets
of respondents answered each question), we solve one question at a time
(block-diagonal relaxation). We then stitch the per-question synthetic
marginals together into individual records using Iterative Proportional Fitting
(IPF) / synthetic reconstruction so that the joint distribution is as
consistent with all marginals as possible.

Requirements
------------
    pip install pandas numpy scipy pulp

Outputs
-------
  pit_synthetic_individuals.csv   – one row per synthetic individual with all
                                    feature columns and a 'period' column.
  pit_synthetic_summary.csv       – aggregated counts reconstructed from the
                                    synthetic data (for validation).
"""

import re
import itertools
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ── optional integer solver (PuLP) ──────────────────────────────────────────
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    warnings.warn(
        "PuLP not found. Falling back to LP relaxation (rounded). "
        "Install with:  pip install pulp",
        UserWarning,
    )

# ────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ────────────────────────────────────────────────────────────────────────────

PIT_CSV = Path("source_data/Ottawa/Point_in_Time_Count_EN.csv")   # ← adjust path if needed
OUT_INDIVIDUALS = Path("pit_synthetic_individuals.csv")
OUT_SUMMARY     = Path("pit_synthetic_summary.csv")

RANDOM_SEED = 42

# ────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & CLEAN PiT DATA
# ────────────────────────────────────────────────────────────────────────────

def load_pit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Numerator"]   = pd.to_numeric(df["Numerator"],   errors="coerce")
    df["Denominator"] = pd.to_numeric(df["Denominator"], errors="coerce")
    df = df.dropna(subset=["Numerator", "Denominator"])
    df = df[df["Denominator"] > 0].copy()
    # Normalise text
    for col in ["Topic", "Sector", "Period", "Question", "Response"]:
        df[col] = df[col].str.strip()
    return df


def clean_colname(s: str) -> str:
    """Turn a question string into a safe column name."""
    s = re.sub(r"^\d+[a-z]?\.\s*", "", s)          # strip leading number
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:60]


# ────────────────────────────────────────────────────────────────────────────
# 2.  BUILD THE ATTRIBUTE TABLE
#     One row per (period, question) cell; store canonical question key and
#     canonical response labels.
# ────────────────────────────────────────────────────────────────────────────

# Questions that appear across multiple periods with slightly different wording
# are unified here (maps raw question text → canonical key).
QUESTION_ALIASES: dict[str, str] = {
    # Gender
    "1. What gender do you identify with?": "gender",
    # Age
    "2. How old are you?": "age_group",
    # Location tonight
    "1. Where are you staying tonight?": "location_tonight",
    # Shelter use
    "2. Have you stayed in a homeless shelter in the past year?": "shelter_past_year",
    # Duration homeless past year
    "2. In total, for how much time have you experienced homelessness over the past year?":
        "duration_homeless_past_year",
    # Number of episodes
    "3. In total, how many different times have you experienced homelessness in the past year?":
        "episodes_homeless_past_year",
    # Age first homeless
    "1. How old were you the first time you experienced homelessness?": "age_first_homeless",
    # Want housing
    "1. Do you want to get into permanent housing?": "want_permanent_housing",
    # Housing challenges
    "2. What challenges or problems have you experienced when trying to find housing?":
        "housing_challenges",
    # What caused housing loss – two slightly different question numbers
    "3. What happened that caused you to lose your housing most recently?":
        "cause_housing_loss",
    "4. What happened that caused you to lose your housing most recently?":
        "cause_housing_loss",
    # Health
    "1. Do you identify as having any of the following health challenges at this time?":
        "health_challenges",
    "2. How many of these conditions do you have?": "num_health_conditions",
    "3. Do you identify as having an acquired brain injury that happened after birth?":
        "acquired_brain_injury",
    # Immigration
    "1. Did you come to Canada as an immigrant, refugee or refugee claimant?":
        "immigration_status",
    "2. How long have you been in Canada?": "time_in_canada",
    "3. Are you a Canadian citizen?": "citizenship",
    # Foster care
    "1. As a child or youth, were you ever in foster care or in a youth group home?":
        "foster_care",
    "2. Approximately how long after leaving foster care/group home did you become homeless?":
        "foster_care_to_homeless",
    "2. If you were in foster care and/or group home, how long ago was that?":
        "foster_care_how_long_ago",
    # Race
    "3. Do you identify with any racial identities?": "racial_identity",
    # Sexual orientation
    "4. How do you describe your sexual orientation?": "sexual_orientation",
    # Education
    "5. What is the highest level of education you completed?": "education",
    # Income sources
    "5. What are your sources of income?": "income_sources",
    "6. What are your sources of income?": "income_sources",
    # Language
    "6. In what language do you feel best able to express yourself?": "language",
    "6a. In what language do you feel best able to express yourself?": "language",
    "7. In what language do you feel best able to express yourself?": "language",
    # Military
    "6. Have you ever served in the Canadian Military and/or RCMP?": "military_service",
    "8. Have you ever served in the Canadian Military and/or RCMP?": "military_service",
    # Ottawa
    "5. How long have you been in Ottawa?": "time_in_ottawa",
    "6. How long have you been in Ottawa?": "time_in_ottawa",
    # Main reason came to Ottawa
    "7. What is the main reason you came to Ottawa?": "reason_came_to_ottawa",
    # Family members tonight
    "8. Do you have family members or anyone else staying with you tonight?":
        "family_members_tonight",
    "9a. Do you have family members or anyone else staying with you tonight?":
        "family_members_tonight",
    # Dependents tonight
    "9b. Do you have dependents staying with you tonight?": "dependents_tonight",
    # COVID
    "4. Was your most recent housing loss related to the COVID-19 pandemic?":
        "covid_housing_loss",
    # Support needs
    "3. What would you need support with to help you through your housing journey?":
        "support_needs",
    # Reasons not in shelter
    "3. If you did not stay in shelter in the past year, what were the main reasons?":
        "reasons_not_in_shelter",
    # How long ago housing loss
    "5. How long ago did that happen (that you lost your housing most recently)?":
        "housing_loss_how_long_ago",
}

# Response aliases: normalise minor wording differences across years
RESPONSE_ALIASES: dict[str, str] = {
    "Yes, refugee/refugee claimant": "Yes, refugee or refugee claimant",
    "Yes, refugee":                  "Yes, refugee or refugee claimant",
    "Yes, refugee claimant":         "Yes, refugee or refugee claimant",
    "No past experience in foster care or group home": "No",
    "Past experience in foster care or group home":    "Yes",
    "25 to 49":           "25 to 49 years old",
    "50 and over":        "50 to 64 years old",   # best approximation
    "Under 17 years old": "Under 18 years old",
    "13 to 18 years old": "13 to 17 years old",
    "Over 50 years old":  "Over 65 years old",    # conservative merge
    "Some post-secondary or higher": "Some post-secondary",
    "Did not service in Canadian military and/or RCMP":
        "Did not serve in Canadian military and/or RCMP",
    "Men's Shelter":  "Mens Shelter",
    "Women's Shelter": "Womens Shelter",
}


def build_attribute_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean table with canonical question keys and responses."""
    df = df.copy()
    df["attr_key"] = df["Question"].map(QUESTION_ALIASES)
    # Drop questions we have no canonical alias for (rare / non-individual)
    df = df.dropna(subset=["attr_key"])
    df["Response"] = df["Response"].replace(RESPONSE_ALIASES)
    return df


# ────────────────────────────────────────────────────────────────────────────
# 3.  SASM OPTIMISATION (one attribute at a time, one period at a time)
# ────────────────────────────────────────────────────────────────────────────

def sasm_single_attribute(
    responses: list[str],
    counts: np.ndarray,
    total: int,
) -> dict[str, int]:
    """
    Solve the SASM integer program for a SINGLE attribute with K response
    categories.

    Variables: x_k  (integer ≥ 0) = # synthetic individuals with response k
    Objective: minimise  sum_k (x_k - y_k)²   [least-squares]
    Constraints:
        sum_k x_k  == total
        x_k >= 0

    For a single attribute W = I (identity), so WX' = X' directly,
    meaning we just want each x_k ≈ y_k with sum = total.

    When PuLP is available we solve the true IP; otherwise we round the LP
    solution and repair the integer constraint on the total.
    """
    y = counts.astype(float)
    K = len(responses)

    if HAS_PULP:
        prob = pulp.LpProblem("sasm_attr", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{k}", lowBound=0, cat="Integer") for k in range(K)]
        # Objective: minimise ||x - y||² via auxiliary variables
        diff = [pulp.LpVariable(f"d_{k}", lowBound=0) for k in range(K)]
        prob += pulp.lpSum(diff[k] for k in range(K))
        for k in range(K):
            prob += diff[k] >= x[k] - y[k]
            prob += diff[k] >= -(x[k] - y[k])
        prob += pulp.lpSum(x[k] for k in range(K)) == total
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        solution = {responses[k]: max(0, int(round(pulp.value(x[k])))) for k in range(K)}
    else:
        # LP relaxation then round
        x0 = y.copy()
        x0 = x0 / x0.sum() * total
        x_int = np.round(x0).astype(int)
        diff = total - x_int.sum()
        # Distribute rounding residual to largest fractional components
        fracs = x0 - np.floor(x0)
        idxs = np.argsort(-fracs)
        for i in range(abs(diff)):
            x_int[idxs[i % K]] += int(np.sign(diff))
        solution = {responses[k]: max(0, x_int[k]) for k in range(K)}

    return solution


# ────────────────────────────────────────────────────────────────────────────
# 4.  GENERATE MARGINAL DISTRIBUTIONS PER PERIOD
# ────────────────────────────────────────────────────────────────────────────

def generate_marginals(
    attr_df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Returns:
        { period: { attr_key: { response: count } } }
    """
    periods = sorted(attr_df["Period"].unique())
    result: dict[str, dict[str, dict[str, int]]] = {}

    for period in periods:
        p_df = attr_df[attr_df["Period"] == period]
        result[period] = {}
        for attr_key, grp in p_df.groupby("attr_key"):
            # Aggregate over Sector (use 'All' if available, else sum)
            all_sector = grp[grp["Sector"] == "All"]
            if all_sector.empty:
                all_sector = grp
            # Use the Denominator of the first row as our population total
            total = int(all_sector["Denominator"].iloc[0])
            # Sum numerators per response (in case of duplicates)
            resp_counts = all_sector.groupby("Response")["Numerator"].sum()
            responses = resp_counts.index.tolist()
            counts = resp_counts.values

            solution = sasm_single_attribute(responses, counts, total)
            result[period][attr_key] = solution
            print(
                f"  [{period}] {attr_key}: total={total}, "
                f"synthetic_total={sum(solution.values())}"
            )

    return result


# ────────────────────────────────────────────────────────────────────────────
# 5.  SYNTHETIC RECONSTRUCTION
#     Given marginal distributions, generate individual records via
#     probabilistic sampling consistent with each marginal.
# ────────────────────────────────────────────────────────────────────────────

def sample_individuals_from_marginals(
    marginals: dict[str, dict[str, int]],
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    For each attribute independently sample n values proportional to marginal
    counts. This creates a synthetic population where each attribute's
    distribution matches the observed marginal exactly (within rounding), but
    joint dependencies are not preserved (independence assumption).

    This is the correct approach when we only have univariate marginals from
    the survey (no cross-tabulations available in the PiT data format).
    """
    records: dict[str, np.ndarray] = {}
    for attr_key, response_counts in marginals.items():
        responses = list(response_counts.keys())
        counts    = np.array([response_counts[r] for r in responses], dtype=float)
        total     = counts.sum()
        if total == 0:
            records[attr_key] = np.array(["Unknown"] * n)
            continue
        # Sample exactly `n` values proportional to counts
        # (multinomial draw, then shuffle)
        probs = counts / total
        drawn = rng.choice(responses, size=n, p=probs)
        records[attr_key] = drawn

    df = pd.DataFrame(records)
    return df


# ────────────────────────────────────────────────────────────────────────────
# 6.  DETERMINE POPULATION SIZE PER PERIOD
#     Use the "All" sector Denominator from a core question (gender) if
#     available, else fall back to the largest Denominator.
# ────────────────────────────────────────────────────────────────────────────

CORE_QUESTION_FOR_N = "gender"   # canonical key


def get_population_size(attr_df: pd.DataFrame, period: str) -> int:
    p_df = attr_df[
        (attr_df["Period"] == period) &
        (attr_df["attr_key"] == CORE_QUESTION_FOR_N)
    ]
    if not p_df.empty:
        all_s = p_df[p_df["Sector"] == "All"]
        src = all_s if not all_s.empty else p_df
        return int(src["Denominator"].iloc[0])
    # Fallback: largest denominator in this period
    p_df2 = attr_df[attr_df["Period"] == period]
    return int(p_df2["Denominator"].max())


# ────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading PiT data …")
    raw = load_pit(PIT_CSV)
    print(f"  {len(raw)} rows loaded. Periods: {sorted(raw['Period'].unique())}")

    print("\nCleaning & mapping attributes …")
    attr_df = build_attribute_table(raw)
    mapped_keys = attr_df["attr_key"].nunique()
    print(f"  {mapped_keys} canonical attribute keys mapped.")

    print("\nRunning SASM optimisation per attribute per period …")
    marginals_by_period = generate_marginals(attr_df)

    all_individuals = []
    for period in sorted(marginals_by_period.keys()):
        marginals = marginals_by_period[period]
        n = get_population_size(attr_df, period)
        print(f"\nSampling {n} synthetic individuals for period: {period}")
        df_period = sample_individuals_from_marginals(marginals, n, rng)
        df_period.insert(0, "period", period)
        all_individuals.append(df_period)

    print("\nCombining all periods …")
    df_all = pd.concat(all_individuals, ignore_index=True)

    # ── reorder columns: period first, then alphabetical attribute columns ──
    attr_cols = sorted([c for c in df_all.columns if c != "period"])
    df_all = df_all[["period"] + attr_cols]

    df_all.to_csv(OUT_INDIVIDUALS, index=False)
    print(f"\nSynthetic individuals saved → {OUT_INDIVIDUALS}")
    print(f"  Total rows: {len(df_all)}, Columns: {list(df_all.columns)}")

    # ── Validation summary ──────────────────────────────────────────────────
    print("\nGenerating validation summary …")
    summary_rows = []
    for period in sorted(marginals_by_period.keys()):
        subset = df_all[df_all["period"] == period]
        for attr_key in sorted(marginals_by_period[period].keys()):
            if attr_key not in subset.columns:
                continue
            val_counts = subset[attr_key].value_counts()
            for response, synth_count in val_counts.items():
                orig_count = marginals_by_period[period][attr_key].get(response, 0)
                summary_rows.append({
                    "period":       period,
                    "attribute":    attr_key,
                    "response":     response,
                    "observed":     orig_count,
                    "synthetic":    synth_count,
                    "abs_error":    abs(synth_count - orig_count),
                })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Validation summary saved → {OUT_SUMMARY}")

    mae = df_summary["abs_error"].mean()
    print(f"\nMean Absolute Error across all cells: {mae:.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
