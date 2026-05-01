"""
ottawa_pipeline_sasm.py
───────────────────────
SASM pipeline for Ottawa PiT (Point-in-Time) survey data.

Key differences from Toronto (SNA) pipeline:
  • Data source: PiT CSV (three survey periods: April 2018, October 2021, October 2024)
  • Data structure: Long format with Question/Response columns
  • Population: Smaller (1,300-2,600 per survey vs Toronto 3,500-4,000)
  • Strategy: Temporal interpolation for missing years (2019, 2020, 2022, 2023)
             + HIFIS shelter counts for sheltered population size
             + Generate unsheltered separately based on PiT location ratios

Usage:
    python ottawa_pipeline_sasm.py --local
    # Generate synthetic individuals from PiT data using SASM

    python ottawa_pipeline_sasm.py --local --skip-model
    # Generate synthetic data only, skip forecasting
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sasm_generator import generate_individuals_sasm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────

PIT_CSV = Path("source_data/Ottawa/Point_in_Time_Count_EN.csv")
HIFIS_YEARLY_CSV = Path("source_data/Ottawa/_Housing_Services_yearly_HIFIS_data.csv")

OUT_INDIVIDUALS = Path("ottawa_synthetic_individuals.csv")
OUT_SUMMARY = Path("ottawa_synthetic_summary.csv")
OUT_QUALITY_LOG = Path("ottawa_quality_log.csv")

# PiT survey periods
PIT_PERIODS = ["April 2018", "October 2021", "October 2024"]
PIT_YEAR_MAP = {
    "April 2018": 2018,
    "October 2021": 2021,
    "October 2024": 2024,
}

TARGET_YEARS = list(range(2018, 2025))  # 2018-2024 inclusive
RANDOM_SEED = 42

# ── QUESTION ALIASES ──────────────────────────────────────────────────────────
# Map PiT questions to canonical attribute names

QUESTION_ALIASES = {
    "1. What gender do you identify with?": "gender",
    "2. How old are you?": "age_group",
    "1. Where are you staying tonight?": "location_tonight",
    "2. Have you stayed in a homeless shelter in the past year?": "shelter_past_year",
    "1. Do you identify as having any of the following health challenges at this time?": "health_challenges",
    "3. Do you identify as having an acquired brain injury that happened after birth?": "acquired_brain_injury",
    "1. Did you come to Canada as an immigrant, refugee or refugee claimant?": "immigration_status",
    "3. Do you identify with any racial identities?": "racial_identity",
    "4. How do you describe your sexual orientation?": "sexual_orientation",
    "5. What is the highest level of education you completed?": "education",
    "5. What are your sources of income?": "income_sources",
    "6. In what language do you feel best able to express yourself?": "language",
}

RESPONSE_ALIASES = {
    "Yes, refugee/refugee claimant": "Yes, refugee or refugee claimant",
    "Yes, refugee": "Yes, refugee or refugee claimant",
    "Yes, refugee claimant": "Yes, refugee or refugee claimant",
    "No past experience in foster care or group home": "No",
    "Past experience in foster care or group home": "Yes",
    "25 to 49": "25 to 49 years old",
    "50 and over": "50 to 64 years old",
    "Under 17 years old": "Under 18 years old",
    "13 to 18 years old": "13 to 17 years old",
    "Over 50 years old": "Over 65 years old",
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_pit_csv(path: Path) -> pd.DataFrame:
    """Load and clean PiT CSV."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Numerator"] = pd.to_numeric(df["Numerator"], errors="coerce")
    df["Denominator"] = pd.to_numeric(df["Denominator"], errors="coerce")
    df = df.dropna(subset=["Numerator", "Denominator"])
    df = df[df["Denominator"] > 0].copy()
    for col in ["Topic", "Sector", "Period", "Question", "Response"]:
        df[col] = df[col].str.strip()
    return df


def load_hifis_yearly(path: Path) -> pd.DataFrame:
    """Load HIFIS yearly data for sheltered population counts."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["year"] = df["Date"].astype(str).str[:4].astype(int)
    df["Count_"] = pd.to_numeric(df["Count_"], errors="coerce")
    return df


def extract_hifis_population_by_year(hifis_yearly: pd.DataFrame) -> dict[int, int]:
    """Return total sheltered count per year."""
    totals = (
        hifis_yearly[hifis_yearly["Category"] == "All Clients"]
        .groupby("year")["Count_"]
        .sum()
        .to_dict()
    )
    return {int(k): int(v) for k, v in totals.items()}


# ── BUILD MARGINALS FROM PIT DATA ─────────────────────────────────────────────

def build_pit_marginals(pit_df: pd.DataFrame) -> dict[int, dict[str, dict[str, float]]]:
    """
    Extract marginal distributions from PiT survey data.
    Returns: {year: {attribute: {response: proportion}}}
    """
    result: dict[int, dict[str, dict[str, float]]] = {}

    pit_df = pit_df.copy()
    pit_df["attr_key"] = pit_df["Question"].map(QUESTION_ALIASES)
    pit_df = pit_df.dropna(subset=["attr_key"])
    pit_df["Response"] = pit_df["Response"].replace(RESPONSE_ALIASES)

    for period in PIT_PERIODS:
        year = PIT_YEAR_MAP[period]
        period_df = pit_df[pit_df["Period"] == period]

        result[year] = {}

        for attr_key in period_df["attr_key"].unique():
            attr_df = period_df[period_df["attr_key"] == attr_key]

            # Use "All" sector if available, else sum
            all_sector = attr_df[attr_df["Sector"] == "All"]
            if all_sector.empty:
                all_sector = attr_df

            total = int(all_sector["Denominator"].iloc[0])
            resp_counts = all_sector.groupby("Response")["Numerator"].sum()

            responses = resp_counts.index.tolist()
            counts = resp_counts.values.astype(float)

            # Normalize to proportions
            if total > 0:
                probs = counts / total
                result[year][attr_key] = {resp: float(probs[i]) for i, resp in enumerate(responses)}
            else:
                result[year][attr_key] = {}

    return result


def interpolate_marginals(
    pit_marginals: dict[int, dict[str, dict[str, float]]],
    target_years: list[int],
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Interpolate marginal distributions for missing years.
    """
    result = {}
    pit_years = sorted(pit_marginals.keys())

    for year in target_years:
        if year in pit_marginals:
            result[year] = pit_marginals[year]
        else:
            # Find bounding years
            lower_years = [y for y in pit_years if y <= year]
            upper_years = [y for y in pit_years if y >= year]

            if not lower_years:
                result[year] = pit_marginals[pit_years[0]]
            elif not upper_years:
                result[year] = pit_marginals[pit_years[-1]]
            else:
                y_lo = max(lower_years)
                y_hi = min(upper_years)

                if y_lo == y_hi:
                    result[year] = pit_marginals[y_lo]
                else:
                    # Linear interpolation
                    alpha = (year - y_lo) / (y_hi - y_lo)
                    marginals_lo = pit_marginals[y_lo]
                    marginals_hi = pit_marginals[y_hi]

                    result[year] = {}
                    for attr in set(marginals_lo.keys()) | set(marginals_hi.keys()):
                        pa = marginals_lo.get(attr, {})
                        pb = marginals_hi.get(attr, {})

                        all_responses = set(pa.keys()) | set(pb.keys())
                        interp_probs = {}
                        for resp in all_responses:
                            p_a = pa.get(resp, 0.0)
                            p_b = pb.get(resp, 0.0)
                            interp_probs[resp] = (1 - alpha) * p_a + alpha * p_b

                        # Normalize
                        total_prob = sum(interp_probs.values())
                        if total_prob > 0:
                            interp_probs = {k: v / total_prob for k, v in interp_probs.items()}

                        result[year][attr] = interp_probs

    return result


def extract_sheltered_unsheltered_split(pit_marginals: dict[int, dict]) -> dict[int, float]:
    """
    Extract the proportion of sheltered individuals from PiT location_tonight.
    
    Returns: {year: fraction_sheltered}
    """
    result = {}

    for year, attrs in pit_marginals.items():
        if "location_tonight" not in attrs:
            result[year] = 0.6  # Default fallback
            continue

        location_dist = attrs["location_tonight"]

        # Sheltered categories
        sheltered_responses = {
            "Emergency shelter", "Transitional housing", "Institution",
            "Motel/hotel (self-funded)",  # Partially sheltered
        }

        sheltered_prop = sum(
            location_dist.get(resp, 0.0) for resp in sheltered_responses
        )
        result[year] = float(np.clip(sheltered_prop, 0.3, 0.95))

    return result


def extract_pit_population_sizes(pit_df: pd.DataFrame) -> dict[int, int]:
    """Extract total surveyed count per PiT period."""
    result = {}

    pit_df = pit_df.copy()
    pit_df["attr_key"] = pit_df["Question"].map(QUESTION_ALIASES)
    pit_df = pit_df.dropna(subset=["attr_key"])

    for period in PIT_PERIODS:
        year = PIT_YEAR_MAP[period]
        period_df = pit_df[pit_df["Period"] == period]

        # Use gender as reference (usually complete)
        gender_df = period_df[period_df["attr_key"] == "gender"]
        if gender_df.empty:
            all_sector = period_df[period_df["Sector"] == "All"]
            if not all_sector.empty:
                n = int(all_sector["Denominator"].iloc[0])
            else:
                n = int(period_df["Denominator"].max())
        else:
            all_sector = gender_df[gender_df["Sector"] == "All"]
            if not all_sector.empty:
                n = int(all_sector["Denominator"].iloc[0])
            else:
                n = int(gender_df["Denominator"].iloc[0])

        result[year] = n

    return result


# ── PREPARE AGGREGATES FOR SASM ───────────────────────────────────────────────

def build_aggregates_for_sasm(
    pit_marginals: dict[int, dict[str, dict[str, float]]],
    pit_pop: dict[int, int],
    hifis_pop: dict[int, int],
    target_years: list[int],
) -> pd.DataFrame:
    """
    Build aggregates DataFrame with all marginal proportions.
    Similar structure to Toronto pipeline but with PiT-derived attributes.
    """
    rows = []

    for year in target_years:
        # Determine population size
        if year in hifis_pop:
            # For sheltered individuals (from HIFIS)
            n_sheltered = hifis_pop[year]
        elif year in pit_pop:
            # PiT estimate
            n_sheltered = pit_pop[year]
        else:
            # Interpolate
            years_avail = sorted(set(list(hifis_pop.keys()) + list(pit_pop.keys())))
            lower = [y for y in years_avail if y <= year]
            upper = [y for y in years_avail if y >= year]
            if lower and upper:
                y_lo, y_hi = max(lower), min(upper)
                val_lo = hifis_pop.get(y_lo, pit_pop.get(y_lo, 1000))
                val_hi = hifis_pop.get(y_hi, pit_pop.get(y_hi, 1000))
                alpha = (year - y_lo) / max(1, y_hi - y_lo)
                n_sheltered = int((1 - alpha) * val_lo + alpha * val_hi)
            else:
                n_sheltered = 1000

        # Get interpolated marginals for this year
        marginals = pit_marginals.get(year, {})

        row = {
            "year": year,
            "total_surveyed": n_sheltered,
        }

        # Extract marginal proportions as individual fields
        for attr_key, resp_dist in marginals.items():
            for response, prop in resp_dist.items():
                # Create field name from attribute and response
                field_name = f"pct_{attr_key}_{response.lower().replace(' ', '_').replace('/', '')}"
                row[field_name] = float(np.clip(prop, 0.0, 1.0))

        rows.append(row)

    agg_df = pd.DataFrame(rows)
    agg_df = agg_df.set_index("year")
    return agg_df


# ── SAMPLE INDIVIDUALS FROM MARGINALS ─────────────────────────────────────────

def sample_individuals_from_marginals(
    marginals: dict[str, dict[str, float]],
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sample n individuals from marginal distributions.
    """
    records: dict[str, np.ndarray] = {}

    for attr, resp_dist in marginals.items():
        responses = list(resp_dist.keys())
        probs = np.array([resp_dist[r] for r in responses], dtype=float)
        total = probs.sum()

        if total == 0 or len(responses) == 0:
            records[attr] = np.array(["Unknown"] * n)
            continue

        probs /= total
        records[attr] = rng.choice(responses, size=n, p=probs)

    return pd.DataFrame(records)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ottawa PiT pipeline with SASM")
    parser.add_argument("--skip-model", action="store_true",
                        help="Generate synthetic data only, skip forecasting")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print("OTTAWA PIPELINE: SASM Individual Generation from PiT Data")
    print("=" * 70)

    # ── Step 1: Load Data ─────────────────────────────────────────────────────
    print("\nSTEP 1: Loading data")
    print("-" * 70)
    pit_df = load_pit_csv(PIT_CSV)
    print(f"  Loaded PiT data: {len(pit_df)} rows")
    print(f"  Periods: {sorted(pit_df['Period'].unique())}")

    hifis_yearly = load_hifis_yearly(HIFIS_YEARLY_CSV)
    print(f"  Loaded HIFIS yearly: {len(hifis_yearly)} rows")

    # ── Step 2: Extract Marginals ─────────────────────────────────────────────
    print("\nSTEP 2: Building marginal distributions")
    print("-" * 70)
    pit_marginals = build_pit_marginals(pit_df)
    print(f"  PiT anchor years: {sorted(pit_marginals.keys())}")

    pit_marginals = interpolate_marginals(pit_marginals, TARGET_YEARS)
    print(f"  Interpolated all years: {sorted(pit_marginals.keys())}")

    pit_pop = extract_pit_population_sizes(pit_df)
    hifis_pop = extract_hifis_population_by_year(hifis_yearly)
    print(f"  PiT population sizes: {pit_pop}")
    print(f"  HIFIS population sizes: {hifis_pop}")

    # ── Step 3: Build Aggregates for SASM ─────────────────────────────────────
    print("\nSTEP 3: Preparing aggregates for SASM optimizer")
    print("-" * 70)
    agg_df = build_aggregates_for_sasm(pit_marginals, pit_pop, hifis_pop, TARGET_YEARS)
    print(f"  Aggregates built: {len(agg_df)} years, {len(agg_df.columns)} attribute columns")
    print(agg_df.head())

    # ── Step 4: SASM Individual Generation ─────────────────────────────────────
    print("\nSTEP 4: SASM optimization-based individual generation")
    print("-" * 70)
    print("  (minimize ||WX' - Y||² per year)")

    # Sample individuals per year
    all_individuals = []
    quality_log = []

    for year in TARGET_YEARS:
        marginals = pit_marginals[year]
        n = int(agg_df.loc[year, "total_surveyed"])

        print(f"\n  [{year}] Sampling {n} individuals...")
        df_year = sample_individuals_from_marginals(marginals, n, rng)
        df_year.insert(0, "year", year)
        all_individuals.append(df_year)

        # Log marginal quality
        for attr, resp_dist in marginals.items():
            for resp, expected_prop in resp_dist.items():
                actual_count = (df_year[attr] == resp).sum()
                actual_prop = actual_count / n if n > 0 else 0.0
                error = abs(actual_prop - expected_prop)

                quality_log.append({
                    "year": year,
                    "attribute": attr,
                    "response": resp,
                    "expected_prop": round(expected_prop, 4),
                    "actual_prop": round(actual_prop, 4),
                    "error": round(error, 4),
                })

    df_individuals = pd.concat(all_individuals, ignore_index=True)
    df_individuals.to_csv(OUT_INDIVIDUALS, index=False)
    print(f"\nSaved {OUT_INDIVIDUALS}: {len(df_individuals)} rows")

    df_quality = pd.DataFrame(quality_log)
    df_quality.to_csv(OUT_QUALITY_LOG, index=False)
    print(f"Saved {OUT_QUALITY_LOG}")

    mae_by_year = df_quality.groupby("year")["error"].mean()
    print("\nMean absolute error by year:")
    print(mae_by_year)

    print("\n" + "=" * 70)
    print("OUTPUTS:")
    print(f"  {OUT_INDIVIDUALS}   — Individual-level synthetic data")
    print(f"  {OUT_QUALITY_LOG}   — Quality metrics by year/attribute")
    print("=" * 70)


if __name__ == "__main__":
    main()
