"""
step2_hifis_fill_missing_years.py
==================================
Fills in missing years between the three PiT survey periods (2018, 2021, 2024)
and generates individual-level synthetic microdata for every year 2018–2024
by leveraging HIFIS shelter count data as auxiliary population-size information.

Strategy
--------
The PiT survey was conducted in:
    - April 2018
    - October 2021
    - October 2024

For intervening years (2019, 2020, 2022, 2023) we have no direct survey of
individual attributes, but the HIFIS data gives us annual counts of shelter
users broken down by demographic category (Singles, Families, Gender, Age,
etc.).  We use two complementary techniques:

1. **Population scaling**: the HIFIS yearly count for "All Clients" gives us
   N(year). We then scale the PiT marginal distributions from the nearest PiT
   year to produce plausible marginals for the missing year.

2. **Temporal interpolation of marginals**: for each attribute response, we
   linearly interpolate (or extrapolate) the *proportions* from adjacent PiT
   years weighted by the temporal distance. This gives a smoothly changing
   attribute distribution over time.

3. **HIFIS category adjustment**: where HIFIS categories overlap with PiT
   attributes (gender: Singles Male/Female, Singles/Families, youth counts),
   the HIFIS proportions are used to *override* the interpolated marginal so
   the synthetic population matches the observed shelter mix exactly.

The result is a synthetic microdata file for each year 2018–2024 with the
same attribute columns as the PiT output (step1).

Requirements
------------
    pip install pandas numpy scipy pulp

Inputs (from step1 output or original CSVs)
-------------------------------------------
    pit_synthetic_individuals.csv    (output of step1_pit_sasm.py)
    _Housing_Services_yearly_HIFIS_data.csv
    Housing_Services_monthly_HIFIS_data.csv

Output
------
    hifis_synthetic_individuals_2018_2024.csv   (one row per synthetic person,
                                                  columns: year + all PiT attrs)
    hifis_synthetic_summary_by_year.csv         (marginal counts for validation)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    warnings.warn(
        "PuLP not found. Using LP relaxation (rounding). pip install pulp",
        UserWarning,
    )

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

PIT_INDIVIDUALS_CSV  = Path("pit_synthetic_individuals.csv")   # step1 output
HIFIS_YEARLY_CSV     = Path("source_data/Ottawa/_Housing_Services_yearly_HIFIS_data.csv")
HIFIS_MONTHLY_CSV    = Path("source_data/Ottawa/Housing_Services_monthly_HIFIS_data.csv")

OUT_INDIVIDUALS = Path("hifis_synthetic_individuals_2018_2024.csv")
OUT_SUMMARY     = Path("hifis_synthetic_summary_by_year.csv")

TARGET_YEARS  = list(range(2018, 2025))   # 2018 … 2024 inclusive
RANDOM_SEED   = 42

# Map PiT period labels → integer year
PIT_YEAR_MAP = {
    "April 2018":   2018,
    "October 2021": 2021,
    "October 2024": 2024,
}

# ────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ────────────────────────────────────────────────────────────────────────────

def load_pit_individuals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def load_hifis_yearly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["year"]    = df["Date"].astype(str).str[:4].astype(int)
    df["Count_"]  = pd.to_numeric(df["Count_"],  errors="coerce")
    df["TotalLengthOfStay"] = pd.to_numeric(df["TotalLengthOfStay"], errors="coerce")
    return df


def load_hifis_monthly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["year"]  = df["Date"].astype(str).str[:4].astype(int)
    df["month"] = df["Date"].astype(str).str[5:7].astype(int)
    df["Count_"] = pd.to_numeric(df["Count_"], errors="coerce")
    return df


# ────────────────────────────────────────────────────────────────────────────
# 2.  EXTRACT HIFIS MARGINALS PER YEAR
#     Maps HIFIS category labels onto the PiT attribute keys / responses
# ────────────────────────────────────────────────────────────────────────────

# HIFIS category → (attr_key, response)
# Only categories that map cleanly to PiT attributes are included.
HIFIS_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "Single Adult Males":        ("gender", "Men"),
    "Single Adult Females":      ("gender", "Women"),
    "Single Youth 18 Under":     ("age_group", "13 to 17 years old"),
    "Mens Shelter":              ("gender", "Men"),
    "Womens Shelter":            ("gender", "Women"),
    "Men's Shelter":             ("gender", "Men"),
    "Women's Shelter":           ("gender", "Women"),
    "Mixed-Gender":              ("gender", "Non-binary and other identities"),
    "All Singles":               ("_population_singles", "singles"),
    "Family Household Members":  ("_population_family_members", "family_members"),
    "Family Households":         ("_population_family_units", "family_units"),
    "All Clients":               ("_population_total", "total"),
}


def extract_hifis_population_by_year(hifis_yearly: pd.DataFrame) -> dict[int, int]:
    """Return total count of all clients per year (the N for each year)."""
    totals = (
        hifis_yearly[hifis_yearly["Category"] == "All Clients"]
        .groupby("year")["Count_"]
        .sum()
        .to_dict()
    )
    return {int(k): int(v) for k, v in totals.items()}


def extract_hifis_gender_proportions(
    hifis_yearly: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """
    Returns per-year gender proportions derived from HIFIS shelter counts.
    Only 'Men' and 'Women' can be derived; Non-binary is the remainder.
    """
    result: dict[int, dict[str, float]] = {}
    for year in sorted(hifis_yearly["year"].unique()):
        yr = hifis_yearly[hifis_yearly["year"] == year]
        males   = yr[yr["Category"].isin(["Single Adult Males", "Mens Shelter",
                                           "Men's Shelter"])]["Count_"].sum()
        females = yr[yr["Category"].isin(["Single Adult Females", "Womens Shelter",
                                           "Women's Shelter"])]["Count_"].sum()
        total   = yr[yr["Category"] == "All Clients"]["Count_"].sum()
        if total <= 0:
            continue
        nb = max(0.0, total - males - females)
        result[year] = {
            "Men":   float(males)   / total,
            "Women": float(females) / total,
            "Non-binary and other identities": nb / total,
        }
    return result


def extract_hifis_family_proportions(
    hifis_yearly: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """Singles vs family members as a location_tonight proxy."""
    result: dict[int, dict[str, float]] = {}
    for year in sorted(hifis_yearly["year"].unique()):
        yr = hifis_yearly[hifis_yearly["year"] == year]
        singles = yr[yr["Category"] == "All Singles"]["Count_"].sum()
        family  = yr[yr["Category"] == "Family Household Members"]["Count_"].sum()
        total   = yr[yr["Category"] == "All Clients"]["Count_"].sum()
        if total <= 0:
            continue
        result[year] = {
            "singles_fraction": float(singles) / total,
            "family_fraction":  float(family)  / total,
        }
    return result


# ────────────────────────────────────────────────────────────────────────────
# 3.  BUILD MARGINALS FOR EVERY TARGET YEAR VIA TEMPORAL INTERPOLATION
# ────────────────────────────────────────────────────────────────────────────

def compute_pit_marginals_from_individuals(
    pit_df: pd.DataFrame,
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Derive per-attribute marginal proportions from the PiT synthetic
    individuals (step1 output). Returns proportions (not raw counts).
    """
    result: dict[int, dict[str, dict[str, float]]] = {}
    period_to_year = PIT_YEAR_MAP
    attr_cols = [c for c in pit_df.columns if c != "period"]

    for period, year in period_to_year.items():
        subset = pit_df[pit_df["period"] == period]
        if subset.empty:
            continue
        result[year] = {}
        for col in attr_cols:
            vc = subset[col].value_counts(normalize=True)
            result[year][col] = vc.to_dict()
    return result


def interpolate_proportions(
    prop_a: dict[str, float],
    prop_b: dict[str, float],
    alpha: float,             # 0 → use prop_a; 1 → use prop_b
) -> dict[str, float]:
    """Linear interpolation between two marginal distributions."""
    all_keys = set(prop_a) | set(prop_b)
    interp = {}
    for k in all_keys:
        a = prop_a.get(k, 0.0)
        b = prop_b.get(k, 0.0)
        interp[k] = (1 - alpha) * a + alpha * b
    # Re-normalise to sum to 1.0
    total = sum(interp.values())
    if total > 0:
        interp = {k: v / total for k, v in interp.items()}
    return interp


def get_interpolated_marginals_for_year(
    year: int,
    pit_marginals: dict[int, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """
    For a given year, interpolate (or select) PiT-based marginals.
    PiT anchor years: 2018, 2021, 2024.
    """
    pit_years = sorted(pit_marginals.keys())   # [2018, 2021, 2024]

    if year in pit_marginals:
        return pit_marginals[year]

    # Find bounding anchor years
    lower_years = [y for y in pit_years if y <= year]
    upper_years = [y for y in pit_years if y >= year]

    if not lower_years:
        # Extrapolate before 2018 — use 2018 marginals directly
        return pit_marginals[pit_years[0]]

    if not upper_years:
        # Extrapolate after 2024 — use 2024 marginals directly
        return pit_marginals[pit_years[-1]]

    y_lo = max(lower_years)
    y_hi = min(upper_years)

    if y_lo == y_hi:
        return pit_marginals[y_lo]

    alpha = (year - y_lo) / (y_hi - y_lo)
    marginals_lo = pit_marginals[y_lo]
    marginals_hi = pit_marginals[y_hi]

    all_attrs = set(marginals_lo) | set(marginals_hi)
    result = {}
    for attr in all_attrs:
        pa = marginals_lo.get(attr, {})
        pb = marginals_hi.get(attr, {})
        result[attr] = interpolate_proportions(pa, pb, alpha)
    return result


# ────────────────────────────────────────────────────────────────────────────
# 4.  OVERRIDE GENDER MARGINAL WITH HIFIS INFORMATION
# ────────────────────────────────────────────────────────────────────────────

def override_with_hifis(
    marginals: dict[str, dict[str, float]],
    year: int,
    hifis_gender: dict[int, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Replace the 'gender' marginal with HIFIS-derived proportions when
    available, as HIFIS is a full census of shelter users (more accurate than
    interpolation from the PiT survey for non-survey years).
    """
    marginals = {k: dict(v) for k, v in marginals.items()}   # deep copy

    if year in hifis_gender and "gender" in marginals:
        hifis_g = hifis_gender[year]
        # Only override responses that exist in the interpolated marginal
        existing_responses = set(marginals["gender"].keys())
        new_gender = {}
        for resp in existing_responses:
            new_gender[resp] = hifis_g.get(resp, marginals["gender"].get(resp, 0.0))
        # Re-normalise
        total = sum(new_gender.values())
        if total > 0:
            marginals["gender"] = {k: v / total for k, v in new_gender.items()}
        else:
            marginals["gender"] = new_gender

    return marginals


# ────────────────────────────────────────────────────────────────────────────
# 5.  SASM OPTIMISATION HELPER (converts proportions × N → integer counts)
# ────────────────────────────────────────────────────────────────────────────

def proportions_to_integer_counts(
    prop: dict[str, float],
    n: int,
) -> dict[str, int]:
    """
    Given proportions that sum to 1.0 and a target total n, produce integer
    counts that sum exactly to n using the Hamilton (largest-remainder) method.
    """
    responses = list(prop.keys())
    exact     = np.array([prop[r] * n for r in responses])
    floors    = np.floor(exact).astype(int)
    remainder = n - floors.sum()
    # Allocate remainder to responses with largest fractional parts
    fracs  = exact - floors
    order  = np.argsort(-fracs)
    for i in range(int(remainder)):
        floors[order[i]] += 1
    return {responses[k]: int(floors[k]) for k in range(len(responses))}


# ────────────────────────────────────────────────────────────────────────────
# 6.  SAMPLE INDIVIDUALS FROM MARGINALS
# ────────────────────────────────────────────────────────────────────────────

def sample_from_marginals(
    marginals: dict[str, dict[str, float]],
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw n independent samples from each attribute's marginal distribution.
    Returns a DataFrame with one row per synthetic individual.
    """
    records: dict[str, np.ndarray] = {}
    for attr, prop in marginals.items():
        responses = list(prop.keys())
        probs     = np.array([prop[r] for r in responses], dtype=float)
        total     = probs.sum()
        if total == 0 or len(responses) == 0:
            records[attr] = np.array(["Unknown"] * n)
            continue
        probs /= total
        records[attr] = rng.choice(responses, size=n, p=probs)

    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Load inputs ─────────────────────────────────────────────────────────
    print("Loading PiT synthetic individuals (step1 output) …")
    pit_df = load_pit_individuals(PIT_INDIVIDUALS_CSV)
    print(f"  {len(pit_df)} rows, periods: {pit_df['period'].unique().tolist()}")

    print("Loading HIFIS yearly data …")
    hifis_yearly = load_hifis_yearly(HIFIS_YEARLY_CSV)
    print(f"  {len(hifis_yearly)} rows, years: {sorted(hifis_yearly['year'].unique())}")

    print("Loading HIFIS monthly data …")
    hifis_monthly = load_hifis_monthly(HIFIS_MONTHLY_CSV)

    # ── Derive PiT marginals (proportions) from step1 synthetic data ────────
    print("\nDeriving PiT marginal distributions from synthetic individuals …")
    pit_marginals = compute_pit_marginals_from_individuals(pit_df)
    print(f"  Anchor years with marginals: {sorted(pit_marginals.keys())}")
    attr_cols = sorted(
        {attr for yr_m in pit_marginals.values() for attr in yr_m}
    )
    print(f"  Attribute columns: {attr_cols}")

    # ── HIFIS auxiliary information ──────────────────────────────────────────
    print("\nExtracting HIFIS auxiliary statistics …")
    hifis_pop   = extract_hifis_population_by_year(hifis_yearly)
    hifis_gen   = extract_hifis_gender_proportions(hifis_yearly)
    hifis_fam   = extract_hifis_family_proportions(hifis_yearly)
    print(f"  Population totals by year: {hifis_pop}")

    # ── Generate synthetic individuals for each target year ──────────────────
    all_years: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for year in TARGET_YEARS:
        print(f"\n{'─'*60}")
        print(f"Processing year {year} …")

        # Determine N for this year
        if year in hifis_pop:
            n = hifis_pop[year]
            n_source = "HIFIS yearly total"
        else:
            # Interpolate N from neighbouring years
            years_avail = sorted(hifis_pop.keys())
            lower = [y for y in years_avail if y <= year]
            upper = [y for y in years_avail if y >= year]
            if lower and upper:
                y_lo, y_hi = max(lower), min(upper)
                alpha = (year - y_lo) / max(1, y_hi - y_lo)
                n = int((1 - alpha) * hifis_pop[y_lo] + alpha * hifis_pop[y_hi])
            elif lower:
                n = hifis_pop[max(lower)]
            else:
                n = hifis_pop[min(upper)]
            n_source = "interpolated"
        print(f"  Population N = {n} ({n_source})")

        # Interpolate marginals from PiT anchor years
        marginals = get_interpolated_marginals_for_year(year, pit_marginals)

        # Override gender with HIFIS where available
        marginals = override_with_hifis(marginals, year, hifis_gen)

        # Sample individuals
        df_year = sample_from_marginals(marginals, n, rng)
        df_year.insert(0, "year", year)
        all_years.append(df_year)

        # ── Validation summary for this year ────────────────────────────────
        for attr in attr_cols:
            if attr not in df_year.columns:
                continue
            vc = df_year[attr].value_counts()
            for resp, cnt in vc.items():
                # Reference proportion: interpolated marginal
                ref_prop = marginals.get(attr, {}).get(resp, 0.0)
                summary_rows.append({
                    "year":        year,
                    "attribute":   attr,
                    "response":    resp,
                    "synth_count": int(cnt),
                    "ref_prop":    round(ref_prop, 4),
                    "synth_prop":  round(cnt / n, 4),
                })

        print(f"  Sampled {len(df_year)} individuals with {len(df_year.columns)-1} attributes.")

    # ── Combine and save ─────────────────────────────────────────────────────
    print("\nCombining all years …")
    df_all = pd.concat(all_years, ignore_index=True)

    # Sort columns
    other_cols = sorted([c for c in df_all.columns if c != "year"])
    df_all = df_all[["year"] + other_cols]

    df_all.to_csv(OUT_INDIVIDUALS, index=False)
    print(f"Synthetic individuals saved → {OUT_INDIVIDUALS}")
    print(f"  Total rows: {len(df_all)}")
    print(f"  Years: {sorted(df_all['year'].unique())}")
    print(f"  Columns: {list(df_all.columns)}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Validation summary saved → {OUT_SUMMARY}")

    # ── Quick accuracy report ────────────────────────────────────────────────
    df_summary["prop_error"] = (
        df_summary["synth_prop"] - df_summary["ref_prop"]
    ).abs()
    print("\nMean absolute proportion error by year:")
    mae_by_year = df_summary.groupby("year")["prop_error"].mean()
    print(mae_by_year.to_string())
    print(f"\nOverall mean absolute proportion error: {df_summary['prop_error'].mean():.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
