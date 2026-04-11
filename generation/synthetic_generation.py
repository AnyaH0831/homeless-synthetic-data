import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm as sp_norm

BINARY_FEATURES = [
    "mental_health", "substance_use", "physical_health",
    "outdoor_sleeping", "chronic_homeless",
    "lgbtq", "indigenous", "immigrant",
    "foster_care_history", "incarceration_history",
    "no_income", "housing_loss_income", "housing_loss_health",
]

CORR_MATRIX = np.array([
    [1.00, 0.55, 0.40, 0.35, 0.38, 0.20, 0.15, 0.05, 0.25, 0.20, 0.18, 0.12, 0.30],
    [0.55, 1.00, 0.35, 0.42, 0.40, 0.18, 0.18, 0.04, 0.28, 0.30, 0.20, 0.10, 0.25],
    [0.40, 0.35, 1.00, 0.25, 0.30, 0.10, 0.12, 0.08, 0.15, 0.15, 0.15, 0.10, 0.35],
    [0.35, 0.42, 0.25, 1.00, 0.55, 0.22, 0.20, 0.06, 0.20, 0.25, 0.25, 0.08, 0.12],
    [0.38, 0.40, 0.30, 0.55, 1.00, 0.18, 0.22, 0.05, 0.22, 0.28, 0.22, 0.10, 0.15],
    [0.20, 0.18, 0.10, 0.22, 0.18, 1.00, 0.10, 0.08, 0.30, 0.10, 0.12, 0.05, 0.08],
    [0.15, 0.18, 0.12, 0.20, 0.22, 0.10, 1.00, 0.05, 0.25, 0.22, 0.18, 0.08, 0.10],
    [0.05, 0.04, 0.08, 0.06, 0.05, 0.08, 0.05, 1.00, 0.05, 0.04, 0.10, 0.15, 0.05],
    [0.25, 0.28, 0.15, 0.20, 0.22, 0.30, 0.25, 0.05, 1.00, 0.30, 0.20, 0.08, 0.12],
    [0.20, 0.30, 0.15, 0.25, 0.28, 0.10, 0.22, 0.04, 0.30, 1.00, 0.22, 0.10, 0.12],
    [0.18, 0.20, 0.15, 0.25, 0.22, 0.12, 0.18, 0.10, 0.20, 0.22, 1.00, 0.15, 0.10],
    [0.12, 0.10, 0.10, 0.08, 0.10, 0.05, 0.08, 0.15, 0.08, 0.10, 0.15, 1.00, 0.15],
    [0.30, 0.25, 0.35, 0.12, 0.15, 0.08, 0.10, 0.05, 0.12, 0.12, 0.10, 0.15, 1.00],
])

CHOL = cholesky(CORR_MATRIX, lower=True)
DEFAULT_SAMPLE_SIZE_PER_YEAR = 2000


def sample_correlated_binaries(n: int, probs: dict) -> pd.DataFrame:
    z_corr = np.random.randn(n, len(BINARY_FEATURES)) @ CHOL.T
    u = sp_norm.cdf(z_corr)
    return pd.DataFrame({feat: (u[:, i] < probs.get(feat, 0.2)).astype(int) for i, feat in enumerate(BINARY_FEATURES)})


def sample_years_homeless(n: int, avg: float, pct_chronic: float) -> np.ndarray:
    is_chronic = np.random.binomial(1, pct_chronic, n).astype(bool)
    yrs = np.where(is_chronic, np.random.pareto(1.5, n) * avg + 1.0, np.random.exponential(max(avg * 0.5, 0.5), n))
    return np.clip(yrs, 0.05, 30.0)


def generate_individuals(
    agg_df: pd.DataFrame,
    sample_size_per_year: int | None = None,
    use_observed_totals: bool = True,
) -> pd.DataFrame:
    records = []
    for year in agg_df.index:
        row = agg_df.loc[year]
        if use_observed_totals:
            n = int(max(round(float(row.get("total_surveyed", 0))), 1))
        else:
            n = int(sample_size_per_year or DEFAULT_SAMPLE_SIZE_PER_YEAR)

        age = np.clip(np.random.normal(row["age_avg"], row.get("age_std", 14.0), n), 15, 85)
        years_homeless = sample_years_homeless(n, row["years_homeless_avg"], row["pct_chronic"])

        probs = {
            "mental_health": row.get("pct_mental_health", 0.2),
            "substance_use": row.get("pct_substance_use", 0.2),
            "physical_health": row.get("pct_physical_health", 0.2),
            "outdoor_sleeping": row.get("pct_outdoor_sleeping", 0.2),
            "chronic_homeless": row.get("pct_chronic", 0.2),
            "lgbtq": row.get("pct_lgbtq", 0.2),
            "indigenous": row.get("pct_indigenous", 0.2),
            "immigrant": row.get("pct_immigrant", 0.2),
            "foster_care_history": row.get("pct_foster_care_history", 0.2),
            "incarceration_history": row.get("pct_incarceration_history", 0.2),
            "no_income": row.get("pct_no_income", 0.2),
            "housing_loss_income": row.get("pct_housing_loss_income", 0.2),
            "housing_loss_health": row.get("pct_housing_loss_health", 0.2),
        }
        binary_df = sample_correlated_binaries(n, probs)

        df_year = pd.DataFrame({
            "year": year,
            "age": age.round(1),
            "years_homeless": years_homeless.round(2),
        })
        df_year = pd.concat([df_year, binary_df], axis=1)
        df_year["youth"] = (df_year["age"] < 25).astype(int)
        records.append(df_year)

    return pd.concat(records, ignore_index=True)


def build_region_year(df_ind: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    ry = df_ind.groupby("year").agg(
        age_avg=("age", "mean"),
        years_homeless_avg=("years_homeless", "mean"),
        pct_mental_health=("mental_health", "mean"),
        pct_substance_use=("substance_use", "mean"),
        pct_physical_health=("physical_health", "mean"),
        pct_outdoor_sleeping=("outdoor_sleeping", "mean"),
        pct_chronic=("chronic_homeless", "mean"),
        pct_lgbtq=("lgbtq", "mean"),
        pct_indigenous=("indigenous", "mean"),
        pct_immigrant=("immigrant", "mean"),
        pct_youth=("youth", "mean"),
        pct_no_income=("no_income", "mean"),
        pct_foster_care=("foster_care_history", "mean"),
        pct_incarceration=("incarceration_history", "mean"),
        pct_housing_loss_income=("housing_loss_income", "mean"),
    ).reset_index()

    total_lookup = agg_df[["total_surveyed"]].reset_index().rename(columns={"total_surveyed": "true_total"})
    ry = ry.merge(total_lookup, on="year", how="left")
    ry["true_total"] = ry["true_total"].round().astype("Int64")
    return ry
