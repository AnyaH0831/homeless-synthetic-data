"""
Toronto SNA Synthetic Data Generator + Homelessness Forecasting Model

- Loads SNA workbooks (API or local source_data)
- Interpolates 2013–2026 aggregates
- Calibrates with Toronto shelter flow (toronto-shelter-system-flow.csv)
- Generates synthetic individuals + region-year features
- Trains simple forecasting models
"""

import argparse
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from generation.synthetic_generation import build_region_year, generate_individuals
from shelter_flow import calibrate_agg_df, enrich_region_year, load_flow
from training.forecast_training import train_and_forecast
from visualization.results_visualization import create_plots

warnings.filterwarnings("ignore")
np.random.seed(42)

CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
OUTPUT_DIR = Path("synthetic_data")
FLOW_LOCAL_DEFAULT = Path("source_data/toronto-shelter-system-flow.csv")

PACKAGE_IDS = {
    2013: "2013-street-needs-assessment-results",
    2018: "2018-street-needs-assessment-results",
    2021: "2021-street-needs-assessment-results",
}

LOCAL_FILES = {
    2013: "source_data/2013-street-needs-assessment-results.xlsx",
    2018: "source_data/2018-street-needs-assessment-results.xlsx",
    2021: "source_data/2021-street-needs-assessment-results.xlsx",
}

ROW_MAP = {
    "TotalSurveys": "total_surveyed",
    "2_AgeAverage": "age_avg",
    "2_AgeCount": "age_count",
    "4_YearHomelessAverage": "years_homeless_avg",
    "26_GenderIdentityCount": "gender_count",
    "26_GenderIdentityMale": "n_male",
    "26_GenderIdentityFemale": "n_female",
    "26_GenderIdentityTransMale": "n_trans_male",
    "26_GenderIdentityTransFemale": "n_trans_female",
    "26_GenderIdentityTwoSpirit": "n_two_spirit",
    "26_GenderIdentityNonBinary": "n_nonbinary",
    "18_IndigenousCount": "indigenous_count",
    "18_FirstNations": "n_first_nations",
    "18_Metis": "n_metis",
    "18_Inuit": "n_inuit",
    "20_RaceEthnicityCount": "race_count",
    "20_RaceEthnicityBlackCanadianAmerican": "n_black_cdn",
    "20_RaceEthnicityBlackAfrican": "n_black_african",
    "20_RaceEthnicityBlackAfroCaribbean": "n_black_caribbean",
    "20_RaceEthnicityWhite": "n_white",
    "23_HealthChallengesCount": "health_count",
    "23_MentalHealthIssueYes": "n_mental_health",
    "23_SubstanceUseIssueYes": "n_substance_use",
    "23_PhysicalLimitationYes": "n_physical_health",
    "9_OverNightLocationCount": "overnight_count",
    "9_OvernightLocationOutdoorsYes": "n_outdoor",
    "9_EmergencyShelterYes": "n_emergency_shelter",
    "11_ImmigrantStatusCount": "immigrant_count",
    "11_Immigrant": "n_immigrant",
    "11_Refugee": "n_refugee",
    "11_RefugeeClaimant": "n_refugee_claimant",
    "28_LGBTQS2Count": "lgbtq_count",
    "28_Yes": "n_lgbtq",
    "22_FosterCount": "foster_count",
    "22_Yes": "n_foster",
    "6_HousingLossCount": "housing_loss_count",
    "6_HousingLossNotEnoughIncome": "n_housing_loss_income",
    "6_HousingLossMentalHealth": "n_housing_loss_mental",
    "6_HousingLossSubstanceUse": "n_housing_loss_substance",
    "29_IncomeCount": "income_count",
    "29_IncomeNoIncome": "n_no_income",
    "32_ServiceUseCount": "service_count",
    "32_PrisonOrJailYes": "n_incarceration",
}

RATIO_MAP = {
    ("n_male", "gender_count"): "pct_male",
    ("n_female", "gender_count"): "pct_female",
    ("n_trans_male", "gender_count"): "pct_trans_male",
    ("n_trans_female", "gender_count"): "pct_trans_female",
    ("n_two_spirit", "gender_count"): "pct_two_spirit",
    ("n_nonbinary", "gender_count"): "pct_nonbinary",
    ("n_first_nations", "indigenous_count"): "pct_first_nations",
    ("n_metis", "indigenous_count"): "pct_metis",
    ("n_inuit", "indigenous_count"): "pct_inuit",
    ("n_white", "race_count"): "pct_white",
    ("n_mental_health", "health_count"): "pct_mental_health",
    ("n_substance_use", "health_count"): "pct_substance_use",
    ("n_physical_health", "health_count"): "pct_physical_health",
    ("n_outdoor", "overnight_count"): "pct_outdoor_sleeping",
    ("n_emergency_shelter", "overnight_count"): "pct_emergency_shelter",
    ("n_immigrant", "immigrant_count"): "pct_immigrant",
    ("n_refugee", "immigrant_count"): "pct_refugee",
    ("n_refugee_claimant", "immigrant_count"): "pct_refugee_claimant",
    ("n_lgbtq", "lgbtq_count"): "pct_lgbtq",
    ("n_foster", "foster_count"): "pct_foster_care_history",
    ("n_housing_loss_income", "housing_loss_count"): "pct_housing_loss_income",
    ("n_housing_loss_mental", "housing_loss_count"): "pct_housing_loss_health",
    ("n_housing_loss_substance", "housing_loss_count"): "pct_housing_loss_substance",
    ("n_no_income", "income_count"): "pct_no_income",
    ("n_incarceration", "service_count"): "pct_incarceration_history",
}

PROPORTION_COLS = [v for v in RATIO_MAP.values()]


# ---- utilities ----

def _normalize_text(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_normalize_text)


def _pick_value_column(columns) -> str:
    normalized = {_normalize_text(col): col for col in columns}
    for candidate in ("totalaverage", "total"):
        if candidate in normalized:
            return normalized[candidate]
    for col in reversed(list(columns)):
        if str(col).strip() and not str(col).startswith("Unnamed"):
            return col
    return columns[-1]


def _find_value(df: pd.DataFrame, row_terms=(), question_terms=(), response_terms=(), sum_matches=False) -> float:
    mask = pd.Series(True, index=df.index)
    if row_terms:
        row_norm = _normalize_series(df["row_name"])
        for term in row_terms:
            mask &= row_norm.str.contains(_normalize_text(term), na=False)
    if question_terms and "question" in df.columns:
        q_norm = _normalize_series(df["question"])
        for term in question_terms:
            mask &= q_norm.str.contains(_normalize_text(term), na=False)
    if response_terms and "response" in df.columns:
        r_norm = _normalize_series(df["response"])
        for term in response_terms:
            mask &= r_norm.str.contains(_normalize_text(term), na=False)

    matches = df.loc[mask, "value"].fillna(0.0)
    if matches.empty:
        return 0.0
    return float(matches.sum() if sum_matches else matches.iloc[0])


def compute_derived(agg: dict) -> dict:
    rc = max(agg.get("race_count", 1), 1)
    agg["pct_black"] = float(np.clip((agg.get("n_black_cdn", 0) + agg.get("n_black_african", 0) + agg.get("n_black_caribbean", 0)) / rc, 0.0, 1.0))

    ic = max(agg.get("indigenous_count", 1), 1)
    agg["pct_indigenous"] = float(np.clip((agg.get("n_first_nations", 0) + agg.get("n_metis", 0) + agg.get("n_inuit", 0)) / ic, 0.0, 1.0))

    avg = max(agg.get("years_homeless_avg", 3.0), 0.1)
    agg["pct_chronic"] = float(np.clip(1 - np.exp(-avg / 3.5), 0.1, 0.9))

    agg["pct_other_race"] = float(np.clip(1 - agg.get("pct_black", 0) - agg.get("pct_white", 0) - agg.get("pct_indigenous", 0), 0.01, 0.99))
    agg["pct_trans_nonbinary"] = float(np.clip(agg.get("pct_trans_male", 0) + agg.get("pct_trans_female", 0) + agg.get("pct_two_spirit", 0) + agg.get("pct_nonbinary", 0), 0.01, 0.99))
    agg.setdefault("age_std", 14.0)
    return agg


# ---- load SNA ----

def fetch_xlsx_from_api(year: int) -> bytes:
    resp = requests.get(
        f"{CKAN_BASE}/api/3/action/package_show",
        params={"id": PACKAGE_IDS[year]},
        timeout=30,
    )
    resp.raise_for_status()
    resources = resp.json()["result"]["resources"]

    for res in resources:
        fmt = (res.get("format") or "").lower()
        url = res.get("url", "")
        if "xlsx" in fmt or url.endswith(".xlsx"):
            print(f"  [{year}] Downloading: {url}")
            dl = requests.get(url, timeout=60)
            dl.raise_for_status()
            return dl.content

    raise ValueError(f"No xlsx resource found for SNA {year}.")


def load_sna_xlsx(source) -> pd.DataFrame:
    buf = io.BytesIO(source) if isinstance(source, bytes) else source
    export = pd.read_excel(buf, sheet_name="Export")
    export.columns = [str(c).strip() if c is not None else "" for c in export.columns]

    row_col = export.columns[0]
    value_col = _pick_value_column(export.columns[1:])
    export = export[[row_col, value_col]].copy()
    export.columns = ["row_name", "value"]

    key_rows = pd.read_excel(buf, sheet_name="Key-Rows")
    key_rows.columns = [str(c).strip() if c is not None else "" for c in key_rows.columns]
    key_rows = key_rows.iloc[:, :5].copy()
    key_rows.columns = ["row_name", "question", "response", "meta_value", "notes"]

    df = key_rows.merge(export, on="row_name", how="left")
    df["row_name"] = df["row_name"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def extract_aggregates(df: pd.DataFrame) -> dict:
    lookup = df.assign(_row_norm=_normalize_series(df["row_name"]))
    lookup = lookup.set_index("_row_norm")["value"].to_dict()

    def from_rows(*labels: str) -> float:
        for label in labels:
            value = lookup.get(_normalize_text(label))
            if value is not None and not pd.isna(value):
                return float(value)
        return 0.0

    agg = {key: from_rows(sna_row) for sna_row, key in ROW_MAP.items()}

    # semantic fallbacks for older workbook naming
    agg["total_surveyed"] = max(agg.get("total_surveyed", 0.0), _find_value(df, question_terms=("total surveys completed",)))
    agg["age_avg"] = max(agg.get("age_avg", 0.0), _find_value(df, question_terms=("how old",)))
    agg["years_homeless_avg"] = max(agg.get("years_homeless_avg", 0.0), _find_value(df, question_terms=("how long", "homeless")))

    for (num_key, den_key), pct_key in RATIO_MAP.items():
        num = agg.get(num_key, 0.0)
        den = max(agg.get(den_key, 1.0), 1.0)
        agg[pct_key] = float(np.clip(num / den, 0.0, 1.0))

    return compute_derived(agg)


def load_all_years(use_local: bool = False) -> dict:
    results = {}
    for year in PACKAGE_IDS:
        print(f"Loading SNA {year}...")
        raw = None

        if not use_local:
            try:
                raw = fetch_xlsx_from_api(year)
                print(f"  [{year}] Loaded from API.")
            except Exception as e:
                print(f"  [{year}] API failed ({e}), falling back to local file...")

        if raw is None:
            local_path = Path(LOCAL_FILES[year])
            if not local_path.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            raw = local_path.read_bytes()
            print(f"  [{year}] Loaded from local file: {local_path}")

        df_sheet = load_sna_xlsx(raw)
        results[year] = extract_aggregates(df_sheet)
        results[year]["year"] = year

    return results


def interpolate_aggregates(observed: dict, all_years: list) -> pd.DataFrame:
    obs_df = pd.DataFrame(observed).T.astype(float)
    obs_df.index = obs_df.index.astype(int)
    obs_df.index.name = "year"

    interp = obs_df.reindex(pd.Index(all_years, name="year"))
    interp = interp.interpolate(method="index", limit_direction="both")

    obs_years = sorted(observed.keys())
    first_yr, last_yr = obs_years[0], obs_years[-1]
    span = last_yr - first_yr

    for col in interp.columns:
        if col == "year":
            continue
        v_first = observed[first_yr].get(col)
        v_last = observed[last_yr].get(col)
        if v_first is None or v_last is None:
            continue
        slope = (v_last - v_first) / span if span else 0
        for yr in all_years:
            if yr > last_yr:
                val = v_last + slope * (yr - last_yr)
                if col in PROPORTION_COLS or col.startswith("pct_"):
                    val = float(np.clip(val, 0.01, 0.99))
                interp.loc[yr, col] = val

    interp["total_surveyed"] = interp["total_surveyed"].round().astype(int)
    return interp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Load SNA from local xlsx files")
    parser.add_argument("--skip-model", action="store_true", help="Only generate synthetic data")
    parser.add_argument("--flow-local", default=str(FLOW_LOCAL_DEFAULT), help="Local shelter flow CSV path")
    args = parser.parse_args()

    all_years = list(range(2013, 2027))
    forecast_years = [2024, 2025, 2026]

    observed = load_all_years(use_local=args.local)
    agg_df = interpolate_aggregates(observed, all_years)

    flow_path = Path(args.flow_local)
    flow_yearly = load_flow(local_path=flow_path) if flow_path.exists() else load_flow()
    agg_df = calibrate_agg_df(agg_df, flow_yearly)

    print("\nInterpolated + calibrated aggregates (key columns):")
    print(agg_df[["total_surveyed", "pct_mental_health", "pct_outdoor_sleeping", "pct_chronic"]].round(3))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating synthetic individuals using yearly total_surveyed counts...")
    df_individuals = generate_individuals(agg_df)
    individuals_path = OUTPUT_DIR / "synthetic_individuals.csv"
    df_individuals.to_csv(individuals_path, index=False)
    print(f"Saved {individuals_path}  ({len(df_individuals):,} rows)")

    ry = build_region_year(df_individuals, agg_df)
    ry = enrich_region_year(ry, flow_yearly)
    region_path = OUTPUT_DIR / "region_year_features.csv"
    ry.to_csv(region_path, index=False)
    print(f"Saved {region_path}")

    plots_dir = OUTPUT_DIR / "plots"

    if args.skip_model:
        create_plots(agg_df=agg_df, region_year_df=ry, output_dir=plots_dir)
        print(f"Saved plots to {plots_dir}")
        print("\n--skip-model set, done.")
        return

    results = train_and_forecast(ry, sorted(observed.keys()), forecast_years)
    forecast_path = OUTPUT_DIR / "forecast_results.csv"
    results.to_csv(forecast_path, index=False)
    create_plots(agg_df=agg_df, region_year_df=ry, output_dir=plots_dir, forecast_results=results)
    print(f"Saved plots to {plots_dir}")
    print(f"\nSaved {forecast_path}")


if __name__ == "__main__":
    main()
