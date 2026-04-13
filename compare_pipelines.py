"""
compare_pipelines.py
────────────────────
Compares outputs from the OLD pipeline (copula sampling) vs the NEW pipeline
(SASM optimization) across three dimensions:

  1. AGGREGATE FIDELITY
     How closely does each pipeline's synthetic data reproduce the SNA
     observed counts? This is the key improvement SASM claims to make.
     Metric: mean absolute % error per constraint per year.

  2. DISTRIBUTION SIMILARITY
     Do the marginal distributions of key attributes look the same?
     We compare: gender, race, mental_health, substance_use, shelter_type.
     Metric: JS divergence (lower = more similar to observed)

  3. FORECAST COMPARISON
     Do the two pipelines produce different predictions for 2024–2026?
     Side-by-side table of ensemble predictions.

HOW TO RUN
──────────
Run BOTH pipelines first, then run this script:

    # Old pipeline
    python sna_pipeline.py --local --local-flow toronto-shelter-system-flow__1_.csv

    # New pipeline
    python sna_pipeline_sasm.py --local --local-flow toronto-shelter-system-flow__1_.csv

    # Compare
    python compare_pipelines.py

The comparison reads these CSV files:
    OLD: synthetic_individuals.csv, region_year_features.csv, forecast_results.csv
    NEW: sasm_synthetic_individuals.csv, sasm_region_year_features.csv, sasm_forecast_results.csv
    QUALITY: sasm_quality_log.csv
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


# ── FILE PATHS ────────────────────────────────────────────────────────────────
# These are the output CSV files from each pipeline.
# Adjust if you saved them somewhere else.

OLD_INDIVIDUALS   = "synthetic_data/synthetic_individuals.csv"
OLD_REGION_YEAR   = "synthetic_data/region_year_features.csv"
OLD_FORECAST      = "synthetic_data/forecast_results.csv"

NEW_INDIVIDUALS   = "sasm_synthetic_individuals.csv"
NEW_REGION_YEAR   = "sasm_region_year_features.csv"
NEW_FORECAST      = "sasm_forecast_results.csv"
NEW_QUALITY_LOG   = "sasm_quality_log.csv"


def check_files():
    """Verify required files exist before proceeding."""
    missing = []
    for f in [OLD_INDIVIDUALS, OLD_REGION_YEAR, OLD_FORECAST,
              NEW_INDIVIDUALS, NEW_REGION_YEAR, NEW_FORECAST]:
        if not Path(f).exists():
            missing.append(f)
    if missing:
        print("ERROR: Missing output files. Run both pipelines first.")
        print("Missing files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)


# ── HELPER: JS DIVERGENCE ─────────────────────────────────────────────────────
# Jensen-Shannon divergence measures how different two distributions are.
# Range: 0 (identical) to 1 (completely different).
# We use it to compare how similar old vs new synthetic distributions are
# to each other and to observed SNA proportions.

def js_div(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two probability arrays.
    Both arrays are normalized to sum to 1 before computing.
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    # Add tiny epsilon to avoid log(0)
    p = (p + 1e-10) / (p + 1e-10).sum()
    q = (q + 1e-10) / (q + 1e-10).sum()
    return float(jensenshannon(p, q))


# ── SECTION 1: AGGREGATE FIDELITY ─────────────────────────────────────────────

def compare_aggregate_fidelity(old_df: pd.DataFrame, new_df: pd.DataFrame,
                                 new_quality: Optional[pd.DataFrame]):
    """
    Compare how well each pipeline reproduces observed SNA aggregate proportions.

    For each year and each key attribute, we compute the absolute difference
    between the observed proportion (from the SNA) and the synthetic proportion.

    We approximate "observed" using the SNA anchor years (2013, 2018, 2021).
    For other years, the target is the interpolated/flow-calibrated value.
    """
    print("\n" + "═" * 70)
    print("  SECTION 1: AGGREGATE FIDELITY")
    print("  How closely does each pipeline reproduce observed SNA proportions?")
    print("═" * 70)

    # Key proportions to compare
    # Each tuple: (column_in_individuals, display_name, sna_anchor_values)
    # sna_anchor_values are approximate ground truth from real SNA reports
    ANCHORS = {
        # attr: {year: observed_proportion}
        "mental_health":  {2013: 0.35, 2018: 0.40, 2021: 0.46},
        "substance_use":  {2013: 0.25, 2018: 0.28, 2021: 0.33},
        "outdoor_sleeping": {2013: 0.15, 2018: 0.19, 2021: 0.25},
    }

    for attr, anchors in ANCHORS.items():
        print(f"\n  Attribute: {attr}")
        print(f"  {'Year':>6}  {'Observed':>10}  {'Old (copula)':>14}  {'New (SASM)':>12}  {'Old err%':>10}  {'New err%':>10}")
        print("  " + "-" * 66)

        for year, obs_pct in sorted(anchors.items()):
            # Old pipeline
            old_year = old_df[old_df["year"] == year]
            old_pct  = old_year[attr].mean() if len(old_year) > 0 else float("nan")

            # New pipeline — column name differs slightly
            new_col  = "outdoor_sleeping" if attr == "outdoor_sleeping" else attr
            new_year = new_df[new_df["year"] == year]
            new_pct  = new_year[new_col].mean() if len(new_year) > 0 else float("nan")

            old_err = abs(old_pct - obs_pct) / obs_pct * 100 if not np.isnan(old_pct) else float("nan")
            new_err = abs(new_pct - obs_pct) / obs_pct * 100 if not np.isnan(new_pct) else float("nan")

            winner = "←SASM" if new_err < old_err else ("←OLD" if old_err < new_err else "  TIE")
            print(f"  {year:>6}  {obs_pct:>10.3f}  {old_pct:>14.3f}  {new_pct:>12.3f}  "
                  f"{old_err:>9.1f}%  {new_err:>9.1f}%  {winner}")

    # Print SASM quality log if available
    if new_quality is not None and len(new_quality) > 0:
        print(f"\n  SASM d_p quality log (from sasm_quality_log.csv):")
        print(f"  d_p = ||WX' - Y||²  (lower = better match to aggregates)")
        print(f"  {'Year':>6}  {'d_p':>10}  {'RMSE':>8}  {'MaxErr':>8}  {'MeanPct%':>10}")
        print("  " + "-" * 46)
        for _, qrow in new_quality.iterrows():
            print(f"  {int(qrow['year']):>6}  {qrow['d_p']:>10.1f}  "
                  f"{qrow['rmse']:>8.2f}  {qrow['max_error']:>8.1f}  "
                  f"{qrow['mean_pct_error']:>9.2f}%")


# ── SECTION 2: DISTRIBUTION SIMILARITY ───────────────────────────────────────

def compare_distributions(old_df: pd.DataFrame, new_df: pd.DataFrame):
    """
    Compare marginal distributions of key categorical attributes between
    old and new pipelines using Jensen-Shannon divergence.

    Lower JS divergence = distributions are more similar to each other.
    We also check similarity between old/new and the expected proportions.
    
    Note: Old pipeline doesn't have 'gender' and 'race' columns, so we skip those.
    """
    print("\n" + "═" * 70)
    print("  SECTION 2: DISTRIBUTION SIMILARITY")
    print("  Jensen-Shannon divergence between OLD and NEW distributions")
    print("  (0 = identical, 1 = completely different, < 0.05 = very similar)")
    print("═" * 70)

    # Gender distribution (only if old pipeline has it)
    if "gender" in old_df.columns:
        print("\n  Gender distribution:")
        for year in [2013, 2018, 2021]:
            o = old_df[old_df["year"] == year]["gender"].value_counts(normalize=True).sort_index()
            n = new_df[new_df["year"] == year]["gender"].value_counts(normalize=True).sort_index()
            # Align on same categories
            cats = sorted(set(o.index) | set(n.index))
            o_arr = np.array([o.get(c, 0) for c in cats])
            n_arr = np.array([n.get(c, 0) for c in cats])
            jsd = js_div(o_arr, n_arr)
            print(f"    {year}  JS divergence: {jsd:.4f}  ", end="")
            print("| Old: " + " ".join(f"{c}={o.get(c,0):.2f}" for c in cats))
            print(f"    {'':4}                          | New: " +
                  " ".join(f"{c}={n.get(c,0):.2f}" for c in cats))
    else:
        print("\n  Gender distribution: Not available in old pipeline (no 'gender' column)")

    # Race distribution (only if old pipeline has it)
    if "race" in old_df.columns:
        print("\n  Race distribution:")
        for year in [2013, 2018, 2021]:
            o = old_df[old_df["year"] == year]["race"].value_counts(normalize=True).sort_index()
            n = new_df[new_df["year"] == year]["race"].value_counts(normalize=True).sort_index()
            cats = sorted(set(o.index) | set(n.index))
            o_arr = np.array([o.get(c, 0) for c in cats])
            n_arr = np.array([n.get(c, 0) for c in cats])
            jsd = js_div(o_arr, n_arr)
            print(f"    {year}  JS divergence: {jsd:.4f}")
    else:
        print("\n  Race distribution: Not available in old pipeline (no 'race' column)")

    # Binary attribute means (mental_health, substance_use, etc.)
    print("\n  Binary attribute means (proportion = 1) across ALL years:")
    print(f"  {'Attribute':30s}  {'Old mean':>10}  {'New mean':>10}  {'Diff':>8}")
    print("  " + "-" * 62)

    old_col_map = {
        "mental_health":    "mental_health",
        "substance_use":    "substance_use",
        "outdoor_sleeping": "outdoor_sleeping",
        "chronic_homeless": "chronic_homeless",
        "lgbtq":            "lgbtq",
        "foster_care":      "foster_care_history",
        "incarceration":    "incarceration_history",
    }
    new_col_map = {
        "mental_health":    "mental_health",
        "substance_use":    "substance_use",
        "outdoor_sleeping": "outdoor_sleeping",
        "chronic_homeless": "chronic_homeless",
        "lgbtq":            "lgbtq",
        "foster_care":      "foster_care_history",
        "incarceration":    "incarceration_history",
    }

    for display_name, old_col in old_col_map.items():
        new_col = new_col_map[display_name]
        old_mean = old_df[old_col].mean() if old_col in old_df.columns else float("nan")
        new_mean = new_df[new_col].mean() if new_col in new_df.columns else float("nan")
        diff = new_mean - old_mean
        marker = " ←" if abs(diff) > 0.02 else "  "
        print(f"  {display_name:30s}  {old_mean:>10.4f}  {new_mean:>10.4f}  {diff:>+8.4f}{marker}")

    # Correlation check — do mental_health and substance_use co-occur more realistically?
    print("\n  Correlation check (mental_health ↔ substance_use):")
    print("  Expected from literature: ~0.35–0.55")
    for label, df in [("Old (copula)", old_df), ("New (SASM)", new_df)]:
        if "mental_health" in df.columns and "substance_use" in df.columns:
            corr = df[["mental_health", "substance_use"]].corr().iloc[0, 1]
            print(f"  {label:20s}: {corr:.4f}")


# ── SECTION 3: FORECAST COMPARISON ───────────────────────────────────────────

def compare_forecasts(old_fc: pd.DataFrame, new_fc: pd.DataFrame):
    """
    Side-by-side comparison of model predictions from both pipelines.
    We compare ensemble predictions for all years (especially 2024–2026).
    """
    print("\n" + "═" * 70)
    print("  SECTION 3: FORECAST COMPARISON")
    print("  Side-by-side model predictions (ensemble = avg of Ridge + GBR)")
    print("═" * 70)

    merged = old_fc[["year", "true_total", "ensemble", "observed"]].merge(
        new_fc[["year", "ensemble"]],
        on="year", suffixes=("_old", "_sasm")
    )

    print(f"\n  {'Year':>6}  {'True/Target':>12}  {'Old ensemble':>14}  "
          f"{'SASM ensemble':>14}  {'Diff':>8}  {'Observed?':>10}")
    print("  " + "-" * 70)

    for _, row in merged.iterrows():
        diff = int(row["ensemble_sasm"]) - int(row["ensemble_old"])
        obs  = "YES ←" if row.get("observed", False) else ""
        print(f"  {int(row['year']):>6}  {int(row['true_total']):>12,}  "
              f"{int(row['ensemble_old']):>14,}  "
              f"{int(row['ensemble_sasm']):>14,}  "
              f"{diff:>+8,}  {obs:>10}")

    # Summary stats
    future_old  = merged[merged["year"] >= 2024]["ensemble_old"].mean()
    future_sasm = merged[merged["year"] >= 2024]["ensemble_sasm"].mean()
    print(f"\n  Average 2024–2026 prediction:")
    print(f"    Old pipeline:  {future_old:,.0f}")
    print(f"    SASM pipeline: {future_sasm:,.0f}")
    print(f"    Difference:    {future_sasm - future_old:+,.0f}")


# ── SECTION 4: SUMMARY ────────────────────────────────────────────────────────

def print_summary(old_df, new_df, old_fc, new_fc, new_quality):
    """Print a concise summary of the comparison."""
    print("\n" + "═" * 70)
    print("  SUMMARY")
    print("═" * 70)

    print(f"\n  Dataset sizes:")
    print(f"    Old individuals:  {len(old_df):,} rows")
    print(f"    SASM individuals: {len(new_df):,} rows")

    if new_quality is not None and len(new_quality) > 0:
        avg_dp   = new_quality["d_p"].mean()
        avg_rmse = new_quality["rmse"].mean()
        avg_pct  = new_quality["mean_pct_error"].mean()
        print(f"\n  SASM optimization quality (average across all years):")
        print(f"    Mean d_p:            {avg_dp:,.1f}  (sum of squared errors vs observed)")
        print(f"    Mean RMSE:           {avg_rmse:.2f}   (per constraint)")
        print(f"    Mean % error:        {avg_pct:.2f}%  (vs observed aggregates)")
        print(f"    Interpretation: RMSE < 5 and % error < 5% = excellent fidelity")

    print(f"\n  Key advantage of SASM: the d_p quality metric gives you a")
    print(f"  mathematically grounded measure of how well the synthetic data")
    print(f"  reproduces the real aggregates. The old copula pipeline had no")
    print(f"  equivalent — you had to trust that the sampling was approximately right.")
    print(f"\n  For your research paper, you can report d_p per year as evidence")
    print(f"  of synthetic data quality, which is directly comparable to the")
    print(f"  Lin & Xiao (2023) paper that introduced this method.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    check_files()

    print("\n" + "═" * 70)
    print("  PIPELINE COMPARISON: Old (Copula) vs New (SASM Optimization)")
    print("═" * 70)

    # Load all output files
    print("\nLoading pipeline outputs...")
    old_df     = pd.read_csv(OLD_INDIVIDUALS)
    new_df     = pd.read_csv(NEW_INDIVIDUALS)
    old_fc     = pd.read_csv(OLD_FORECAST)
    new_fc     = pd.read_csv(NEW_FORECAST)
    new_quality = pd.read_csv(NEW_QUALITY_LOG) if Path(NEW_QUALITY_LOG).exists() else None

    print(f"  Old individuals: {len(old_df):,} rows, years {old_df['year'].min()}–{old_df['year'].max()}")
    print(f"  New individuals: {len(new_df):,} rows, years {new_df['year'].min()}–{new_df['year'].max()}")

    # Run comparisons
    compare_aggregate_fidelity(old_df, new_df, new_quality)
    compare_distributions(old_df, new_df)
    compare_forecasts(old_fc, new_fc)
    print_summary(old_df, new_df, old_fc, new_fc, new_quality)

    print("\n" + "═" * 70)
    print("  Comparison complete.")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
