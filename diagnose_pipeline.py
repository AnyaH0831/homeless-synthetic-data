"""
diagnose_pipeline.py
────────────────────
Prints a diagnostic table of every value that feeds into the SASM optimizer
(the Y vector) for each year. Run this whenever d_p is unexpectedly high to
find the bad values before they reach the solver.

Usage:
    python diagnose_pipeline.py --local --local-flow source_data/toronto-shelter-system-flow__1_.csv

What to look for:
    - Any pct_* column that is 0.0 or very close to 0 → means it was overwritten with
      a missing/NaN value from shelter flow calibration. SASM will struggle here.
    - pct_male + pct_female + pct_trans_nonbinary should sum to ~1.0
    - pct_black + pct_white + pct_indigenous + pct_other_race should sum to ~1.0
    - total_surveyed should be 5000–12000 for Toronto (not 0, not millions)
    - pct_mental_health: 0.30–0.55
    - pct_substance_use: 0.20–0.40
    - pct_outdoor_sleeping: 0.10–0.35
    - pct_chronic: 0.25–0.55
"""

import argparse
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Import pipeline components ─────────────────────────────────────────────────
# We reuse the same loading code from sna_pipeline_sasm.py so the diagnosis
# reflects exactly what the optimizer will see.
from sna_pipeline_sasm import (
    load_all_years, interpolate_aggregates,
    compute_derived, PROPORTION_COLS
)
from shelter_flow import load_flow, calibrate_agg_df
from sasm_generator import extract_Y, CONSTRAINT_NAMES

# ── EXPECTED RANGES from literature ───────────────────────────────────────────
# Based on published homeless health studies and Toronto SNA reports.
# Used to flag values that are outside plausible ranges.
EXPECTED_RANGES = {
    "pct_male":              (0.55, 0.80),
    "pct_female":            (0.15, 0.40),
    "pct_trans_nonbinary":   (0.01, 0.10),
    "pct_black":             (0.20, 0.45),
    "pct_white":             (0.20, 0.50),
    "pct_indigenous":        (0.10, 0.30),
    "pct_other_race":        (0.05, 0.35),
    "pct_mental_health":     (0.25, 0.60),
    "pct_substance_use":     (0.15, 0.45),
    "pct_outdoor_sleeping":  (0.08, 0.40),
    "pct_chronic":           (0.20, 0.55),
    "pct_lgbtq":             (0.05, 0.25),
    "pct_indigenous_flow":   (0.08, 0.30),
    "total_surveyed":        (4000, 15000),
    "age_avg":               (30.0, 55.0),
    "years_homeless_avg":    (1.0, 10.0),
}


def flag(val, col):
    """Return ✓ if value is in expected range, ✗ if not."""
    if col not in EXPECTED_RANGES:
        return "  "
    lo, hi = EXPECTED_RANGES[col]
    return "✓ " if lo <= float(val) <= hi else "✗ "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--local-flow", type=str, default=None)
    args = parser.parse_args()

    ALL_YEARS = list(range(2013, 2027))

    print("=" * 80)
    print("DIAGNOSTIC: What does agg_df look like before SASM receives it?")
    print("=" * 80)

    # Load everything the same way the pipeline does
    observed    = load_all_years(use_local=args.local)
    agg_df      = interpolate_aggregates(observed, ALL_YEARS)
    flow_yearly = load_flow(local_path=args.local_flow)
    agg_df      = calibrate_agg_df(agg_df, flow_yearly)

    # ── Table 1: key proportions per year ──────────────────────────────────────
    check_cols = [
        "total_surveyed", "age_avg", "years_homeless_avg",
        "pct_male", "pct_female", "pct_trans_nonbinary",
        "pct_black", "pct_white", "pct_indigenous", "pct_other_race",
        "pct_mental_health", "pct_substance_use", "pct_outdoor_sleeping",
        "pct_chronic", "pct_lgbtq",
    ]

    print(f"\n{'Column':<28} ", end="")
    for yr in ALL_YEARS:
        print(f" {yr}", end="")
    print()
    print("-" * (28 + len(ALL_YEARS) * 7))

    for col in check_cols:
        if col not in agg_df.columns:
            print(f"  {col:<26}  [MISSING]")
            continue
        print(f"  {col:<26}", end="")
        for yr in ALL_YEARS:
            val = agg_df.loc[yr, col] if yr in agg_df.index else float("nan")
            f   = flag(val, col)
            if pd.isna(val):
                print(f"   NaN ", end="")
            elif col == "total_surveyed":
                print(f" {f}{int(val):5d}", end="")
            else:
                print(f" {f}{float(val):.2f}", end="")
        print()

    # ── Table 2: Y vector for each observed year ───────────────────────────────
    print(f"\n{'Y VECTOR PER YEAR (what the SASM optimizer sees, at sample_size=2000)':}")
    print("-" * 60)
    for yr in [2013, 2018, 2021, 2024]:
        if yr not in agg_df.index:
            continue
        row = agg_df.loc[yr]
        y   = extract_Y(row, 2000)
        print(f"\n  Year {yr}:")
        for i, name in enumerate(CONSTRAINT_NAMES):
            val = y[i]
            ok  = "✓" if val > 0 else "✗ ZERO — will cause optimizer error"
            print(f"    {name:<20} {val:>8.1f}  {ok}")

    # ── Table 3: Sum checks ────────────────────────────────────────────────────
    print(f"\n{'SUM CHECKS (should all be ~1.0)':}")
    print("-" * 40)
    for yr in [2013, 2018, 2021]:
        if yr not in agg_df.index:
            continue
        row  = agg_df.loc[yr]
        g_sum = (row.get("pct_male",0) + row.get("pct_female",0)
                 + row.get("pct_trans_nonbinary",0))
        r_sum = (row.get("pct_black",0) + row.get("pct_white",0)
                 + row.get("pct_indigenous",0) + row.get("pct_other_race",0))
        print(f"  {yr}  gender sum: {g_sum:.3f}  race sum: {r_sum:.3f}")

    print("\n✓ = in expected range   ✗ = outside expected range or zero")
    print("Fix: any ✗ on pct_mental_health/substance_use/outdoor_sleeping means")
    print("shelter_flow calibration overwrote them. Use shelter_flow_v2.py instead.")


if __name__ == "__main__":
    main()
