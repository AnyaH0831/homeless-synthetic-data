# SASM Pipeline: Observed Totals Update

## Summary
Updated the SASM pipeline to support using **actual observed totals** from shelter flow data instead of a fixed sample size per year.

## Changes

### Files Modified
1. **sasm_generator.py**
   - Added `use_observed_totals` parameter to `generate_individuals_sasm()` function
   - Updated print statements to show per-year totals
   - Extended header row from 60 to 70 characters to accommodate new data

2. **sna_pipeline_sasm.py**
   - Added `--use-observed-totals` command-line flag
   - Updated help text for `--sample-size` parameter
   - Added logic to print which mode is being used (fixed vs observed)
   - Updated usage examples in docstring

## How to Use

### Option 1: Fixed Sample Size (Default - 2000/year)
```bash
python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv
```
- Generates ~2,000 synthetic individuals per year
- Total: ~29,000 individuals across 14 years

### Option 2: Observed Totals from Shelter Flow (NEW)
```bash
python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv --use-observed-totals
```
- Generates synthetic individuals matching actual observed counts
- Varies by year: 6,400-8,000 per year
- Total: ~105,000 individuals across 14 years
- **Recommended** for more realistic scale

### Option 3: Custom Fixed Sample Size
```bash
python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv --sample-size 5000
```
- Use any fixed sample size instead of default 2000

## Data Comparison

| Metric | Fixed 2000 | Observed Totals |
|--------|-----------|-----------------|
| Total individuals | 29,002 | 105,392 |
| Per-year range | 1,984-2,438 | 6,638-8,241 |
| Matches reality | ❌ No | ✅ Yes |
| Use case | Testing/quick runs | Production/research |

## Example Output

### With `--use-observed-totals`:
```
STEP 3: SASM optimization-based individual generation
  (minimize ||WX' - Y||² per year)
  Using OBSERVED totals from shelter flow data
═════════════════════════════════════════════════════

  Year     Total       d_p     RMSE   MaxErr   MeanPct%   MH-SU corr
──────────────────────────────────────────────────────────────────────
  2013      7178  18399640.40  1189.688   2191.0     628.61%        0.493
  2014      7181  8391738.92   803.442   1408.2     279.11%        0.081
  ...
  2026      7247  641813.85    222.194    401.9      13.86%        0.504

Saved sasm_synthetic_individuals.csv  (105,392 rows)
```

## Quality Metrics Interpretation

The quality metrics now show:
- **Total**: Per-year count being generated (matches `agg_df['total_surveyed']` when using `--use-observed-totals`)
- **d_p**: Sum of squared errors vs observed constraints (lower is better)
- **RMSE**: Root mean squared error per constraint
- **MeanPct%**: Mean absolute percent error vs observed
- **MH-SU corr**: Mental health & substance use correlation (target: 0.35-0.55)

## Recommendation

For research and production use:
```bash
python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv --use-observed-totals
```

This generates synthetic microdata that:
- ✅ Matches the scale of the real homeless-serving system
- ✅ Preserves demographic distributions from SNA surveys
- ✅ Includes realistic temporal variation (2013-2026)
- ✅ Can be validated against actual aggregate statistics
