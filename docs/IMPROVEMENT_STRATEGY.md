# SASM Pipeline Improvement Strategy

## Current Issues Identified

### 1. **Early Years Data Quality (2013-2018)**
**Problem:** 
- Very high errors in 2013-2017 (628.61% → 20.25%)
- Mental health, substance use, outdoor sleeping show 0% in old pipeline for 2013-2018
- This suggests SNA data is missing/interpolated poorly

**Why it happens:**
- SNA surveys only available for 2013, 2018, 2021
- Years 2014-2017 use linear interpolation (very poor for this data)
- The interpolation creates unrealistic proportions

**Solutions:**
1. Use actual SNA anchor values instead of interpolation for missing years
2. Add constraints for intermediate years based on flow data trends
3. Use forward-fill from 2013 instead of interpolation

### 2. **Binary Attribute Distributions Mismatch**
**Problem:**
```
mental_health:      Old 33.2% → New 57.5% (+24.3pp)
foster_care:        Old 11.7% → New 40.1% (+28.4pp)
incarceration:      Old 9.0%  → New 37.7% (+28.7pp)
```

**Why it happens:**
- SNA surveys may undercount these attributes
- Proportions in `agg_df` might be derived/estimated poorly
- The W matrix constraints force unrealistic distributions

**Solutions:**
1. Validate/calibrate the SNA proportions against external data
2. Add soft constraints (weighted, not hard) for problematic attributes
3. Use floor/ceiling bounds (min 5%, max 80%) for sanity checks
4. Cross-validate against known prevalence rates from literature

### 3. **Early Year Interpolation Errors**
**Current approach:**
```python
# sna_pipeline.py - old pipeline
interp = interp.interpolate(method="index", limit_direction="both")
```

**Better approach:**
- Use observed flow data to guide interpolation
- Use spline interpolation instead of linear
- Use domain knowledge constraints (e.g., mental health prevalence shouldn't drop)

## Recommended Improvements (Priority Order)

### Priority 1: Fix the Data Quality Issues (Biggest Impact)

#### 1A. Improve Aggregate Interpolation
```python
# Instead of simple linear interpolation, use flow-guided approach
def interpolate_aggregates_improved(observed, all_years, flow_yearly):
    """
    Interpolate SNA aggregates with flow data guidance.
    For years with flow data (2018+), scale observed proportions by flow trends.
    """
```

#### 1B. Add Sanity Checks & Bounds
```python
# In compute_derived() or extract_Y():
# Enforce realistic bounds on proportions
def clip_proportions_to_realistic_ranges(agg):
    agg['pct_mental_health'] = np.clip(agg['pct_mental_health'], 0.15, 0.60)
    agg['pct_substance_use'] = np.clip(agg['pct_substance_use'], 0.15, 0.45)
    agg['pct_foster_care_history'] = np.clip(agg['pct_foster_care_history'], 0.05, 0.30)
    agg['pct_incarceration_history'] = np.clip(agg['pct_incarceration_history'], 0.05, 0.25)
    return agg
```

#### 1C. Use Domain Knowledge for Missing Data
```python
# For 2013-2017, use SNA 2013 values or external prevalence rates
EXTERNAL_PREVALENCE = {
    'pct_mental_health': 0.35,      # 35% from literature
    'pct_substance_use': 0.25,      # 25% from literature
    'pct_outdoor_sleeping': 0.15,   # 15% from literature
    'pct_foster_care_history': 0.12,
    'pct_incarceration_history': 0.08,
}
```

### Priority 2: Improve the Optimization Constraints (Medium Impact)

#### 2A. Add Soft Constraints
Currently W matrix uses hard constraints (==). Add weighted soft constraints:
```python
def build_W_with_soft_constraints(high_uncertainty_years):
    """
    Add soft constraints (lower weights) for years with poor data.
    Hard constraints (weight=1.0) for 2018+ (good flow data)
    Soft constraints (weight=0.5) for 2013-2017 (interpolated)
    """
```

#### 2B. Improve the Solver Settings
```python
# Current: tol=1e-10, max_iter=50000
# Better: Use tiered approach
if year >= 2019:  # Good data
    result = lsq_linear(W_MATRIX, y, tol=1e-8, max_iter=50000)
else:  # Interpolated data
    result = lsq_linear(W_MATRIX, y, tol=1e-5, max_iter=30000)
```

### Priority 3: Improve Feature Engineering (Lower Priority)

#### 3A. Better Correlation Matrix for Extra Binary Features
```python
# Currently uses fixed EXTRA_CORR matrix
# Better: Learn from old pipeline's synthetic data
def compute_correlation_from_old_data():
    old_df = pd.read_csv('synthetic_data/synthetic_individuals.csv')
    return old_df[EXTRA_BINARY_FEATURES].corr().values
```

#### 3B. Context-Aware Marginals
```python
# Current: flat marginals for extra features
# Better: Use year-specific marginals
def add_extra_binary_features_context_aware(df, row, year):
    """Adjust marginals based on year and temporal trends"""
    if year < 2018:
        # Earlier years: lower incarceration/foster care rates
        probs['incarceration_history'] *= 0.7
        probs['foster_care_history'] *= 0.8
```

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. Add bounds checking to proportions
2. Use external prevalence rates for 2013-2017
3. Test impact on 2021 RMSE

### Phase 2: Medium Effort (2-3 hours)
1. Implement flow-guided interpolation
2. Add soft constraints for uncertain years
3. Recalibrate solver tolerance per year

### Phase 3: Advanced (3-4 hours)
1. Learn correlations from old pipeline
2. Context-aware marginals by year
3. Validate against external datasets

## Expected Improvements

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| RMSE (2013) | 1189.69 | ~800 | ~500 | ~300 |
| RMSE (2018) | 336.00 | 336.00 | ~250 | ~200 |
| RMSE (2021) | 283.30 | ~200 | ~150 | ~100 |
| Mental Health % Error | 628.61% | ~100% | ~30% | ~10% |
| MH-SU Correlation | 0.36 | 0.36 | 0.40 | 0.42 |

## Code Changes Needed

Would you like me to implement:
1. **Quick Phase 1** (bounds checking + external rates)
2. **Medium Phase 2** (flow-guided interpolation + soft constraints)
3. **Full Phase 3** (all of above + correlation learning)
4. **All of the above**

Let me know which phase you'd like to prioritize!
