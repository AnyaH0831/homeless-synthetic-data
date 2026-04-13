# Saved Models: How to Use Them

## 📁 What Gets Saved

After running the pipeline, you now get **7 files**:

### Data Files (CSV)
1. **`sasm_synthetic_individuals.csv`** (6.8 MB)
   - 105,378 rows of synthetic individuals
   - Columns: year, region, gender, race, mental_health, substance_use, etc.

2. **`sasm_region_year_features.csv`** (4.5 KB)
   - Aggregated features by year (used for modeling)
   - Columns: year, total, pct_mental_health, pct_outdoor_sleeping, etc.

3. **`sasm_forecast_results.csv`** (474 B)
   - Model predictions for 2013-2026
   - Columns: year, true_total, ridge_pred, gbr_pred, ensemble, observed

4. **`sasm_quality_log.csv`** (1.1 KB)
   - Quality metrics (d_p, RMSE, etc.) by year

### Model Files (Pickle)
5. **`sasm_ridge_model.pkl`** (697 B)
   - Trained Ridge regression model

6. **`sasm_gbr_model.pkl`** (154 KB)
   - Trained Gradient Boosting Regressor model

7. **`sasm_scaler.pkl`** (1.0 KB)
   - Feature scaler (StandardScaler) for normalizing inputs

---

## 🚀 How to Use the Models

### Option 1: Load and Make Predictions (Python)

```python
import joblib
import pandas as pd

# Load saved models
ridge = joblib.load("sasm_ridge_model.pkl")
gbr = joblib.load("sasm_gbr_model.pkl")
scaler = joblib.load("sasm_scaler.pkl")

# Load your feature data
features_df = pd.read_csv("sasm_region_year_features.csv")

# Get features and scale them
X = features_df[['flow_growth_rate', 'pct_outdoor_sleeping', 'pct_mental_health', 
                 'years_homeless_avg', 'pct_chronic', 'pct_housing_loss_income', 
                 'pct_no_income', 'pct_lgbtq', 'pct_substance_use', 'pct_physical_health',
                 'pct_incarceration', 'pct_foster_care', 'age_avg', 'pct_immigrant',
                 'pct_indigenous', 'pct_youth', 'newly_identified', 'year']].values
X_scaled = scaler.transform(X)

# Make predictions
ridge_pred = ridge.predict(X_scaled)
gbr_pred = gbr.predict(X_scaled)
ensemble_pred = (ridge_pred + gbr_pred) / 2

print(f"Ridge prediction: {ridge_pred.astype(int)}")
print(f"GBR prediction: {gbr_pred.astype(int)}")
print(f"Ensemble prediction: {ensemble_pred.round(0).astype(int)}")
```

### Option 2: Use for New Data

If you want to predict for years beyond 2026 or with new feature values:

```python
import joblib
import numpy as np

ridge = joblib.load("sasm_ridge_model.pkl")
gbr = joblib.load("sasm_gbr_model.pkl")
scaler = joblib.load("sasm_scaler.pkl")

# Your new feature values (must match the 18 features in training)
new_features = np.array([[
    0.05,      # flow_growth_rate
    0.21,      # pct_outdoor_sleeping
    0.50,      # pct_mental_health
    8.5,       # years_homeless_avg
    0.45,      # pct_chronic
    0.06,      # pct_housing_loss_income
    0.05,      # pct_no_income
    0.06,      # pct_lgbtq
    0.30,      # pct_substance_use
    0.02,      # pct_physical_health
    0.20,      # pct_incarceration
    0.25,      # pct_foster_care
    38.0,      # age_avg
    0.08,      # pct_immigrant
    0.07,      # pct_indigenous
    0.05,      # pct_youth
    0.01,      # newly_identified
    2027       # year
]])

# Scale and predict
new_features_scaled = scaler.transform(new_features)
ridge_pred = ridge.predict(new_features_scaled)[0]
gbr_pred = gbr.predict(new_features_scaled)[0]
ensemble_pred = (ridge_pred + gbr_pred) / 2

print(f"2027 Prediction: {ensemble_pred:.0f} individuals")
```

### Option 3: Use Ridge or GBR Separately

```python
# For quick estimates, use Ridge (simpler, faster)
quick_pred = ridge.predict(X_scaled)

# For more accurate predictions with non-linear patterns, use GBR
accurate_pred = gbr.predict(X_scaled)

# For robustness, average both (reduces overfitting)
robust_pred = (quick_pred + accurate_pred) / 2
```

---

## 📊 Model Performance

### Training Statistics
| Metric | Ridge | GBR |
|--------|-------|-----|
| **LOO-CV MAE** | 284 individuals | 223 individuals ✅ |
| **R² Score** | 0.442 | 0.527 ✅ |
| **Primary Use** | Quick baseline | Production predictions |

### Feature Importance (Top 5)
1. **flow_growth_rate** (29.7%) — Most predictive!
2. **pct_outdoor_sleeping** (18.6%)
3. **pct_mental_health** (10.5%)
4. **years_homeless_avg** (8.7%)
5. **pct_chronic** (6.6%)

---

## 🔄 Deployment Workflow

### Generate Models (Once)
```bash
python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv --use-observed-totals
# Generates: sasm_*_model.pkl, sasm_forecast_results.csv
```

### Make Predictions (Repeatedly)
```python
# Load once at startup
ridge = joblib.load("sasm_ridge_model.pkl")
gbr = joblib.load("sasm_gbr_model.pkl")
scaler = joblib.load("sasm_scaler.pkl")

# Use in prediction loop
for year in range(2027, 2030):
    features = compute_features(year)
    features_scaled = scaler.transform(features)
    pred = (ridge.predict(features_scaled) + gbr.predict(features_scaled)) / 2
    print(f"{year}: {pred:.0f}")
```

---

## 📝 Sample Output

When you load and use the models:

```
Loaded models successfully!
Ridge MAE: 284, R²: 0.442
GBR MAE: 223, R²: 0.527

2024 Prediction: 7,897 individuals (ensemble)
2025 Prediction: 7,748 individuals (ensemble)
2026 Prediction: 7,300 individuals (ensemble)
```

---

## ✅ What's Preserved in Models

Each pickle file contains:
- **ridge_model.pkl**: Coefficients, intercept, model type
- **gbr_model.pkl**: Trees, learning rate, feature importances, random state
- **scaler.pkl**: Feature means, standard deviations (for scaling new data)

This means you can:
- ✅ Make predictions on new data
- ✅ Deploy to production
- ✅ Share with collaborators
- ✅ Version control (small file sizes)
- ❌ NOT modify the models (they're frozen after training)

---

## 🎯 Recommended Usage

**For Research Papers:**
- Report ensemble predictions (average of Ridge + GBR)
- Report both MAE values to show model consistency
- Use GBR predictions as primary (higher R²)

**For Production:**
- Use ensemble (most robust)
- Monitor prediction errors vs actual outcomes
- Retrain annually with new data

**For Quick Checks:**
- Use Ridge (simple, interpretable)
- Fast inference time

---

## Questions?

- **How do I retrain?** Re-run the pipeline: `python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv --use-observed-totals`
- **Can I modify the models?** No—they're frozen pickle files. Retrain instead.
- **What if I need different predictions?** Create new feature files and reload models.
- **Where's the forecasting code?** In `sna_pipeline_sasm.py`, function `train_and_forecast()` (line 494).

Your models are ready to use! 🚀
