# RBC Borealis X United Way East Ontario - Chronic Homeless Predictor

This is the backend and model for the chronic homeless predictor. It uses the Toronto SNA data and calibrates it with shelter-flow data, extrapolates from the data, and trains a predictor model. It also tests the Lanark County data by extrapolating the missing features.

## Setup

1. Create/activate your virtual environment.
2. Install dependencies:

```bash
py -m pip install -r requirements.txt
```

## Generate Toronto SNA data

### Run full pipeline (local SNA + local shelter-flow)

```bash
.\.venv\Scripts\python.exe sna_pipeline.py --local
```
python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv --use-observed-totals && \
python compare_pipelines.py

Outputs:

- `synthetic_data/synthetic_individuals.csv`
- `synthetic_data/region_year_features.csv`
- `synthetic_data/forecast_results.csv`
- `synthetic_data/plots/total_individuals_by_year.png`
- `synthetic_data/plots/pct_chronic_by_year.png`
- `synthetic_data/plots/pct_youth_by_year.png`
- `synthetic_data/plots/true_vs_predicted_totals.png`
- `synthetic_data/plots/observed_ensemble_error.png`

### Generate synthetic data only (skip training)

```bash
.\.venv\Scripts\python.exe sna_pipeline.py --local --skip-model
```

Outputs:

- `synthetic_data/synthetic_individuals.csv`
- `synthetic_data/region_year_features.csv`
- `synthetic_data/plots/total_individuals_by_year.png`
- `synthetic_data/plots/pct_chronic_by_year.png`
- `synthetic_data/plots/pct_youth_by_year.png`

### API mode (SNA and flow from Toronto Open Data)

```bash
.\.venv\Scripts\python.exe sna_pipeline.py
```

### Override local shelter-flow path

```bash
.\.venv\Scripts\python.exe sna_pipeline.py --local --flow-local source_data/toronto-shelter-system-flow.csv
```

### Occupancy/capacity data auto-detection

When running locally, the pipeline also auto-loads daily occupancy files from `source_data/` if present:

- Retired format: `Daily shelter occupancy YYYY.csv`
- New format: `daily-shelter-overnight-service-occupancy-capacity-YYYY.csv`

These are annualized and used to improve `actively_homeless` totals during calibration.

### Validate synthetic quality

```bash
.\.venv\Scripts\python.exe validation\validate_synthetic.py
```

Outputs:

- `synthetic_data/validation/validation_summary.csv`
- `synthetic_data/validation/yearly_comparison.csv`
- `synthetic_data/validation/validation_totals.png`
- `synthetic_data/validation/forecast_vs_flow.png`
- `synthetic_data/validation/validation_mae_by_metric.png`
- `synthetic_data/validation/validation_corr_by_metric.png`


## Run model

This is model that predicts whether an individual will become chronically homeless from the Toronto SNA data.

Run with:
- `python final/new_no_year.py synthetic_data/synthetic_individuals.csv`

Outputs:
- `youth_no_year_confusion_matrix.png`
- `youth_no_year_features_importance.png`
- `youth_no_year_roc_curve.png`
- `youth_no_year_shap_summary.png`

## Predict on Lanark

Run this in terminal to generate the missing features for the Lanark County Data
```
python final/generate_model_csv.py \
        --bnl  final/new_lanark.csv \
        --syn  synthetic_data/synthetic_individuals.csv \
        --out  final/new_output.csv
```

It outputs `new_output.csv` which is the new Lanark County Data with the required features for the model.

Now run 
`python final/predict_bnl_no_year.py --bnl final/new_output.csv --model final/youth_xgboost_no_year.pkl --all-ages`
to verify the model on the Lanark County data

which outputs
- `bnl_confusion_matrix.png`
- `bnl_feature_importance.png`
- `bnl_roc_curve.png`
- `bnl_shap_bar.png`
- `bnl_shap_summary.png`
- `bnl_predictions.csv`

## Old versions

### youth folder

- `youth_chornic_xgboost.py` to run the model

Outputs:
- classification report
- confusion matrix
- feature importance
- roc auc
- shap bar
- shap summary

### old folder

Models:
- XGBoost (`sasm_xgboost_detailed_model.pkl`)
- Gradient Boosting (`sasm_gbr_model.pkl`)
- Clusters (`sasm_cluster_scaler.pk`)
- Ridge model (`sasm_ridge_model.pkl`)

## Project structure

- `synthetic_data/synthetic_individuals.csv`: Toronto SNA extrapolated data
- `sna_pipeline.py`: orchestration and SNA loading/interpolation
- `shelter_flow.py`: shelter flow loading + calibration
- `generation/synthetic_generation.py`: synthetic individual and region-year feature generation
- `training/forecast_training.py`: model training and forecasting
- `source_data/`: local input files
- `synthetic_data/`: generated outputs

- `final/new_no_year.py`: model
- `final/generate_model_csv.py`: fill in missing Lanark features
- `final/predict_bnl_no_year.py`: predict on Lanark data