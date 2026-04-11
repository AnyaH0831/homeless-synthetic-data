# Homeless Synthetic Data Pipeline

This project loads Toronto SNA data, calibrates it with shelter-flow data, generates synthetic records, and trains a forecast model.

## Setup

1. Create/activate your virtual environment.
2. Install dependencies:

```bash
py -m pip install -r requirements.txt
```

## Commands

### Run full pipeline (local SNA + local shelter-flow)

```bash
.\.venv\Scripts\python.exe sna_pipeline.py --local
```

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

## Project structure

- `sna_pipeline.py`: orchestration and SNA loading/interpolation
- `shelter_flow.py`: shelter flow loading + calibration
- `generation/synthetic_generation.py`: synthetic individual and region-year feature generation
- `training/forecast_training.py`: model training and forecasting
- `source_data/`: local input files
- `synthetic_data/`: generated outputs
