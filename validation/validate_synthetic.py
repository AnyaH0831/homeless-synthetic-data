from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SYNTHETIC_PATH = Path("synthetic_data/synthetic_individuals.csv")
REGION_YEAR_PATH = Path("synthetic_data/region_year_features.csv")
FORECAST_PATH = Path("synthetic_data/forecast_results.csv")
OUT_DIR = Path("synthetic_data/validation")

BINARY_COLS = [
    "mental_health",
    "substance_use",
    "physical_health",
    "outdoor_sleeping",
    "chronic_homeless",
    "lgbtq",
    "indigenous",
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
    "housing_loss_health",
    "youth",
]

TARGET_MAP = {
    "age_avg": "age_avg",
    "years_homeless_avg": "years_homeless_avg",
    "pct_mental_health": "mental_health",
    "pct_substance_use": "substance_use",
    "pct_physical_health": "physical_health",
    "pct_outdoor_sleeping": "outdoor_sleeping",
    "pct_chronic": "chronic_homeless",
    "pct_lgbtq": "lgbtq",
    "pct_indigenous": "indigenous",
    "pct_immigrant": "immigrant",
    "pct_youth": "youth",
    "pct_no_income": "no_income",
    "pct_foster_care": "foster_care_history",
    "pct_incarceration": "incarceration_history",
    "pct_housing_loss_income": "housing_loss_income",
}

CONTINUOUS_COLS = ["age", "years_homeless"]


def _mean_abs_error(a: pd.Series, b: pd.Series) -> float:
    return float(np.nanmean(np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))))


def _relative_error(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    denom = actual.replace(0, np.nan).astype(float)
    return (predicted.astype(float) - actual.astype(float)).abs() / denom


def aggregate_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("year").agg(
        age_avg=("age", "mean"),
        years_homeless_avg=("years_homeless", "mean"),
        total_individuals=("year", "size"),
    ).reset_index()

    for col in BINARY_COLS:
        if col in df.columns:
            agg[f"pct_{col}"] = df.groupby("year")[col].mean().values

    return agg


def summarize_metric(name: str, actual: pd.Series, predicted: pd.Series) -> dict:
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    mae = _mean_abs_error(actual, predicted)
    rmse = float(np.sqrt(np.nanmean((actual - predicted) ** 2)))
    if len(actual.dropna()) > 1 and len(predicted.dropna()) > 1:
        corr = float(actual.corr(predicted))
    else:
        corr = np.nan
    return {
        "metric": name,
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "actual_mean": float(actual.mean()),
        "predicted_mean": float(predicted.mean()),
    }


def add_external_metrics(rows: list[dict], merged: pd.DataFrame, forecast: pd.DataFrame | None) -> None:
    if "flow_active_homeless" not in merged.columns:
        return

    valid = merged[["year", "true_total", "flow_active_homeless"]].dropna()
    if valid.empty:
        return

    rows.append(summarize_metric("true_total_vs_flow", valid["flow_active_homeless"], valid["true_total"]))

    if forecast is not None and not forecast.empty and "ensemble" in forecast.columns:
        forecast_merged = forecast.merge(
            merged[["year", "flow_active_homeless"]],
            on="year",
            how="left",
        ).dropna(subset=["flow_active_homeless"])
        if not forecast_merged.empty:
            rows.append(summarize_metric("ensemble_vs_flow", forecast_merged["flow_active_homeless"], forecast_merged["ensemble"]))
            rows.append(summarize_metric("gbr_vs_flow", forecast_merged["flow_active_homeless"], forecast_merged["gbr_pred"]))
            rows.append(summarize_metric("ridge_vs_flow", forecast_merged["flow_active_homeless"], forecast_merged["ridge_pred"]))


def main() -> None:
    if not SYNTHETIC_PATH.exists():
        raise FileNotFoundError(f"Missing synthetic file: {SYNTHETIC_PATH}")
    if not REGION_YEAR_PATH.exists():
        raise FileNotFoundError(f"Missing region-year file: {REGION_YEAR_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_error = OUT_DIR / "plot_error.txt"
    if plot_error.exists():
        plot_error.unlink()

    synthetic = pd.read_csv(SYNTHETIC_PATH)
    target = pd.read_csv(REGION_YEAR_PATH)
    forecast = pd.read_csv(FORECAST_PATH) if FORECAST_PATH.exists() else None

    syn_agg = aggregate_synthetic(synthetic).rename(
        columns={col: f"{col}_syn" for col in aggregate_synthetic(synthetic).columns if col != "year"}
    )
    merged = target.merge(syn_agg, on="year", how="left")

    rows = []
    for target_col, syn_col in TARGET_MAP.items():
        syn_name = f"{syn_col}_syn" if not syn_col.startswith("pct_") else f"{syn_col}_syn"
        if target_col in merged.columns and syn_name in merged.columns:
            rows.append(summarize_metric(target_col, merged[target_col], merged[syn_name]))

    if "true_total" in merged.columns:
        rows.append(summarize_metric("true_total_vs_rowcount", merged["true_total"], merged["total_individuals_syn"]))

    add_external_metrics(rows, merged, forecast)

    validation = pd.DataFrame(rows)
    validation["good_threshold"] = np.where(
        validation["metric"].isin(["age_avg", "years_homeless_avg"]),
        0.10,
        np.where(validation["metric"].isin(["true_total_vs_rowcount", "true_total_vs_flow"]), 0.05, 0.05),
    )

    validation["status"] = np.where(validation["mae"] <= validation["good_threshold"], "good", "needs_work")

    validation.to_csv(OUT_DIR / "validation_summary.csv", index=False)

    # Year-by-year comparison table
    year_cols = ["year", "true_total", "total_individuals_syn", "age_avg", "age_avg_syn", "years_homeless_avg", "years_homeless_avg_syn"]
    year_compare = merged[[c for c in year_cols if c in merged.columns]].copy()
    for target_col, syn_col in TARGET_MAP.items():
        syn_name = f"{syn_col}_syn"
        if target_col in merged.columns and syn_name in merged.columns:
            year_compare[f"{target_col}_abs_error"] = (merged[target_col] - merged[syn_name]).abs()
    year_compare.to_csv(OUT_DIR / "yearly_comparison.csv", index=False)

    # Plotting
    try:
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")

        def save_plot(name: str):
            plt.tight_layout()
            plt.savefig(OUT_DIR / name, dpi=160)
            plt.close()

        if "true_total" in merged.columns:
            plt.figure(figsize=(9, 4.5))
            plt.plot(merged["year"], merged["true_total"], marker="o", label="Target total")
            plt.plot(merged["year"], merged["total_individuals_syn"], marker="o", label="Synthetic row count")
            if "flow_active_homeless" in merged.columns:
                plt.plot(merged["year"], merged["flow_active_homeless"], marker="o", label="Flow active homeless")
            plt.title("Total Individuals: Target vs Synthetic")
            plt.xlabel("Year")
            plt.ylabel("Count")
            plt.legend()
            save_plot("validation_totals.png")

        if forecast is not None and not forecast.empty:
            plt.figure(figsize=(9, 4.5))
            f = forecast.copy()
            if "flow_active_homeless" in merged.columns:
                f = f.merge(merged[["year", "flow_active_homeless"]], on="year", how="left")
            if "flow_active_homeless" in f.columns:
                plt.plot(f["year"], f["flow_active_homeless"], marker="o", label="Flow active homeless")
            plt.plot(f["year"], f["ensemble"], marker="o", label="Ensemble")
            plt.plot(f["year"], f["gbr_pred"], marker="o", label="GBR")
            plt.plot(f["year"], f["ridge_pred"], marker="o", label="Ridge")
            plt.title("Forecast vs Flow Signal")
            plt.xlabel("Year")
            plt.ylabel("Count")
            plt.legend()
            save_plot("forecast_vs_flow.png")

        plt.figure(figsize=(9, 4.5))
        plt.bar(validation["metric"], validation["mae"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Validation MAE by Metric")
        plt.ylabel("MAE")
        save_plot("validation_mae_by_metric.png")

        plt.figure(figsize=(9, 4.5))
        plt.bar(validation["metric"], validation["corr"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Validation Correlation by Metric")
        plt.ylabel("Correlation")
        save_plot("validation_corr_by_metric.png")

    except Exception as exc:
        plot_error.write_text(str(exc), encoding="utf-8")

    summary = {
        "rows_compared": int(len(validation)),
        "good_metrics": int((validation["status"] == "good").sum()),
        "needs_work_metrics": int((validation["status"] == "needs_work").sum()),
        "overall_mean_mae": float(validation["mae"].mean()) if not validation.empty else np.nan,
    }
    (OUT_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Validation summary:")
    print(validation.sort_values("mae").to_string(index=False))
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
