from pathlib import Path
import importlib

import pandas as pd


def _save_plot(path: Path, plt):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def create_plots(
    agg_df: pd.DataFrame,
    region_year_df: pd.DataFrame,
    output_dir: str | Path,
    forecast_results: pd.DataFrame | None = None,
):
    plt = importlib.import_module("matplotlib.pyplot")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "year" in agg_df.columns:
        yearly = agg_df[["year", "total_surveyed", "pct_chronic"]].copy()
    else:
        yearly = agg_df.reset_index()[["year", "total_surveyed", "pct_chronic"]].copy()

    plt.figure(figsize=(9, 4.5))
    plt.plot(yearly["year"], yearly["total_surveyed"], marker="o")
    plt.title("Calibrated Total Individuals by Year")
    plt.xlabel("Year")
    plt.ylabel("Individuals")
    _save_plot(out / "total_individuals_by_year.png", plt)

    plt.figure(figsize=(9, 4.5))
    plt.plot(yearly["year"], yearly["pct_chronic"], marker="o")
    plt.title("Chronic Share by Year")
    plt.xlabel("Year")
    plt.ylabel("Proportion")
    _save_plot(out / "pct_chronic_by_year.png", plt)

    ry = region_year_df.sort_values("year").copy()
    if "pct_youth" in ry.columns:
        plt.figure(figsize=(9, 4.5))
        plt.plot(ry["year"], ry["pct_youth"], marker="o")
        plt.title("Youth Share by Year")
        plt.xlabel("Year")
        plt.ylabel("Proportion")
        _save_plot(out / "pct_youth_by_year.png", plt)

    if forecast_results is not None and not forecast_results.empty:
        fr = forecast_results.sort_values("year").copy()

        plt.figure(figsize=(9, 4.5))
        plt.plot(fr["year"], fr["true_total"], marker="o", label="True Total")
        plt.plot(fr["year"], fr["ridge_pred"], marker="o", label="Ridge")
        plt.plot(fr["year"], fr["gbr_pred"], marker="o", label="GBR")
        plt.plot(fr["year"], fr["ensemble"], marker="o", label="Ensemble")
        plt.title("True vs Predicted Totals")
        plt.xlabel("Year")
        plt.ylabel("Individuals")
        plt.legend()
        _save_plot(out / "true_vs_predicted_totals.png", plt)

        observed = fr[fr["observed"] == True].copy()
        if not observed.empty:
            observed["ensemble_error"] = observed["ensemble"] - observed["true_total"]
            plt.figure(figsize=(9, 4.5))
            plt.bar(observed["year"].astype(str), observed["ensemble_error"])
            plt.axhline(0, color="black", linewidth=1)
            plt.title("Observed-Year Ensemble Error")
            plt.xlabel("Year")
            plt.ylabel("Prediction Error")
            _save_plot(out / "observed_ensemble_error.png", plt)
