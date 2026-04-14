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
        total_col = None
        pred_total_col = None
        for candidate in ("true_total", "actual_total"):
            if candidate in fr.columns:
                total_col = candidate
                break
        for candidate in ("pred_total", "ensemble", "gbr_pred"):
            if candidate in fr.columns:
                pred_total_col = candidate
                break

        if total_col is not None:
            plt.figure(figsize=(9, 4.5))
            plt.plot(fr["year"], fr[total_col], marker="o", label="Actual Total")
            if pred_total_col is not None:
                plt.plot(fr["year"], fr[pred_total_col], marker="o", label="Predicted Total")
            plt.title("Total Individuals by Year")
            plt.xlabel("Year")
            plt.ylabel("Individuals")
            plt.legend()
            _save_plot(out / "total_individuals_comparison.png", plt)

        detailed_targets = []
        for target in ["mental_health", "substance_use", "outdoor_sleeping", "chronic_homeless", "lgbtq", "immigrant"]:
            actual_col = f"actual_{target}_count"
            pred_col = f"pred_{target}_count"
            if actual_col in fr.columns and pred_col in fr.columns:
                detailed_targets.append(target)

        if detailed_targets:
            fig, axes = plt.subplots(len(detailed_targets), 1, figsize=(10, 3.2 * len(detailed_targets)), sharex=True)
            if len(detailed_targets) == 1:
                axes = [axes]
            for ax, target in zip(axes, detailed_targets):
                actual_col = f"actual_{target}_count"
                pred_col = f"pred_{target}_count"
                ax.plot(fr["year"], fr[actual_col], marker="o", label=f"Actual {target}")
                ax.plot(fr["year"], fr[pred_col], marker="o", label=f"Predicted {target}")
                ax.set_title(target.replace("_", " ").title())
                ax.set_ylabel("Count")
                ax.legend(loc="best")
            axes[-1].set_xlabel("Year")
            _save_plot(out / "detailed_attribute_counts.png", plt)

            fig, axes = plt.subplots(len(detailed_targets), 1, figsize=(10, 3.2 * len(detailed_targets)), sharex=True)
            if len(detailed_targets) == 1:
                axes = [axes]
            for ax, target in zip(axes, detailed_targets):
                actual_col = f"actual_{target}_rate"
                pred_col = f"pred_{target}_rate"
                ax.plot(fr["year"], fr[actual_col], marker="o", label=f"Actual {target}")
                ax.plot(fr["year"], fr[pred_col], marker="o", label=f"Predicted {target}")
                ax.set_title(f"{target.replace('_', ' ').title()} Rate")
                ax.set_ylabel("Rate")
                ax.legend(loc="best")
            axes[-1].set_xlabel("Year")
            _save_plot(out / "detailed_attribute_rates.png", plt)
