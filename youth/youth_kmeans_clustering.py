"""
Youth-Only K-Means Clustering
==============================
Filters the dataset to youth=1 records ONLY, then clusters them using
k-means along a second axis you choose (e.g. chronic_homeless, lgbtq).

Every record in the model is a youth. Adults are excluded entirely.
Clusters are named purely by the second column (e.g. "Chronic Homeless: Yes").

Usage
-----
  python youth_kmeans_clustering.py                          # interactive
  python youth_kmeans_clustering.py --col chronic_homeless   # non-interactive
  python youth_kmeans_clustering.py --col lgbtq --k 8       # custom k
  python youth_kmeans_clustering.py --list-cols              # show valid cols

Outputs
-------
  clustered_output.csv   - original data + cluster label + cluster_name
  cluster_summary.csv    - counts and feature means per cluster
  cluster_plot.png       - heatmap of cluster distribution (youth × col2)
  youth_kmeans_model.pkl - trained model bundle (KMeans + scaler + metadata)

Load the model later with:
  import pickle
  with open("youth_kmeans_model.pkl", "rb") as f:
      model = pickle.load(f)
  # model keys: kmeans, scaler, feature_cols, second_col, cluster_name_map
  labels = model["kmeans"].predict(model["scaler"].transform(new_df[model["feature_cols"]]))
"""

import argparse
import sys
import warnings
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ── Feature config ────────────────────────────────────────────────────────────

# All columns that can serve as the second axis (binary or low-cardinality).
VALID_SECOND_COLS = [
    "chronic_homeless",
    "mental_health",
    "substance_use",
    "outdoor_sleeping",
    "lgbtq",
    "indigenous_flag",
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
    "housing_loss_health",
]

# Numeric features used for k-means (youth is NOT included — all rows are youth=1).
BASE_FEATURES = [
    "age",
    "years_homeless",
    "mental_health",
    "substance_use",
    "physical_health",  # present in synthetic_individuals.csv
    "outdoor_sleeping",
    "chronic_homeless",
    "lgbtq",
    "indigenous",       # column name in synthetic_individuals.csv
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
    "housing_loss_health",
]

# Aliases: some CSVs use slightly different column names.
COL_ALIASES = {
    "indigenous_flag": "indigenous",
    "indigenous":      "indigenous_flag",
}


def resolve_col(df: pd.DataFrame, name: str) -> str | None:
    """Return the actual column name in df, trying aliases if needed."""
    if name in df.columns:
        return name
    alias = COL_ALIASES.get(name)
    if alias and alias in df.columns:
        return alias
    return None


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading {path} …")
    df = pd.read_csv(path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns (full dataset)")

    youth_col = resolve_col(df, "youth")
    if youth_col is None:
        sys.exit("ERROR: 'youth' column not found in the CSV.")

    df_youth = df[df[youth_col] == 1].copy().reset_index(drop=True)
    n_dropped = len(df) - len(df_youth)
    print(f"  Filtered to youth=1 only: {len(df_youth):,} records kept, {n_dropped:,} adults dropped")
    return df_youth


def choose_column(df: pd.DataFrame) -> str:
    """Interactive column picker (df is already youth-only)."""
    available = [c for c in VALID_SECOND_COLS if resolve_col(df, c) is not None]
    print("\nAvailable columns for the second cluster axis (youth records only):")
    for i, c in enumerate(available, 1):
        actual = resolve_col(df, c)
        val_counts = df[actual].value_counts().to_dict()
        print(f"  {i:2d}. {c:<30s}  values: {val_counts}")
    while True:
        raw = input("\nEnter column name or number: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(available):
                return available[idx]
        elif raw in available or raw in [resolve_col(df, c) for c in available]:
            return raw
        print("  ✗ Not recognised — try again.")


def build_feature_matrix(df: pd.DataFrame, second_col_actual: str) -> np.ndarray:
    """Build and scale the feature matrix used for k-means."""
    candidate_features = BASE_FEATURES + [second_col_actual]
    features = [f for f in candidate_features if f in df.columns]
    # Deduplicate while preserving order
    seen = set()
    features = [f for f in features if not (f in seen or seen.add(f))]

    X = df[features].copy()
    # Fill any NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Features used: {features}")
    return X_scaled, features, scaler


def pick_k(second_vals: np.ndarray) -> int:
    """
    k = number of non-empty values in col2 × a multiplier for sub-clusters.
    Since all rows are youth=1, we only split along the second column.
    """
    n_cells = len(set(second_vals))
    suggested = min(n_cells * 3, 12)
    print(f"\n  Unique values in col2: {n_cells}  →  suggested k = {suggested}")
    return suggested


def cluster_label(col2_val: int, col2_name: str) -> str:
    col2_str = col2_name.replace("_", " ").title()
    flag_str = "Yes" if col2_val == 1 else "No"
    return f"Youth | {col2_str}: {flag_str}"


def run_kmeans(X_scaled: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, KMeans]:
    print(f"\nRunning k-means  k={k} …")
    km = KMeans(n_clusters=k, random_state=seed, n_init=15, max_iter=500)
    labels = km.fit_predict(X_scaled)
    if len(X_scaled) > 50_000:
        sample_idx = np.random.choice(len(X_scaled), 50_000, replace=False)
        sil = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
    else:
        sil = silhouette_score(X_scaled, labels)
    print(f"  Silhouette score: {sil:.4f}  (higher is better, range −1 to 1)")
    return labels, km


def assign_cluster_names(
    df: pd.DataFrame,
    labels: np.ndarray,
    second_col: str,
    second_col_canonical: str,
) -> tuple[pd.Series, dict]:
    """
    Each raw k-means cluster gets a human-readable name based on the
    majority col2 value within it. All records are already youth=1.
    """
    tmp = df[[second_col]].copy()
    tmp["_label"] = labels

    def majority(series):
        return int(series.mode()[0])

    agg = tmp.groupby("_label").agg(
        col2_maj=(second_col, majority),
    )

    name_counter: dict[str, int] = {}
    raw_to_name: dict[int, str] = {}
    for raw_cluster, row in agg.iterrows():
        base = cluster_label(row["col2_maj"], second_col_canonical)
        count = name_counter.get(base, 0)
        name_counter[base] = count + 1
        display = base if count == 0 else f"{base} (variant {count})"
        raw_to_name[raw_cluster] = display

    return pd.Series(labels, index=df.index).map(raw_to_name), raw_to_name


def summarise(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: pd.Series,
    second_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    tmp = df.copy()
    tmp["cluster_id"] = labels
    tmp["cluster_name"] = cluster_names

    numeric_cols = [c for c in feature_cols if c in tmp.columns]

    # Build summary in two steps to avoid named-agg conflicts with str columns
    counts = tmp.groupby("cluster_id").size().rename("count")
    names_series = tmp.groupby("cluster_id")["cluster_name"].first()
    means = tmp.groupby("cluster_id")[numeric_cols].mean().round(3)

    summary = pd.concat([counts, names_series, means], axis=1).reset_index()
    # Round feature means for readability
    for c in numeric_cols:
        if c in summary.columns:
            summary[c] = summary[c].round(3)

    summary = summary.sort_values("count", ascending=False)
    return summary


def plot_heatmap(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: pd.Series,
    second_col: str,
    second_col_canonical: str,
    output_path: str,
):
    """Bar chart of record counts per cluster, grouped by col2 value."""
    tmp = df[[second_col]].copy()
    tmp["cluster_name"] = cluster_names.values

    pivot = (
        tmp.groupby([second_col, "cluster_name"])
        .size()
        .reset_index(name="count")
    )

    unique_names = sorted(pivot["cluster_name"].unique())
    cmap = plt.cm.get_cmap("tab20", len(unique_names))
    name_to_colour = {n: cmap(i) for i, n in enumerate(unique_names)}

    col2_vals = sorted(tmp[second_col].unique())
    col2_label = second_col_canonical.replace("_", " ").title()

    fig, axes = plt.subplots(
        1, len(col2_vals),
        figsize=(5 * len(col2_vals), 5),
        squeeze=False,
    )

    for j, cv in enumerate(col2_vals):
        ax = axes[0][j]
        subset = pivot[pivot[second_col] == cv]

        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=12, color="grey")
        else:
            ax.bar(
                range(len(subset)),
                subset["count"],
                color=[name_to_colour[n] for n in subset["cluster_name"]],
                edgecolor="white",
                linewidth=0.8,
            )
            ax.set_xticks([])
            ax.set_ylabel("Youth Count", fontsize=9)
            total = subset["count"].sum()
            ax.set_title(f"n = {total:,}", fontsize=9, pad=3)

        col2_str = "Yes" if cv == 1 else "No"
        ax.set_xlabel(
            f"{col2_label}: {col2_str}",
            fontsize=11,
            fontweight="bold",
        )

    patches = [
        mpatches.Patch(color=name_to_colour[n], label=n) for n in unique_names
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=min(3, len(unique_names)),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
    )

    fig.suptitle(
        f"Youth Clusters by {col2_label}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Youth-anchored k-means clustering")
    parser.add_argument("input", nargs="?",
                        default="synthetic_data/synthetic_individuals.csv",
                        help="Path to synthetic_individuals.csv")
    parser.add_argument("--col", default=None,
                        help="Second axis column (e.g. chronic_homeless)")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of clusters (default: auto)")
    parser.add_argument("--out-csv", default="clustered_output.csv")
    parser.add_argument("--out-summary", default="cluster_summary.csv")
    parser.add_argument("--out-plot", default="cluster_plot.png")
    parser.add_argument("--out-model", default="youth_kmeans_model.pkl",
                        help="Path to save the trained model bundle (.pkl)")
    parser.add_argument("--list-cols", action="store_true",
                        help="Print valid second-axis columns and exit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.input)

    if args.list_cols:
        available = [c for c in VALID_SECOND_COLS if resolve_col(df, c) is not None]
        print("Valid second-axis columns:")
        for c in available:
            print(f"  {c}")
        sys.exit(0)

    # ── Resolve youth column (used only for filtering, already done in load_data) ─
    youth_col = resolve_col(df, "youth")
    if youth_col is None:
        sys.exit("ERROR: 'youth' column not found in the CSV.")

    # ── Resolve second column ─────────────────────────────────────────────────
    second_col_canonical = args.col
    if second_col_canonical is None:
        second_col_canonical = choose_column(df)

    second_col = resolve_col(df, second_col_canonical)
    if second_col is None:
        sys.exit(f"ERROR: column '{second_col_canonical}' not found in the CSV.")

    print(f"\n  All records: youth=1")
    print(f"  Cluster axis: {second_col}")

    # ── Build features & run k-means ──────────────────────────────────────────
    X_scaled, feature_cols, scaler = build_feature_matrix(df, second_col)

    k = args.k or pick_k(df[second_col].values)
    labels, km = run_kmeans(X_scaled, k, seed=args.seed)

    # ── Name clusters ─────────────────────────────────────────────────────────
    cluster_names, raw_to_name = assign_cluster_names(
        df, labels, second_col, second_col_canonical
    )

    # ── Summarise ─────────────────────────────────────────────────────────────
    summary = summarise(df, labels, cluster_names, second_col, feature_cols)

    print("\nCluster summary:")
    print(summary[["cluster_id", "cluster_name", "count"]].to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    df_out = df.copy()
    df_out["cluster_id"] = labels
    df_out["cluster_name"] = cluster_names.values
    df_out.to_csv(args.out_csv, index=False)
    print(f"\n  Saved clustered data → {args.out_csv}")

    summary.to_csv(args.out_summary, index=False)
    print(f"  Saved cluster summary → {args.out_summary}")

    plot_heatmap(
        df, labels, cluster_names,
        second_col, second_col_canonical,
        args.out_plot,
    )

    # ── Save model bundle ─────────────────────────────────────────────────────
    model_bundle = {
        "kmeans":               km,
        "scaler":               scaler,
        "feature_cols":         feature_cols,
        "second_col":           second_col,
        "second_col_canonical": second_col_canonical,
        "youth_col":            youth_col,
        "cluster_name_map":     raw_to_name,
        "k":                    k,
        "note":                 "Model trained on youth=1 records only. Always filter new data to youth=1 before predicting.",
    }
    with open(args.out_model, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"  Saved model bundle → {args.out_model}")
    print(f"""
  To use the model on new data (youth=1 only):
    import pickle, pandas as pd
    with open("{args.out_model}", "rb") as f:
        model = pickle.load(f)
    new_df = pd.read_csv("new_data.csv")
    new_df = new_df[new_df[model["youth_col"]] == 1]          # filter to youth
    X = new_df[model["feature_cols"]].fillna(new_df[model["feature_cols"]].median())
    X_scaled = model["scaler"].transform(X)
    cluster_ids = model["kmeans"].predict(X_scaled)
    cluster_labels = [model["cluster_name_map"][c] for c in cluster_ids]
""")
    print("Done ✓")


if __name__ == "__main__":
    main()
