"""
sasm_analysis.py
────────────────
Three-layer analysis of SASM synthetic individual data.

WHY THREE LAYERS?
──────────────────
The SASM generates 162k individual rows. Just predicting a total count per year
throws away most of that richness. This module extracts three levels of insight:

  LAYER 1 — WHO ARE THEY? (Clustering / unsupervised)
  ─────────────────────────────────────────────────────
  K-Means on individual features finds natural "need profiles" — groups of
  people who tend to have the same combination of characteristics. For a
  rural dashboard this is the most immediately useful output: "This region
  has 40% high-acuity chronically homeless males and 25% vulnerable youth"
  is more actionable than "there are 500 homeless people."

  LAYER 2 — HOW MANY? (Supervised count forecasting per cluster)
  ──────────────────────────────────────────────────────────────
  Instead of predicting one total count, we predict how many people in EACH
  cluster will be homeless each year. This gives service planners a breakdown
  like "cluster 3 (outdoor chronic) will grow 12% while cluster 1 (sheltered
  family) stays flat." GBR is used here — it handles the small-n well.

  LAYER 3 — WHAT SERVICES? (XGBoost per-individual outcome prediction)
  ─────────────────────────────────────────────────────────────────────
  Train XGBoost on individual rows to predict binary outcomes:
  mental_health, substance_use, chronic_homeless, etc.
  This answers "given this person's profile, what services will they need?"
  which is what intake workers actually ask.

WHY NOT UNSUPERVISED ONLY?
──────────────────────────
Clustering alone tells you who is here NOW but not what will happen NEXT YEAR.
Supervised forecasting tells you future counts but not who those people are.
Combined: cluster assignment tells you the "type" of need, supervised tells
you how many of each type to expect. That's what a forecasting dashboard needs.

OUTPUTS
────────
  sasm_clusters.csv          — individual rows with cluster assignment
  sasm_cluster_profiles.csv  — mean features per cluster (the "who" table)
  sasm_cluster_forecast.csv  — predicted cluster counts per year 2024-2026
  sasm_service_forecast.csv  — predicted service need rates per year
  sasm_analysis_summary.txt  — human-readable summary for dashboard

Usage:
    python sasm_analysis.py
    # Reads sasm_synthetic_individuals.csv (must exist — run sna_pipeline_sasm.py first)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_FILE    = "sasm_synthetic_individuals.csv"
N_CLUSTERS    = 6       # number of need profiles; 5-7 works well for homelessness
FORECAST_YEARS = [2024, 2025, 2026]
OBSERVED_YEARS = [2013, 2018, 2021]    # real SNA anchor years

# Features used for clustering — these define "who the person is"
# We use ALL available individual attributes so clusters are maximally informative
CLUSTER_FEATURES = [
    "age",
    "years_homeless",
    "youth",                 # age < 25
    "indigenous_flag",
    "mental_health",
    "substance_use",
    "outdoor_sleeping",
    "chronic_homeless",
    "lgbtq",
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
    "housing_loss_health",
    # Gender encoded as numeric (male=0, female=1, trans/nb=2)
    "gender_encoded",
    # Race encoded as numeric
    "race_encoded",
    # Shelter type encoded
    "shelter_encoded",
]

# Features used for forecasting models
FORECAST_FEATURES = [
    "year",
    "age",
    "years_homeless",
    "youth",
    "indigenous_flag",
    "mental_health",
    "substance_use",
    "outdoor_sleeping",
    "chronic_homeless",
    "lgbtq",
    "immigrant",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
    "housing_loss_health",
    "gender_encoded",
    "race_encoded",
    "shelter_encoded",
]

# Binary outcomes we want to forecast per person
BINARY_TARGETS = [
    "mental_health",
    "substance_use",
    "outdoor_sleeping",
    "chronic_homeless",
    "lgbtq",
    "foster_care_history",
    "incarceration_history",
    "no_income",
    "housing_loss_income",
]

# Human-readable cluster names — assigned after inspecting cluster profiles
# These will be auto-generated based on dominant characteristics
CLUSTER_LABEL_MAP = {}   # filled in after clustering


# ── STEP 1: LOAD AND ENCODE ───────────────────────────────────────────────────

def load_and_encode(path: str) -> pd.DataFrame:
    """
    Load individual-level synthetic data and add encoded categorical columns.
    
    We encode gender, race, shelter_type as integers so they can be used in
    numerical models (KMeans, GBR) without one-hot explosion.
    
    Label encoding is fine here because:
    - KMeans uses Euclidean distance, so we normalize afterward anyway
    - GBR handles ordinal-like encodings well with tree splits
    - The cluster visualization shows the original text values
    """
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} individual records, years {df['year'].min()}–{df['year'].max()}")
    
    # Encode categoricals as integers (needed for KMeans and GBR)
    gender_order = {"male": 0, "female": 1, "trans_nonbinary": 2}
    race_order   = {"black": 0, "white": 1, "indigenous": 2, "other": 3}
    shelter_order = {"emergency_shelter": 0, "respite": 1, "other": 2, "outdoor": 3}
    
    df["gender_encoded"]  = df["gender"].map(gender_order).fillna(0).astype(int)
    df["race_encoded"]    = df["race"].map(race_order).fillna(3).astype(int)
    df["shelter_encoded"] = df["shelter_type"].map(shelter_order).fillna(2).astype(int)
    
    # Ensure all cluster feature columns exist
    for col in CLUSTER_FEATURES:
        if col not in df.columns:
            df[col] = 0
            print(f"  Warning: column '{col}' not found, filled with 0")
    
    return df


# ── STEP 2: CLUSTERING (Layer 1 — WHO ARE THEY?) ─────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(3, 9)) -> int:
    """
    Find optimal number of clusters using silhouette score.
    
    Silhouette score measures how similar each point is to its own cluster
    vs other clusters. Range: -1 (wrong cluster) to +1 (perfect cluster).
    We try k=3 through k=8 and pick the best.
    
    We use a sample for speed (silhouette is O(n²)).
    """
    sample_size = min(5000, len(X_scaled))
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[idx]
    
    best_k, best_score = 4, -1
    print("\n  Silhouette scores by k:")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        marker = " ←" if score > best_score else ""
        print(f"    k={k}: {score:.4f}{marker}")
        if score > best_score:
            best_score, best_k = score, k
    
    print(f"  Best k: {best_k} (silhouette={best_score:.4f})")
    return best_k


def cluster_individuals(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> tuple:
    """
    Apply K-Means clustering to find natural need profiles.
    
    Returns:
        df         : original DataFrame with 'cluster' column added
        km         : fitted KMeans model
        scaler     : fitted StandardScaler (needed to encode new data)
        profiles   : DataFrame with mean features per cluster
    
    Why K-Means?
    - Fast on 162k rows
    - Interpretable (cluster centers = "average person" in that group)
    - Works well when we normalize first (which we do)
    
    Alternative we considered: HDBSCAN (density-based, handles noise)
    — better for irregular shapes but harder to explain to non-technical
    stakeholders. K-Means cluster profiles are directly presentable in a report.
    """
    X = df[CLUSTER_FEATURES].values.astype(float)
    
    # Normalize: KMeans is distance-based so scale matters
    # Without this, 'age' (range 15-85) would dominate 'mental_health' (0/1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optionally find best k (comment out to use N_CLUSTERS directly)
    # best_k = find_optimal_k(X_scaled)
    # For speed, use fixed N_CLUSTERS
    best_k = n_clusters
    
    print(f"\n  Fitting K-Means with k={best_k} on {len(df):,} individuals...")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20, max_iter=500)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_scaled)
    
    # Build cluster profiles (mean of each feature per cluster)
    readable_cols = [
        "age", "years_homeless", "youth", "indigenous_flag",
        "mental_health", "substance_use", "outdoor_sleeping", "chronic_homeless",
        "lgbtq", "immigrant", "foster_care_history", "incarceration_history",
        "no_income", "housing_loss_income", "housing_loss_health",
    ]
    profiles = df.groupby("cluster")[readable_cols].mean().round(3)
    profiles["n_total"] = df.groupby("cluster").size()
    profiles["pct_of_total"] = (profiles["n_total"] / len(df) * 100).round(1)
    
    # Add dominant gender and race per cluster
    profiles["dominant_gender"] = df.groupby("cluster")["gender"].agg(lambda x: x.value_counts().index[0])
    profiles["dominant_race"]   = df.groupby("cluster")["race"].agg(lambda x: x.value_counts().index[0])
    profiles["dominant_shelter"]= df.groupby("cluster")["shelter_type"].agg(lambda x: x.value_counts().index[0])
    
    return df, km, scaler, profiles


def auto_label_clusters(profiles: pd.DataFrame) -> dict:
    """
    Automatically generate a human-readable label for each cluster based on
    the cluster's dominant characteristics.
    
    This is used to make the dashboard output interpretable.
    Labels are constructed from the top 2-3 distinguishing features.
    
    For example:
      - High age + chronic + mental_health → "Older Chronic High-Needs"
      - Youth + outdoor + no_income        → "Unsheltered Youth"
      - Indigenous + foster_care           → "Indigenous Trauma-History"
    """
    labels = {}
    for cluster_id, row in profiles.iterrows():
        parts = []
        
        # Age group
        if row["age"] < 28:
            parts.append("Youth")
        elif row["age"] > 50:
            parts.append("Older Adult")
        
        # Indigenous flag — important for service planning
        if row.get("indigenous_flag", 0) > 0.35:
            parts.append("Indigenous")
        
        # Housing situation
        if row["outdoor_sleeping"] > 0.40:
            parts.append("Unsheltered")
        elif row.get("dominant_shelter", "") == "emergency_shelter":
            parts.append("Sheltered")
        
        # Primary need
        if row["mental_health"] > 0.55 and row["substance_use"] > 0.45:
            parts.append("Dual-Diagnosis")
        elif row["mental_health"] > 0.55:
            parts.append("Mental Health")
        elif row["substance_use"] > 0.45:
            parts.append("Substance Use")
        
        # Chronic
        if row["chronic_homeless"] > 0.55:
            parts.append("Chronic")
        
        # History factors
        if row.get("foster_care_history", 0) > 0.30 or row.get("incarceration_history", 0) > 0.30:
            parts.append("System-Involved")
        
        label = " / ".join(parts[:3]) if parts else f"Cluster {cluster_id}"
        labels[cluster_id] = f"[{cluster_id}] {label}"
    
    return labels


# ── STEP 3: CLUSTER FORECASTING (Layer 2 — HOW MANY OF EACH TYPE?) ───────────

def forecast_cluster_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster, predict how many people will be in that cluster each year.
    
    This is a regression problem: features = year + cluster characteristics,
    target = count of people in that cluster in that year.
    
    WHY THIS MATTERS:
    A rural service provider doesn't just need "500 people will be homeless."
    They need "200 will be high-acuity dual-diagnosis (need intensive services),
    150 will be sheltered families (need housing support), 80 will be youth
    (need youth-specific programs)." That's what this produces.
    
    We use Gradient Boosting Regressor per cluster because:
    - Small training set (14 years × 1 cluster count = 14 data points per cluster)
    - GBR handles small n well with depth=2 and high regularization
    - No need for deep learning here — the signal is simple
    """
    # Build year-level cluster count table
    cluster_year_counts = (
        df.groupby(["year", "cluster"])
        .size()
        .reset_index(name="count")
    )
    
    # Also compute cluster proportions (more stable than raw counts)
    year_totals = df.groupby("year").size().reset_index(name="year_total")
    cluster_year_counts = cluster_year_counts.merge(year_totals, on="year")
    cluster_year_counts["proportion"] = (
        cluster_year_counts["count"] / cluster_year_counts["year_total"]
    )
    
    results = []
    all_years = sorted(df["year"].unique())
    # Deduplicate forecast years with observed years
    forecast_years_unique = [y for y in FORECAST_YEARS if y not in all_years]
    years_to_predict = sorted(set(all_years + forecast_years_unique))
    
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_data = cluster_year_counts[
            cluster_year_counts["cluster"] == cluster_id
        ].sort_values("year")
        
        # Use proportion as target (more stable than raw count across years)
        X = cluster_data["year"].values.reshape(-1, 1).astype(float)
        y = cluster_data["proportion"].values.astype(float)
        
        # Train GBR on all observed years
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=2,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        )
        gbr.fit(X, y)
        
        # Predict for all years including forecast
        for year in years_to_predict:
            is_forecast = year not in OBSERVED_YEARS
            if not is_forecast:
                # Observed anchor year: use actual count
                row_data = cluster_data[cluster_data["year"] == year]
                actual_count = int(row_data["count"].iloc[0]) if len(row_data) > 0 else 0
                actual_prop  = float(row_data["proportion"].iloc[0]) if len(row_data) > 0 else 0.0
                pred_prop    = float(gbr.predict([[float(year)]])[0])
            else:
                # Forecast year: no actual
                actual_count = None
                actual_prop  = None
                pred_prop    = float(np.clip(gbr.predict([[float(year)]])[0], 0.0, 1.0))
            
            results.append({
                "year":          year,
                "cluster":       cluster_id,
                "actual_count":  actual_count,
                "actual_pct":    actual_prop,
                "pred_pct":      pred_prop,
                "is_forecast":   is_forecast,
            })
    
    return pd.DataFrame(results)


# ── STEP 4: SERVICE NEED FORECASTING (Layer 3 — WHAT SERVICES?) ──────────────

def forecast_service_needs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use GBR to predict what proportion of the homeless population will need
    each type of service in each year, broken down by cluster.
    
    This produces outputs like:
      "In 2025, Cluster 2 (Unsheltered Chronic) will have:
        - 68% needing mental health services
        - 55% needing substance use treatment
        - 42% needing ID/income support"
    
    We train a separate GBR for each binary target × cluster combination.
    With 6 clusters × 9 targets = 54 models — fast because each is tiny.
    
    Training target: proportion of cluster with that attribute = 1, per year.
    This is a proportion regression (0-1), not binary classification.
    Using proportions instead of individual labels avoids the class imbalance
    problem and produces smoother, more interpretable forecasts.
    """
    all_years = sorted(df["year"].unique())
    # Deduplicate forecast years with observed years
    forecast_years_unique = [y for y in FORECAST_YEARS if y not in all_years]
    years_to_predict = sorted(set(all_years + forecast_years_unique))
    # Annual proportion of each target within each cluster
    results = []
    
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        
        # Year × target proportion table
        year_props = cluster_df.groupby("year")[BINARY_TARGETS].mean()
        
        for target in BINARY_TARGETS:
            X = year_props.index.values.reshape(-1, 1).astype(float)
            y = year_props[target].values.astype(float)
            
            if len(y) < 3:
                continue  # Not enough data points
            
            gbr = GradientBoostingRegressor(
                n_estimators=100, max_depth=2,
                learning_rate=0.05, random_state=42
            )
            gbr.fit(X, y)
            
            for year in years_to_predict:
                is_forecast = year not in OBSERVED_YEARS
                if not is_forecast:
                    pred_rate   = float(gbr.predict([[float(year)]])[0])
                    actual_rate = float(year_props.loc[year, target]) if year in year_props.index else None
                else:
                    pred_rate = float(np.clip(gbr.predict([[float(year)]])[0], 0.0, 1.0))
                    actual_rate = None
                
                results.append({
                    "year":        year,
                    "cluster":     cluster_id,
                    "target":      target,
                    "actual_rate": actual_rate,
                    "pred_rate":   float(np.clip(pred_rate, 0.0, 1.0)),
                    "is_forecast": is_forecast,
                })
    
    return pd.DataFrame(results)


# ── STEP 5: GENERATE HUMAN-READABLE SUMMARY ──────────────────────────────────

def generate_summary(
    profiles: pd.DataFrame,
    cluster_labels: dict,
    cluster_forecast: pd.DataFrame,
    service_forecast: pd.DataFrame,
    df: pd.DataFrame,
) -> str:
    """
    Generate a plain-English summary suitable for a dashboard or report.
    
    This is what a rural service planner would actually read — not raw numbers
    but interpreted findings like "the youth unsheltered cluster is growing fastest."
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SYNTHETIC DATA ANALYSIS SUMMARY")
    lines.append("Based on SASM-generated individual-level data")
    lines.append("=" * 70)
    
    # Overall stats
    lines.append(f"\nTotal synthetic individuals: {len(df):,}")
    lines.append(f"Years covered: {df['year'].min()}–{df['year'].max()}")
    lines.append(f"Number of need profiles (clusters): {profiles.shape[0]}")
    
    # Cluster profiles
    lines.append("\n" + "─" * 70)
    lines.append("NEED PROFILES (Who is experiencing homelessness?)")
    lines.append("─" * 70)
    for cid, label in cluster_labels.items():
        p = profiles.loc[cid]
        lines.append(f"\n{label}  ({p['pct_of_total']:.1f}% of population)")
        lines.append(f"  Average age:          {p['age']:.1f} years")
        lines.append(f"  Avg years homeless:   {p['years_homeless']:.1f}")
        lines.append(f"  Dominant gender:      {p['dominant_gender']}")
        lines.append(f"  Dominant race:        {p['dominant_race']}")
        lines.append(f"  Mental health:        {p['mental_health']*100:.0f}%")
        lines.append(f"  Substance use:        {p['substance_use']*100:.0f}%")
        lines.append(f"  Outdoor/unsheltered:  {p['outdoor_sleeping']*100:.0f}%")
        lines.append(f"  Chronic homeless:     {p['chronic_homeless']*100:.0f}%")
        lines.append(f"  Indigenous:           {p['indigenous_flag']*100:.0f}%")
        lines.append(f"  Youth (under 25):     {p['youth']*100:.0f}%")
        lines.append(f"  System-involved:      foster={p['foster_care_history']*100:.0f}%  "
                     f"incarceration={p['incarceration_history']*100:.0f}%")
    
    # Forecast summary
    lines.append("\n" + "─" * 70)
    lines.append("FORECAST SUMMARY (2024–2026)")
    lines.append("─" * 70)
    
    forecast_df = cluster_forecast[cluster_forecast["is_forecast"]]
    for year in FORECAST_YEARS:
        yr_data = forecast_df[forecast_df["year"] == year]
        lines.append(f"\n  {year}:")
        total_pred_pct = yr_data["pred_pct"].sum()
        for _, row in yr_data.sort_values("pred_pct", ascending=False).iterrows():
            cid   = int(row["cluster"])
            label = cluster_labels.get(cid, f"Cluster {cid}")
            pct   = row["pred_pct"] * 100
            lines.append(f"    {label}: {pct:.1f}% of population")
    
    # Fastest growing cluster
    if len(forecast_df) > 0:
        try:
            growth = []
            for cid in sorted(df["cluster"].unique()):
                cdata = cluster_forecast[cluster_forecast["cluster"] == cid]
                obs   = cdata[~cdata["is_forecast"]]["actual_pct"].dropna()
                fc    = cdata[cdata["is_forecast"]]["pred_pct"]
                if len(obs) > 0 and len(fc) > 0:
                    g = float(fc.mean()) - float(obs.mean())
                    growth.append((cid, g))
            if growth:
                fastest_cid = max(growth, key=lambda x: x[1])[0]
                fastest_label = cluster_labels.get(fastest_cid, f"Cluster {fastest_cid}")
                lines.append(f"\n  Fastest growing profile: {fastest_label}")
        except Exception:
            pass
    
    # Service needs for forecast years
    lines.append("\n" + "─" * 70)
    lines.append("PROJECTED SERVICE NEEDS (2024 forecast)")
    lines.append("─" * 70)
    svc_2024 = service_forecast[
        (service_forecast["year"] == 2024) & service_forecast["is_forecast"]
    ]
    if len(svc_2024) > 0:
        overall = svc_2024.groupby("target")["pred_rate"].mean().sort_values(ascending=False)
        lines.append("\n  Average across all clusters:")
        for target, rate in overall.items():
            lines.append(f"    {target:<28}: {rate*100:.1f}%")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SASM ANALYSIS: Clustering + Service Forecasting")
    print("=" * 70)
    
    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nStep 1: Loading synthetic individual data...")
    df = load_and_encode(INPUT_FILE)
    
    # ── Cluster ───────────────────────────────────────────────────────────────
    print("\nStep 2: Clustering individuals into need profiles...")
    df, km, scaler, profiles = cluster_individuals(df, n_clusters=N_CLUSTERS)
    
    # Auto-generate readable cluster labels
    cluster_labels = auto_label_clusters(profiles)
    profiles["label"] = profiles.index.map(cluster_labels)
    
    print("\nCluster profiles:")
    display_cols = ["label","n_total","pct_of_total","age","years_homeless",
                    "mental_health","substance_use","outdoor_sleeping","chronic_homeless",
                    "indigenous_flag","youth"]
    print(profiles[display_cols].to_string())
    
    # ── Forecast cluster counts ────────────────────────────────────────────────
    print("\nStep 3: Forecasting cluster size over time...")
    cluster_forecast = forecast_cluster_counts(df)
    
    # Show forecast for target years
    fc_show = cluster_forecast[
        cluster_forecast["year"].isin(FORECAST_YEARS) & cluster_forecast["is_forecast"]
    ][["year","cluster","pred_pct"]].copy()
    fc_show["cluster_label"] = fc_show["cluster"].map(cluster_labels)
    fc_show["pred_pct"] = (fc_show["pred_pct"] * 100).round(1)
    print("\nPredicted cluster proportions for 2024-2026:")
    print(fc_show.to_string(index=False))
    
    # ── Forecast service needs ─────────────────────────────────────────────────
    print("\nStep 4: Forecasting service need rates by cluster...")
    service_forecast = forecast_service_needs(df)
    
    # Show 2024 service needs per cluster
    svc_2024 = service_forecast[
        (service_forecast["year"] == 2024) & service_forecast["is_forecast"]
    ][["cluster","target","pred_rate"]].copy()
    svc_2024["cluster_label"] = svc_2024["cluster"].map(cluster_labels)
    svc_2024["pred_rate_pct"] = (svc_2024["pred_rate"] * 100).round(1)
    print("\n2024 service need rates by cluster:")
    pivot = svc_2024.pivot(index="cluster_label", columns="target", values="pred_rate_pct")
    print(pivot.to_string())
    
    # ── Generate summary ───────────────────────────────────────────────────────
    print("\nStep 5: Generating human-readable summary...")
    summary = generate_summary(profiles, cluster_labels, cluster_forecast, service_forecast, df)
    print(summary)
    
    # ── Save outputs ───────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    df.to_csv("sasm_clusters.csv", index=False)
    profiles.to_csv("sasm_cluster_profiles.csv")
    cluster_forecast.to_csv("sasm_cluster_forecast.csv", index=False)
    service_forecast.to_csv("sasm_service_forecast.csv", index=False)
    
    with open("sasm_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    joblib.dump(km,     "sasm_kmeans_model.pkl")
    joblib.dump(scaler, "sasm_cluster_scaler.pkl")
    
    print("\nOutputs saved:")
    print("  sasm_clusters.csv           — individuals with cluster assignment")
    print("  sasm_cluster_profiles.csv   — mean features per cluster")
    print("  sasm_cluster_forecast.csv   — cluster size predictions 2024-2026")
    print("  sasm_service_forecast.csv   — service need rates by cluster+year")
    print("  sasm_analysis_summary.txt   — human-readable summary")
    print("  sasm_kmeans_model.pkl       — fitted clustering model")
    print("  sasm_cluster_scaler.pkl     — feature scaler for new data")


if __name__ == "__main__":
    main()
