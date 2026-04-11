import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


def train_and_forecast(ry: pd.DataFrame, observed_years: list, forecast_years: list) -> pd.DataFrame:
    base_features = [
        "year", "age_avg", "years_homeless_avg", "pct_mental_health", "pct_substance_use", "pct_physical_health",
        "pct_outdoor_sleeping", "pct_chronic", "pct_lgbtq", "pct_indigenous", "pct_immigrant", "pct_youth",
        "pct_no_income", "pct_foster_care", "pct_incarceration", "pct_housing_loss_income",
    ]
    extra_features = [c for c in ["flow_active_homeless", "newly_identified", "flow_growth_rate"] if c in ry.columns]
    feature_cols = base_features + extra_features

    train_df = ry[ry["year"] <= max(observed_years)].copy()
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df["true_total"].values

    ridge = Ridge(alpha=10.0).fit(X_train_sc, y_train)
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)
    gbr.fit(X_train_sc, y_train)

    obs_mask = train_df["year"].isin(observed_years)
    X_obs = scaler.transform(train_df.loc[obs_mask, feature_cols].values)
    y_obs = train_df.loc[obs_mask, "true_total"].values

    loo = LeaveOneOut()
    ridge_loo, gbr_loo = [], []
    for tr_idx, te_idx in loo.split(X_obs):
        r = Ridge(alpha=10.0).fit(X_obs[tr_idx], y_obs[tr_idx])
        g = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
        g.fit(X_obs[tr_idx], y_obs[tr_idx])
        ridge_loo.append(r.predict(X_obs[te_idx])[0])
        gbr_loo.append(g.predict(X_obs[te_idx])[0])

    print("\nLOO-CV on observed years:")
    print(f"  Ridge  MAE: {mean_absolute_error(y_obs, ridge_loo):,.0f}  R²: {r2_score(y_obs, ridge_loo):.3f}")
    print(f"  GBR    MAE: {mean_absolute_error(y_obs, gbr_loo):,.0f}  R²: {r2_score(y_obs, gbr_loo):.3f}")

    all_X_sc = scaler.transform(ry[feature_cols].values)
    results = ry[["year", "true_total"]].copy()
    results["ridge_pred"] = ridge.predict(all_X_sc).round(0).astype(int)
    results["gbr_pred"] = gbr.predict(all_X_sc).round(0).astype(int)
    results["ensemble"] = ((results["ridge_pred"] + results["gbr_pred"]) / 2).round(0).astype(int)
    results["observed"] = results["year"].isin(observed_years)

    future = results[results["year"].isin(forecast_years)][["year", "ridge_pred", "gbr_pred", "ensemble"]]
    print("\nForecast for future years:")
    print(future.to_string(index=False))

    return results
