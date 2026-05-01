import pickle, pandas as pd

with open("youth_chronic_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("synthetic_data/synthetic_individuals.csv")
df = df[df[model["youth_col"]] == 1]
X = df[model["feature_cols"]].fillna(df[model["feature_cols"]].median())
X_scaled = model["scaler"].transform(X)

df["chronic_pred"] = model["xgb"].predict(X_scaled)
df["chronic_prob"] = model["xgb"].predict_proba(X_scaled)[:, 1].round(3)
df.to_csv("youth_chronic_predictions.csv", index=False)