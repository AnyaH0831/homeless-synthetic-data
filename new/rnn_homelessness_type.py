"""
rnn_homelessness_type.py
────────────────────────
LSTM-based model to predict homelessness type (cluster) from individual features.

Why an RNN/LSTM here?
─────────────────────
Your data has a 'year' dimension — the same population is observed across
2013–2026. An LSTM can treat each individual's feature vector as a timestep
in a sequence, learning how the *combination* of features interacts temporally.
More practically, it learns non-linear feature interactions that a single
tree split can't capture in one step.

The input is shaped as (samples, timesteps=1, features=N). Using timesteps=1
means we're using the LSTM as a deep feature extractor rather than a true
sequence model — this is the right approach when you have one snapshot per
person rather than a longitudinal record per person.

If you later have multiple records per person over time, change build_sequences()
to group by person_id and feed real sequences.

Reads:   sasm_clusters.csv  (preferred — already has cluster labels)
      OR synthetic_data/validation/synthetic_individuals.csv

Outputs:
  rnn_type_model.keras         — saved Keras model
  rnn_type_model_weights.h5    — weights only (for portability)
  rnn_training_history.png     — loss + accuracy curves
  rnn_confusion_matrix.png     — per-cluster confusion matrix
  rnn_feature_importance.png   — permutation importance (model-agnostic)
  rnn_feature_importance.csv   — importance table

Usage:
  pip install tensorflow scikit-learn matplotlib pandas numpy
  python rnn_homelessness_type.py

  # Predict on new data:
  python rnn_homelessness_type.py --predict my_new_people.csv
"""

import argparse
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info logs
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# ── CONFIG ─────────────────────────────────────────────────────────────────────

RAW_CSV   = "synthetic_data/validation/synthetic_individuals.csv"
CLUST_CSV = "sasm_clusters.csv"

N_CLUSTERS  = 6
BATCH_SIZE  = 512
MAX_EPOCHS  = 80
PATIENCE    = 10     # early stopping patience

CLUSTER_NAMES = {
    0: "Indigenous / Sheltered",
    1: "Sheltered / MH / Chronic",
    2: "Youth / Sheltered",
    3: "Sheltered / Dual-Diagnosis / Chronic",
    4: "Sheltered",
    5: "Unsheltered",
}

# Raw feature columns (same as XGBoost script)
FEATURE_COLS = [
    "age",
    "years_homeless",
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
    "gender_encoded",
    "race_encoded",
    "shelter_encoded",
]


# ── 1. LOAD & ENCODE ───────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    gender_map  = {"male": 0, "female": 1, "trans_nonbinary": 2}
    race_map    = {"black": 0, "white": 1, "indigenous": 2, "other": 3}
    shelter_map = {"emergency_shelter": 0, "respite": 1, "other": 2, "outdoor": 3}

    df["gender_encoded"]  = df.get("gender",  pd.Series(dtype=str)).map(gender_map).fillna(0).astype(int)
    df["race_encoded"]    = df.get("race",     pd.Series(dtype=str)).map(race_map).fillna(3).astype(int)
    df["shelter_encoded"] = df.get("shelter_type", pd.Series(dtype=str)).map(shelter_map).fillna(2).astype(int)
    return df


def load_data() -> pd.DataFrame:
    for path, label in [(CLUST_CSV, "pre-clustered"), (RAW_CSV, "raw")]:
        if Path(path).exists():
            print(f"Loading {label} data from {path} …")
            df = pd.read_csv(path)
            df = encode_categoricals(df)
            if "cluster" not in df.columns:
                print(f"  No cluster column — running KMeans (k={N_CLUSTERS}) …")
                df = assign_clusters(df)
            print(f"  {len(df):,} records loaded.")
            return df
    raise FileNotFoundError(
        f"Neither '{CLUST_CSV}' nor '{RAW_CSV}' found.\n"
        "Run sasm_analysis.py first, or adjust path constants at the top of this file."
    )


def assign_clusters(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_sc)
    return df


# ── 2. PREPARE DATA ────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Returns X (scaled, shape: n_samples × n_features),
            X_seq (shape: n_samples × 1 × n_features)  ← LSTM input
            y (integer class labels)
            scaler  (fitted StandardScaler for reuse at inference)
            feature_names (list)
    """
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    X_raw = df[FEATURE_COLS].fillna(0).astype(float)
    y     = df["cluster"].astype(int).values

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_raw)

    # LSTM expects (samples, timesteps, features)
    # timesteps=1: one snapshot per person
    X_seq = X_sc.reshape(X_sc.shape[0], 1, X_sc.shape[1])

    print(f"\nData shapes  : X_seq={X_seq.shape}  y={y.shape}")
    print(f"Class balance:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Cluster {cls} ({CLUSTER_NAMES.get(cls,'?')}): {cnt:,}  ({cnt/len(y)*100:.1f}%)")

    return X_sc, X_seq, y, scaler, list(X_raw.columns)


# ── 3. BUILD MODEL ─────────────────────────────────────────────────────────────

def build_model(n_features: int, n_classes: int) -> keras.Model:
    """
    Architecture:
      Input (1 timestep × n_features)
        → LSTM(128, return_sequences=True)   extract temporal/interaction patterns
        → Dropout(0.3)
        → LSTM(64)                           compress to fixed-size representation
        → Dropout(0.3)
        → Dense(64, relu) + L2              non-linear mixing
        → Dropout(0.2)
        → Dense(n_classes, softmax)          probability per cluster

    Why two LSTM layers?
    The first layer (return_sequences=True) passes its full hidden state at every
    timestep to the second layer. With timesteps=1 this is equivalent to a deep
    MLP with recurrent weight structure — it learns richer feature interactions
    than a single layer.

    Why not a plain MLP?
    You could. But the LSTM's gating mechanism (input/forget/output gates) acts as
    a learned feature selector at each layer, which often works better than a dense
    layer on tabular data with many binary/categorical features like yours.
    """
    inp = keras.Input(shape=(1, n_features), name="input")

    x = layers.LSTM(
        128,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-4),
        name="lstm_1",
    )(inp)
    x = layers.Dropout(0.3, name="drop_1")(x)

    x = layers.LSTM(
        64,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(1e-4),
        name="lstm_2",
    )(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = layers.Dropout(0.2, name="drop_3")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="rnn_homelessness_type")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 4. TRAIN ───────────────────────────────────────────────────────────────────

def train_model(X_seq: np.ndarray, y: np.ndarray):
    """
    Train with:
      - 80/20 stratified train/val split
      - Early stopping on val_loss (restores best weights)
      - ReduceLROnPlateau to decay learning rate when progress stalls
      - Class weights to handle imbalance (Cluster 1 is only 5.7%)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights — inverse frequency
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, cw))
    print(f"\nClass weights: { {CLUSTER_NAMES.get(k,k): round(v,2) for k,v in class_weight.items()} }")

    model = build_model(n_features=X_seq.shape[2], n_classes=N_CLUSTERS)
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
    ]

    print(f"\nTraining (max {MAX_EPOCHS} epochs, early stopping patience={PATIENCE}) …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cb_list,
        verbose=1,
    )

    return model, history, X_val, y_val


# ── 5. PLOT TRAINING HISTORY ───────────────────────────────────────────────────

def plot_history(history) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"],     label="Train loss")
    ax1.plot(history.history["val_loss"], label="Val loss")
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sparse categorical crossentropy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history["accuracy"],     label="Train acc")
    ax2.plot(history.history["val_accuracy"], label="Val acc")
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("RNN Training History — Homelessness Type Classifier", fontweight="bold")
    plt.tight_layout()
    plt.savefig("rnn_training_history.png", dpi=150)
    plt.close()
    print("Saved: rnn_training_history.png")


# ── 6. EVALUATE ────────────────────────────────────────────────────────────────

def evaluate(model, X_seq: np.ndarray, y: np.ndarray, label: str = "validation") -> None:
    y_pred = np.argmax(model.predict(X_seq, verbose=0), axis=1)
    target_names = [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in range(N_CLUSTERS)]

    print(f"\nClassification report ({label}):")
    print(classification_report(y, y_pred, target_names=target_names))
    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.3f}")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=30)
    ax.set_title(f"RNN — Homelessness Type Prediction\nConfusion Matrix ({label})")
    plt.tight_layout()
    plt.savefig("rnn_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: rnn_confusion_matrix.png")


# ── 7. PERMUTATION FEATURE IMPORTANCE ─────────────────────────────────────────

def permutation_feature_importance(
    model, X_sc: np.ndarray, y: np.ndarray, feature_names: list
) -> pd.DataFrame:
    """
    Keras models aren't sklearn estimators, so we wrap the model in a
    sklearn-compatible class to use permutation_importance.

    Permutation importance: for each feature, randomly shuffle its values
    across all samples and measure how much accuracy drops. A large drop
    means the model relies heavily on that feature.

    This is model-agnostic and works for any black-box model.
    """
    print("\nComputing permutation feature importance (this may take a minute) …")

    class KerasClassifierWrapper:
        def __init__(self, keras_model):
            self.model = keras_model

        def fit(self, X, y):
            """Dummy fit method for sklearn compatibility."""
            return self

        def predict(self, X):
            X_seq = X.reshape(X.shape[0], 1, X.shape[1])
            return np.argmax(self.model.predict(X_seq, verbose=0), axis=1)

        def score(self, X, y):
            preds = self.predict(X)
            return balanced_accuracy_score(y, preds)

    # Use a sample for speed
    n_sample = min(10_000, len(X_sc))
    idx = np.random.choice(len(X_sc), n_sample, replace=False)
    X_sample = X_sc[idx]
    y_sample = y[idx]

    wrapper = KerasClassifierWrapper(model)
    result  = permutation_importance(
        wrapper, X_sample, y_sample,
        n_repeats=10, random_state=42, n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False)

    imp_df.to_csv("rnn_feature_importance.csv", index=False)
    print("Saved: rnn_feature_importance.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_df = imp_df.sort_values("importance", ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_df)))
    ax.barh(sorted_df["feature"], sorted_df["importance"],
            xerr=sorted_df["std"], color=colors, capsize=3)
    ax.set_title("RNN Permutation Feature Importance\n(drop in balanced accuracy when feature is shuffled)")
    ax.set_xlabel("Mean accuracy drop")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("rnn_feature_importance.png", dpi=150)
    plt.close()
    print("Saved: rnn_feature_importance.png")

    return imp_df


# ── 8. PREDICT ON NEW DATA ────────────────────────────────────────────────────

def predict_new(model_path: str, input_csv: str) -> None:
    print(f"\nLoading model from {model_path} …")
    model = keras.models.load_model(model_path)

    df = pd.read_csv(input_csv)
    df = encode_categoricals(df)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    X_raw = df[FEATURE_COLS].fillna(0).astype(float)
    scaler = StandardScaler()          # Note: for production, save/load the scaler too
    X_sc  = scaler.fit_transform(X_raw)
    X_seq = X_sc.reshape(X_sc.shape[0], 1, X_sc.shape[1])

    proba       = model.predict(X_seq, verbose=0)
    pred_class  = np.argmax(proba, axis=1)

    df["predicted_cluster"]      = pred_class
    df["predicted_cluster_name"] = [CLUSTER_NAMES.get(c, str(c)) for c in pred_class]
    for i in range(N_CLUSTERS):
        col = f"prob_cluster_{i}_{CLUSTER_NAMES.get(i,'').split('/')[0].strip()}"
        df[col] = proba[:, i].round(3)

    out = Path(input_csv).stem + "_rnn_predictions.csv"
    df.to_csv(out, index=False)
    print(f"Predictions saved: {out}")
    print(df["predicted_cluster_name"].value_counts().to_string())


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main(predict_csv: str = None) -> None:

    if predict_csv:
        predict_new("rnn_type_model.keras", predict_csv)
        return

    # Load
    df = load_data()
    X_sc, X_seq, y, scaler, feature_names = prepare_data(df)

    # Train
    model, history, X_val_seq, y_val = train_model(X_seq, y)

    # Plots
    plot_history(history)

    # Evaluate on validation set
    evaluate(model, X_val_seq, y_val, label="validation")

    # Feature importance (permutation, model-agnostic)
    imp_df = permutation_feature_importance(model, X_sc, y, feature_names)
    print("\nTop features by permutation importance:")
    print(imp_df.head(10).to_string(index=False))

    # Save model
    model.save("rnn_type_model.keras")
    model.save_weights("rnn_type_model_weights.weights.h5")
    import joblib
    joblib.dump(scaler, "rnn_scaler.pkl")   # save scaler for inference
    print("\nSaved: rnn_type_model.keras")
    print("Saved: rnn_type_model_weights.weights.h5")
    print("Saved: rnn_scaler.pkl")

    print("\n" + "=" * 60)
    print("Done. Outputs:")
    print("  rnn_type_model.keras         — full saved model")
    print("  rnn_type_model_weights.h5    — weights only")
    print("  rnn_scaler.pkl               — fitted feature scaler")
    print("  rnn_training_history.png     — loss + accuracy curves")
    print("  rnn_confusion_matrix.png     — per-cluster confusion matrix")
    print("  rnn_feature_importance.png   — permutation importance chart")
    print("  rnn_feature_importance.csv   — importance table")
    print("=" * 60)
    print("""
To predict on new individuals:
  python rnn_homelessness_type.py --predict my_new_people.csv
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNN homelessness type classifier")
    parser.add_argument(
        "--predict", metavar="CSV", default=None,
        help="Path to new individuals CSV to predict (uses saved model)"
    )
    args = parser.parse_args()
    main(predict_csv=args.predict)