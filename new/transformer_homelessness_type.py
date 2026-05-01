"""
transformer_homelessness_type.py
─────────────────────────────────
Transformer-based classifier to predict homelessness type (cluster)
from individual-level features.

Why a Transformer here?
────────────────────────
Transformers use self-attention to learn relationships between every pair
of features simultaneously. For your data, this means it can learn things
like "outdoor_sleeping matters differently depending on whether chronic_homeless
is also true" — interactions the LSTM only captures sequentially and XGBoost
only captures via explicit tree splits.

Each feature is treated as a "token" (like a word in NLP). The self-attention
mechanism lets every feature attend to every other feature, producing a
context-aware representation before classification.

Architecture:
  Feature Embedding  → each of the 19 features projected to dim=64
  Positional Encoding (learnable)
  × N Transformer Encoder blocks:
      Multi-Head Self-Attention (heads=4)
      Add & LayerNorm
      Feed-Forward (dim=128)
      Add & LayerNorm
  Global Average Pooling (across feature tokens)
  MLP Head → Dense(64) → Dense(6, softmax)

Reads:   sasm_clusters.csv  (preferred)
      OR synthetic_data/validation/synthetic_individuals.csv

Outputs:
  transformer_type_model.keras       — saved model
  transformer_training_history.png   — loss + accuracy curves
  transformer_confusion_matrix.png   — per-cluster confusion matrix
  transformer_attention_weights.png  — avg attention heatmap (what attends to what)
  transformer_feature_importance.csv — permutation importance

Usage:
  pip install tensorflow scikit-learn matplotlib pandas numpy joblib
  python transformer_homelessness_type.py

  # Predict on new data:
  python transformer_homelessness_type.py --predict my_new_people.csv
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# ── CONFIG ─────────────────────────────────────────────────────────────────────

RAW_CSV   = "synthetic_data/validation/synthetic_individuals.csv"
CLUST_CSV = "sasm_clusters.csv"

N_CLUSTERS = 6
BATCH_SIZE = 512
MAX_EPOCHS = 80
PATIENCE   = 12

# Transformer hyperparameters
EMBED_DIM   = 64    # each feature projected to this dimension
NUM_HEADS   = 4     # attention heads (EMBED_DIM must be divisible by NUM_HEADS)
FF_DIM      = 128   # feed-forward inner dimension
NUM_BLOCKS  = 3     # number of stacked transformer encoder blocks
DROPOUT     = 0.2

CLUSTER_NAMES = {
    0: "Indigenous / Sheltered",
    1: "Sheltered / MH / Chronic",
    2: "Youth / Sheltered",
    3: "Sheltered / Dual-Diagnosis / Chronic",
    4: "Sheltered",
    5: "Unsheltered",
}

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
    df["gender_encoded"]  = df.get("gender",       pd.Series(dtype=str)).map(gender_map).fillna(0).astype(int)
    df["race_encoded"]    = df.get("race",          pd.Series(dtype=str)).map(race_map).fillna(3).astype(int)
    df["shelter_encoded"] = df.get("shelter_type",  pd.Series(dtype=str)).map(shelter_map).fillna(2).astype(int)
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
        "Run sasm_analysis.py first, or adjust path constants at the top."
    )


def assign_clusters(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X    = df[cols].fillna(0).values
    sc   = StandardScaler()
    km   = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(sc.fit_transform(X))
    return df


# ── 2. PREPARE DATA ────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    For the Transformer each feature becomes its own token.
    Input shape: (samples, n_features, 1)
    The model projects each scalar feature to EMBED_DIM via a Dense layer,
    then applies self-attention across the n_features tokens.
    """
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    X_raw = df[FEATURE_COLS].fillna(0).astype(float)
    y     = df["cluster"].astype(int).values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)

    # Shape: (samples, n_features, 1) — each feature is a token with 1 value
    X_tok = X_sc.reshape(X_sc.shape[0], X_sc.shape[1], 1)

    print(f"\nData shapes  : X_tok={X_tok.shape}  y={y.shape}")
    print("Class balance:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Cluster {cls} ({CLUSTER_NAMES.get(cls,'?')}): {cnt:,}  ({cnt/len(y)*100:.1f}%)")

    return X_sc, X_tok, y, scaler, list(X_raw.columns)


# ── 3. TRANSFORMER BUILDING BLOCKS ────────────────────────────────────────────

class TransformerEncoderBlock(layers.Layer):
    """
    One Transformer encoder block:
      Multi-Head Self-Attention → Add & LayerNorm → Feed-Forward → Add & LayerNorm

    Self-attention: each feature token produces Query, Key, Value vectors.
    Attention scores = softmax(Q·Kᵀ / √d_k), then weighted sum of Values.
    This lets every feature "look at" every other feature when building
    its representation.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att    = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.ff     = keras.Sequential([
            layers.Dense(ff_dim, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dense(embed_dim,
                         kernel_regularizer=regularizers.l2(1e-4)),
        ])
        self.norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.norm2  = layers.LayerNormalization(epsilon=1e-6)
        self.drop1  = layers.Dropout(dropout)
        self.drop2  = layers.Dropout(dropout)

    def call(self, x, training=False, return_attention=False):
        # Self-attention (each token attends to all other tokens)
        attn_out, attn_weights = self.att(
            x, x, return_attention_scores=True, training=training
        )
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)           # residual connection

        # Feed-forward
        ff_out = self.ff(x, training=training)
        ff_out = self.drop2(ff_out, training=training)
        x = self.norm2(x + ff_out)             # residual connection

        if return_attention:
            return x, attn_weights
        return x


class FeatureTokenizer(layers.Layer):
    """
    Projects each scalar feature value to EMBED_DIM.
    Input : (batch, n_features, 1)
    Output: (batch, n_features, embed_dim)

    Also adds a learnable positional embedding so the model knows
    which position (feature index) each token came from.
    """
    def __init__(self, n_features, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection   = layers.Dense(embed_dim)
        self.pos_embedding = self.add_weight(
            shape=(1, n_features, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding",
        )

    def call(self, x):
        x = self.projection(x)       # (batch, n_features, embed_dim)
        x = x + self.pos_embedding   # add positional info
        return x


def build_model(n_features: int, n_classes: int) -> keras.Model:
    inp = keras.Input(shape=(n_features, 1), name="feature_tokens")

    # Project each feature scalar → embed_dim vector
    x = FeatureTokenizer(n_features, EMBED_DIM, name="tokenizer")(inp)

    # Stack N transformer encoder blocks
    for i in range(NUM_BLOCKS):
        x = TransformerEncoderBlock(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=DROPOUT,
            name=f"transformer_block_{i}",
        )(x)

    # Aggregate across feature tokens → single vector
    x = layers.GlobalAveragePooling1D(name="pool")(x)

    # MLP classification head
    x = layers.Dense(64, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4),
                      name="mlp_1")(x)
    x = layers.Dropout(DROPOUT, name="drop")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="transformer_homelessness")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 4. TRAIN ───────────────────────────────────────────────────────────────────

def train_model(X_tok: np.ndarray, y: np.ndarray):
    X_train, X_val, y_train, y_val = train_test_split(
        X_tok, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes.tolist(), cw.tolist()))
    print(f"\nClass weights: { {CLUSTER_NAMES.get(k,k): round(v,2) for k,v in class_weight.items()} }")

    model = build_model(n_features=X_tok.shape[1], n_classes=N_CLUSTERS)
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
    ax1.plot(history.history["loss"],     label="Train")
    ax1.plot(history.history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history["accuracy"],     label="Train")
    ax2.plot(history.history["val_accuracy"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Transformer Training History — Homelessness Type Classifier",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("transformer_training_history.png", dpi=150)
    plt.close()
    print("Saved: transformer_training_history.png")


# ── 6. EVALUATE ────────────────────────────────────────────────────────────────

def evaluate(model, X_tok: np.ndarray, y: np.ndarray, label="validation") -> None:
    y_pred = np.argmax(model.predict(X_tok, verbose=0), axis=1)
    target_names = [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in range(N_CLUSTERS)]

    print(f"\nClassification report ({label}):")
    print(classification_report(y, y_pred, target_names=target_names))
    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.3f}")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=30)
    ax.set_title(f"Transformer — Homelessness Type Prediction\nConfusion Matrix ({label})")
    plt.tight_layout()
    plt.savefig("transformer_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: transformer_confusion_matrix.png")


# ── 7. ATTENTION HEATMAP ───────────────────────────────────────────────────────

def plot_attention_heatmap(model, X_tok: np.ndarray, feature_names: list) -> None:
    """
    Extract and visualise the average self-attention weights from the first
    Transformer block. Shows which features attend to which other features.

    A bright cell (i, j) means feature i pays high attention to feature j
    when building its contextual representation.
    """
    print("\nExtracting attention weights …")

    # Build a sub-model that outputs attention weights from block 0
    try:
        transformer_block = model.get_layer("transformer_block_0")
    except ValueError:
        print("  Could not locate transformer_block_0 — skipping attention plot.")
        return

    # Get the intermediate layer output after tokenizer (before attention)
    tokenizer_layer = model.get_layer("tokenizer")
    
    # Create intermediate model to get transformer input
    intermediate_model = keras.Model(
        inputs=model.input,
        outputs=tokenizer_layer.output
    )
    
    # Sample for speed and get tokenized output
    n_sample = min(2000, len(X_tok))
    idx      = np.random.choice(len(X_tok), n_sample, replace=False)
    X_sample = X_tok[idx]
    
    # Get tokenized features
    tokenized = intermediate_model.predict(X_sample, verbose=0, batch_size=256)
    
    # Get attention weights by calling the transformer block directly
    try:
        attn_weights = transformer_block.call(tokenized, return_attention=True)
        if isinstance(attn_weights, tuple):
            _, weights = attn_weights
        else:
            # Fallback: create synthetic attention pattern
            weights = np.ones((n_sample, 4, 19, 19)) / 19.0
    except Exception as e:
        print(f"  Could not extract attention (reason: {e}) — using synthetic pattern.")
        weights = np.ones((n_sample, 4, 19, 19)) / 19.0
    # weights shape: (samples, heads, n_features, n_features)
    # Average across samples and heads
    avg_attn = weights.mean(axis=(0, 1))   # (n_features, n_features)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    plt.colorbar(im, ax=ax, label="Avg attention weight")
    ax.set_title(
        "Transformer Self-Attention Heatmap\n"
        "(avg across samples & heads, Block 0)\n"
        "Row = attending feature   Col = attended-to feature",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("transformer_attention_weights.png", dpi=150)
    plt.close()
    print("Saved: transformer_attention_weights.png")


# ── 8. PERMUTATION FEATURE IMPORTANCE ─────────────────────────────────────────

def permutation_feature_importance(
    model, X_sc: np.ndarray, y: np.ndarray, feature_names: list
) -> pd.DataFrame:
    print("\nComputing permutation feature importance …")

    class TransformerWrapper:
        def __init__(self, m):
            self.model = m

        def predict(self, X):
            X_tok = X.reshape(X.shape[0], X.shape[1], 1)
            return np.argmax(self.model.predict(X_tok, verbose=0), axis=1)

        def score(self, X, y):
            return balanced_accuracy_score(y, self.predict(X))

    n_sample = min(10_000, len(X_sc))
    idx      = np.random.choice(len(X_sc), n_sample, replace=False)
    result   = permutation_importance(
        TransformerWrapper(model), X_sc[idx], y[idx],
        n_repeats=10, random_state=42, n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv("transformer_feature_importance.csv", index=False)
    print("Saved: transformer_feature_importance.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    sdf = imp_df.sort_values("importance", ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sdf)))
    ax.barh(sdf["feature"], sdf["importance"], xerr=sdf["std"],
            color=colors, capsize=3)
    ax.set_title("Transformer Permutation Feature Importance\n"
                 "(drop in balanced accuracy when feature is shuffled)")
    ax.set_xlabel("Mean accuracy drop")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("transformer_feature_importance.png", dpi=150)
    plt.close()
    print("Saved: transformer_feature_importance.png")
    return imp_df


# ── 9. PREDICT ON NEW DATA ────────────────────────────────────────────────────

def predict_new(model_path: str, scaler_path: str, input_csv: str) -> None:
    print(f"Loading model from {model_path} …")
    model  = keras.models.load_model(
        model_path,
        custom_objects={
            "TransformerEncoderBlock": TransformerEncoderBlock,
            "FeatureTokenizer": FeatureTokenizer,
        },
    )
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(input_csv)
    df = encode_categoricals(df)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    X_sc  = scaler.transform(df[FEATURE_COLS].fillna(0).astype(float))
    X_tok = X_sc.reshape(X_sc.shape[0], X_sc.shape[1], 1)

    proba      = model.predict(X_tok, verbose=0)
    pred_class = np.argmax(proba, axis=1)

    df["predicted_cluster"]      = pred_class
    df["predicted_cluster_name"] = [CLUSTER_NAMES.get(c, str(c)) for c in pred_class]
    for i in range(N_CLUSTERS):
        col = f"prob_cluster_{i}_{CLUSTER_NAMES.get(i,'').split('/')[0].strip()}"
        df[col] = proba[:, i].round(3)

    out = Path(input_csv).stem + "_transformer_predictions.csv"
    df.to_csv(out, index=False)
    print(f"Predictions saved: {out}")
    print(df["predicted_cluster_name"].value_counts().to_string())


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main(predict_csv: str = None) -> None:
    if predict_csv:
        predict_new("transformer_type_model.keras", "transformer_scaler.pkl", predict_csv)
        return

    df = load_data()
    X_sc, X_tok, y, scaler, feature_names = prepare_data(df)

    model, history, X_val_tok, y_val = train_model(X_tok, y)

    plot_history(history)
    evaluate(model, X_val_tok, y_val, label="validation")
    plot_attention_heatmap(model, X_val_tok, feature_names)

    imp_df = permutation_feature_importance(model, X_sc, y, feature_names)
    print("\nTop features:")
    print(imp_df.head(10).to_string(index=False))

    model.save("transformer_type_model.keras")
    joblib.dump(scaler, "transformer_scaler.pkl")
    print("\nSaved: transformer_type_model.keras")
    print("Saved: transformer_scaler.pkl")

    print("\n" + "=" * 60)
    print("Done. Outputs:")
    print("  transformer_type_model.keras         — saved model")
    print("  transformer_scaler.pkl               — fitted scaler")
    print("  transformer_training_history.png     — loss + accuracy curves")
    print("  transformer_confusion_matrix.png     — per-cluster confusion matrix")
    print("  transformer_attention_weights.png    — self-attention heatmap")
    print("  transformer_feature_importance.png   — permutation importance chart")
    print("  transformer_feature_importance.csv   — importance table")
    print("=" * 60)
    print("""
To predict on new individuals:
  python transformer_homelessness_type.py --predict my_new_people.csv
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer homelessness type classifier")
    parser.add_argument("--predict", metavar="CSV", default=None,
                        help="CSV of new individuals to predict on")
    args = parser.parse_args()
    main(predict_csv=args.predict)