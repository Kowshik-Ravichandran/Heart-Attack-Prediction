# ============================================================
# clinical_model.py — Train & Save Clinical Heart Disease Model
# ============================================================
# This script trains a Random Forest classifier on the Heart
# Disease dataset (heart.csv) and saves the model + scaler to
# disk for use in the web application.
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "clinical_rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "clinical_scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "clinical_features.pkl")

# Features used for clinical prediction
# These correspond to the user-input fields in the web app
CLINICAL_FEATURES = [
    "age",       # Age in years
    "sex",       # Sex (1 = male, 0 = female)
    "cp",        # Chest pain type (0-3)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",      # Serum cholesterol (mg/dl)
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # Resting ECG results (0-2)
    "thalach",   # Maximum heart rate achieved
    "exang",     # Exercise induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # Slope of peak exercise ST segment (0-2)
    "ca",        # Number of major vessels (0-4)
    "thal",      # Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect, 3 = not described)
]


def train_clinical_model():
    """
    Train a Random Forest model on the Heart Disease dataset.

    Steps:
      1. Load the CSV dataset
      2. Split into features (X) and target (y)
      3. Standardize features using StandardScaler
      4. Train a Random Forest classifier
      5. Evaluate and print metrics
      6. Save model, scaler, and feature names to disk

    Returns:
        tuple: (model, scaler, accuracy)
    """
    print("=" * 60)
    print("  CLINICAL HEART DISEASE MODEL TRAINING")
    print("=" * 60)

    # ── 1. Load data ──
    print(f"\n📂 Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}")
    print(f"   Target distribution:\n{df['target'].value_counts().to_string()}")

    # ── 2. Prepare features and labels ──
    X = df[CLINICAL_FEATURES].values
    y = df["target"].values

    print(f"\n📊 Features: {len(CLINICAL_FEATURES)}")
    print(f"   Samples: {len(y)}")
    print(f"   Positive (Heart Disease): {np.sum(y == 1)}")
    print(f"   Negative (No Disease):    {np.sum(y == 0)}")

    # ── 3. Train/Test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📋 Train set: {len(X_train)} | Test set: {len(X_test)}")

    # ── 4. Standardize features ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 5. Train Random Forest ──
    print("\n🌲 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # ── 6. Evaluate ──
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("📊 Feature Importance (Top 5):")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"   {CLINICAL_FEATURES[idx]:12s}: {importances[idx]:.4f}")

    # ── 7. Save model and scaler ──
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to: {MODEL_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to: {SCALER_PATH}")

    with open(FEATURE_NAMES_PATH, "wb") as f:
        pickle.dump(CLINICAL_FEATURES, f)
    print(f"💾 Feature names saved to: {FEATURE_NAMES_PATH}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE ✅")
    print("=" * 60)

    return model, scaler, accuracy


# ────────────────────────────────────────────────────────────
# RUN
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_clinical_model()
