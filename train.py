"""
Training script for the Housing Price Prediction model.

Usage:
    python train.py

This script:
1. Loads the housing dataset from data/housing.csv
2. Prepares features and target
3. Trains a LinearRegression model (sklearn or numpy fallback)
4. Evaluates on test set
5. Saves the model to model/model.pkl
"""

import pickle
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.utils.config import DATA_PATH, MODEL_PATH, FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _train_with_sklearn(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train split: %d samples, Test split: %d samples", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    logger.info("Model trained with scikit-learn")

    y_pred = model.predict(X_test_scaled)

    r2 = round(r2_score(y_test, y_pred), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 2)

    coefficients = {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLUMNS, model.coef_)}
    intercept = round(float(model.intercept_), 4)

    return model, scaler, coefficients, intercept, {"r_squared": r2, "rmse": rmse, "mae": mae}


def _train_with_numpy(df):
    from app.services.simple_model import SimpleScaler, SimpleLinearRegression

    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df[TARGET_COLUMN].values.astype(float)

    n_total = len(df)
    n_train = int(n_total * 0.8)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_total)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info("Train split: %d samples, Test split: %d samples", len(X_train), len(X_test))

    scaler = SimpleScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SimpleLinearRegression()
    model.fit(X_train_scaled, y_train)
    logger.info("Model trained with numpy fallback")

    y_pred = model.predict(X_test_scaled)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = round(float(1 - ss_res / ss_tot), 4)
    rmse = round(float(np.sqrt(np.mean((y_test - y_pred) ** 2))), 2)
    mae = round(float(np.mean(np.abs(y_test - y_pred))), 2)

    coefficients = {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLUMNS, model.coef_)}
    intercept = round(float(model.intercept_), 4)

    return model, scaler, coefficients, intercept, {"r_squared": r2, "rmse": rmse, "mae": mae}


def main():
    logger.info("Starting model training...")

    if not DATA_PATH.exists():
        logger.error("Dataset not found at %s", DATA_PATH)
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d samples with columns: %s", len(df), list(df.columns))

    try:
        model, scaler, coefficients, intercept, metrics = _train_with_sklearn(df)
    except ImportError:
        logger.warning("scikit-learn not available, falling back to numpy...")
        model, scaler, coefficients, intercept, metrics = _train_with_numpy(df)

    logger.info("Metrics: R2=%.4f, RMSE=%.2f, MAE=%.2f", metrics["r_squared"], metrics["rmse"], metrics["mae"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_date = datetime.now(timezone.utc).isoformat()

    data = {
        "model": model,
        "scaler": scaler,
        "coefficients": coefficients,
        "intercept": intercept,
        "metrics": metrics,
        "training_date": training_date,
        "n_samples": len(df),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)

    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
