"""
Fallback training script using only numpy/pandas (no scikit-learn required).
Produces a model.pkl compatible with app/services/model.py.

Usage:
    python train_numpy.py
"""

import pickle
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.utils.config import DATA_PATH, MODEL_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from app.services.simple_model import SimpleScaler, SimpleLinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting numpy-based model training...")

    if not DATA_PATH.exists():
        logger.error("Dataset not found at %s", DATA_PATH)
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d samples", len(df))

    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df[TARGET_COLUMN].values.astype(float)

    n_total = len(df)
    n_train = int(n_total * 0.8)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_total)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    scaler = SimpleScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SimpleLinearRegression()
    model.fit(X_train_scaled, y_train)
    logger.info("Model trained successfully")

    y_pred = model.predict(X_test_scaled)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = round(float(1 - ss_res / ss_tot), 4)
    rmse = round(float(np.sqrt(np.mean((y_test - y_pred) ** 2))), 2)
    mae = round(float(np.mean(np.abs(y_test - y_pred))), 2)

    metrics = {"r_squared": r2, "rmse": rmse, "mae": mae}
    logger.info("Metrics: R2=%.4f, RMSE=%.2f, MAE=%.2f", r2, rmse, mae)

    coefficients = {
        col: round(float(coef), 4)
        for col, coef in zip(FEATURE_COLUMNS, model.coef_)
    }
    intercept = round(float(model.intercept_), 4)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_date = datetime.now(timezone.utc).isoformat()

    data = {
        "model": model,
        "scaler": scaler,
        "coefficients": coefficients,
        "intercept": intercept,
        "metrics": metrics,
        "training_date": training_date,
        "n_samples": n_total,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)

    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
