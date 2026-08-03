"""
Training script for the Housing Price Prediction model.

Usage:
    python train.py

Workflow:
    1. Load the housing dataset from ``data/housing.csv``.
    2. Split 80/20 train/test with a fixed random seed (42) so results
       are reproducible across runs.
    3. Train a LinearRegression model with StandardScaler via scikit-learn.
    4. Evaluate with R², RMSE, MAE on the held-out test set.
    5. Serialise everything to ``model/model.pkl`` as a single dict.
"""

import pickle
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.utils.config import DATA_PATH, MODEL_PATH, FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Load data, train model, and persist the artifact to disk.

    Raises:
        FileNotFoundError: If ``data/housing.csv`` is missing.
    """
    logger.info("Starting model training...")

    if not DATA_PATH.exists():
        logger.error("Dataset not found at %s", DATA_PATH)
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    logger.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d samples with columns: %s", len(df), list(df.columns))

    # Extract feature matrix X and target vector y.
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Deterministic 80/20 split — same seed every run for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train split: %d samples, Test split: %d samples", len(X_train), len(X_test))

    # Z-score normalisation — critical for linear regression when
    # features are on different scales.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    logger.info("Model trained with scikit-learn")

    # Evaluate on the held-out test set.
    y_pred = model.predict(X_test_scaled)

    r2 = round(r2_score(y_test, y_pred), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
    metrics = {"r_squared": r2, "rmse": rmse, "mae": mae}

    # Build human-readable coefficient dict for the /model-info endpoint.
    coefficients = {col: round(float(coef), 4) for col, coef in zip(FEATURE_COLUMNS, model.coef_)}
    intercept = round(float(model.intercept_), 4)

    logger.info(
        "Metrics: R2=%.4f, RMSE=%.2f, MAE=%.2f",
        metrics["r_squared"],
        metrics["rmse"],
        metrics["mae"],
    )

    # Bundle everything the service layer needs into a single pickle.
    # The model service (app/services/model.py) expects exactly this schema.
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
