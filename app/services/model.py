"""
Model service — loading, inference, and metadata access.

This module is the single point of contact for interacting with the
trained regression model.  It loads the pickled artifact from disk on
construction (or on first call via :func:`get_model`) and exposes three
read-only operations:

* :meth:`HousePriceModel.predict_single` — one property → one price
* :meth:`HousePriceModel.predict_batch` — N properties → N prices
* :meth:`HousePriceModel.get_model_info` — model metadata for review

The design follows the **service layer** pattern — the FastAPI routers
never touch scikit-learn or pandas directly.
"""

import pickle
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.utils.config import MODEL_PATH, DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class HousePriceModel:
    """Thin wrapper around the fitted regression pipeline.

    On instantiation the class eagerly loads ``model/model.pkl``.
    If the file is missing the object still constructs (``model is None``)
    and ``is_loaded`` returns ``False`` — callers are expected to
    check the flag before calling predict methods.

    Attributes:
        model: The fitted scikit-learn estimator (e.g. LinearRegression)
            or our numpy ``SimpleLinearRegression`` fallback.
        scaler: Optional fitted ``StandardScaler`` or ``SimpleScaler``.
        coefficients: Dict mapping feature name → learned weight.
        intercept: Bias term (float).
        metrics: Dict with R², RMSE, MAE from the training evaluation.
        training_date: ISO-8601 string of when the model was trained.
        n_samples: Number of rows used during training.
    """

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.coefficients: Dict[str, float] = {}
        self.intercept: float = 0.0
        self.metrics: Dict[str, float] = {}
        self.training_date: str | None = None
        self.n_samples: int = 0
        self._load_model()

    def _load_model(self) -> None:
        """Deserialise the pickled model artifact from ``MODEL_PATH``.

        The expected pickle structure is a dict with keys:
        ``model``, ``scaler`` (optional), ``coefficients``,
        ``intercept``, ``metrics``, ``training_date``, ``n_samples``.

        Raises:
            pickle.UnpicklingError / EOFError: If the file is corrupted.
        """
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data.get("scaler")
                self.coefficients = data.get("coefficients", {})
                self.intercept = data.get("intercept", 0.0)
                self.metrics = data.get("metrics", {})
                self.training_date = data.get("training_date")
                self.n_samples = data.get("n_samples", 0)
                logger.info("Model loaded successfully from %s", MODEL_PATH)
            except Exception as e:
                logger.error("Failed to load model: %s", e)
                raise
        else:
            logger.warning(
                "Model file not found at %s. Please run train.py first.", MODEL_PATH
            )

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` when the regression estimator is ready."""
        return self.model is not None

    def predict_single(self, features: Dict[str, Any]) -> float:
        """Predict the price of a single property.

        Builds a one-row DataFrame with the exact column order expected
        by the fitted scaler, scales the features, runs the model, and
        returns the prediction rounded to two decimals.

        Args:
            features: Dict keyed by feature name (must contain exactly
                the keys in ``FEATURE_COLUMNS``).

        Returns:
            Predicted price as a float rounded to 2 decimal places.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        # Build DataFrame with columns in the exact order the model was
        # trained on — scikit-learn pipelines are order-sensitive.
        feature_df = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])

        if self.scaler:
            feature_df = pd.DataFrame(
                self.scaler.transform(feature_df),
                columns=FEATURE_COLUMNS,
            )

        prediction = self.model.predict(feature_df)[0]
        return round(float(prediction), 2)

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Predict prices for multiple properties in a single call.

        Vectorised inference — the scaler and model are applied to all
        rows at once, which is dramatically faster than calling
        ``predict_single`` in a loop.

        Args:
            features_list: List of dicts, each with keys matching
                ``FEATURE_COLUMNS``.

        Returns:
            List of predicted prices, each rounded to 2 decimal places.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        feature_df = pd.DataFrame(
            [{col: f[col] for col in FEATURE_COLUMNS} for f in features_list]
        )

        if self.scaler:
            feature_df = pd.DataFrame(
                self.scaler.transform(feature_df),
                columns=FEATURE_COLUMNS,
            )

        predictions = self.model.predict(feature_df)
        return [round(float(p), 2) for p in predictions]

    def get_model_info(self) -> Dict[str, Any]:
        """Return a dictionary of model metadata for the /model-info endpoint.

        Returns:
            Dict with keys: model_type, coefficients, intercept,
            metrics, training_date, n_samples_trained, excluded_features.
        """
        return {
            "model_type": type(self.model).__name__,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "metrics": self.metrics,
            "training_date": self.training_date,
            "n_samples_trained": self.n_samples,
            "excluded_features": ["id", "price"],
        }


# ---------------------------------------------------------------------------
# Module-level singleton — one HousePriceModel instance for the entire process.
# A global is acceptable here because:
#   1. The model is immutable after loading (read-only at runtime).
#   2. Loading is fast (<10 ms) and happens once.
#   3. It avoids the overhead of re-parsing the pickle on every request.
# ---------------------------------------------------------------------------
_model_instance: HousePriceModel | None = None


def get_model() -> HousePriceModel:
    """Return the process-wide singleton ``HousePriceModel`` instance.

    Thread-safe enough for FastAPI's async concurrency model because
    the singleton is created exactly once at startup (inside the
    ``lifespan`` context manager) and never mutated afterwards.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = HousePriceModel()
    return _model_instance
