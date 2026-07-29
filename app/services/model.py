import pickle
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from app.utils.config import MODEL_PATH, DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class HousePriceModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.coefficients = {}
        self.intercept = 0.0
        self.metrics = {}
        self.training_date = None
        self.n_samples = 0
        self._load_model()

    def _load_model(self):
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
            logger.warning("Model file not found at %s. Please run train.py first.", MODEL_PATH)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_single(self, features: Dict[str, Any]) -> float:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        feature_values = np.array([[features[col] for col in FEATURE_COLUMNS]])

        if self.scaler:
            feature_values = self.scaler.transform(feature_values)

        prediction = self.model.predict(feature_values)[0]
        return round(float(prediction), 2)

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        feature_values = np.array(
            [[features[col] for col in FEATURE_COLUMNS] for features in features_list]
        )

        if self.scaler:
            feature_values = self.scaler.transform(feature_values)

        predictions = self.model.predict(feature_values)
        return [round(float(p), 2) for p in predictions]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": type(self.model).__name__,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "metrics": self.metrics,
            "training_date": self.training_date,
            "n_samples_trained": self.n_samples,
            "excluded_features": ["id", "price"],
        }


_model_instance = None


def get_model() -> HousePriceModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = HousePriceModel()
    return _model_instance
