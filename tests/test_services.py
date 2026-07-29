"""
Unit tests for the service layer — model.py, simple_model.py, config.py.

These tests validate the core business logic in isolation (no HTTP layer),
covering normal paths, edge cases, error conditions, and singleton
behaviour.  The model pickle at ``model/model.pkl`` is used as the
fixture for integration-style unit tests.

Run with:
    pytest tests/test_services.py -v
"""

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.services.simple_model import SimpleLinearRegression, SimpleScaler
from app.utils.config import (
    BASE_DIR,
    DATA_DIR,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_PATH,
    TARGET_COLUMN,
)


# ---------------------------------------------------------------------------
# SimpleScaler tests
# ---------------------------------------------------------------------------

class TestSimpleScaler:
    """Unit tests for the numpy SimpleScaler."""

    def test_fit_stores_mean_and_std(self):
        """fit() should compute mean_ and std_ from the training data."""
        scaler = SimpleScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler.fit(X)

        np.testing.assert_array_almost_equal(scaler.mean_, [3.0, 4.0])
        np.testing.assert_array_almost_equal(scaler.std_, [np.std([1, 3, 5]), np.std([2, 4, 6])])

    def test_transform_applies_z_score(self):
        """transform() should return (X - mean) / std."""
        scaler = SimpleScaler()
        X = np.array([[1.0], [3.0], [5.0]])
        scaler.fit(X)

        scaled = scaler.transform(np.array([[3.0]]))
        # mean=3, std=std of [1,3,5] ≈ 1.633
        expected = (3.0 - 3.0) / np.std([1, 3, 5])
        np.testing.assert_almost_equal(scaled[0, 0], expected)

    def test_zero_std_column_handled(self):
        """Constant columns (std=0) should not cause division-by-zero."""
        scaler = SimpleScaler()
        X = np.array([[5.0, 2.0], [5.0, 4.0], [5.0, 6.0]])
        scaler.fit(X)

        # First column is constant 5.0 → std is replaced with 1.0
        assert scaler.std_[0] == 1.0
        # Transform should not raise
        result = scaler.transform(X)
        assert not np.any(np.isnan(result))

    def test_fit_returns_self_for_chaining(self):
        """fit() should return self for method chaining compatibility."""
        scaler = SimpleScaler()
        X = np.array([[1.0], [2.0]])
        result = scaler.fit(X)
        assert result is scaler

    def test_single_sample_fit(self):
        """fit() with a single sample — std becomes 0 → replaced with 1.0."""
        scaler = SimpleScaler()
        X = np.array([[42.0, 100.0]])
        scaler.fit(X)
        assert scaler.std_[0] == 1.0
        assert scaler.std_[1] == 1.0

    def test_negative_values(self):
        """Scaling should work with negative values."""
        scaler = SimpleScaler()
        X = np.array([[-10.0], [0.0], [10.0]])
        scaler.fit(X)
        scaled = scaler.transform(np.array([[0.0]]))
        # mean=0, std=~8.16 → 0/8.16 ≈ 0
        assert abs(scaled[0, 0]) < 1e-10


# ---------------------------------------------------------------------------
# SimpleLinearRegression tests
# ---------------------------------------------------------------------------

class TestSimpleLinearRegression:
    """Unit tests for the numpy SimpleLinearRegression."""

    def test_fit_stores_coef_and_intercept(self):
        """fit() should learn non-None coefficients and intercept."""
        model = SimpleLinearRegression()
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0])  # y = 2x
        model.fit(X, y)

        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert len(model.coef_) == 1

    def test_fit_returns_self(self):
        """fit() should return self for method chaining."""
        model = SimpleLinearRegression()
        result = model.fit(np.array([[1.0]]), np.array([2.0]))
        assert result is model

    def test_predict_multiple_samples(self):
        """predict() with multiple rows should return correct shape."""
        model = SimpleLinearRegression()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model.fit(X, y)

        preds = model.predict(np.array([[1.0], [5.0], [10.0]]))
        assert preds.shape == (3,)
        assert all(np.isfinite(preds))

    def test_predict_single_sample_2d(self):
        """predict() with a single 2D row should work."""
        model = SimpleLinearRegression()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model.fit(X, y)

        pred = model.predict(np.array([[5.0]]))
        assert pred.ndim == 1
        assert len(pred) == 1

    def test_predict_single_sample_1d(self):
        """predict() with a 1D array (single feature vector) should auto-reshape."""
        model = SimpleLinearRegression()
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model.fit(X, y)

        pred = model.predict(np.array([5.0]))
        assert pred.ndim == 1
        assert len(pred) == 1

    def test_perfect_fit(self):
        """With perfectly linear data, predictions should be near-exact."""
        model = SimpleLinearRegression()
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # y = 10 + 10x
        model.fit(X, y)

        pred = model.predict(np.array([[2.0]]))
        np.testing.assert_almost_equal(pred[0], 30.0, decimal=5)

    def test_multi_feature(self):
        """Regression with multiple features should work correctly."""
        model = SimpleLinearRegression()
        # y = 3*x1 + 5*x2
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0, 8.0, 11.0])
        model.fit(X, y)

        pred = model.predict(np.array([[2.0, 3.0]]))
        # Expected: 3*2 + 5*3 = 21 (plus small intercept)
        assert np.isfinite(pred[0])
        # Check it's close (allowing for numerical noise)
        assert abs(pred[0] - 21.0) < 1.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Unit tests for app.utils.config."""

    def test_base_dir_is_project_root(self):
        """BASE_DIR should resolve to the project root (contains app/, data/, model/)."""
        assert (BASE_DIR / "app").is_dir()
        assert (BASE_DIR / "data").is_dir()
        assert (BASE_DIR / "model").is_dir()
        assert (BASE_DIR / "app" / "utils" / "config.py").is_file()

    def test_model_dir_exists(self):
        """MODEL_DIR should point to the model/ directory."""
        assert MODEL_DIR.name == "model"
        assert MODEL_DIR == BASE_DIR / "model"

    def test_data_dir_exists(self):
        """DATA_DIR should point to the data/ directory."""
        assert DATA_DIR.name == "data"
        assert DATA_DIR == BASE_DIR / "data"

    def test_model_path(self):
        """MODEL_PATH should be model/model.pkl."""
        assert MODEL_PATH.name == "model.pkl"
        assert MODEL_PATH.parent == MODEL_DIR

    def test_data_path(self):
        """DATA_PATH should be data/housing.csv."""
        assert DATA_PATH.name == "housing.csv"
        assert DATA_PATH.parent == DATA_DIR

    def test_feature_columns_count(self):
        """FEATURE_COLUMNS should have exactly 7 features."""
        assert len(FEATURE_COLUMNS) == 7

    def test_feature_columns_order(self):
        """FEATURE_COLUMNS should be in the expected order."""
        expected = [
            "square_footage",
            "bedrooms",
            "bathrooms",
            "year_built",
            "lot_size",
            "distance_to_city_center",
            "school_rating",
        ]
        assert FEATURE_COLUMNS == expected

    def test_target_column(self):
        """TARGET_COLUMN should be 'price'."""
        assert TARGET_COLUMN == "price"


# ---------------------------------------------------------------------------
# HousePriceModel unit tests (requires model.pkl to exist)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def loaded_model():
    """Fixture that returns a HousePriceModel with the real pickle loaded."""
    from app.services.model import HousePriceModel
    return HousePriceModel()


@pytest.fixture(scope="class")
def unloaded_model():
    """Fixture that returns a HousePriceModel without loading (no pickle)."""
    from app.services.model import HousePriceModel
    with patch.object(HousePriceModel, "_load_model", lambda self: None):
        return HousePriceModel()


class TestHousePriceModel:
    """Unit tests for HousePriceModel."""

    # --- is_loaded ---

    def test_is_loaded_true(self, loaded_model):
        """is_loaded should be True when model.pkl is loaded."""
        assert loaded_model.is_loaded is True

    def test_is_loaded_false(self, unloaded_model):
        """is_loaded should be False when no model is loaded."""
        assert unloaded_model.is_loaded is False

    # --- predict_single ---

    def test_predict_single_valid(self, loaded_model):
        """predict_single with valid features returns a finite float."""
        features = {
            "square_footage": 1550,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1997,
            "lot_size": 6800,
            "distance_to_city_center": 4.1,
            "school_rating": 7.6,
        }
        result = loaded_model.predict_single(features)
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result > 0

    def test_predict_single_returns_two_decimals(self, loaded_model):
        """predict_single should round to 2 decimal places."""
        features = {
            "square_footage": 1550,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1997,
            "lot_size": 6800,
            "distance_to_city_center": 4.1,
            "school_rating": 7.6,
        }
        result = loaded_model.predict_single(features)
        # Check that the result has at most 2 decimal places
        assert result == round(result, 2)

    def test_predict_single_unloaded_raises(self, unloaded_model):
        """predict_single on an unloaded model should raise RuntimeError."""
        features = {
            "square_footage": 1550,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1997,
            "lot_size": 6800,
            "distance_to_city_center": 4.1,
            "school_rating": 7.6,
        }
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            unloaded_model.predict_single(features)

    def test_predict_single_with_missing_feature_key(self, loaded_model):
        """predict_single should raise KeyError if a required feature is missing."""
        # Missing 'school_rating' key
        bad_features = {
            "square_footage": 1550,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1997,
            "lot_size": 6800,
            "distance_to_city_center": 4.1,
            # school_rating is missing
        }
        with pytest.raises(KeyError):
            loaded_model.predict_single(bad_features)

    # --- predict_batch ---

    def test_predict_batch_two_items(self, loaded_model):
        """predict_batch with 2 items returns 2 predictions."""
        features_list = [
            {
                "square_footage": 1550,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 1997,
                "lot_size": 6800,
                "distance_to_city_center": 4.1,
                "school_rating": 7.6,
            },
            {
                "square_footage": 2800,
                "bedrooms": 4,
                "bathrooms": 3,
                "year_built": 2018,
                "lot_size": 9500,
                "distance_to_city_center": 2.3,
                "school_rating": 8.9,
            },
        ]
        results = loaded_model.predict_batch(features_list)
        assert len(results) == 2
        assert all(isinstance(r, float) for r in results)
        assert all(np.isfinite(r) for r in results)

    def test_predict_batch_single_item(self, loaded_model):
        """predict_batch with a single item works (not a special case)."""
        features_list = [
            {
                "square_footage": 1550,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 1997,
                "lot_size": 6800,
                "distance_to_city_center": 4.1,
                "school_rating": 7.6,
            }
        ]
        results = loaded_model.predict_batch(features_list)
        assert len(results) == 1
        assert isinstance(results[0], float)

    def test_predict_batch_unloaded_raises(self, unloaded_model):
        """predict_batch on an unloaded model should raise RuntimeError."""
        features_list = [
            {
                "square_footage": 1550,
                "bedrooms": 3,
                "bathrooms": 2,
                "year_built": 1997,
                "lot_size": 6800,
                "distance_to_city_center": 4.1,
                "school_rating": 7.6,
            }
        ]
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            unloaded_model.predict_batch(features_list)

    # --- get_model_info ---

    def test_get_model_info_keys(self, loaded_model):
        """get_model_info should return a dict with all expected keys."""
        info = loaded_model.get_model_info()
        expected_keys = {
            "model_type", "coefficients", "intercept",
            "metrics", "training_date", "n_samples_trained",
            "excluded_features",
        }
        assert set(info.keys()) == expected_keys

    def test_get_model_info_coefficients(self, loaded_model):
        """coefficients should map feature names to numeric values."""
        info = loaded_model.get_model_info()
        coefs = info["coefficients"]
        assert set(coefs.keys()) == set(FEATURE_COLUMNS)
        for name, value in coefs.items():
            assert isinstance(value, float)
            assert np.isfinite(value)

    def test_get_model_info_metrics(self, loaded_model):
        """metrics should contain r_squared, rmse, mae."""
        info = loaded_model.get_model_info()
        metrics = info["metrics"]
        assert "r_squared" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["r_squared"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_get_model_info_intercept(self, loaded_model):
        """intercept should be a finite number."""
        info = loaded_model.get_model_info()
        assert isinstance(info["intercept"], float)
        assert np.isfinite(info["intercept"])

    def test_get_model_info_training_date(self, loaded_model):
        """training_date should be an ISO-format string."""
        info = loaded_model.get_model_info()
        assert isinstance(info["training_date"], str)
        # Basic ISO-8601 check: starts with a year
        assert len(info["training_date"]) >= 10

    def test_get_model_info_n_samples(self, loaded_model):
        """n_samples_trained should be a positive integer."""
        info = loaded_model.get_model_info()
        assert isinstance(info["n_samples_trained"], int)
        assert info["n_samples_trained"] > 0

    def test_get_model_info_excluded_features(self, loaded_model):
        """excluded_features should be ['id', 'price']."""
        info = loaded_model.get_model_info()
        assert info["excluded_features"] == ["id", "price"]

    def test_get_model_info_model_type(self, loaded_model):
        """model_type should be a reasonable class name."""
        info = loaded_model.get_model_info()
        assert isinstance(info["model_type"], str)
        assert len(info["model_type"]) > 0


# ---------------------------------------------------------------------------
# get_model() singleton tests
# ---------------------------------------------------------------------------

class TestGetModelSingleton:
    """Unit tests for the get_model() singleton factory."""

    def test_get_model_returns_loaded_instance(self):
        """get_model() should return an already-loaded model instance."""
        from app.services.model import get_model
        model = get_model()
        assert model.is_loaded is True

    def test_get_model_returns_same_object(self):
        """get_model() should return the same object on repeated calls."""
        from app.services.model import get_model, _model_instance
        # Clear cache first
        import app.services.model as model_module
        original = model_module._model_instance
        model_module._model_instance = None
        try:
            a = get_model()
            b = get_model()
            assert a is b
        finally:
            model_module._model_instance = original


# ---------------------------------------------------------------------------
# Model pickle integration tests
# ---------------------------------------------------------------------------

class TestModelPickle:
    """Verify the model.pkl artifact can be loaded and has expected structure."""

    def test_model_pkl_exists(self):
        """model.pkl file should exist on disk."""
        assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"

    def test_model_pkl_loads(self):
        """model.pkl should be a valid pickle with expected keys."""
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        expected_keys = {"model", "scaler", "coefficients", "intercept", "metrics", "training_date", "n_samples"}
        assert set(data.keys()) >= expected_keys

    def test_model_pkl_coefficients_match_features(self):
        """coefficients dict keys should match FEATURE_COLUMNS."""
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        assert set(data["coefficients"].keys()) == set(FEATURE_COLUMNS)

    def test_model_pkl_metrics_valid(self):
        """metrics should contain valid numeric values."""
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        metrics = data["metrics"]
        assert metrics["r_squared"] >= 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
