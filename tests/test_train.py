"""
Unit tests for the training script (train.py).

These tests validate the core workflow of ``train.main()``: data loading,
train/test split, scaling, model fitting, evaluation, and pickle
serialisation.  All file I/O is redirected to temporary directories so
no real data or model files are touched.
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from app.utils.config import FEATURE_COLUMNS, TARGET_COLUMN

# Suppress logging noise during tests.
logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_train_paths(monkeypatch, data_path: Path, model_path: Path) -> None:
    """Patch path references in both the config module and the train module.

    ``train.py`` uses ``from app.utils.config import DATA_PATH, MODEL_PATH``,
    which creates local module-level bindings.  After the train module is
    first imported, ``monkeypatch`` on ``app.utils.config`` alone has no
    effect on those local names.  This helper patches both the config
    module (for fresh imports) and the already-imported ``train`` module
    (for cached imports), so tests work regardless of import order.
    """
    import train as train_mod

    # Patch the source of truth (app.utils.config).
    monkeypatch.setattr("app.utils.config.DATA_PATH", data_path)
    monkeypatch.setattr("app.utils.config.MODEL_PATH", model_path)
    # Patch the local references inside the already-imported train module.
    train_mod.DATA_PATH = data_path
    train_mod.MODEL_PATH = model_path


def _make_sample_csv(dest: Path, n: int = 20, seed: int = 42) -> Path:
    """Generate a deterministic synthetic housing CSV at ``dest``."""
    rng = np.random.RandomState(seed)
    rows = []

    for i in range(n):
        square_footage = rng.randint(800, 4000)
        bedrooms = rng.randint(1, 6)
        bathrooms = round(rng.uniform(1, 4), 1)
        year_built = rng.randint(1920, 2024)
        lot_size = rng.randint(2000, 15000)
        distance_to_city = round(rng.uniform(0.5, 15.0), 1)
        school_rating = round(rng.uniform(1, 10), 1)

        base_price = (
            square_footage * 150
            + bedrooms * 5000
            + bathrooms * 8000
            + (year_built - 1920) * 200
            + lot_size * 0.5
            - distance_to_city * 3000
            + school_rating * 8000
        )
        noise = rng.normal(0, 20000)
        price = max(50000, round(base_price + noise, 0))

        rows.append({
            "id": i + 1,
            "square_footage": square_footage,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "lot_size": lot_size,
            "distance_to_city_center": distance_to_city,
            "school_rating": school_rating,
            "price": int(price),
        })

    pd.DataFrame(rows).to_csv(dest, index=False)
    return dest


# ---------------------------------------------------------------------------
# Happy path — main() succeeds
# ---------------------------------------------------------------------------

class TestTrainMain:
    """Tests for the train.main() entry point."""

    def test_main_success(self, tmp_path, monkeypatch):
        """main() should complete without raising any exceptions."""
        csv_path = _make_sample_csv(tmp_path / "housing.csv")
        model_path = tmp_path / "model" / "model.pkl"
        _patch_train_paths(monkeypatch, csv_path, model_path)

        import train
        train.main()  # should not raise

    def test_main_creates_pickle(self, tmp_path, monkeypatch):
        """main() should create a model.pkl file on disk."""
        csv_path = _make_sample_csv(tmp_path / "housing.csv")
        model_path = tmp_path / "model" / "model.pkl"
        _patch_train_paths(monkeypatch, csv_path, model_path)

        import train
        train.main()
        assert model_path.exists(), "model.pkl was not created"

    def test_main_missing_data_raises(self, tmp_path, monkeypatch):
        """main() should raise FileNotFoundError when data CSV is missing."""
        missing_csv = tmp_path / "nonexistent.csv"
        model_path = tmp_path / "model" / "model.pkl"
        _patch_train_paths(monkeypatch, missing_csv, model_path)

        import train
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            train.main()


# ---------------------------------------------------------------------------
# Pickle artifact structure
# ---------------------------------------------------------------------------

class TestPickleArtifact:
    """Tests for the structure and content of the saved model.pkl."""

    @pytest.fixture(autouse=True)
    def _run_training(self, tmp_path, monkeypatch):
        """Run train.main() once before all tests in this class."""
        csv_path = _make_sample_csv(tmp_path / "housing.csv")
        model_path = tmp_path / "model" / "model.pkl"
        _patch_train_paths(monkeypatch, csv_path, model_path)

        import train
        train.main()

        with open(model_path, "rb") as f:
            self._data = pickle.load(f)
        self._model_path = model_path

    def test_pickle_has_all_expected_keys(self):
        """The pickle dict should contain every key the model service expects."""
        expected_keys = {"model", "scaler", "coefficients", "intercept",
                         "metrics", "training_date", "n_samples"}
        assert set(self._data.keys()) >= expected_keys

    def test_pickle_model_is_linear_regression(self):
        """The 'model' field should be a scikit-learn LinearRegression."""
        assert isinstance(self._data["model"], LinearRegression)

    def test_pickle_scaler_is_standard_scaler(self):
        """The 'scaler' field should be a scikit-learn StandardScaler."""
        assert isinstance(self._data["scaler"], StandardScaler)

    def test_pickle_coefficients_mapping(self):
        """Coefficients should map each feature name to a float."""
        coefs = self._data["coefficients"]
        assert set(coefs.keys()) == set(FEATURE_COLUMNS)
        for name, value in coefs.items():
            assert isinstance(value, float), f"Coefficient {name} is not float"
            assert np.isfinite(value), f"Coefficient {name} is not finite"

    def test_pickle_intercept_is_float(self):
        """Intercept should be a finite float."""
        assert isinstance(self._data["intercept"], float)
        assert np.isfinite(self._data["intercept"])

    def test_pickle_metrics_are_valid(self):
        """Metrics should be present and have sensible values."""
        metrics = self._data["metrics"]
        assert "r_squared" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        # R² should be > 0 (reasonable for synthetic linear data).
        assert metrics["r_squared"] > 0.5
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_pickle_training_date_is_string(self):
        """Training date should be a non-empty ISO-format string."""
        date = self._data["training_date"]
        assert isinstance(date, str)
        assert len(date) >= 10  # at least YYYY-MM-DD

    def test_pickle_n_samples_matches_input(self):
        """n_samples should equal the number of rows in the source CSV."""
        assert self._data["n_samples"] == 20

    def test_pickle_model_is_fitted(self):
        """The stored model should have learned coefficients (not all zeros)."""
        coef = self._data["model"].coef_
        assert coef is not None
        # At least one coefficient should be meaningfully non-zero.
        assert np.any(np.abs(coef) > 1e-6)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Training with the same data and seed should produce identical results."""

    def test_training_is_deterministic(self, tmp_path, monkeypatch):
        """Two consecutive runs should produce the same pickle content."""
        csv_path = _make_sample_csv(tmp_path / "housing.csv")
        model_path = tmp_path / "model" / "model.pkl"
        _patch_train_paths(monkeypatch, csv_path, model_path)

        import train

        # First run.
        train.main()
        with open(model_path, "rb") as f:
            data1 = pickle.load(f)

        # Second run — overwrites the same file.
        train.main()
        with open(model_path, "rb") as f:
            data2 = pickle.load(f)

        # Coefficients and intercept must match exactly.
        assert data1["coefficients"] == data2["coefficients"]
        assert data1["intercept"] == data2["intercept"]
        assert data1["metrics"] == data2["metrics"]
        assert data1["n_samples"] == data2["n_samples"]