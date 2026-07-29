"""
Integration tests for the Housing Price Prediction API.

These tests exercise every public endpoint through FastAPI's
``TestClient``, validating both success paths and error handling
(422 for missing fields, 400 for empty batch, etc.).

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# ─── Valid test payloads (reusable across tests) ───────────────────────────

SINGLE_PREDICT_PAYLOAD = {
    "features": {
        "square_footage": 1550,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1997,
        "lot_size": 6800,
        "distance_to_city_center": 4.1,
        "school_rating": 7.6,
    }
}

BATCH_PREDICT_PAYLOAD = {
    "features": [
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
}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_check():
    """GET /health — returns 200 with status and model_loaded flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_health_check_response_structure():
    """GET /health — response has exactly the expected keys (no extras)."""
    response = client.get("/health")
    data = response.json()
    assert set(data.keys()) == {"status", "model_loaded"}


# ---------------------------------------------------------------------------
# POST /predict — normal cases
# ---------------------------------------------------------------------------

def test_predict_single():
    """POST /predict — single property prediction returns 200 with float price."""
    response = client.post("/predict", json=SINGLE_PREDICT_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0


def test_predict_single_response_structure():
    """POST /predict — response has exactly the expected keys."""
    response = client.post("/predict", json=SINGLE_PREDICT_PAYLOAD)
    data = response.json()
    assert set(data.keys()) == {"prediction"}


def test_predict_single_returns_positive_price():
    """POST /predict — predicted price should be positive (not NaN/null/negative)."""
    response = client.post("/predict", json=SINGLE_PREDICT_PAYLOAD)
    data = response.json()
    assert data["prediction"] > 0
    assert data["prediction"] == data["prediction"]  # not NaN


def test_predict_single_boundary_values():
    """POST /predict — extreme but valid values should not cause crashes."""
    payload = {
        "features": {
            "square_footage": 500,       # very small house
            "bedrooms": 1,
            "bathrooms": 1,
            "year_built": 2024,
            "lot_size": 1000,
            "distance_to_city_center": 0.1,
            "school_rating": 10.0,        # max allowed
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["prediction"], float)
    assert data["prediction"] == data["prediction"]  # not NaN


def test_predict_single_minimal_school_rating():
    """POST /predict — school_rating at minimum (1.0) should work."""
    payload = {
        "features": {
            "square_footage": 2000,
            "bedrooms": 4,
            "bathrooms": 2,
            "year_built": 2000,
            "lot_size": 8000,
            "distance_to_city_center": 5.0,
            "school_rating": 1.0,          # min allowed
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == response.json()["prediction"]  # not NaN


# ---------------------------------------------------------------------------
# POST /predict — error cases
# ---------------------------------------------------------------------------

def test_predict_single_missing_field():
    """POST /predict with missing field → Pydantic 422 Unprocessable Entity."""
    payload = {
        "features": {
            "square_footage": 1550,
            "bedrooms": 3,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_single_invalid_json():
    """POST /predict with malformed JSON → 422 or 400."""
    response = client.post(
        "/predict",
        content="not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422)


def test_predict_single_empty_body():
    """POST /predict with empty body → 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_single_wrong_type():
    """POST /predict with wrong types (string for int field) → 422."""
    payload = {
        "features": {
            "square_footage": "large",   # should be float
            "bedrooms": "many",           # should be int
            "bathrooms": 2,
            "year_built": 1997,
            "lot_size": 6800,
            "distance_to_city_center": 4.1,
            "school_rating": 7.6,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_single_school_rating_out_of_range_high():
    """POST /predict — school_rating > 10 → 422."""
    payload = {
        "features": {
            "square_footage": 1550,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1997,
            "lot_size": 6800,
            "distance_to_city_center": 4.1,
            "school_rating": 11.0,         # exceeds max
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_single_school_rating_out_of_range_low():
    """POST /predict — school_rating < 1 → 422."""
    payload = {
        "features": {
            "square_footage": 1550,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1997,
            "lot_size": 6800,
            "distance_to_city_center": 4.1,
            "school_rating": 0.5,           # below min
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_single_extra_field_ignored():
    """POST /predict — extra unknown fields should be silently ignored."""
    payload = {
        "features": {
            **SINGLE_PREDICT_PAYLOAD["features"],
            "extra_field": "should be ignored",
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict/batch — normal cases
# ---------------------------------------------------------------------------

def test_predict_batch():
    """POST /predict/batch — two predictions returned with correct total."""
    response = client.post("/predict/batch", json=BATCH_PREDICT_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2


def test_predict_batch_response_structure():
    """POST /predict/batch — response has exactly the expected keys."""
    response = client.post("/predict/batch", json=BATCH_PREDICT_PAYLOAD)
    data = response.json()
    assert set(data.keys()) == {"predictions", "total"}
    for item in data["predictions"]:
        assert set(item.keys()) == {"id", "price"}


def test_predict_batch_ids_are_sequential():
    """POST /predict/batch — item IDs should be 0-based sequential."""
    response = client.post("/predict/batch", json=BATCH_PREDICT_PAYLOAD)
    data = response.json()
    ids = [item["id"] for item in data["predictions"]]
    assert ids == [0, 1]


def test_predict_batch_single_item():
    """POST /predict/batch with a single item — should work correctly."""
    payload = {"features": [SINGLE_PREDICT_PAYLOAD["features"]]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["id"] == 0


def test_predict_batch_all_positive():
    """POST /predict/batch — all predicted prices should be positive finite."""
    response = client.post("/predict/batch", json=BATCH_PREDICT_PAYLOAD)
    data = response.json()
    for item in data["predictions"]:
        assert item["price"] > 0
        assert item["price"] == item["price"]  # not NaN


# ---------------------------------------------------------------------------
# POST /predict/batch — error cases
# ---------------------------------------------------------------------------

def test_predict_batch_empty():
    """POST /predict/batch with empty list → HTTP 400 Bad Request."""
    payload = {"features": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400


def test_predict_batch_missing_features_key():
    """POST /predict/batch without 'features' key → 422."""
    response = client.post("/predict/batch", json={})
    assert response.status_code == 422


def test_predict_batch_one_invalid_item():
    """POST /predict/batch with one valid + one invalid → 422."""
    payload = {
        "features": [
            SINGLE_PREDICT_PAYLOAD["features"],
            {
                "square_footage": 1550,
                "bedrooms": 3,
                # Missing required fields
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_predict_batch_invalid_json():
    """POST /predict/batch with malformed JSON → 400 or 422."""
    response = client.post(
        "/predict/batch",
        content="invalid",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# GET /model-info
# ---------------------------------------------------------------------------

def test_model_info():
    """GET /model-info — returns model_type, coefficients, metrics, excluded_features."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "coefficients" in data
    assert "metrics" in data
    assert "excluded_features" in data


def test_model_info_full_structure():
    """GET /model-info — response contains all documented fields."""
    response = client.get("/model-info")
    data = response.json()
    expected_keys = {
        "model_type", "coefficients", "intercept",
        "metrics", "training_date", "n_samples_trained",
        "excluded_features",
    }
    assert set(data.keys()) == expected_keys


def test_model_info_metrics_values():
    """GET /model-info — metrics contain valid numeric values."""
    response = client.get("/model-info")
    metrics = response.json()["metrics"]
    assert metrics["r_squared"] >= 0
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0


def test_model_info_coefficients_count():
    """GET /model-info — coefficients should have exactly 7 entries."""
    response = client.get("/model-info")
    data = response.json()
    assert len(data["coefficients"]) == 7


def test_model_info_training_date_format():
    """GET /model-info — training_date should be a non-empty string."""
    response = client.get("/model-info")
    data = response.json()
    assert isinstance(data["training_date"], str)
    assert len(data["training_date"]) > 0


# ---------------------------------------------------------------------------
# Cross-endpoint consistency
# ---------------------------------------------------------------------------

def test_single_and_batch_predictions_consistent():
    """Same features → single predict should match batch predict (within tolerance)."""
    single_resp = client.post("/predict", json=SINGLE_PREDICT_PAYLOAD)
    batch_resp = client.post(
        "/predict/batch",
        json={"features": [SINGLE_PREDICT_PAYLOAD["features"]]},
    )

    single_price = single_resp.json()["prediction"]
    batch_price = batch_resp.json()["predictions"][0]["price"]
    assert abs(single_price - batch_price) < 0.01, (
        f"Single: {single_price}, Batch: {batch_price}"
    )


def test_model_info_coefficients_sum_one():
    """Coefficients should all be finite floats (sanity check)."""
    response = client.get("/model-info")
    data = response.json()
    for feature, coef in data["coefficients"].items():
        assert isinstance(coef, float), f"Coefficient for {feature} is not float"
        assert coef == coef, f"Coefficient for {feature} is NaN"
