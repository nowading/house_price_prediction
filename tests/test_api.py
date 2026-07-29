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


def test_health_check():
    """GET /health — returns 200 with status and model_loaded flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_single():
    """POST /predict — single property prediction returns 200 with float price."""
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


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


def test_predict_batch():
    """POST /predict/batch — two predictions returned with correct total."""
    payload = {
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
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2


def test_predict_batch_empty():
    """POST /predict/batch with empty list → HTTP 400 Bad Request."""
    payload = {"features": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400


def test_model_info():
    """GET /model-info — returns model_type, coefficients, metrics, excluded_features."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "coefficients" in data
    assert "metrics" in data
    assert "excluded_features" in data
