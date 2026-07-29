import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_single():
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
    payload = {
        "features": {
            "square_footage": 1550,
            "bedrooms": 3,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch():
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
    payload = {"features": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "coefficients" in data
    assert "metrics" in data
    assert "excluded_features" in data
