"""
Pydantic request and response schemas for the Housing Price Prediction API.

These schemas define the contract between the API and its consumers.
Pydantic validates incoming JSON automatically and generates the
OpenAPI / Swagger documentation from the ``Field`` metadata.

Note:
    ORM models (SQLAlchemy) are intentionally kept separate from these
    schemas to prevent accidental leakage of database fields (e.g. id,
    internal timestamps) in API responses.
"""

from typing import List

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """Core feature vector for a single property.

    All seven features are required.  ``school_rating`` is constrained
    to the 1-10 range via ``ge`` / ``le``.
    """

    square_footage: float = Field(
        ..., description="Total living area in square feet"
    )
    bedrooms: int = Field(
        ..., description="Number of bedrooms"
    )
    bathrooms: float = Field(
        ..., description="Number of bathrooms"
    )
    year_built: int = Field(
        ..., description="Year the house was built"
    )
    lot_size: float = Field(
        ..., description="Lot size in square feet"
    )
    distance_to_city_center: float = Field(
        ..., description="Distance to city center in miles"
    )
    school_rating: float = Field(
        ..., ge=1, le=10, description="Local school rating (1-10)"
    )


class PredictRequest(BaseModel):
    """Request body for POST /predict — a single property prediction."""

    features: HouseFeatures


class BatchPredictRequest(BaseModel):
    """Request body for POST /predict/batch — multiple property predictions.

    Consumers can send up to 100 properties per request for efficient
    vectorised inference.
    """

    features: List[HouseFeatures] = Field(
        ..., min_length=1, max_length=100,
        description="List of property features (1-100 items)",
    )


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    prediction: float = Field(
        ..., description="Predicted house price in USD, rounded to 2 dp"
    )


class BatchPredictItem(BaseModel):
    """Single item inside a batch prediction response."""

    id: int = Field(
        ..., description="Zero-based index matching the input list position"
    )
    price: float = Field(
        ..., description="Predicted house price in USD, rounded to 2 dp"
    )


class BatchPredictResponse(BaseModel):
    """Response body for POST /predict/batch."""

    predictions: List[BatchPredictItem]
    total: int = Field(..., description="Number of predictions returned")


class ModelInfoResponse(BaseModel):
    """Response body for GET /model-info — model metadata for review."""

    model_type: str
    coefficients: dict
    intercept: float
    metrics: dict
    training_date: str
    n_samples_trained: int
    excluded_features: List[str]


class HealthResponse(BaseModel):
    """Response body for GET /health — liveness + readiness probe."""

    status: str
    model_loaded: bool
