from pydantic import BaseModel, Field
from typing import List, Optional


class HouseFeatures(BaseModel):
    square_footage: float = Field(..., description="Total living area in square feet")
    bedrooms: int = Field(..., description="Number of bedrooms")
    bathrooms: float = Field(..., description="Number of bathrooms")
    year_built: int = Field(..., description="Year the house was built")
    lot_size: float = Field(..., description="Lot size in square feet")
    distance_to_city_center: float = Field(..., description="Distance to city center in miles")
    school_rating: float = Field(..., ge=1, le=10, description="Local school rating (1-10)")


class PredictRequest(BaseModel):
    features: HouseFeatures


class BatchPredictRequest(BaseModel):
    features: List[HouseFeatures]


class PredictResponse(BaseModel):
    prediction: float


class BatchPredictItem(BaseModel):
    id: int
    price: float


class BatchPredictResponse(BaseModel):
    predictions: List[BatchPredictItem]
    total: int


class ModelInfoResponse(BaseModel):
    model_type: str
    coefficients: dict
    intercept: float
    metrics: dict
    training_date: str
    n_samples_trained: int
    excluded_features: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
