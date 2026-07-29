"""
Prediction route handlers — single and batch housing price inference.

All heavy lifting (DataFrame construction, scaling, model invocation) is
delegated to :class:`app.services.model.HousePriceModel`.  The route layer
only handles HTTP concerns: input validation, error mapping, and response
serialisation.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.request import (
    PredictRequest,
    BatchPredictRequest,
    PredictResponse,
    BatchPredictItem,
    BatchPredictResponse,
)
from app.services.model import get_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
async def predict_single(request: PredictRequest):
    """Predict the price of a single property.

    Args:
        request: A JSON body conforming to ``PredictRequest``, which
            wraps a ``HouseFeatures`` object with the seven property
            features (square_footage, bedrooms, bathrooms, year_built,
            lot_size, distance_to_city_center, school_rating).

    Returns:
        ``PredictResponse`` containing the predicted price rounded to two
        decimal places.

    Raises:
        HTTPException(500): If the regression model has not been loaded
            (missing ``model/model.pkl``).
        HTTPException(500): If the underlying model call raises an
            unexpected error.
    """
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # Convert Pydantic model to plain dict so the service layer can build
    # its own DataFrame with the exact column order required by the
    # fitted scikit-learn pipeline.
    features = request.features.model_dump()
    try:
        prediction = model.predict_single(features)
        return PredictResponse(prediction=prediction)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict prices for a batch of properties in one request.

    This endpoint is more efficient than issuing N single predictions
    because scikit-learn's predict() is vectorised — all rows are
    scaled and inferred in a single call.

    Args:
        request: ``BatchPredictRequest`` containing a list of
            ``HouseFeatures`` objects (max 100 items enforced by the
            schema).

    Returns:
        ``BatchPredictResponse`` with an ``items`` list mapping each
        input to its predicted price, plus a ``total`` count.

    Raises:
        HTTPException(400): If the input list is empty.
        HTTPException(500): If the model is not loaded or inference
            throws an unexpected error.
    """
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    if not request.features:
        raise HTTPException(status_code=400, detail="Features list cannot be empty")

    # Convert each Pydantic feature object to a plain dict.
    features_list = [f.model_dump() for f in request.features]
    try:
        predictions = model.predict_batch(features_list)
        items = [
            BatchPredictItem(id=i, price=pred)
            for i, pred in enumerate(predictions)
        ]
        return BatchPredictResponse(predictions=items, total=len(items))
    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
