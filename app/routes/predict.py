import logging

from fastapi import APIRouter, HTTPException

from app.schemas.request import (
    HouseFeatures,
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
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    features = request.features.model_dump()
    try:
        prediction = model.predict_single(features)
        return PredictResponse(prediction=prediction)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    if not request.features:
        raise HTTPException(status_code=400, detail="Features list cannot be empty")

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
