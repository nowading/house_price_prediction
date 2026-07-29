import logging

from fastapi import APIRouter, HTTPException

from app.schemas.request import ModelInfoResponse
from app.services.model import get_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/model-info", tags=["model-info"])


@router.get("", response_model=ModelInfoResponse)
async def get_model_info():
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    info = model.get_model_info()
    return ModelInfoResponse(**info)
