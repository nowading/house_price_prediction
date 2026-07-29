import logging

from fastapi import APIRouter

from app.schemas.request import HealthResponse
from app.services.model import get_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    model = get_model()
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
    )
