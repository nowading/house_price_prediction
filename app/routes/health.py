"""
Health-check route handler — liveness & readiness probe.

Returns a simple JSON payload so Kubernetes / load-balancers / CI pipelines
can verify that the API process is up and (optionally) that the regression
model has been loaded successfully.
"""

import logging

from fastapi import APIRouter

from app.schemas.request import HealthResponse
from app.services.model import get_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API liveness and model-loading readiness.

    Unlike a traditional liveness probe, this endpoint also reports
    whether the pickled regression model is loaded.  If ``model_loaded``
    is ``false`` the process is alive but /predict will return 500 —
    useful for distinguishing *not ready* from *not running*.

    Returns:
        ``HealthResponse`` with ``status`` ("healthy") and
        ``model_loaded`` (bool).
    """
    model = get_model()
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded,
    )
