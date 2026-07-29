"""
Model-info route handler — exposes model metadata for inspection.

Returns the regression coefficients, intercept, performance metrics,
and training date so that a reviewer can verify the model without
needing to read the pickled file directly.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.request import ModelInfoResponse
from app.services.model import get_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/model-info", tags=["model-info"])


@router.get("", response_model=ModelInfoResponse)
async def get_model_info():
    """Return a human-readable summary of the trained regression model.

    This endpoint is useful during code review and interviews — it
    exposes the learned coefficients, the model's R² / RMSE / MAE on
    the held-out test set, and the training timestamp.

    Returns:
        ``ModelInfoResponse`` containing:
        - ``model_type``: e.g. "LinearRegression"
        - ``coefficients``: per-feature weight dict
        - ``intercept``: bias term
        - ``metrics``: R², RMSE, MAE
        - ``training_date``: ISO-8601 timestamp
        - ``n_samples_trained``: number of training rows

    Raises:
        HTTPException(500): If the model has not been loaded.
    """
    model = get_model()
    if not model.is_loaded:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    info = model.get_model_info()
    return ModelInfoResponse(**info)
