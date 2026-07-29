"""
Housing Price Prediction API — FastAPI application entry point.

This module creates the FastAPI app instance, wires up all route routers,
and manages the application lifecycle (startup / shutdown) via the `lifespan`
context manager.  At startup it eagerly loads the pickled regression model so
that the first prediction request pays no cold-start penalty.
"""

import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.predict import router as predict_router
from app.routes.model_info import router as model_info_router
from app.routes.health import router as health_router
from app.services.model import get_model

# Suppress the sklearn UserWarning that fires because StandardScaler was
# fitted with feature names but we pass a plain ndarray at predict time.
# The scaling itself is still mathematically correct — only the warning is
# noise for our demo.
warnings.filterwarnings("ignore", message=".*feature names.*fitted without feature names.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — eager model loading on startup.

    The pickled model is loaded *before* the server accepts traffic so that
    the first client hitting /predict doesn't experience a cold-start delay.
    If loading fails (missing file, corrupted pkl, ...) the error is logged
    but the server still starts — the health endpoint will report
    ``model_loaded: false`` and predict endpoints return HTTP 500 until a
    valid model is placed at ``model/model.pkl``.

    Args:
        app: The FastAPI application instance (injected by the framework).
    """
    logger.info("Starting Housing Price Prediction API...")
    try:
        model = get_model()
        if model.is_loaded:
            logger.info("Model loaded successfully at startup")
        else:
            logger.warning(
                "Model not loaded. Run 'python train.py' to train and save the model."
            )
    except Exception as e:
        logger.error("Failed to initialize model: %s", e)
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Housing Price Prediction API",
    description=(
        "A RESTful API that predicts housing prices based on property features "
        "using a Linear Regression model trained with scikit-learn."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register all route routers.  No URL prefix is used so that the public
# endpoint paths (/predict, /predict/batch, /model-info, /health) match
# the README and the task specification exactly.
app.include_router(predict_router)
app.include_router(model_info_router)
app.include_router(health_router)
