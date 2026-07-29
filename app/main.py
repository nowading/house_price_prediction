import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.predict import router as predict_router
from app.routes.model_info import router as model_info_router
from app.routes.health import router as health_router
from app.services.model import get_model

warnings.filterwarnings("ignore", message=".*feature names.*fitted without feature names.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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

app.include_router(predict_router)
app.include_router(model_info_router)
app.include_router(health_router)
