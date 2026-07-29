"""
Application configuration — file paths and feature column definitions.

All path constants are derived from ``BASE_DIR`` (the project root) so
that the application works regardless of where it is cloned or mounted.
If the project layout changes, only this file needs to be updated.
"""

from pathlib import Path

# Project root — three levels up from this file (app/utils/config.py).
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

MODEL_DIR: Path = BASE_DIR / "model"
DATA_DIR: Path = BASE_DIR / "data"

# Absolute paths consumed by the training script and the model service.
MODEL_PATH: Path = MODEL_DIR / "model.pkl"
DATA_PATH: Path = DATA_DIR / "housing.csv"

# Columns used as input features (X) and target (y).
# Order matters — the fitted scaler and model expect columns in this
# exact sequence.
FEATURE_COLUMNS: list[str] = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]

TARGET_COLUMN: str = "price"
