from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "model.pkl"
DATA_PATH = DATA_DIR / "housing.csv"

FEATURE_COLUMNS = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]

TARGET_COLUMN = "price"
