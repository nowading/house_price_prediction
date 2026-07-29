# 🏠 Housing Price Prediction API

A machine learning API that predicts housing prices based on property features, built with **FastAPI** and **Scikit-learn**, containerized with Docker for easy deployment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Docker Deployment](#docker-deployment)
- [Model Information](#model-information)
- [Testing](#testing)
- [License](#license)

---

## Overview

This project implements a regression model trained on a housing price dataset and exposes it through a RESTful API. The API supports both **single** and **batch** predictions, along with endpoints for model introspection and health checks.

### Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Predict** | Single & batch housing price predictions |
| 📊 **Model Info** | Retrieve model coefficients, metrics, and parameters |
| ❤️ **Health Check** | Quick liveness probe for monitoring |
| 📖 **Interactive Docs** | Auto-generated Swagger UI & OpenAPI spec |
| 🐳 **Docker Ready** | One-command container deployment |

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.12+ |
| Framework | FastAPI | Latest stable |
| ML Library | Scikit-learn | Latest stable |
| Data | Pandas / NumPy | Latest stable |
| Container | Docker | 24+ |
| ASGI Server | Uvicorn | Latest stable |

---

## Project Structure

```
house_price_prediction/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py       # /predict & /predict/batch endpoints
│   │   ├── model_info.py    # /model-info endpoint
│   │   └── health.py        # /health endpoint
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── request.py       # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   └── model.py         # Model loading, prediction logic
│   └── utils/
│       └── config.py        # Configuration constants
├── data/
│   └── housing.csv          # Housing dataset
├── model/
│   └── model.pkl            # Saved trained model artifact
├── notebooks/
│   └── model_training.ipynb # EDA & model training notebook
├── tests/
│   ├── __init__.py
│   ├── test_predict.py
│   ├── test_model_info.py
│   └── test_health.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## API Endpoints

### 1. `POST /predict` — Single Prediction

Predict the price of a single house given its features.

**Request Body:**

```json
{
  "features": {
    "square_footage": 1550,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 1997,
    "lot_size": 6800,
    "distance_to_city_center": 4.1,
    "school_rating": 7.6
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `square_footage` | float | Total living area (sq ft) |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms |
| `year_built` | int | Year the house was built |
| `lot_size` | float | Lot size (sq ft) |
| `distance_to_city_center` | float | Distance to city center (miles) |
| `school_rating` | float | Local school rating (1-10) |

**Response:**

```json
{
  "prediction": 245620.35
}
```

---

### 2. `POST /predict/batch` — Batch Predictions

Predict prices for multiple houses at once.

**Request Body:**

```json
{
  "features": [
    {
      "square_footage": 1550,
      "bedrooms": 3,
      "bathrooms": 2,
      "year_built": 1997,
      "lot_size": 6800,
      "distance_to_city_center": 4.1,
      "school_rating": 7.6
    },
    {
      "square_footage": 2800,
      "bedrooms": 4,
      "bathrooms": 3,
      "year_built": 2018,
      "lot_size": 9500,
      "distance_to_city_center": 2.3,
      "school_rating": 8.9
    }
  ]
}
```

**Response:**

```json
{
  "predictions": [
    {"id": 0, "price": 245620.35},
    {"id": 1, "price": 438900.72}
  ],
  "total": 2
}
```

---

### 3. `GET /model-info` — Model Information

Returns the trained model's coefficients, performance metrics, and metadata.

**Response:**

```json
{
  "model_type": "LinearRegression",
  "coefficients": {
    "square_footage": 112.45,
    "bedrooms": -2345.60,
    "bathrooms": 5678.30,
    "year_built": 210.75,
    "lot_size": 0.52,
    "distance_to_city_center": -8934.20,
    "school_rating": 12540.80
  },
  "intercept": -186320.45,
  "metrics": {
    "r_squared": 0.87,
    "rmse": 25430.15,
    "mae": 18200.42
  },
  "training_date": "2026-07-29T10:00:00Z",
  "n_samples_trained": "<dynamic>",
  "excluded_features": ["id", "price"]
}
```

---

### 4. `GET /health` — Health Check

Simple liveness probe for load balancers and monitoring systems.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2026-07-29T10:00:00Z",
  "model_loaded": true
}
```

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- pip / conda
- Docker (for containerization)

### Option 1: Local Development

```bash
# Clone the repository
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (if model.pkl doesn't exist)
jupyter notebook notebooks/model_training.ipynb

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

### Option 2: Docker Deployment

#### Build the Image

```bash
docker build -t house-price-prediction:latest .
```

#### Run the Container

```bash
docker run -d \
  --name house-price-api \
  -p 8000:8000 \
  house-price-prediction:latest
```

#### Stop & Remove

```bash
docker stop house-price-api
docker rm house-price-api
```

#### Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "square_footage": 1550,
      "bedrooms": 3,
      "bathrooms": 2,
      "year_built": 1997,
      "lot_size": 6800,
      "distance_to_city_center": 4.1,
      "school_rating": 7.6
    }
  }'
```

---

## Model Information

### Algorithm

The model uses **Linear Regression** (or optionally Ridge/Lasso Regression) from Scikit-learn to predict housing prices based on multiple features.

### Training Pipeline

1. **Data Loading**: Housing dataset loaded from CSV
2. **Preprocessing**: Feature scaling & train/test split (80/20)
3. **Model Training**: Fitted on training data
4. **Evaluation**: Metrics computed on test set (R², RMSE, MAE)
5. **Serialization**: Model saved as pickle file (`model.pkl`)

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| square_footage | continuous | Total living area in square feet |
| bedrooms | discrete | Number of bedrooms |
| bathrooms | continuous | Number of bathrooms |
| year_built | discrete | Year the property was constructed |
| lot_size | continuous | Lot size in square feet |
| distance_to_city_center | continuous | Distance to city center in miles |
| school_rating | continuous | Local school rating (1-10 scale) |

> **Note:** `id` and `price` columns are excluded from input features. `price` is the prediction target, and `id` is a non-informative identifier.

---

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_predict.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Test Coverage Targets

| Endpoint | Test Case |
|----------|-----------|
| `/predict` | Valid request, missing fields, invalid types |
| `/predict/batch` | Multiple entries, empty list, oversized batch |
| `/model-info` | Returns correct structure, model loaded |
| `/health` | Healthy status, model loaded check |

---

## Live Demo

This API is designed for live demonstration during interviews:

1. **Swagger UI** at `/docs` — Interactive documentation where you can:
   - Try every endpoint directly from the browser
   - View request/response schemas
   - Execute real predictions with sample data

2. **ReDoc** at `/redoc` — Alternative API documentation view

3. **OpenAPI Spec** at `/openapi.json` — Machine-readable API specification

To demo live:
```bash
docker run -d --name demo -p 8000:8000 house-price-prediction:latest
# Open browser → http://localhost:8000/docs
```

---

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Source code | GitHub repository |
| 2 | Dockerfile | `./Dockerfile` |
| 3 | Live demo | Swagger UI at `/docs` |

---

## License

MIT License © 2026
