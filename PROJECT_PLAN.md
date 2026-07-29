# 🏠 Housing Price Prediction API — Development Plan

> **Objective**: Build, containerise, and deploy a simple regression model that predicts housing prices based on provided features.

---

## 📋 Requirements Recap

| Category | Item |
|----------|------|
| **API Endpoints** | `POST /predict` (single), `POST /predict/batch` (batch), `GET /model-info`, `GET /health` |
| **Tech Stack** | Python 3.12+, FastAPI, Scikit-learn |
| **Deliverables** | 1) Source code on GitHub  2) Dockerfile  3) Live demo via Swagger/OpenAPI |
| **Dataset** | 50 rows, 7 features (`square_footage`, `bedrooms`, `bathrooms`, `year_built`, `lot_size`, `distance_to_city_center`, `school_rating`) → target: `price` |

---

## 🗺 Phase Roadmap

```
Phase 1: Code Foundation ──► Phase 2: Docker Build ──► Phase 3: API Verification
                                                                    │
                                                                    ▼
                                                             Phase 4: Swagger Demo
                                                                    │
                                                                    ▼
                                                             Phase 5: Test Suite
```

---

## Phase 1: Code Foundation

**Goal**: Ensure all source code is correct, consistent, and ready for containerization.

### Tasks

| # | Task | Status | Output |
|---|------|--------|--------|
| 1.1 | Fix route prefix inconsistency (`/api/v1` → `/`) | ⬜ | `main.py` uses root-level routes |
| 1.2 | Update Dockerfile to train model during build | ⬜ | `Dockerfile` includes `RUN python train.py` |
| 1.3 | Verify `.dockerignore` does not exclude model/ | ⬜ | Model artifacts are included |
| 1.4 | Update README to match final route structure | ⬜ | Docs reflect `/predict`, `/health`, etc. |
| 1.5 | Run code syntax check on all Python files | ⬜ | Zero syntax errors |

### Acceptance Criteria
- [ ] All route paths are consistent across `main.py`, code, and README
- [ ] Dockerfile builds a trained model inside the container
- [ ] No Python syntax errors in any source file

---

## Phase 2: Docker Build

**Goal**: Build a production-ready Docker image with the trained model embedded.

### Tasks

| # | Task | Status | Output |
|---|------|--------|--------|
| 2.1 | Build Docker image: `docker build -t house-price-api .` | ⬜ | Image built successfully |
| 2.2 | Verify image size and contents | ⬜ | Reasonable size (~300-500MB) |
| 2.3 | Run container: `docker run -d -p 8000:8000 house-price-api` | ⬜ | Container running, port 8000 |

### Acceptance Criteria
- [ ] Docker image builds without errors
- [ ] Model training completes inside the container
- [ ] Container starts and stays healthy

---

## Phase 3: API Verification

**Goal**: Validate all 4 API endpoints return correct responses.

### Tasks

| # | Task | Method | Endpoint | Expected |
|---|------|--------|----------|----------|
| 3.1 | Health check | GET | `/health` | `{"status":"healthy","model_loaded":true}` |
| 3.2 | Single prediction | POST | `/predict` | Returns `{"prediction": <float>}` |
| 3.3 | Batch prediction | POST | `/predict/batch` | Returns 2 predictions with IDs |
| 3.4 | Model info | GET | `/model-info` | Returns coefficients, metrics, metadata |
| 3.5 | Error handling | POST | `/predict` with bad data | Returns 422 |

### Acceptance Criteria
- [ ] All 4 endpoints return 200 with correct JSON structure
- [ ] Input validation returns 422 for invalid data
- [ ] Predictions return reasonable price values (not NaN/null)

---

## Phase 4: Swagger/OpenAPI Demo

**Goal**: Verify the interactive API documentation works for live interview demonstration.

### Tasks

| # | Task | URL | Expected |
|---|------|-----|----------|
| 4.1 | Swagger UI accessible | `http://localhost:8000/docs` | Page loads, shows all endpoints |
| 4.2 | ReDoc accessible | `http://localhost:8000/redoc` | Alternative docs page loads |
| 4.3 | OpenAPI JSON spec | `http://localhost:8000/openapi.json` | Valid JSON schema |
| 4.4 | Try `/predict` in Swagger UI | Interactive "Try it out" | Returns prediction |
| 4.5 | Try `/model-info` in Swagger UI | Interactive "Try it out" | Returns model metadata |

### Acceptance Criteria
- [ ] Swagger UI loads without errors
- [ ] All endpoints visible and documented
- [ ] "Try it out" feature works for at least 2 endpoints

---

## Phase 5: Test Suite

**Goal**: Run automated tests to ensure code quality.

### Tasks

| # | Task | Command | Expected |
|---|------|---------|----------|
| 5.1 | Install test deps in container | `docker exec <id> pip install pytest` | pytest installed |
| 5.2 | Run API tests | `docker exec <id> python -m pytest tests/ -v` | All tests pass |
| 5.3 | Verify test coverage | Review output | Coverage for all 4 endpoints |

### Acceptance Criteria
- [ ] All test cases pass
- [ ] Tests cover: health, single predict, batch predict, model-info, error cases

---

## 📊 Final Deliverables Checklist

| # | Deliverable | Status | Location |
|---|-------------|--------|----------|
| 1 | Source code | ⬜ | GitHub repository (full project) |
| 2 | Dockerfile | ⬜ | `./Dockerfile` (builds + trains model) |
| 3 | Live demo | ⬜ | `http://localhost:8000/docs` (Swagger UI) |

---

## ⚠ Known Limitations

- **Small dataset**: Only 50 rows → 10 test samples after 80/20 split. Metrics (R², RMSE, MAE) should be interpreted with caution.
- **Simple model**: Linear Regression may underfit. Could be improved with Ridge/Lasso in production.
- **No authentication**: API is open — suitable for interview demo only.
