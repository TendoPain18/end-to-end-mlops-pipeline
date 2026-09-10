# End-to-End MLOps Pipeline — VPN Traffic Classification

A complete MLOps workflow for classifying network traffic type (e.g. `BROWSING`, `VOIP`, `VPN-P2P`, ...) from flow-level statistics. Built to demonstrate the full lifecycle: data exploration → experiment tracking → reproducible training → containerized serving → automated testing in CI.

## Overview

| Stage | Tool |
|---|---|
| Experiment tracking | MLflow |
| Model training | scikit-learn (RandomForestClassifier) |
| Serving | FastAPI + Uvicorn |
| Containerization | Docker |
| CI | GitHub Actions |

The model predicts one of 14 traffic classes (7 regular + 7 VPN-tunneled equivalents) from 23 flow-based features (packet timing, inter-arrival times, active/idle periods, etc.), based on the ISCX VPN-nonVPN dataset.

## Project structure

```
├── data/                   # Training dataset (consolidated_traffic_data.csv)
├── notebooks/              # Exploratory data analysis
├── src/
│   ├── preprocess.py       # Shared preprocessing (negative-sentinel → NaN → median impute)
│   ├── train.py            # Trains the pipeline, logs to MLflow, saves to models/
│   └── api.py              # FastAPI app serving predictions
├── tests/
│   └── test_api.py         # API tests (root, health, predict, validation)
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## How it works

1. **Preprocessing** (`src/preprocess.py`): the dataset uses `-1` as a sentinel for unavailable measurements. These are converted to `NaN` and then median-imputed. This logic is wrapped in a scikit-learn `Pipeline` step so it's applied identically during training and inference — no train/serve skew.

2. **Training** (`src/train.py`):
   - Splits data with `GroupShuffleSplit`, grouped by exact feature combination, so duplicate rows (~31% of the dataset) can't leak between train and test.
   - Trains a `RandomForestClassifier` inside a single `Pipeline` (preprocessing + model), so the saved artifact is self-contained.
   - Logs parameters, metrics (accuracy, macro F1, weighted F1), and the model itself to MLflow.
   - Saves the trained pipeline to `models/traffic_classifier.joblib`.

3. **Serving** (`src/api.py`): a FastAPI app that loads the saved pipeline once at startup and exposes:
   - `GET /` — basic status message
   - `GET /health` — health check
   - `POST /predict` — takes the 23 raw features as JSON, returns the predicted traffic type

4. **CI** (`.github/workflows/ci.yml`): on every push/PR, GitHub Actions installs dependencies, **trains the model from scratch** (training takes under a minute), then runs the test suite against the freshly trained model. This proves the whole pipeline is reproducible from nothing but the code and the committed dataset — not just tested against a pre-baked model file.

## Running locally

**Train the model:**
```bash
python -m src.train
```
This reads `data/consolidated_traffic_data.csv`, trains the pipeline, logs the run to MLflow (local SQLite backend), and saves `models/traffic_classifier.joblib`.

**View experiment results:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open `http://localhost:5000`.

**Run the API directly:**
```bash
uvicorn src.api:app --reload
```

**Run tests:**
```bash
pytest
```

## Running with Docker

```bash
docker build -t mlops-pipeline .
docker run -p 8000:8000 mlops-pipeline
```

The image expects `models/traffic_classifier.joblib` to already exist locally (run training first — the model is not trained inside the image build).

**Test it:**
```bash
curl http://localhost:8000/health
```

Example prediction request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0.0, "total_fiat": 0.0, "total_biat": 0.0,
    "min_fiat": -1.0, "min_biat": -1.0, "max_fiat": -1.0, "max_biat": -1.0,
    "mean_fiat": 0.0, "mean_biat": 0.0,
    "flowPktsPerSecond": 0.0, "flowBytesPerSecond": 0.0,
    "min_flowiat": -1.0, "max_flowiat": -1.0, "mean_flowiat": 0.0, "std_flowiat": 0.0,
    "min_active": -1.0, "mean_active": 0.0, "max_active": -1.0, "std_active": 0.0,
    "min_idle": -1.0, "mean_idle": 0.0, "max_idle": -1.0, "std_idle": 0.0
  }'
```

## Dataset

Based on the ISCX VPN-nonVPN flow-statistics dataset. 14 classes across regular and VPN-tunneled traffic: `BROWSING`, `CHAT`, `FT`, `MAIL`, `P2P`, `STREAMING`, `VOIP`, and their `VPN-` equivalents.

## Current limitations / next steps

- The API loads a static local model file rather than pulling "the best" version from the MLflow Model Registry — model selection is currently manual.
- `requirements.txt` is unpinned; exact versions aren't locked for full reproducibility.
- No Kubernetes deployment yet (planned as a follow-up project, adding autoscaling with an HPA).
