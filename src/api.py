from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Import the preprocessing module before loading the model.
# The trained pipeline contains a reference to this module.
import src.preprocess


MODEL_PATH = Path("models/traffic_classifier.joblib")


app = FastAPI(
    title="VPN Traffic Classification API",
    description="Predicts the application type of network traffic.",
    version="1.0.0",
)


# Load the complete preprocessing + model pipeline.
model = joblib.load(MODEL_PATH)


class TrafficFeatures(BaseModel):
    duration: float
    total_fiat: float
    total_biat: float
    min_fiat: float
    min_biat: float
    max_fiat: float
    max_biat: float
    mean_fiat: float
    mean_biat: float
    flowPktsPerSecond: float
    flowBytesPerSecond: float
    min_flowiat: float
    max_flowiat: float
    mean_flowiat: float
    std_flowiat: float
    min_active: float
    mean_active: float
    max_active: float
    std_active: float
    min_idle: float
    mean_idle: float
    max_idle: float
    std_idle: float


@app.get("/")
def root():
    return {
        "message": "VPN Traffic Classification API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(features: TrafficFeatures):
    input_data = pd.DataFrame([
        features.model_dump()
    ])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": prediction
    }
