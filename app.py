"""Food delivery prediction API (workshop example; not a production risk model)."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Food Delivery Prediction API", description="MLOps Workshop Day 4", version="1.0.0")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "delivery_model.joblib"
FEATURES = ["distance_km", "raining", "heavy_traffic"]
model = joblib.load(MODEL_PATH)  # Only load a trusted model artifact.


class OrderRequest(BaseModel):
    distance_km: float = Field(ge=0, description="Delivery distance in kilometers")
    raining: int = Field(ge=0, le=1, description="0=No, 1=Yes")
    heavy_traffic: int = Field(ge=0, le=1, description="0=No, 1=Yes")


class PredictionResponse(BaseModel):
    prediction: int
    status: str


@app.get("/")
def home():
    return {"message": "Welcome to the Food Delivery Prediction API"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_delivery(order: OrderRequest):
    values = pd.DataFrame([{
        "distance_km": order.distance_km,
        "raining": order.raining,
        "heavy_traffic": order.heavy_traffic,
    }], columns=FEATURES)
    prediction = int(model.predict(values)[0])
    return PredictionResponse(prediction=prediction,
                              status="Late Delivery" if prediction == 1 else "On Time")
