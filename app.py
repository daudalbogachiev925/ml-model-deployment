"""FastAPI сервис для ML-модели."""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API")

# Загрузка модели
model = joblib.load("model.pkl")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: float
    probability: float | None = None

@app.get("/")
def root():
    return {"status": "ok", "model": "linear_regression"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X = np.array(request.features).reshape(1, -1)
    pred = float(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    return PredictResponse(prediction=pred, probability=proba)

@app.get("/health")
def health():
    return {"status": "healthy"}
