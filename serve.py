"""Serve the registered model with uv run uvicorn serve:app."""
from pathlib import Path
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from train import FEATURES
mlflow.set_tracking_uri((Path(__file__).resolve().parent / 'mlruns').as_uri())
app = FastAPI(title='Telco churn predictor')
_model = None
class PredictionRequest(BaseModel):
    features: dict
@app.get('/health')
def health():
    return {'status': 'ok'}
@app.post('/predict')
def predict(req: PredictionRequest):
    global _model
    missing = set(FEATURES) - req.features.keys()
    if missing:
        raise HTTPException(status_code=422, detail=f'Missing features: {sorted(missing)}')
    if _model is None:
        _model = mlflow.sklearn.load_model('models:/telco-churn-week17/Production')
    row = pd.DataFrame([{name: req.features[name] for name in FEATURES}])
    row['TotalCharges'] = pd.to_numeric(row['TotalCharges'], errors='coerce')
    score = float(_model.predict_proba(row)[0, 1])
    return {'churn_probability': score, 'prediction': 'Yes' if score >= 0.5 else 'No'}
