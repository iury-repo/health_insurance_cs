import panda as pd
import typer
import uvicorn
import joblib

from pathlib import Path
from loguru import logger
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import PredictionRequest, PredictionResponse
from insurance_classifier.modeling.config import load_yaml_config



base_path = "config/base.yaml"
model_config = load_yaml_config(base_path)

# context manager (model load)
@asynccontextmanager
async def lifespan(app:FastAPI):
    model = joblib.load(model_config["paths"]["model_dir"] + "/xgb_v01.joblib")
    yield
    model.clear()

app = FastAPI()

@app.get("/v1/health")
async def health_check():
    return {"status": "ok"}

@app.post("/v1/predict", response_model=PredictionRequest)
def predict_request():
    model.predict()
    model.predict_proba()
    return


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

