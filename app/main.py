from contextlib import asynccontextmanager

from fastapi import FastAPI
import joblib
import uvicorn

from pathlib import Path
from app.schemas import PredictionRequest
from insurance_classifier.config import load_yaml_config
from insurance_classifier.modeling.predict import predict_request
from insurance_classifier.pipeline import PreprocessingPipeline
from app.preprocessing.features import feature_engineering

ml_model = {}
model_config = load_yaml_config("config/base.yaml")

# context manager (model load)
@asynccontextmanager
async def lifespan(app:FastAPI):
    ml_model["xgb_v01"] = joblib.load(Path(model_config["paths"]["model_dir"]) / "xgb_v01.joblib")
    ml_model["preprocessing"] = PreprocessingPipeline(model_config) 
    yield
    ml_model.clear()

app = FastAPI(title="Insurance Classifier API", 
              version="v1", 
              lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionRequest)
def predict_endpoint(request: PredictionRequest):
    model = ml_model["xgb_v01"]
    pipeline = ml_model["preprocessing"]

    return predict_request(model, request, pipeline=pipeline)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)

