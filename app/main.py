from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
import joblib
import uvicorn

from app.schemas import PredictionRequest, PredictionResponse
from insurance_classifier.config import load_yaml_config
from insurance_classifier.modeling.predict import predict_request
from insurance_classifier.pipeline import PreprocessingPipeline

ml_model = {}
model_config = load_yaml_config("config/base.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["xgb_v01"] = joblib.load(Path(model_config["paths"]["model_dir"]) / "xgb_v01.joblib")
    ml_model["preprocessing"] = PreprocessingPipeline(model_config)
    yield
    ml_model.clear()


app = FastAPI(title="Insurance Classifier API", version="v1", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    model = ml_model["xgb_v01"]
    pipeline = ml_model["preprocessing"]

    try:
        return predict_request(model, request, pipeline=pipeline)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
