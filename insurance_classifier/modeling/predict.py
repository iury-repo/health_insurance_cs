from app.schemas import PredictionRequest, PredictionResponse
import pandas as pd
import json

from insurance_classifier.config import load_yaml_config
from app.preprocessing.features import feature_engineering

config = load_yaml_config('config/base.yaml')

def predict_request(model, data: PredictionRequest, pipeline):
    df = pd.DataFrame(data.model_dump()["data"])
    
    # Preprocessing

    df2 = pipeline.feature_engineering(df)
    df3 = pipeline.data_preparation(df2)

    # Prediction
    # preds = model.predict(df3)
    probas = model.predict_proba(df3)[:,1]

    results = [
    {
        "probability": float(proba)
    }
    for proba in zip(probas)
    ]

    return PredictionResponse(predictions=results)