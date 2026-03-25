from app.schemas import PredictionRequest, PredictionResponse
import pandas as pd


def predict_request(model, data: PredictionRequest, pipeline):
    df = pd.DataFrame(data.model_dump()["data"])

    engineered_df = pipeline.feature_engineering(df)
    prepared_df = pipeline.data_preparation(engineered_df)

    probabilities = model.predict_proba(prepared_df)[:, 1]

    return PredictionResponse(
        predictions=[{"probability": float(probability)} for probability in probabilities]
    )
