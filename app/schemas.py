from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: list[dict[str, Any]]


class PredictionItem(BaseModel):
    probability: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]
