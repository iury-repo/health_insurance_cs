from pydantic import BaseModel, ConfigDict
from typing import List, Any, Dict

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    data: List[Dict[str, Any]]

# class PredictionItem(BaseModel):
#     prediction: int
#     probability: float

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    