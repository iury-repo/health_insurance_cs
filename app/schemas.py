from pydantic import BaseModel

class PredictionRequest(BaseModel):
    age: int
    vintage: float
    annual_premium: float
    region_code: float
    vehicle_damage: int
    policy_sales_channel: float
    previously_insured: int

class PredictionResponse(BaseModel):
    predict_class: int
    predict_proba: float