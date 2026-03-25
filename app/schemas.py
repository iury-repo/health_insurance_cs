from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: list[dict[str, Any]]

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, value: Any) -> dict[str, Any]:
        """Accept payloads as {"data": [...]} or directly as a list of rows."""
        if isinstance(value, list):
            return {"data": value}

        if isinstance(value, dict):
            if "data" in value:
                return value

            # Backward-compatible: a single row sent without the `data` key.
            return {"data": [value]}

        raise ValueError("Payload must be either a list of rows or an object with `data`.")


class PredictionItem(BaseModel):
    probability: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]
