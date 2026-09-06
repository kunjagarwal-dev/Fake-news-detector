from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News article text to classify")


class PredictionResponse(BaseModel):
    prediction: str  # "real" or "fake"
    label: int        # 1 = real, 0 = fake
    confidence: float