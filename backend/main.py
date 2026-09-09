from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.schemas import PredictionRequest, PredictionResponse
from backend.model_loader import load_artifacts
from backend.logger import logger
from src.text_cleaning import preprocess

app = FastAPI(title="Fake News Detector API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer, model = load_artifacts()


@app.get("/")
def root():
    return {"status": "ok", "message": "Fake News Detector API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        cleaned_text = preprocess(request.text, mode="heavy")

        if not cleaned_text.strip():
            raise HTTPException(status_code=400, detail="Text became empty after cleaning")

        vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized)[0]

        # LinearSVC has no predict_proba; use decision_function distance as a confidence proxy
        decision_score = model.decision_function(vectorized)[0]
        confidence = 1 / (1 + pow(2.718281828, -abs(decision_score)))  # sigmoid squashing

        label_str = "real" if prediction == 1 else "fake"

        logger.info(f"Prediction: {label_str} (confidence={confidence:.4f})")

        return PredictionResponse(
            prediction=label_str,
            label=int(prediction),
            confidence=round(confidence, 4),
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))