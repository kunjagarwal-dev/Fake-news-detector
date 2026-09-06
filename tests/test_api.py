"""
test_api.py
Basic tests for the Fake News Detector FastAPI backend.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_valid_response_shape():
    response = client.post(
        "/predict",
        json={"text": "The Federal Reserve announced on Wednesday that it would hold interest rates steady."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["real", "fake"]
    assert "label" in data
    assert data["label"] in [0, 1]
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_rejects_too_short_text():
    response = client.post("/predict", json={"text": "short"})
    assert response.status_code == 422  # pydantic validation error, min_length=10


def test_predict_rejects_missing_text_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422