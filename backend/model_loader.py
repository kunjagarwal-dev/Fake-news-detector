import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_vectorizer = None
_model = None


def load_artifacts():
    global _vectorizer, _model
    if _vectorizer is None or _model is None:
        _vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer_v1.pkl"))
        _model = joblib.load(os.path.join(MODELS_DIR, "svm_v1.pkl"))
    return _vectorizer, _model