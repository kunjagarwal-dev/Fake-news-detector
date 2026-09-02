"""
utils.py
Reusable evaluation utilities for model training and comparison.
"""

import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Evaluates a trained model on test data and prints/returns key metrics.

    Args:
        model: A fitted model with a .predict() method.
        X_test: Test features.
        y_test: True test labels.
        model_name (str): Name used for printing/logging.

    Returns:
        dict: Dictionary of computed metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"===== {model_name} =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),  # tolist() so it's JSON-serializable
    }


def save_results_json(results_dict, filepath):
    """
    Saves a results dictionary (or list of them) to a JSON file.

    Args:
        results_dict (dict or list): Results to save.
        filepath (str): Path to save the JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {filepath}")