from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / "outputs" / "models" / "final_pipeline.pkl"

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained pipeline not found at {MODEL_PATH}")
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_customer_segment(recency: float, frequency: float, monetary: float) -> int:
    """Predict customer segment using the trained clustering pipeline."""
    if recency < 0 or frequency < 0 or monetary < 0:
        raise ValueError("Recency, Frequency and Monetary must be non-negative.")

    customer = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary],
    })
    cluster = _get_pipeline().predict(customer)
    return int(cluster[0])
