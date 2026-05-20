import joblib
import pandas as pd

# load trained pipeline
pipeline = joblib.load("../../outputs/models/final_pipeline.pkl")


def predict_customer_segment(recency, frequency, monetary):
    """
    Predict customer segment using trained clustering pipeline.
    """

    customer = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    cluster = pipeline.predict(customer)

    return int(cluster[0])