import joblib
import pandas as pd

pipeline = joblib.load("../../outputs/models/final_pipeline.pkl")


def predict_customer_segment(recency, frequency, monetary):

    customer = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    cluster = pipeline.predict(customer)

    return int(cluster[0])