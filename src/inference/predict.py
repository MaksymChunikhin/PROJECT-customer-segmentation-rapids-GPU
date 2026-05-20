import joblib
import pandas as pd


# загрузка scaler и модели
scaler = joblib.load("../../outputs/models/scaler.pkl")
model = joblib.load("../../outputs/models/kmeans_model.pkl")


def predict_customer_segment(recency, frequency, monetary):

    # dataframe нового клиента
    customer = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    # scaling
    customer_scaled = scaler.transform(customer)

    # prediction
    cluster = model.predict(customer_scaled)

    return int(cluster[0])