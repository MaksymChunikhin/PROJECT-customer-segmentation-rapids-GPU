# Customer Segmentation and Market Intelligence System

GPU-accelerated customer segmentation project using RAPIDS, cuML, and clustering algorithms for customer analytics and personalized marketing strategies.

---

# Project Overview

The goal of this project is to segment customers of an online retail store based on transactional behavior and purchasing history.

The project combines:
- RFM analysis
- GPU-accelerated clustering
- RAPIDS ecosystem
- preprocessing pipelines
- UMAP visualization
- inference pipeline for new customers

The system is designed in a production-oriented style and demonstrates a complete customer segmentation workflow.

---

# Business Goal

The company wants to better understand customer behavior and move from mass marketing to personalized customer interaction strategies.

The resulting customer segments can be used for:
- personalized marketing;
- retention strategies;
- customer analytics;
- recommendation systems;
- churn analysis.

---

# Tech Stack

## GPU / RAPIDS
- NVIDIA RTX 3090
- RAPIDS cuDF
- RAPIDS cuML
- CuPy

## Machine Learning
- RAPIDS KMeans
- RAPIDS DBSCAN
- RAPIDS HDBSCAN
- sklearn Pipeline

## Data Processing
- pandas
- NumPy

## Visualization
- matplotlib
- seaborn
- plotly
- UMAP

---

# Dataset

Online Retail transactional dataset containing:
- invoices;
- products;
- quantities;
- prices;
- countries;
- customer IDs.

---

# Project Pipeline

## 1. Data Cleaning

Implemented:
- missing values handling;
- duplicate removal;
- return filtering;
- special transaction filtering;
- zero-price removal;
- TotalPrice feature creation.

Only service-related transaction codes are removed:
- POST
- D
- M
- BANK CHARGES
- DOT
- CRUK

---

## 2. Exploratory Data Analysis (EDA)

Performed analysis:
- top countries;
- monthly revenue;
- customer activity;
- orders by hour;
- top customers by revenue.

---

## 3. Feature Engineering

## RFM Features
- Recency
- Frequency
- Monetary

## Additional Features
- AvgBasketSize
- UniqueProducts

---

## 4. Outlier Handling

Removed:
- top 5% Frequency;
- top 5% Monetary.

---

## 5. Preprocessing Pipeline

Implemented reusable preprocessing pipeline:

```python
Pipeline([
    ("log_transform", FunctionTransformer(np.log1p)),
    ("scaler", StandardScaler())
])
```

The pipeline:
- centralizes preprocessing;
- improves reproducibility;
- simplifies inference;
- is saved as a model artifact.

---

## 6. GPU Clustering

Compared clustering algorithms:
- RAPIDS KMeans
- RAPIDS DBSCAN
- RAPIDS HDBSCAN

GPU acceleration is implemented using:
- cuDF
- cuML
- CuPy

---

## 7. Visualization

Implemented:
- UMAP visualization;
- customer cluster visualization;
- clustering comparison;
- customer analytics charts.

Visualization was built for:
- KMeans;
- DBSCAN;
- HDBSCAN.

---

## 8. Model Evaluation

Used:
- silhouette score;
- clustering comparison;
- baseline vs extended features comparison.

GPU evaluation implemented with:

```python
cython_silhouette_score
```

---

## 9. Customer Segments

The final KMeans model identified 4 customer segments:

| Segment | Description |
|---|---|
| VIP Customers | High-value loyal customers |
| New Customers | Recently active customers |
| Potential Churn | Customers with churn risk |
| Low Value Customers | Low-engagement customers |

---

# Results

## Final Model
- RAPIDS KMeans
- 4 clusters
- Silhouette Score: **0.3318**

## Key Findings
- Classical RFM features outperformed extended feature set.
- GPU acceleration significantly improved clustering workflow.
- UMAP visualization showed clear cluster separation.
- KMeans achieved the best clustering quality for this dataset.

---

# Inference Pipeline

Implemented inference function:

```python
predict_customer_segment(...)
```

The inference pipeline:
- loads preprocessing pipeline;
- preprocesses new customer data;
- predicts customer segment.

---

# Model Artifacts

Saved artifacts:
- `kmeans_model.pkl`
- `preprocessing_pipeline.pkl`
- `rfm_clusters.csv`

---

# How to Run

## Install dependencies

```bash
pip install -r requirements.txt
```

## Open notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/PROJECT-Customer Segmentation.ipynb
```

---

# Future Improvements

Potential improvements:
- radar charts;
- PCA visualization;
- advanced hyperparameter tuning;
- seasonal behavior features;
- deployment with FastAPI;
- dashboard integration.

---

# Conclusion

This project demonstrates a production-oriented GPU Data Science workflow for customer segmentation using RAPIDS and clustering algorithms.

The project includes:
- preprocessing pipelines;
- GPU clustering;
- model evaluation;
- inference pipeline;
- business interpretation of customer segments.

It represents a complete end-to-end Customer Segmentation ML system.