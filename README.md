# Customer Segmentation and Market Intelligence System

GPU-accelerated customer segmentation project using RAPIDS, cuML, and clustering algorithms for customer analytics and personalized marketing strategies.

---

# Project Overview

The goal of this project is to segment customers of an online retail store based on transactional behavior and purchasing history.

The project combines:
- RFM analysis
- GPU-accelerated clustering
- RAPIDS ecosystem
- production-ready ML pipeline
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

Additional features were tested experimentally but did not improve clustering quality compared to classical RFM features.

---

## 4. Outlier Handling

Removed:
- top 5% Frequency;
- top 5% Monetary.

---

## 5. Final Production Pipeline

Implemented final production-ready pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=4,
        random_state=42
    ))
])