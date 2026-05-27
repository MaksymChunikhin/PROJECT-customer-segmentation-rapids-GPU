# Customer Segmentation & Market Intelligence System

GPU-accelerated customer segmentation system for customer analytics, behavioral clustering, and market intelligence using RAPIDS and cuML.

---

## Overview

End-to-end machine learning project for customer segmentation based on e-commerce transaction history.

The project combines:

- RFM customer analytics
- GPU-accelerated clustering
- customer behavior analysis
- cohort retention analysis
- production-style ML pipelines
- interactive visual analytics

---

## Tech Stack

### Machine Learning & GPU
- RAPIDS cuDF / cuML
- CuPy
- KMeans
- DBSCAN
- HDBSCAN
- sklearn Pipeline

### Data & Visualization
- pandas
- NumPy
- UMAP
- matplotlib
- seaborn
- plotly

---

## Features

- transaction preprocessing
- RFM feature engineering
- GPU clustering
- clustering evaluation
- UMAP visualization
- inference pipeline
- customer segment analytics

---

## Final Model

### Selected Algorithm
- RAPIDS GPU KMeans
- 4 customer segments
- Silhouette Score: **0.3318**

### Customer Segments
- VIP Customers
- New Customers
- Potential Churn
- Low Value Customers

---

## Key Findings

- Classical RFM features outperformed extended feature engineering.
- KMeans achieved the best balance between clustering quality and production usability.
- GPU acceleration significantly improved clustering experimentation workflow.
- The final pipeline supports reusable customer inference.

---

## Project Structure

```text
cluster_project/
│
├── notebooks/
├── src/
├── outputs/
├── tests/
├── requirements.txt
└── README.md
```

---

## Business Applications

- personalized marketing
- churn prediction support
- customer intelligence
- retention strategies
- recommendation systems
