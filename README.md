# Customer Segmentation and Market Intelligence System

GPU-accelerated customer segmentation project built with RAPIDS and cuML for customer analytics and clustering.

---

## Project Overview

This project implements an end-to-end customer segmentation workflow for an online retail dataset using GPU-accelerated clustering algorithms.

Main features:
- RFM customer analysis
- RAPIDS GPU clustering
- KMeans / DBSCAN / HDBSCAN comparison
- production-ready ML pipeline
- UMAP visualization
- inference pipeline for new customers

The final solution uses a production-style sklearn Pipeline with RAPIDS-based clustering and reusable inference workflow.

---

## Tech Stack

### GPU / RAPIDS
- RAPIDS cuDF
- RAPIDS cuML
- CuPy

### Machine Learning
- RAPIDS KMeans
- RAPIDS DBSCAN
- RAPIDS HDBSCAN
- sklearn Pipeline

### Data Processing & Visualization
- pandas
- NumPy
- matplotlib
- seaborn
- UMAP

---

## Pipeline

### Data Processing
- data cleaning
- duplicate removal
- return filtering
- outlier handling
- RFM feature engineering

### Clustering
Compared:
- RAPIDS KMeans
- RAPIDS DBSCAN
- RAPIDS HDBSCAN

### Final Production Pipeline

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=4,
        random_state=42
    ))
])
```

The final pipeline combines preprocessing and clustering into a reusable inference-ready artifact.

---

## Results

### Final Model
- RAPIDS KMeans
- 4 clusters
- Silhouette Score: **0.3318**

### Key Findings
- Classical RFM features outperformed extended feature engineering.
- KMeans achieved the best clustering quality.
- GPU acceleration significantly improved clustering workflow.
- UMAP visualization showed clear cluster separation.

---

## Inference Pipeline

```python
predict_customer_segment(...)
```

Inference workflow:
- loads final pipeline;
- preprocesses new customer data;
- predicts customer segment.

---

## Model Artifacts

Saved artifacts:
- `final_pipeline.pkl`
- `rfm_clusters.csv`

---

## Project Structure

```text
PROJECT-customer-segmentation-rapids-GPU/
│
├── notebooks/
├── outputs/
├── src/
│   └── inference/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Run Project

```bash
pip install -r requirements.txt
jupyter notebook
```

Open:

```text
notebooks/PROJECT-customer_segmentation_rapids.ipynb
```