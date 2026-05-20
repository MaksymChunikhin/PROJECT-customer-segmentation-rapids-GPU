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

The final solution uses a production-style sklearn `Pipeline` with RAPIDS-based clustering and a reusable inference workflow.

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
- plotly (interactive 3D)
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
    ("log_transform", FunctionTransformer(np.log1p)),
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=4, random_state=42)),
])
```

The final pipeline combines preprocessing and clustering into a reusable, inference-ready artifact.

---

## Results

### Final Model
- RAPIDS KMeans
- 4 clusters
- Silhouette Score: **0.3318**

### Key Findings
- Classical RFM features outperformed extended feature engineering (0.250 vs 0.332 silhouette).
- KMeans (0.332) and DBSCAN (0.325) showed comparable quality; KMeans was chosen for production due to a fixed number of segments, no noise points, and easier inference.
- HDBSCAN (0.145) gave noticeably weaker separation on this dataset.
- GPU acceleration significantly improved the clustering workflow.

---

## Inference Pipeline

```python
from src.inference.predict import predict_customer_segment

cluster_id = predict_customer_segment(recency=30, frequency=5, monetary=250.0)
```

The inference function:
- lazily loads the trained pipeline,
- preprocesses new customer data,
- predicts the customer segment.

Requires a GPU and a working RAPIDS / cuML installation, since the saved pipeline contains a `cuml.KMeans` estimator.

---

## Model Artifacts

Saved artifacts:
- `outputs/models/final_pipeline.pkl`
- `outputs/rfm_clusters.csv` (optional, produced from the notebook)

---

## Project Structure

```text
cluster_project/
│
├── data/
│   └── raw/                # input CSV (gitignored)
├── notebooks/
│   └── PROJECT-customer_segmentation_rapids.ipynb
├── outputs/
│   └── models/             # trained pipeline (committed)
├── src/
│   └── inference/
│       └── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Run Project

### 1. Install RAPIDS (GPU, CUDA required)

RAPIDS packages (`cudf`, `cuml`, `cupy`) **cannot be installed via plain `pip install -r requirements.txt`**.
Use conda or the official RAPIDS pip index matched to your CUDA version: https://docs.rapids.ai/install

Example (conda, CUDA 12):

```bash
conda create -n rapids -c rapidsai -c conda-forge -c nvidia \
    rapids=24.10 python=3.11 cuda-version=12.5
conda activate rapids
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
notebooks/PROJECT-customer_segmentation_rapids.ipynb
```
