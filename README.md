# Delivery ETA Prediction — Machine Learning Project

Predicting food delivery time using gradient boosted tree models,
engineered operational features, and SHAP-based explainability.

---

## Problem Statement

Accurate delivery time estimation is critical for customer trust
in quick-commerce platforms. This project builds a machine learning
pipeline to predict delivery ETA using order characteristics and
real-time partner availability signals.

---

## Dataset

- Source: Porter Delivery Time Estimation (Kaggle)
- Size: 197,428 orders after cleaning
- Target: Delivery time in minutes (derived from timestamps)
- Domain: Food/grocery delivery logistics

---

## Project Structure

```
delivery-eta-prediction/
│
├── data/
│   ├── raw/          ← original dataset (not tracked in git)
│   └── processed/    ← cleaned data
│
├── notebooks/
│   ├── 01_eda.ipynb               ← exploratory data analysis
│   ├── 02_feature_engineering.ipynb ← feature creation
│   └── 03_modelling.ipynb         ← model training and evaluation
│
├── models/
│   ├── xgboost_eta_model.pkl      ← trained model
│   └── feature_list.pkl           ← expected features
│
└── README.md
```

---

## Approach

### 1. Data Cleaning
- Removed physically implausible delivery times (<10 mins, >120 mins)
- Imputed missing values using median (numerical) and mode (categorical)
- Identified and fixed correlated missingness in partner tracking columns

### 2. Feature Engineering
Created 9 engineered features across three categories:

**Time features:**
- `hour`, `is_weekend`, `is_peak_hour`, `is_night`

**Partner pressure features:**
- `partner_utilisation` = busy_partners / onshift_partners
- `orders_per_partner` = outstanding_orders / onshift_partners
- `available_partners` = onshift_partners - busy_partners

**Order complexity features:**
- `price_spread` = max_price - min_price
- `avg_item_price` = subtotal / total_items

### 3. Models Trained

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 11.69 mins | 15.29 | 0.18 |
| XGBoost (default) | 10.91 mins | 14.34 | 0.28 |
| LightGBM | 10.93 mins | 14.35 | 0.28 |
| XGBoost (tuned) | 10.90 mins | 14.32 | 0.28 |

### 4. Cross Validation
5-fold CV on XGBoost showed:
- Mean MAE: 10.93 minutes
- Std: 0.05 minutes
- Range: 10.88 - 11.02 minutes

Extremely stable across folds — model generalises consistently.

### 5. SHAP Feature Importance

Top features by mean absolute SHAP value:

| Feature | Mean |SHAP | Business Meaning |
|---------|-----------|-----------------|
| orders_per_partner | 4.15 mins | Partner workload pressure |
| subtotal | 3.84 mins | Order complexity proxy |
| available_partners | 1.92 mins | Supply availability |
| hour | 1.89 mins | Time-of-day demand pattern |
| order_protocol | 1.54 mins | Order channel friction |

---

## Key Findings

- Partner operational pressure (orders_per_partner) is the
  strongest single predictor of delivery time
- Order value (subtotal) outperforms item count as a
  complexity signal — monetary complexity better captures
  preparation time than quantity
- Peak hour flag showed negative correlation with delivery time —
  reflecting supply-demand balancing where platforms deploy
  more partners during high-demand periods
- Model reaches information ceiling of public dataset —
  adding GPS distance, real-time traffic, and store preparation
  history would be the highest-impact improvements

---

## Residual Analysis

- 30% of predictions within ±5 minutes
- 55% of predictions within ±10 minutes
- Residuals randomly distributed — no systematic bias
- Right-skewed error distribution reflects unpredictable
  edge cases (traffic, store delays) absent from dataset

---

## Limitations and Next Steps

Current limitations (all dataset-related, not model-related):
- No GPS distance between store and customer
- No real-time traffic signal
- No store-level preparation time history
- No delivery partner historical speed data

Production improvements:
- Add H3 geospatial indexing for zone-level demand features
- Integrate real-time traffic API signals
- Build model monitoring and drift detection pipeline
- Implement quantile regression for uncertainty estimation

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Chethanac29/delivery-eta-prediction.git
cd delivery-eta-prediction

# Install dependencies
uv sync

# Download dataset
# Go to: https://www.kaggle.com/datasets/ranitsarkar01/porter-delivery-time-estimation
# Place CSV in data/raw/

# Run notebooks in order
uv run jupyter notebook
```

---

## Tech Stack

Python, XGBoost, LightGBM, Scikit-learn, SHAP,
Pandas, NumPy, Matplotlib, Seaborn, uv

