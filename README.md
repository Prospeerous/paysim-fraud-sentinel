# PaySim Fraud Sentinel

**Detect fraudulent mobile money transactions using machine learning.**

A locally-hosted web application that serves a trained Random Forest classifier for real-time fraud scoring on PaySim synthetic mobile money data. Supports single-transaction analysis and batch CSV scoring via a clean web interface.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Feature Engineering](#feature-engineering)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)

---

## Overview

Mobile money fraud is a critical challenge in fintech, particularly in emerging markets where mobile payments dominate financial transactions. This project trains and deploys a fraud detection model on the PaySim dataset — a synthetic simulation of real M-Pesa transaction logs — and serves it through a FastAPI backend with a browser-based interface.

The model focuses on **TRANSFER** and **CASH_OUT** transaction types, which account for 100% of fraudulent activity in the dataset.

---

## Dataset

**PaySim** — synthetic mobile money transactions generated using agent-based simulation calibrated against real M-Pesa logs from a month of financial activity in an African country.

| Property | Value |
|----------|-------|
| Source | [Kaggle — PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| Transactions | 6,362,620 |
| Fraud cases | 8,213 (0.13%) |
| Time horizon | 30 days (744 steps) |
| Transaction types | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |

The dataset is heavily imbalanced — fraud represents ~0.13% of all transactions — making precision-recall metrics more meaningful than accuracy.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest |
| PR-AUC | **0.997** |
| Decision threshold | 0.9592 (optimised for F1) |
| Large transaction threshold | 518,526 (dataset 75th percentile) |

The model was selected via cross-validated comparison against Logistic Regression, Decision Tree, and Gradient Boosting baselines. PR-AUC was used as the primary metric given the class imbalance.

---

## Feature Engineering

Five features are engineered from raw transaction fields:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `error_balance_orig` | `newbalanceOrig − (oldbalanceOrg − amount)` | Detects unexplained balance discrepancies on the sender side |
| `error_balance_dest` | `newbalanceDest − (oldbalanceDest + amount)` | Detects unexplained balance discrepancies on the receiver side |
| `amount_to_balance_ratio` | `amount / (oldbalanceOrg + 1)` | Flags account-draining behaviour |
| `hour_of_step` | `step % 24` | Captures intra-day transaction timing patterns |
| `is_large_transaction` | `amount > 518,526` (75th percentile) | Binary flag for unusually large transfers |

Transaction type is one-hot encoded into five binary columns.

---

## Architecture

```
Browser
  │
  │  GET /          → index.html (Tailwind CSS, vanilla JS)
  │  POST /predict  → single transaction JSON → fraud score
  │  POST /predict-batch → CSV upload → scored records
  │
FastAPI (app.py)
  │
  │  Pydantic validation → HTTP 422 on bad input
  │  Column validation   → HTTP 400 on bad CSV
  │
Inference (inference.py)
  │
  ├── engineer_features()   — pandas feature transforms
  ├── Predictor.predict()   — single-row scoring
  └── Predictor.predict_batch() — vectorised batch scoring
        │
        ├── StandardScaler  (scaler.joblib)
        └── RandomForest    (best_model.joblib)
```

All ML logic is isolated in `inference.py` and loaded once at startup. The web layer performs no feature computation.

---

## Project Structure

```
paysim-fraud-sentinel/
├── app.py                   # FastAPI application — routes and input validation
├── inference.py             # Feature engineering, Predictor class, artifact loading
├── requirements.txt         # Pinned dependencies
├── models/
│   ├── best_model.joblib    # Trained Random Forest (not in version control)
│   ├── scaler.joblib        # Fitted StandardScaler (not in version control)
│   └── feature_params.json  # Thresholds used during training
├── templates/
│   └── index.html           # Single-page UI (Tailwind CSS CDN, vanilla JS)
└── tests/
    └── test_inference.py    # Unit tests for feature engineering and prediction
```

> **Note:** `best_model.joblib` and `scaler.joblib` are excluded from version control due to file size. Download or re-train them using the training notebook in the parent repository.

---

## Getting Started

### Prerequisites

- Python 3.11+
- `best_model.joblib` and `scaler.joblib` placed in the `models/` directory

### Installation

```bash
git clone https://github.com/Prospeerous/paysim-fraud-sentinel.git
cd paysim-fraud-sentinel

pip install -r requirements.txt
```

Place your trained model artifacts in `models/`:
```
models/
├── best_model.joblib
├── scaler.joblib
└── feature_params.json     # already in the repo
```

### Running the App

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

---

## API Reference

### `POST /predict`

Score a single transaction.

**Request body:**
```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181.00,
  "oldbalanceOrg": 181.00,
  "newbalanceOrig": 0.00,
  "oldbalanceDest": 0.00,
  "newbalanceDest": 0.00
}
```

**Valid types:** `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`

**Response:**
```json
{
  "fraud_probability": 1.0,
  "is_fraud": true,
  "threshold": 0.9592
}
```

---

### `POST /predict-batch`

Score multiple transactions from a CSV file.

**Request:** `multipart/form-data` with a `file` field containing a `.csv`

**Required CSV columns:**
```
step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
```

**Response:** JSON array — original columns plus `fraud_probability`, `is_fraud`, `threshold` appended to each row.

---

## Running Tests

```bash
pytest tests/ -v
```

```
tests/test_inference.py::test_balance_errors          PASSED
tests/test_inference.py::test_ratio_and_hour          PASSED
tests/test_inference.py::test_large_transaction_false PASSED
tests/test_inference.py::test_large_transaction_true  PASSED
tests/test_inference.py::test_predict_keys            PASSED
tests/test_inference.py::test_predict_fraud_flag      PASSED
tests/test_inference.py::test_predict_batch           PASSED
7 passed
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn 1.6, pandas 2.2 |
| Backend | FastAPI 0.111, uvicorn |
| Frontend | Tailwind CSS (CDN), vanilla JS |
| Testing | pytest 8.2 |
| Serialisation | joblib |
