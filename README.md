# Fraud Detection — Credit Card Transactions

End-to-end fraud detection on highly imbalanced transaction data — model comparison, cost-sensitive threshold selection, and a deployable FastAPI + Docker service.

## Problem

Detect fraudulent credit card transactions while accounting for the fact that a missed fraud (false negative) and a wrongly blocked transaction (false positive) carry different real-world costs. Accuracy is not a meaningful metric here given extreme class imbalance — PR-AUC is used instead.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB / Worldline) — 284,807 transactions, 492 frauds (0.172%). `V1`–`V28` are PCA-anonymized; `Time` and `Amount` are raw.

## Approach

1. Stratified train/test split
2. Baseline: Logistic Regression
3. Compared against Random Forest and XGBoost (5-fold stratified CV, PR-AUC)
4. Selected decision threshold via a cost matrix (missed fraud vs. false alarm)
5. Evaluated on held-out test set; error analysis on missed frauds
6. Served via FastAPI, containerised with Docker

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb
```

## Limitations

- Features are PCA-anonymized — limited interpretability
- Static historical dataset — no concept drift handling
- Cost matrix values are illustrative, not real business figures
