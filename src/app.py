"""
FastAPI service for the fraud detection model.

Loads the trained pipeline + chosen threshold (both produced by the notebook)
and exposes a single POST endpoint: /predict

Run locally (from project root):
    uvicorn src.app:app --reload --port 8000

Then visit http://localhost:8000/docs for the auto-generated interactive UI.
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load the trained pipeline and threshold ONCE at startup (not per-request —
# loading from disk on every request would be slow and wasteful).
# ---------------------------------------------------------------------------
MODEL_PATH = "models/fraud_model.joblib"
THRESHOLD_PATH = "models/threshold.joblib"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

app = FastAPI(
    title="Fraud Detection API",
    description="Predicts whether a credit card transaction is fraudulent.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Define the expected input shape.
# The dataset has Time, Amount, and V1..V28 (PCA-anonymized features).
# Pydantic validates incoming JSON against this automatically — if a field
# is missing or the wrong type, FastAPI returns a clear 422 error itself,
# before our code even runs.
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0, "Amount": 123.45,
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
            }
        }


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float


@app.get("/")
def root():
    """Simple health check — confirms the API is up."""
    return {"status": "ok", "message": "Fraud Detection API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Takes a single transaction's features and returns a fraud probability
    plus a yes/no flag based on the threshold selected in the notebook
    (via the cost-sensitive analysis, not the default 0.5 cutoff).
    """
    # Convert the validated Pydantic object into a single-row DataFrame,
    # matching the column structure the model was trained on.
    TRAINED_COLUMN_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    input_df = pd.DataFrame([transaction.model_dump()])
    input_df = input_df[TRAINED_COLUMN_ORDER]

    # predict_proba returns [prob_not_fraud, prob_fraud] — we want index 1
    fraud_prob = model.predict_proba(input_df)[0, 1]
    is_fraud = bool(fraud_prob >= threshold)

    return PredictionResponse(
        fraud_probability=round(float(fraud_prob), 8),
        is_fraud=is_fraud,
        threshold_used=round(float(threshold), 4),
    )
