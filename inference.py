import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
TYPE_COLS  = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
              "type_PAYMENT", "type_TRANSFER"]


def engineer_features(df: pd.DataFrame, large_tx_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["error_balance_orig"]      = df["newbalanceOrig"] - (df["oldbalanceOrg"] - df["amount"])
    df["error_balance_dest"]      = df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"])
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["hour_of_step"]            = df["step"] % 24
    df["is_large_transaction"]    = (df["amount"] > large_tx_threshold).astype(int)
    df = pd.get_dummies(df, columns=["type"], drop_first=False)
    for col in TYPE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df


class Predictor:
    def __init__(self, model, scaler, large_tx_threshold: float, optimal_threshold: float):
        self.model               = model
        self.scaler              = scaler
        self.large_tx_threshold  = large_tx_threshold
        self.optimal_threshold   = optimal_threshold

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        df = engineer_features(df, self.large_tx_threshold)
        for col in self.scaler.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        return self.scaler.transform(df[self.scaler.feature_names_in_])

    def predict(self, transaction: dict) -> dict:
        X     = self._prepare(pd.DataFrame([transaction]))
        proba = float(self.model.predict_proba(X)[0, 1])
        return {
            "fraud_probability": round(proba, 4),
            "is_fraud":          bool(proba >= self.optimal_threshold),
            "threshold":         round(self.optimal_threshold, 4),
        }

    def predict_batch(self, df: pd.DataFrame) -> list[dict]:
        X      = self._prepare(df)
        probas = self.model.predict_proba(X)[:, 1]
        return [
            {
                "fraud_probability": round(float(p), 4),
                "is_fraud":          bool(p >= self.optimal_threshold),
                "threshold":         round(self.optimal_threshold, 4),
            }
            for p in probas
        ]


def load_predictor() -> Predictor:
    for path in [MODELS_DIR / "best_model.joblib",
                 MODELS_DIR / "scaler.joblib",
                 MODELS_DIR / "feature_params.json"]:
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
    params = json.loads((MODELS_DIR / "feature_params.json").read_text())
    return Predictor(
        model               = joblib.load(MODELS_DIR / "best_model.joblib"),
        scaler              = joblib.load(MODELS_DIR / "scaler.joblib"),
        large_tx_threshold  = params["large_tx_threshold"],
        optimal_threshold   = params["optimal_threshold"],
    )
