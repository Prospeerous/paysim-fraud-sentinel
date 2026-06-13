import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

SAMPLE_TX = {
    "step": 1, "type": "TRANSFER", "amount": 181.0,
    "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,  "newbalanceDest": 0.0,
}
THRESHOLD = 500_000.0
FEAT_NAMES = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "error_balance_orig", "error_balance_dest",
    "amount_to_balance_ratio", "is_large_transaction", "hour_of_step",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]


def test_balance_errors():
    from inference import engineer_features
    df = engineer_features(pd.DataFrame([SAMPLE_TX]), THRESHOLD)
    assert df["error_balance_orig"].iloc[0] == pytest.approx(0.0 - (181.0 - 181.0))
    assert df["error_balance_dest"].iloc[0] == pytest.approx(0.0 - (0.0 + 181.0))


def test_ratio_and_hour():
    from inference import engineer_features
    df = engineer_features(pd.DataFrame([SAMPLE_TX]), THRESHOLD)
    assert df["amount_to_balance_ratio"].iloc[0] == pytest.approx(181.0 / 182.0)
    assert df["hour_of_step"].iloc[0] == 1


def test_large_transaction_false():
    from inference import engineer_features
    df = engineer_features(pd.DataFrame([SAMPLE_TX]), THRESHOLD)
    assert df["is_large_transaction"].iloc[0] == 0


def test_large_transaction_true():
    from inference import engineer_features
    tx = {**SAMPLE_TX, "amount": 1_000_000.0}
    df = engineer_features(pd.DataFrame([tx]), THRESHOLD)
    assert df["is_large_transaction"].iloc[0] == 1


def _make_predictor(proba: float):
    from inference import Predictor
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - proba, proba]])
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, len(FEAT_NAMES)))
    scaler.feature_names_in_ = FEAT_NAMES
    return Predictor(model=model, scaler=scaler,
                     large_tx_threshold=THRESHOLD, optimal_threshold=0.5)


def test_predict_keys():
    p = _make_predictor(0.7)
    result = p.predict(SAMPLE_TX)
    assert {"fraud_probability", "is_fraud", "threshold"} <= result.keys()


def test_predict_fraud_flag():
    assert _make_predictor(0.8).predict(SAMPLE_TX)["is_fraud"] is True
    assert _make_predictor(0.2).predict(SAMPLE_TX)["is_fraud"] is False


def test_predict_batch():
    from inference import Predictor
    n = 2
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((n, len(FEAT_NAMES)))
    scaler.feature_names_in_ = FEAT_NAMES
    p = Predictor(model=model, scaler=scaler,
                  large_tx_threshold=THRESHOLD, optimal_threshold=0.5)
    results = p.predict_batch(pd.DataFrame([SAMPLE_TX, SAMPLE_TX]))
    assert len(results) == 2
    assert results[0]["is_fraud"] is True   # 0.8 >= 0.5
    assert results[1]["is_fraud"] is False  # 0.3 < 0.5
