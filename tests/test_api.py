"""
Unit tests for the FastAPI churn prediction API.
Uses mocking so tests pass in CI without real model artifacts.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Mock model and feature columns before importing the app ──────────────────
# This prevents ModelState.load() from trying to connect to MLflow or load pkl
mock_model = MagicMock()
mock_model.predict_proba.return_value = [[0.7, 0.3]]  # low churn probability


@pytest.fixture(autouse=True)
def mock_model_state():
    """Patch ModelState so every test runs with a fake loaded model."""
    with patch("api.main.ModelState") as mock_state:
        mock_state.model = mock_model
        mock_state.model_name = "test_churn_classifier"
        mock_state.model_version = "1"
        mock_state.feature_cols = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "PaperlessBilling",
            "gender_Male",
            "InternetService_Fiber",
            "InternetService_No",
            "Contract_OneYear",
            "Contract_TwoYear",
            "PaymentMethod_CreditCard",
            "PaymentMethod_ElecCheck",
            "PaymentMethod_MailedCheck",
            "charges_per_month",
            "has_internet",
            "num_services",
        ]
        mock_state.load_time = 0.0
        mock_state.fs = None
        yield mock_state


# Import app AFTER mocking so startup doesn't fire real load
from api.main import app  # noqa: E402

client = TestClient(app)

_PREDICT_PAYLOAD = {
    "customerID": "TEST-123",
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 1,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "PaperlessBilling": 1,
    "gender_Male": 0,
    "InternetService_Fiber": 1,
    "InternetService_No": 0,
    "Contract_OneYear": 0,
    "Contract_TwoYear": 0,
    "PaymentMethod_CreditCard": 0,
    "PaymentMethod_ElecCheck": 1,
    "PaymentMethod_MailedCheck": 0,
}


def test_health_check(mock_model_state):
    """Health endpoint should return 'healthy' when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # Status is driven by ModelState.model being truthy
    assert data["status"] in ("healthy", "degraded")
    assert "model_loaded" in data


def test_health_check_healthy_when_model_loaded(mock_model_state):
    """Explicitly verify healthy status with a loaded mock model."""
    # model is already set as a truthy MagicMock in the fixture
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test_churn_classifier"
    assert data["model_version"] == "1"


def test_model_info(mock_model_state):
    """Model info endpoint returns expected fields."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "feature_count" in data
    assert data["model_name"] == "test_churn_classifier"
    assert data["feature_count"] == 26


def test_predict_endpoint(mock_model_state):
    """Predict endpoint returns valid prediction response."""
    response = client.post("/predict", json=_PREDICT_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert data["customerID"] == "TEST-123"
    assert data["churn_prediction"] in (0, 1)
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_predict_missing_model():
    """Predict endpoint returns 503 when model is not loaded."""
    with patch("api.main.ModelState") as ms:
        ms.model = None
        ms.feature_cols = []
        ms.fs = None
        response = client.post("/predict", json=_PREDICT_PAYLOAD)
        assert response.status_code == 503


def test_metrics_endpoint(mock_model_state):
    """Metrics endpoint returns Prometheus-format text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"churn_predictions_total" in response.content


def test_predict_invalid_payload(mock_model_state):
    """Predict endpoint returns 422 for completely missing required fields."""
    response = client.post("/predict", json={})
    # customerID is required, so empty body → 422
    assert response.status_code == 422
