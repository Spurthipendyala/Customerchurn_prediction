import pytest
from fastapi.testclient import TestClient
from api.main import app, ModelState

# Ensure model is loaded for tests
try:
    ModelState.load()
except Exception:
    pass

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "feature_count" in data

def test_predict_endpoint():
    payload = {
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
        "PaymentMethod_MailedCheck": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert data["customerID"] == "TEST-123"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"churn_predictions_total" in response.content
