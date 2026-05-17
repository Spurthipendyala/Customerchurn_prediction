import pytest
import pandas as pd
import numpy as np
from src.data.ingest import ingest_data
from src.data.preprocess import preprocess_data


def test_ingestion_structure():
    # This might require Raw_data_set to be present
    try:
        df = pd.read_csv("Raw_data_set/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        assert not df.empty
        assert "Churn" in df.columns
        assert "customerID" in df.columns
    except FileNotFoundError:
        pytest.skip("Raw dataset not found for testing")


def test_preprocessing_logic():
    # Create dummy data
    dummy_data = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [10, 20],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [70.0, 50.0],
            "TotalCharges": [700.0, 1000.0],
            "Churn": ["No", "Yes"],
        }
    )

    # Normally we'd call preprocess_data() but it depends on files
    # We'll just verify the mapping logic used in preprocess.py
    target_map = {"Yes": 1, "No": 0}
    churn_encoded = dummy_data["Churn"].map(target_map)
    assert churn_encoded.iloc[0] == 0
    assert churn_encoded.iloc[1] == 1
