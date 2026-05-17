"""
FastAPI model serving with Prometheus metrics, ClickHouse prediction logging,
and OpenLineage event emission.
"""

import json
import os
import pickle
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from feast import FeatureStore
from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.monitoring.retraining_trigger import trigger_retraining

load_dotenv()

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production MLOps API for predicting telecom customer churn",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total number of churn predictions",
    ["model_name", "prediction_result"],
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
CHURN_PROBABILITY = Histogram(
    "churn_probability_distribution",
    "Distribution of predicted churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)
ACTIVE_REQUESTS = Gauge("churn_active_requests", "Number of active requests")
MODEL_VERSION_GAUGE = Gauge("churn_model_version", "Current model version", ["model_name"])
DATA_DRIFT_SCORE = Gauge("churn_drift_score", "Latest data drift score")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    customerID: str = Field(..., example="7590-VHVEG")
    tenure: Optional[int] = Field(None, ge=0, le=120, example=12)
    MonthlyCharges: Optional[float] = Field(None, ge=0, le=200, example=65.5)
    TotalCharges: Optional[float] = Field(None, ge=0, example=786.0)
    SeniorCitizen: Optional[int] = Field(None, ge=0, le=1, example=0)
    Partner: Optional[int] = Field(None, ge=0, le=1, example=1)
    Dependents: Optional[int] = Field(None, ge=0, le=1, example=0)
    PhoneService: Optional[int] = Field(None, ge=0, le=1, example=1)
    MultipleLines: Optional[int] = Field(None, ge=0, le=1, example=0)
    OnlineSecurity: Optional[int] = Field(None, ge=0, le=1, example=0)
    OnlineBackup: Optional[int] = Field(None, ge=0, le=1, example=1)
    DeviceProtection: Optional[int] = Field(None, ge=0, le=1, example=0)
    TechSupport: Optional[int] = Field(None, ge=0, le=1, example=0)
    StreamingTV: Optional[int] = Field(None, ge=0, le=1, example=0)
    StreamingMovies: Optional[int] = Field(None, ge=0, le=1, example=0)
    PaperlessBilling: Optional[int] = Field(None, ge=0, le=1, example=1)
    gender_Male: Optional[int] = Field(None, ge=0, le=1, example=0)
    InternetService_Fiber: Optional[int] = Field(None, ge=0, le=1, example=0)
    InternetService_No: Optional[int] = Field(None, ge=0, le=1, example=0)
    Contract_OneYear: Optional[int] = Field(None, ge=0, le=1, example=0)
    Contract_TwoYear: Optional[int] = Field(None, ge=0, le=1, example=0)
    PaymentMethod_CreditCard: Optional[int] = Field(None, ge=0, le=1, example=0)
    PaymentMethod_ElecCheck: Optional[int] = Field(None, ge=0, le=1, example=1)
    PaymentMethod_MailedCheck: Optional[int] = Field(None, ge=0, le=1, example=0)
    charges_per_month: Optional[float] = Field(None, example=65.5)
    has_internet: Optional[int] = Field(None, ge=0, le=1, example=1)
    num_services: Optional[int] = Field(None, ge=0, le=10, example=2)

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
                "SeniorCitizen": 0,
                "Partner": 1,
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
                "InternetService_Fiber": 0,
                "InternetService_No": 0,
                "Contract_OneYear": 0,
                "Contract_TwoYear": 0,
                "PaymentMethod_CreditCard": 0,
                "PaymentMethod_ElecCheck": 1,
                "PaymentMethod_MailedCheck": 0,
            }
        }


class PredictionResponse(BaseModel):
    prediction_id: str
    customerID: str
    churn_probability: float
    churn_prediction: int
    churn_label: str
    risk_level: str
    model_name: str
    model_version: str
    prediction_timestamp: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    churn_count: int
    churn_rate: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float
    timestamp: str


# ── Model State ───────────────────────────────────────────────────────────────
class ModelState:
    model = None
    model_name: str = ""
    model_version: str = "0"
    feature_cols: List[str] = []
    load_time: float = time.time()
    fs: Optional[FeatureStore] = None

    @classmethod
    def load(cls) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(tracking_uri)
        model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "churn_classifier")
        model_stage = os.getenv("MODEL_STAGE", "Production")

        # Initialize Feast
        try:
            repo_path = os.getenv("FEAST_REPO_PATH", "src/features/feature_repo")
            cls.fs = FeatureStore(repo_path=repo_path)
            logger.success("✅ Feast Feature Store initialized")
        except Exception as e:
            logger.warning(f"⚠️  Feast initialization failed: {e}")

        # Try MLflow registry first
        try:
            model_uri = f"models:/{model_name}/{model_stage}"
            cls.model = mlflow.pyfunc.load_model(model_uri)
            cls.model_name = model_name
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[model_stage])
            cls.model_version = versions[0].version if versions else "1"
            logger.success(f"✅ Model loaded from MLflow: {model_name} v{cls.model_version}")
        except Exception as e:
            logger.warning(f"⚠️  MLflow load failed ({e}), loading from local pkl...")
            pkl_path = Path("artifacts/models/best_model.pkl")
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    cls.model = pickle.load(f)
                cls.model_name = "local_churn_classifier"
                cls.model_version = "local"
                logger.success("✅ Model loaded from local pkl")
            else:
                raise RuntimeError("No model available. Run training first.")

        # Load feature columns
        fc_path = Path("artifacts/models/feature_columns.json")
        if fc_path.exists():
            with open(fc_path) as f:
                cls.feature_cols = json.load(f)
        logger.info(f"📋 Feature columns: {len(cls.feature_cols)}")
        MODEL_VERSION_GAUGE.labels(model_name=cls.model_name).set(float(cls.model_version.replace("local", "0")))


def customer_to_features(customer: CustomerFeatures, feature_cols: List[str]) -> pd.DataFrame:
    """Convert CustomerFeatures to model input DataFrame."""
    data = customer.model_dump(exclude={"customerID"})

    # Derive optional features if not provided
    tenure = data["tenure"]
    monthly = data["MonthlyCharges"]
    total = data["TotalCharges"]

    if data.get("charges_per_month") is None:
        data["charges_per_month"] = total / tenure if tenure > 0 else monthly
    if data.get("has_internet") is None:
        data["has_internet"] = 1 - data.get("InternetService_No", 0)
    if data.get("num_services") is None:
        svc = sum(
            [
                data.get(c, 0)
                for c in [
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                ]
            ]
        )
        data["num_services"] = svc

    df = pd.DataFrame([data])

    # Align to model feature columns
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

    return df


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    elif probability < 0.5:
        return "MEDIUM"
    elif probability < 0.7:
        return "HIGH"
    else:
        return "CRITICAL"


async def log_prediction_to_clickhouse(response: PredictionResponse) -> None:
    """Background task: log prediction to ClickHouse."""
    try:
        from src.data.clickhouse_client import get_clickhouse_client

        ch = get_clickhouse_client()
        db = os.getenv("CLICKHOUSE_DATABASE", "churn_mlops")
        ch.client.command(f"""
            INSERT INTO {db}.churn_predictions
            (prediction_id, customerID, churn_probability, churn_prediction,
             model_name, model_version, latency_ms)
            VALUES ('{response.prediction_id}', '{response.customerID}',
                    {response.churn_probability}, {response.churn_prediction},
                    '{response.model_name}', '{response.model_version}',
                    {response.latency_ms})
        """)
    except Exception as e:
        logger.warning(f"⚠️  Failed to log prediction to ClickHouse: {e}")


# ── Lifecycle Events ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Churn Prediction API...")
    ModelState.load()
    logger.success("✅ API ready!")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if ModelState.model else "degraded",
        model_loaded=ModelState.model is not None,
        model_name=ModelState.model_name,
        model_version=ModelState.model_version,
        uptime_seconds=round(time.time() - ModelState.load_time, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    from fastapi.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    customer: CustomerFeatures,
    background_tasks: BackgroundTasks,
):
    """Predict churn probability for a single customer."""
    if not ModelState.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()

    try:
        # Check if we need to fetch features from Feast
        # We fetch from Feast if only customerID is provided (most other fields are None)
        input_data = customer.model_dump()
        filled_fields = [k for k, v in input_data.items() if v is not None and k != "customerID"]

        if len(filled_fields) == 0 and ModelState.fs:
            logger.info(f"🔍 Fetching features from Feast for customer: {customer.customerID}")
            try:
                feature_vector = ModelState.fs.get_online_features(
                    features=[
                        "customer_demographics:tenure",
                        "customer_demographics:SeniorCitizen",
                        "customer_demographics:Partner",
                        "customer_demographics:Dependents",
                        "customer_demographics:gender_Male",
                        "customer_demographics:tenure_group",
                        "customer_services:PhoneService",
                        "customer_services:MultipleLines",
                        "customer_services:InternetService_Fiber",
                        "customer_services:InternetService_No",
                        "customer_services:OnlineSecurity",
                        "customer_services:OnlineBackup",
                        "customer_services:DeviceProtection",
                        "customer_services:TechSupport",
                        "customer_services:StreamingTV",
                        "customer_services:StreamingMovies",
                        "customer_services:has_internet",
                        "customer_services:num_services",
                        "customer_billing:MonthlyCharges",
                        "customer_billing:TotalCharges",
                        "customer_billing:charges_per_month",
                        "customer_billing:PaperlessBilling",
                        "customer_billing:Contract_OneYear",
                        "customer_billing:Contract_TwoYear",
                        "customer_billing:PaymentMethod_CreditCard",
                        "customer_billing:PaymentMethod_ElecCheck",
                        "customer_billing:PaymentMethod_MailedCheck",
                    ],
                    entity_rows=[{"customerID": customer.customerID}],
                ).to_dict()

                # Update input data with fetched features
                for k, v in feature_vector.items():
                    # Feast returns feature names as 'view:feature' or just 'feature'
                    feat_name = k.split(":")[-1] if ":" in k else k
                    if feat_name in input_data:
                        input_data[feat_name] = v[0]

                # Re-create customer object with filled features
                customer = CustomerFeatures(**input_data)
            except Exception as e:
                logger.error(f"❌ Feast fetch failed: {e}")
                raise HTTPException(status_code=500, detail=f"Feast fetch failed: {e}")

        df = customer_to_features(customer, ModelState.feature_cols)
        proba = float(ModelState.model.predict_proba(df)[0][1])
        pred = int(proba >= 0.5)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        response = PredictionResponse(
            prediction_id=str(uuid.uuid4()),
            customerID=customer.customerID,
            churn_probability=round(proba, 4),
            churn_prediction=pred,
            churn_label="Churn" if pred == 1 else "No Churn",
            risk_level=get_risk_level(proba),
            model_name=ModelState.model_name,
            model_version=ModelState.model_version,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
        )

        # Prometheus
        PREDICTION_COUNTER.labels(
            model_name=ModelState.model_name,
            prediction_result="churn" if pred else "no_churn",
        ).inc()
        PREDICTION_LATENCY.observe(latency_ms / 1000)
        CHURN_PROBABILITY.observe(proba)

        # Background: log to ClickHouse
        background_tasks.add_task(log_prediction_to_clickhouse, response)

        return response

    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
):
    """Predict churn probability for a batch of customers."""
    if not ModelState.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    predictions = []

    for customer in request.customers:
        df = customer_to_features(customer, ModelState.feature_cols)
        proba = float(ModelState.model.predict_proba(df)[0][1])
        pred = int(proba >= 0.5)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        pred_resp = PredictionResponse(
            prediction_id=str(uuid.uuid4()),
            customerID=customer.customerID,
            churn_probability=round(proba, 4),
            churn_prediction=pred,
            churn_label="Churn" if pred == 1 else "No Churn",
            risk_level=get_risk_level(proba),
            model_name=ModelState.model_name,
            model_version=ModelState.model_version,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
        )
        predictions.append(pred_resp)
        background_tasks.add_task(log_prediction_to_clickhouse, pred_resp)

    churn_count = sum(p.churn_prediction for p in predictions)
    total_time = round((time.perf_counter() - start) * 1000, 3)

    return BatchPredictionResponse(
        predictions=predictions,
        total_customers=len(predictions),
        churn_count=churn_count,
        churn_rate=round(churn_count / len(predictions), 4),
        processing_time_ms=total_time,
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get current model metadata."""
    return {
        "model_name": ModelState.model_name,
        "model_version": ModelState.model_version,
        "feature_count": len(ModelState.feature_cols),
        "features": ModelState.feature_cols,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/drift/latest", tags=["Monitoring"])
async def get_latest_drift():
    """Get latest Evidently drift report summary."""
    report_dir = Path("artifacts/drift_reports")
    reports = sorted(report_dir.glob("*.json"), reverse=True) if report_dir.exists() else []
    if not reports:
        return {"message": "No drift reports available yet."}
    with open(reports[0]) as f:
        return json.load(f)


@app.get("/predictions/history", tags=["Monitoring"])
async def get_prediction_history(limit: int = 50):
    """Fetch recent predictions from ClickHouse."""
    try:
        from src.data.clickhouse_client import get_clickhouse_client

        ch = get_clickhouse_client()
        db = os.getenv("CLICKHOUSE_DATABASE", "churn_mlops")
        sql = f"""
            SELECT * FROM {db}.churn_predictions
            ORDER BY request_timestamp DESC
            LIMIT {limit}
        """
        df = ch.query_to_dataframe(sql)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return []


@app.post("/retrain", tags=["Model"])
async def manual_retrain(background_tasks: BackgroundTasks):
    """Manually trigger model retraining pipeline."""
    logger.info("🖱️  Manual retraining requested via API")
    background_tasks.add_task(trigger_retraining, reason="Manual trigger via API")
    return {"message": "Retraining pipeline triggered in background."}
