"""
Feast feature definitions for Telco Churn prediction.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import String, Int32, Float64

# ─── Entity ──────────────────────────────────────────────────────────────────
customer = Entity(
    name="customer",
    join_keys=["customerID"],
    description="Telco customer entity",
)

# ─── Data Source ─────────────────────────────────────────────────────────────
churn_source = FileSource(
    path="data/churn_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ─── Customer Demographics Feature View ──────────────────────────────────────
customer_demographics = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="tenure",        dtype=Int32),
        Field(name="SeniorCitizen", dtype=Int32),
        Field(name="Partner",       dtype=Int32),
        Field(name="Dependents",    dtype=Int32),
        Field(name="gender_Male",   dtype=Int32),
        Field(name="tenure_group",  dtype=String),
    ],
    source=churn_source,
    description="Customer demographic features",
    tags={"team": "mlops", "domain": "demographics"},
)

# ─── Service & Billing Feature View ──────────────────────────────────────────
customer_services = FeatureView(
    name="customer_services",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="PhoneService",          dtype=Int32),
        Field(name="MultipleLines",         dtype=Int32),
        Field(name="InternetService_Fiber", dtype=Int32),
        Field(name="InternetService_No",    dtype=Int32),
        Field(name="OnlineSecurity",        dtype=Int32),
        Field(name="OnlineBackup",          dtype=Int32),
        Field(name="DeviceProtection",      dtype=Int32),
        Field(name="TechSupport",           dtype=Int32),
        Field(name="StreamingTV",           dtype=Int32),
        Field(name="StreamingMovies",       dtype=Int32),
        Field(name="has_internet",          dtype=Int32),
        Field(name="num_services",          dtype=Int32),
    ],
    source=churn_source,
    description="Customer service subscription features",
    tags={"team": "mlops", "domain": "services"},
)

# ─── Billing Feature View ─────────────────────────────────────────────────────
customer_billing = FeatureView(
    name="customer_billing",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="MonthlyCharges",             dtype=Float64),
        Field(name="TotalCharges",               dtype=Float64),
        Field(name="charges_per_month",          dtype=Float64),
        Field(name="PaperlessBilling",           dtype=Int32),
        Field(name="Contract_OneYear",           dtype=Int32),
        Field(name="Contract_TwoYear",           dtype=Int32),
        Field(name="PaymentMethod_CreditCard",   dtype=Int32),
        Field(name="PaymentMethod_ElecCheck",    dtype=Int32),
        Field(name="PaymentMethod_MailedCheck",  dtype=Int32),
    ],
    source=churn_source,
    description="Customer billing and contract features",
    tags={"team": "mlops", "domain": "billing"},
)

# ─── Feature Service (combines all views for model inference) ─────────────────
churn_feature_service = FeatureService(
    name="churn_features",
    features=[
        customer_demographics,
        customer_services,
        customer_billing,
    ],
    description="Complete feature set for churn prediction model",
    tags={"model": "churn_classifier", "version": "v1"},
)
