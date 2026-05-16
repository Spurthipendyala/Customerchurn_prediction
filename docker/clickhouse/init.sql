-- ClickHouse initialization SQL
-- Creates the churn_mlops database and all required tables

CREATE DATABASE IF NOT EXISTS churn_mlops;

-- Validated data table (post Great Expectations)
CREATE TABLE IF NOT EXISTS churn_mlops.churn_validated
(
    customerID          String,
    gender              String,
    SeniorCitizen       Int8,
    Partner             String,
    Dependents          String,
    tenure              Int32,
    PhoneService        String,
    MultipleLines       String,
    InternetService     String,
    OnlineSecurity      String,
    OnlineBackup        String,
    DeviceProtection    String,
    TechSupport         String,
    StreamingTV         String,
    StreamingMovies     String,
    Contract            String,
    PaperlessBilling    String,
    PaymentMethod       String,
    MonthlyCharges      Float64,
    TotalCharges        Float64,
    Churn               String,
    ingested_at         DateTime DEFAULT now(),
    pipeline_run_id     String DEFAULT ''
)
ENGINE = MergeTree()
ORDER BY (customerID, ingested_at)
SETTINGS index_granularity = 8192;

-- Processed/feature-engineered data table
CREATE TABLE IF NOT EXISTS churn_mlops.churn_processed
(
    customerID              String,
    tenure                  Int32,
    MonthlyCharges          Float64,
    TotalCharges            Float64,
    SeniorCitizen           Int8,
    Partner                 Int8,
    Dependents              Int8,
    PhoneService            Int8,
    MultipleLines           Int8,
    OnlineSecurity          Int8,
    OnlineBackup            Int8,
    DeviceProtection        Int8,
    TechSupport             Int8,
    StreamingTV             Int8,
    StreamingMovies         Int8,
    PaperlessBilling        Int8,
    gender_Male             Int8,
    InternetService_Fiber   Int8,
    InternetService_No      Int8,
    Contract_OneYear        Int8,
    Contract_TwoYear        Int8,
    PaymentMethod_CreditCard    Int8,
    PaymentMethod_ElecCheck     Int8,
    PaymentMethod_MailedCheck   Int8,
    tenure_group            String,
    charges_per_month       Float64,
    has_internet            Int8,
    num_services            Int8,
    Churn                   Int8,
    ingested_at             DateTime DEFAULT now(),
    pipeline_run_id         String DEFAULT '',
    dvc_commit_hash         String DEFAULT ''
)
ENGINE = MergeTree()
ORDER BY (customerID, ingested_at)
SETTINGS index_granularity = 8192;

-- Predictions table (for API inference logging)
CREATE TABLE IF NOT EXISTS churn_mlops.churn_predictions
(
    prediction_id       String,
    customerID          String,
    churn_probability   Float64,
    churn_prediction    Int8,
    model_name          String,
    model_version       String,
    request_timestamp   DateTime DEFAULT now(),
    latency_ms          Float64,
    pipeline_run_id     String DEFAULT ''
)
ENGINE = MergeTree()
ORDER BY (customerID, request_timestamp)
SETTINGS index_granularity = 8192;

-- Model metrics table (for tracking performance over time)
CREATE TABLE IF NOT EXISTS churn_mlops.model_metrics
(
    run_id          String,
    model_name      String,
    metric_name     String,
    metric_value    Float64,
    recorded_at     DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (model_name, recorded_at)
SETTINGS index_granularity = 8192;
