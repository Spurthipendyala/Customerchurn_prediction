"""
Feast Feature Store definitions for Telco Churn MLOps.
Defines entities, feature views, and feature services.
Reads from ClickHouse-backed processed data (parquet offline store).
"""
import os
from pathlib import Path
from datetime import timedelta

import pandas as pd
import yaml
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def prepare_feature_data() -> None:
    """
    Prepare parquet data for Feast offline store from ClickHouse / local CSV.
    Adds event_timestamp and created_timestamp required by Feast.
    """
    params = load_params()
    processed_path = Path(params["data"]["processed_path"])
    feast_data_dir = Path("src/features/feature_repo/data")
    feast_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🍽️  Preparing feature data for Feast offline store...")

    # Try loading from ClickHouse first, fallback to CSV
    try:
        from src.data.clickhouse_client import get_clickhouse_client
        ch = get_clickhouse_client()
        database = os.getenv("CLICKHOUSE_DATABASE", "churn_mlops")
        processed_table = os.getenv("CLICKHOUSE_PROCESSED_TABLE", "churn_processed")
        df = ch.query_to_dataframe(
            f"SELECT * FROM {database}.{processed_table} "
            f"ORDER BY ingested_at DESC LIMIT 10000"
        )
        logger.success("✅ Feature data loaded from ClickHouse")
    except Exception as e:
        logger.warning(f"⚠️  ClickHouse unavailable ({e}), falling back to CSV")
        df = pd.read_csv(processed_path)

    # Feast requires event_timestamp and created_timestamp
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    df["event_timestamp"] = now
    df["created_timestamp"] = now

    # Ensure customerID is string
    df["customerID"] = df["customerID"].astype(str)

    # Save as parquet for Feast offline store
    output_path = feast_data_dir / "churn_features.parquet"
    df.to_parquet(output_path, index=False)
    logger.success(f"✅ Feast feature data saved → {output_path} ({len(df)} rows)")
    return df


def create_feature_repo() -> None:
    """Generate the Feast feature repository files."""
    repo_dir = Path("src/features/feature_repo")
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Write feature_store.yaml
    feature_store_config = {
        "project": "churn_mlops",
        "registry": "data/registry.db",
        "provider": "local",
        "online_store": {
            "type": "sqlite",
            "path": "data/online_store.db"
        },
        "offline_store": {
            "type": "file"
        },
        "entity_key_serialization_version": 2
    }
    import yaml as _yaml
    with open(repo_dir / "feature_store.yaml", "w", encoding="utf-8") as f:
        _yaml.dump(feature_store_config, f, default_flow_style=False)
    logger.info("✅ feature_store.yaml created")

    # Write features.py
    features_py = '''"""
Feast feature definitions for Telco Churn prediction.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import String, Int32, Int8, Float64

# ─── Entity ──────────────────────────────────────────────────────────────────
customer = Entity(
    name="customer",
    join_keys=["customerID"],
    description="Telco customer entity",
)

# ─── Data Source ─────────────────────────────────────────────────────────────
churn_source = FileSource(
    path="data/churn_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ─── Customer Demographics Feature View ──────────────────────────────────────
customer_demographics = FeatureView(
    name="customer_demographics",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="tenure",        dtype=Int32),
        Field(name="SeniorCitizen", dtype=Int8),
        Field(name="Partner",       dtype=Int8),
        Field(name="Dependents",    dtype=Int8),
        Field(name="gender_Male",   dtype=Int8),
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
        Field(name="PhoneService",          dtype=Int8),
        Field(name="MultipleLines",         dtype=Int8),
        Field(name="InternetService_Fiber", dtype=Int8),
        Field(name="InternetService_No",    dtype=Int8),
        Field(name="OnlineSecurity",        dtype=Int8),
        Field(name="OnlineBackup",          dtype=Int8),
        Field(name="DeviceProtection",      dtype=Int8),
        Field(name="TechSupport",           dtype=Int8),
        Field(name="StreamingTV",           dtype=Int8),
        Field(name="StreamingMovies",       dtype=Int8),
        Field(name="has_internet",          dtype=Int8),
        Field(name="num_services",          dtype=Int8),
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
        Field(name="PaperlessBilling",           dtype=Int8),
        Field(name="Contract_OneYear",           dtype=Int8),
        Field(name="Contract_TwoYear",           dtype=Int8),
        Field(name="PaymentMethod_CreditCard",   dtype=Int8),
        Field(name="PaymentMethod_ElecCheck",    dtype=Int8),
        Field(name="PaymentMethod_MailedCheck",  dtype=Int8),
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
'''
    with open(repo_dir / "features.py", "w", encoding="utf-8") as f:
        f.write(features_py)
    logger.info("✅ features.py created")


def materialize_features() -> None:
    """Apply and materialize features to the online store."""
    try:
        from feast import FeatureStore
        repo_dir = Path("src/features/feature_repo")
        store = FeatureStore(repo_path=str(repo_dir))
        store.apply([])  # Apply repo

        from datetime import datetime, timezone, timedelta
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=365)
        store.materialize(start_date=start_date, end_date=end_date)
        logger.success("✅ Features materialized to online store!")
    except Exception as e:
        logger.error(f"❌ Materialization failed: {e}")


if __name__ == "__main__":
    params = load_params()
    prepare_feature_data()
    create_feature_repo()
    logger.success("Feature store setup complete!")
