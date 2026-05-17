"""
Data Preprocessing & Feature Engineering.
Transforms validated data and stores processed features in ClickHouse.
Also emits OpenLineage events for full data lineage tracking.
"""

import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

from src.data.clickhouse_client import get_clickhouse_client

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def get_dvc_commit_hash() -> str:
    """Get current git commit hash for lineage tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def emit_preprocessing_lineage(run_id: str, state: str, input_path: str, output_path: str) -> None:
    """Emit OpenLineage event for preprocessing step."""
    try:
        from openlineage.client import OpenLineageClient
        from openlineage.client.facet import SchemaDatasetFacet, SchemaField
        from openlineage.client.run import (
            InputDataset,
            Job,
            OutputDataset,
            Run,
            RunEvent,
            RunState,
        )

        ol_url = os.getenv("OPENLINEAGE_URL", "http://localhost:5000")
        ns = os.getenv("OPENLINEAGE_NAMESPACE", "churn_mlops_pipeline")
        client = OpenLineageClient(url=ol_url)

        output_schema = SchemaDatasetFacet(
            fields=[
                SchemaField("customerID", "STRING"),
                SchemaField("tenure", "INTEGER"),
                SchemaField("MonthlyCharges", "DOUBLE"),
                SchemaField("TotalCharges", "DOUBLE"),
                SchemaField("Churn", "INTEGER"),
                SchemaField("num_services", "INTEGER"),
                SchemaField("charges_per_month", "DOUBLE"),
            ]
        )

        event = RunEvent(
            eventType=RunState[state],
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace=ns, name="data_preprocessing"),
            inputs=[InputDataset(namespace=ns, name=input_path)],
            outputs=[
                OutputDataset(namespace=ns, name=output_path),
                OutputDataset(
                    namespace=ns,
                    name="clickhouse://churn_mlops.churn_processed",
                    facets={"schema": output_schema},
                ),
            ],
            producer="https://github.com/your-org/customerchurn_mlops",
        )
        client.emit(event)
        logger.info(f"📡 Preprocessing lineage [{state}] emitted.")
    except Exception as e:
        logger.warning(f"⚠️  Lineage skipped: {e}")


def preprocess_data() -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Load validated data
    2. Encode categorical features
    3. Engineer new features
    4. Save processed data locally (DVC-tracked)
    5. Store in ClickHouse churn_processed table
    6. Emit OpenLineage events
    Returns: processed DataFrame
    """
    params = load_params()
    run_id = str(uuid.uuid4())
    dvc_hash = get_dvc_commit_hash()

    validated_path = Path(params["data"]["validated_path"])
    processed_path = Path(params["data"]["processed_path"])
    reference_path = Path(params["data"]["reference_path"])

    logger.info(f"🔧 Starting preprocessing | run_id={run_id} | commit={dvc_hash}")
    emit_preprocessing_lineage(run_id, "START", str(validated_path), str(processed_path))

    # ── Load validated data ───────────────────────────────────────────────────
    df = pd.read_csv(validated_path)
    logger.info(f"📂 Loaded {len(df)} rows for preprocessing")

    # ── 1. Target encoding ────────────────────────────────────────────────────
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int8")

    # ── 2. Binary categorical encoding (Yes/No → 1/0) ─────────────────────────
    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})
        df[col] = df[col].astype("int8")

    # ── 3. Gender encoding ────────────────────────────────────────────────────
    df["gender_Male"] = (df["gender"] == "Male").astype("int8")
    df.drop(columns=["gender"], inplace=True)

    # ── 4. InternetService one-hot ────────────────────────────────────────────
    df["InternetService_Fiber"] = (df["InternetService"] == "Fiber optic").astype("int8")
    df["InternetService_No"] = (df["InternetService"] == "No").astype("int8")
    df.drop(columns=["InternetService"], inplace=True)

    # ── 5. Contract one-hot ───────────────────────────────────────────────────
    df["Contract_OneYear"] = (df["Contract"] == "One year").astype("int8")
    df["Contract_TwoYear"] = (df["Contract"] == "Two year").astype("int8")
    df.drop(columns=["Contract"], inplace=True)

    # ── 6. PaymentMethod one-hot ──────────────────────────────────────────────
    df["PaymentMethod_CreditCard"] = (df["PaymentMethod"] == "Credit card (automatic)").astype("int8")
    df["PaymentMethod_ElecCheck"] = (df["PaymentMethod"] == "Electronic check").astype("int8")
    df["PaymentMethod_MailedCheck"] = (df["PaymentMethod"] == "Mailed check").astype("int8")
    df.drop(columns=["PaymentMethod"], inplace=True)

    # ── 7. Feature Engineering ────────────────────────────────────────────────
    # Tenure group (for segmentation and Feast feature views)
    def get_tenure_group(tenure: int) -> str:
        if tenure <= 12:
            return "new"
        elif tenure <= 24:
            return "developing"
        elif tenure <= 48:
            return "mature"
        else:
            return "loyal"

    df["tenure_group"] = df["tenure"].apply(get_tenure_group)

    # Charges per month (ratio feature)
    df["charges_per_month"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]).round(
        4
    )

    # Has internet service
    df["has_internet"] = (df["InternetService_No"] == 0).astype("int8")

    # Number of add-on services
    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].sum(axis=1).astype("int8")

    # ── 8. Final column ordering ──────────────────────────────────────────────
    feature_cols = [
        "customerID",
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
        "tenure_group",
        "charges_per_month",
        "has_internet",
        "num_services",
        "Churn",
    ]
    df = df[feature_cols]

    # ── 9. Save processed data locally ────────────────────────────────────────
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.success(f"✅ Processed data saved → {processed_path}")

    # ── 10. Save reference dataset for Evidently drift monitoring ─────────────
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_df = df.sample(n=min(1000, len(df)), random_state=42)
    reference_df.to_csv(reference_path, index=False)
    logger.info(f"📌 Reference dataset saved → {reference_path} ({len(reference_df)} rows)")

    # ── 11. Store processed data in ClickHouse ────────────────────────────────
    try:
        ch = get_clickhouse_client()
        ch.setup_all_tables()

        df_ch = df.copy()
        df_ch["pipeline_run_id"] = run_id
        df_ch["dvc_commit_hash"] = dvc_hash

        # Ensure int8 for ClickHouse
        int8_cols = [
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
            "has_internet",
            "num_services",
            "Churn",
        ]
        for col in int8_cols:
            if col in df_ch.columns:
                df_ch[col] = df_ch[col].astype("int8")

        processed_table = os.getenv("CLICKHOUSE_PROCESSED_TABLE", "churn_processed")
        rows_inserted = ch.insert_dataframe(
            df=df_ch,
            table=processed_table,
            pipeline_run_id=run_id,
            dvc_commit_hash=dvc_hash,
        )
        logger.success(f"🏠 ClickHouse: Inserted {rows_inserted} processed rows → " f"churn_mlops.{processed_table}")

        # Log ClickHouse table stats
        stats = ch.get_stats()
        for table, info in stats.items():
            logger.info(f"   📊 {table}: {info.get('row_count', 'N/A')} rows")

    except Exception as e:
        logger.error(f"❌ ClickHouse insertion failed: {e}")
        logger.warning("⚠️  Continuing without ClickHouse storage...")

    # ── 12. Emit COMPLETE lineage event ───────────────────────────────────────
    emit_preprocessing_lineage(run_id, "COMPLETE", str(validated_path), str(processed_path))

    # ── Log final summary ─────────────────────────────────────────────────────
    churn_rate = df["Churn"].mean()
    logger.info("📊 Preprocessing Summary:")
    logger.info(f"   Input rows: {len(df)}")
    logger.info(f"   Output features: {len(df.columns) - 1}")
    logger.info(f"   Churn rate: {churn_rate:.2%}")
    logger.info("   New features: tenure_group, charges_per_month, has_internet, num_services")
    logger.info(f"   Tenure groups: {df['tenure_group'].value_counts().to_dict()}")
    logger.info(f"   Avg services per customer: {df['num_services'].mean():.1f}")

    return df


if __name__ == "__main__":
    df = preprocess_data()
    print(f"\nPreprocessing complete: {len(df)} rows, {len(df.columns)} columns")
    print("\nFeature dtypes:")
    print(df.dtypes)
    print("\nSample:")
    print(df.head(3))
