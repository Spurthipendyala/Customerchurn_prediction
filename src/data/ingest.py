"""
Data ingestion module with OpenLineage tracking.
Copies raw CSV into DVC-tracked data/raw/ and emits lineage events to Marquez.
"""
import os
import shutil
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ─── OpenLineage ─────────────────────────────────────────────────────────────
try:
    from openlineage.client import OpenLineageClient, set_producer
    from openlineage.client.run import (
        RunEvent, RunState, Run, Job, Dataset,
        InputDataset, OutputDataset
    )
    from openlineage.client.facet import (
        SchemaDatasetFacet, SchemaField,
        SourceCodeLocationJobFacet,
        DataQualityMetricsInputDatasetFacet,
        DataSourceDatasetFacet,
    )
    OPENLINEAGE_AVAILABLE = True
except ImportError:
    OPENLINEAGE_AVAILABLE = False
    logger.warning("⚠️  OpenLineage not available — lineage events will be skipped.")


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file for lineage tracking."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def emit_lineage_event(
    run_id: str,
    state: str,
    input_path: str,
    output_path: str,
    row_count: int = 0,
    namespace: str = "churn_mlops_pipeline",
) -> None:
    """Emit an OpenLineage event to Marquez."""
    if not OPENLINEAGE_AVAILABLE:
        return

    ol_url = os.getenv("OPENLINEAGE_URL", "http://localhost:5000")
    namespace = os.getenv("OPENLINEAGE_NAMESPACE", namespace)

    try:
        client = OpenLineageClient(url=ol_url)

        input_ds = InputDataset(
            namespace=namespace,
            name=input_path,
            facets={
                "schema": SchemaDatasetFacet(
                    fields=[
                        SchemaField("customerID", "STRING"),
                        SchemaField("gender", "STRING"),
                        SchemaField("SeniorCitizen", "INTEGER"),
                        SchemaField("tenure", "INTEGER"),
                        SchemaField("MonthlyCharges", "DOUBLE"),
                        SchemaField("TotalCharges", "DOUBLE"),
                        SchemaField("Churn", "STRING"),
                    ]
                ),
                "dataSource": DataSourceDatasetFacet(
                    name="telco_churn_csv",
                    uri=f"file://{Path(input_path).absolute()}"
                ),
            },
        )

        output_ds = OutputDataset(
            namespace=namespace,
            name=output_path,
            facets={
                "schema": SchemaDatasetFacet(
                    fields=[
                        SchemaField("customerID", "STRING"),
                        SchemaField("Churn", "STRING"),
                        SchemaField("row_count", "INTEGER"),
                    ]
                ),
            },
        )

        event = RunEvent(
            eventType=RunState[state],
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(
                namespace=namespace,
                name="data_ingestion",
                facets={
                    "sourceCodeLocation": SourceCodeLocationJobFacet(
                        type="git",
                        url="https://github.com/your-org/customerchurn_mlops",
                    )
                },
            ),
            inputs=[input_ds],
            outputs=[output_ds],
            producer="https://github.com/your-org/customerchurn_mlops",
        )

        client.emit(event)
        logger.info(f"📡 OpenLineage event emitted: {state} for run {run_id}")

    except Exception as e:
        logger.warning(f"⚠️  Could not emit lineage event: {e}")


def ingest_data() -> pd.DataFrame:
    """
    Main ingestion function:
    1. Loads raw CSV from Raw_data_set/
    2. Copies to data/raw/ (DVC-tracked)
    3. Emits OpenLineage START + COMPLETE events
    4. Returns DataFrame
    """
    params = load_params()
    run_id = str(uuid.uuid4())

    src_path = Path("Raw_data_set/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    dst_path = Path(params["data"]["raw_path"])
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"🔄 Starting data ingestion | run_id={run_id}")

    # ── Emit START lineage event ──────────────────────────────────────────────
    emit_lineage_event(
        run_id=run_id,
        state="START",
        input_path=str(src_path),
        output_path=str(dst_path),
    )

    # ── Load and validate shape ───────────────────────────────────────────────
    df = pd.read_csv(src_path)
    logger.info(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns from {src_path}")

    # ── Basic initial cleaning ────────────────────────────────────────────────
    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # TotalCharges: convert blanks to NaN then fill with 0 (tenure=0 customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_total = df["TotalCharges"].isna().sum()
    if blank_total > 0:
        logger.warning(f"⚠️  Found {blank_total} blank TotalCharges — filling with 0.0")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # ── Save to DVC-tracked location ──────────────────────────────────────────
    df.to_csv(dst_path, index=False)
    file_hash = compute_file_hash(str(dst_path))
    logger.success(f"✅ Raw data saved → {dst_path} | MD5: {file_hash}")

    # ── Emit COMPLETE lineage event ───────────────────────────────────────────
    emit_lineage_event(
        run_id=run_id,
        state="COMPLETE",
        input_path=str(src_path),
        output_path=str(dst_path),
        row_count=len(df),
    )

    # ── Log summary statistics ────────────────────────────────────────────────
    churn_rate = df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    logger.info(f"📈 Dataset Stats:")
    logger.info(f"   Rows: {len(df)} | Columns: {len(df.columns)}")
    logger.info(f"   Churn Rate: {churn_rate:.1f}%")
    logger.info(f"   Tenure range: {df['tenure'].min()}–{df['tenure'].max()} months")
    logger.info(f"   Monthly charges: ${df['MonthlyCharges'].mean():.2f} avg")

    return df


if __name__ == "__main__":
    df = ingest_data()
    print(f"\nIngestion complete: {len(df)} records")
    print(df.head(3))
