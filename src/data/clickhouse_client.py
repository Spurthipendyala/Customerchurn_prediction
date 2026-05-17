"""
ClickHouse client utility for churn MLOps pipeline.
Handles connections, table creation, and data ingestion.
"""

import os
from typing import Optional

import clickhouse_connect
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class ClickHouseClient:
    """Manages ClickHouse connections and data operations for churn pipeline."""

    CREATE_VALIDATED_TABLE = """
    CREATE TABLE IF NOT EXISTS {database}.{table} (
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
    SETTINGS index_granularity = 8192
    """

    CREATE_PROCESSED_TABLE = """
    CREATE TABLE IF NOT EXISTS {database}.{table} (
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
    SETTINGS index_granularity = 8192
    """

    CREATE_PREDICTIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS {database}.{table} (
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
    SETTINGS index_granularity = 8192
    """

    CREATE_METRICS_TABLE = """
    CREATE TABLE IF NOT EXISTS {database}.{table} (
        run_id          String,
        model_name      String,
        metric_name     String,
        metric_value    Float64,
        recorded_at     DateTime DEFAULT now()
    )
    ENGINE = MergeTree()
    ORDER BY (model_name, recorded_at)
    SETTINGS index_granularity = 8192
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.host = host or os.getenv("CLICKHOUSE_HOST", "localhost")
        self.port = int(port or os.getenv("CLICKHOUSE_PORT", 8123))
        self.database = database or os.getenv("CLICKHOUSE_DATABASE", "churn_mlops")
        self.user = user or os.getenv("CLICKHOUSE_USER", "default")
        self.password = password or os.getenv("CLICKHOUSE_PASSWORD", "")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                database=self.database,
                username=self.user,
                password=self.password,
            )
        return self._client

    def ensure_database(self) -> None:
        """Create database if not exists."""
        self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        logger.info(f"✅ ClickHouse database '{self.database}' ready.")

    def create_validated_table(self, table_name: Optional[str] = None) -> None:
        """Create the validated data table."""
        table = table_name or os.getenv("CLICKHOUSE_VALIDATED_TABLE", "churn_validated")
        ddl = self.CREATE_VALIDATED_TABLE.format(database=self.database, table=table)
        self.client.command(ddl)
        logger.info(f"✅ Table '{self.database}.{table}' ready.")

    def create_processed_table(self, table_name: Optional[str] = None) -> None:
        """Create the processed/feature-engineered data table."""
        table = table_name or os.getenv("CLICKHOUSE_PROCESSED_TABLE", "churn_processed")
        ddl = self.CREATE_PROCESSED_TABLE.format(database=self.database, table=table)
        self.client.command(ddl)
        logger.info(f"✅ Table '{self.database}.{table}' ready.")

    def create_predictions_table(self, table_name: str = "churn_predictions") -> None:
        """Create the predictions logging table."""
        ddl = self.CREATE_PREDICTIONS_TABLE.format(
            database=self.database, table=table_name
        )
        self.client.command(ddl)
        logger.info(f"✅ Table '{self.database}.{table_name}' ready.")

    def create_metrics_table(self, table_name: str = "model_metrics") -> None:
        """Create the model metrics table."""
        ddl = self.CREATE_METRICS_TABLE.format(database=self.database, table=table_name)
        self.client.command(ddl)
        logger.info(f"✅ Table '{self.database}.{table_name}' ready.")

    def setup_all_tables(self) -> None:
        """Initialize database and all tables."""
        self.ensure_database()
        self.create_validated_table()
        self.create_processed_table()
        self.create_predictions_table()
        self.create_metrics_table()
        logger.success("🏠 All ClickHouse tables initialized.")

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        pipeline_run_id: str = "",
        dvc_commit_hash: str = "",
    ) -> int:
        """Insert a pandas DataFrame into a ClickHouse table."""
        df = df.copy()

        # Add metadata columns if not present
        if "pipeline_run_id" not in df.columns:
            df["pipeline_run_id"] = pipeline_run_id
        if "dvc_commit_hash" not in df.columns and "processed" in table:
            df["dvc_commit_hash"] = dvc_commit_hash

        rows_before = self._get_row_count(table)
        self.client.insert_df(table, df, database=self.database)
        rows_after = self._get_row_count(table)
        inserted = rows_after - rows_before

        logger.info(
            f"📥 Inserted {inserted} rows into '{self.database}.{table}'. "
            f"Total rows: {rows_after}"
        )
        return inserted

    def _get_row_count(self, table: str) -> int:
        """Get current row count for a table."""
        result = self.client.query(f"SELECT count() FROM {self.database}.{table}")
        return result.result_rows[0][0] if result.result_rows else 0

    def query_to_dataframe(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return as DataFrame."""
        result = self.client.query_df(sql)
        return result

    def get_latest_processed_data(self, table: Optional[str] = None) -> pd.DataFrame:
        """Fetch latest batch of processed data for monitoring."""
        table = table or os.getenv("CLICKHOUSE_PROCESSED_TABLE", "churn_processed")
        sql = """
        SELECT *
        FROM {self.database}.{table}
        WHERE ingested_at = (
            SELECT max(ingested_at) FROM {self.database}.{table}
        )
        """
        return self.query_to_dataframe(sql)

    def get_reference_data(
        self, table: Optional[str] = None, limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch reference dataset for drift monitoring."""
        table = table or os.getenv("CLICKHOUSE_PROCESSED_TABLE", "churn_processed")
        sql = """
        SELECT * FROM {self.database}.{table}
        ORDER BY ingested_at ASC
        LIMIT {limit}
        """
        return self.query_to_dataframe(sql)

    def get_stats(self) -> dict:
        """Get table statistics for monitoring."""
        validated_table = os.getenv("CLICKHOUSE_VALIDATED_TABLE", "churn_validated")
        processed_table = os.getenv("CLICKHOUSE_PROCESSED_TABLE", "churn_processed")

        stats = {}
        for table in [validated_table, processed_table]:
            try:
                count = self._get_row_count(table)
                stats[table] = {"row_count": count}
            except Exception as e:
                stats[table] = {"error": str(e)}
        return stats

    def close(self) -> None:
        """Close ClickHouse connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("🔌 ClickHouse connection closed.")


# Singleton instance
_ch_client: Optional[ClickHouseClient] = None


def get_clickhouse_client() -> ClickHouseClient:
    """Get or create the singleton ClickHouse client."""
    global _ch_client
    if _ch_client is None:
        _ch_client = ClickHouseClient()
    return _ch_client
