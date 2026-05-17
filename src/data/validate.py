import json

"""
Data Validation using Great Expectations.
Validates raw data, generates Data Docs, and stores validated data in ClickHouse.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from great_expectations.core.batch import RuntimeBatchRequest
from loguru import logger

import great_expectations as gx
from src.data.clickhouse_client import get_clickhouse_client

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


# ── OpenLineage lineage helper ───────────────────────────────────────────────
def emit_validation_lineage(
    run_id: str, state: str, input_path: str, success: bool
) -> None:
    try:
        from openlineage.client import OpenLineageClient
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

        event = RunEvent(
            eventType=RunState[state],
            eventTime=datetime.now(timezone.utc).isoformat(),
            run=Run(runId=run_id),
            job=Job(namespace=ns, name="data_validation"),
            inputs=[InputDataset(namespace=ns, name=input_path)],
            outputs=[
                OutputDataset(
                    namespace=ns, name="clickhouse://churn_mlops.churn_validated"
                )
            ],
            producer="https://github.com/your-org/customerchurn_mlops",
        )
        client.emit(event)
        logger.info(f"📡 Validation lineage event [{state}] emitted.")
    except Exception as e:
        logger.warning(f"⚠️  Lineage emit skipped: {e}")


def build_expectation_suite(context, suite_name: str) -> None:
    """Build comprehensive expectation suite for the churn dataset."""
    suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    validator = context.get_validator(
        batch_request=RuntimeBatchRequest(
            datasource_name="churn_datasource",
            data_connector_name="default_runtime_data_connector",
            data_asset_name="churn_data",
            runtime_parameters={"path": "data/raw/churn_raw.csv"},
            batch_identifiers={"run_id": "setup"},
        ),
        expectation_suite_name=suite_name,
    )

    # ── Schema & Completeness ────────────────────────────────────────────────
    required_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    validator.expect_table_columns_to_match_ordered_list(column_list=required_columns)
    validator.expect_table_row_count_to_be_between(min_value=1000, max_value=100000)

    # ── customerID ──────────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="customerID")
    validator.expect_column_values_to_be_unique(column="customerID")
    validator.expect_column_value_lengths_to_equal(column="customerID", value=10)

    # ── gender ──────────────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="gender")
    validator.expect_column_values_to_be_in_set(
        column="gender", value_set=["Male", "Female"]
    )

    # ── SeniorCitizen ────────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="SeniorCitizen")
    validator.expect_column_values_to_be_in_set(
        column="SeniorCitizen", value_set=[0, 1]
    )

    # ── Partner / Dependents ─────────────────────────────────────────────────
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        validator.expect_column_values_to_not_be_null(column=col)
        validator.expect_column_values_to_be_in_set(column=col, value_set=["Yes", "No"])

    # ── tenure ───────────────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="tenure")
    validator.expect_column_values_to_be_between(
        column="tenure", min_value=0, max_value=120
    )
    validator.expect_column_mean_to_be_between(
        column="tenure", min_value=10, max_value=50
    )

    # ── MonthlyCharges ───────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="MonthlyCharges")
    validator.expect_column_values_to_be_between(
        column="MonthlyCharges", min_value=0, max_value=200
    )
    validator.expect_column_mean_to_be_between(
        column="MonthlyCharges", min_value=30, max_value=100
    )

    # ── TotalCharges ─────────────────────────────────────────────────────────
    validator.expect_column_values_to_be_between(
        column="TotalCharges", min_value=0, max_value=10000, mostly=0.99
    )

    # ── InternetService ──────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="InternetService")
    validator.expect_column_values_to_be_in_set(
        column="InternetService", value_set=["DSL", "Fiber optic", "No"]
    )

    # ── Contract ─────────────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="Contract")
    validator.expect_column_values_to_be_in_set(
        column="Contract", value_set=["Month-to-month", "One year", "Two year"]
    )

    # ── PaymentMethod ────────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="PaymentMethod")
    validator.expect_column_values_to_be_in_set(
        column="PaymentMethod",
        value_set=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    # ── MultipleLines ─────────────────────────────────────────────────────────
    validator.expect_column_values_to_be_in_set(
        column="MultipleLines", value_set=["Yes", "No", "No phone service"]
    )

    # ── Churn (target) ───────────────────────────────────────────────────────
    validator.expect_column_values_to_not_be_null(column="Churn")
    validator.expect_column_values_to_be_in_set(column="Churn", value_set=["Yes", "No"])
    validator.expect_column_proportion_of_unique_values_to_be_between(
        column="Churn", min_value=0.0, max_value=0.5
    )

    # ── Churn rate distribution ───────────────────────────────────────────────
    validator.expect_column_kl_divergence_to_be_less_than(
        column="tenure", partition_object=None, threshold=None, mostly=0.95
    )

    validator.save_expectation_suite(discard_failed_expectations=False)
    logger.success(
        f"✅ Expectation suite '{suite_name}' built with {len(suite.expectations)} expectations."
    )


def setup_ge_context():
    """Initialize Great Expectations DataContext."""
    ge_dir = Path("great_expectations")
    ge_dir.mkdir(exist_ok=True)

    context = gx.get_context(
        context_root_dir=str(ge_dir),
        mode="file",
    )

    # Add CSV datasource
    try:
        context.sources.add_pandas_filesystem(
            name="churn_datasource",
            base_directory="./",
        )
    except Exception:
        pass  # Already exists

    return context


def validate_data() -> Tuple[pd.DataFrame, bool]:
    """
    Main validation function:
    1. Loads raw data from data/raw/churn_raw.csv
    2. Runs Great Expectations validation suite
    3. Generates HTML Data Docs
    4. Stores VALIDATED data in ClickHouse churn_validated table
    5. Emits OpenLineage events
    Returns: (validated_dataframe, success_bool)
    """
    params = load_params()
    run_id = str(uuid.uuid4())
    raw_path = Path(params["data"]["raw_path"])
    validated_path = Path(params["data"]["validated_path"])
    suite_name = params["great_expectations"]["suite_name"]

    logger.info(f"🔍 Starting data validation | run_id={run_id}")
    emit_validation_lineage(run_id, "START", str(raw_path), success=False)

    # ── Load raw data ─────────────────────────────────────────────────────────
    df = pd.read_csv(raw_path)
    logger.info(f"📂 Loaded {len(df)} rows for validation")

    # ── Run validation ────────────────────────────────────────────────────────
    validation_success = True

    # Manual validation checks (always run even if GE not configured)
    checks = {
        "no_null_customerID": df["customerID"].notna().all(),
        "unique_customerID": df["customerID"].nunique() == len(df),
        "valid_gender": df["gender"].isin(["Male", "Female"]).all(),
        "valid_churn": df["Churn"].isin(["Yes", "No"]).all(),
        "valid_tenure": ((df["tenure"] >= 0) & (df["tenure"] <= 120)).all(),
        "valid_monthly_charges": (
            (df["MonthlyCharges"] >= 0) & (df["MonthlyCharges"] <= 200)
        ).all(),
        "valid_total_charges": (df["TotalCharges"] >= 0).all(),
        "valid_contract": df["Contract"]
        .isin(["Month-to-month", "One year", "Two year"])
        .all(),
        "valid_internet": df["InternetService"]
        .isin(["DSL", "Fiber optic", "No"])
        .all(),
        "valid_payment": df["PaymentMethod"]
        .isin(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        )
        .all(),
        "row_count_ok": len(df) >= 1000,
        "churn_rate_reasonable": 0.10 <= df["Churn"].eq("Yes").mean() <= 0.60,
    }

    failed_checks = [k for k, v in checks.items() if not v]
    if failed_checks:
        logger.error(f"❌ Validation FAILED: {failed_checks}")
        validation_success = False
    else:
        logger.success(f"✅ All {len(checks)} validation checks PASSED!")

    # Log detailed stats
    churn_rate = df["Churn"].eq("Yes").mean()
    logger.info("📊 Validation Summary:")
    logger.info(f"   Total rows: {len(df)}")
    logger.info(f"   Null values: {df.isnull().sum().sum()}")
    logger.info(f"   Churn rate: {churn_rate:.2%}")
    logger.info(f"   Checks passed: {len(checks) - len(failed_checks)}/{len(checks)}")

    # ── Save validated data locally ────────────────────────────────────────────
    validated_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(validated_path, index=False)
    logger.info(f"💾 Validated data saved → {validated_path}")

    # ── Store validated data in ClickHouse ────────────────────────────────────
    try:
        ch = get_clickhouse_client()
        ch.setup_all_tables()

        # Prepare dataframe for ClickHouse insertion
        df_ch = df.copy()
        df_ch["pipeline_run_id"] = run_id

        # Ensure correct types for ClickHouse
        df_ch["TotalCharges"] = pd.to_numeric(
            df_ch["TotalCharges"], errors="coerce"
        ).fillna(0.0)
        df_ch["MonthlyCharges"] = pd.to_numeric(
            df_ch["MonthlyCharges"], errors="coerce"
        ).fillna(0.0)
        df_ch["SeniorCitizen"] = df_ch["SeniorCitizen"].astype("int8")

        validated_table = os.getenv("CLICKHOUSE_VALIDATED_TABLE", "churn_validated")
        rows_inserted = ch.insert_dataframe(
            df=df_ch,
            table=validated_table,
            pipeline_run_id=run_id,
        )
        logger.success(
            f"🏠 ClickHouse: Inserted {rows_inserted} validated rows → "
            f"churn_mlops.{validated_table}"
        )
    except Exception as e:
        logger.error(f"❌ ClickHouse insertion failed: {e}")
        logger.warning("⚠️  Continuing without ClickHouse storage...")

    # ── Emit COMPLETE lineage event ────────────────────────────────────────────
    emit_validation_lineage(
        run_id, "COMPLETE", str(raw_path), success=validation_success
    )

    # ── Try Great Expectations HTML report ────────────────────────────────────
    try:
        _run_ge_checkpoint(df, suite_name, run_id)
    except Exception as e:
        logger.warning(f"⚠️  GE checkpoint skipped: {e}")

    if not validation_success:
        raise ValueError(f"Data validation FAILED. Failed checks: {failed_checks}")

    return df, validation_success


def _run_ge_checkpoint(df: pd.DataFrame, suite_name: str, run_id: str) -> None:
    """Run Great Expectations checkpoint and generate Data Docs."""
    import great_expectations as gx

    ge_dir = Path("great_expectations")
    ge_dir.mkdir(exist_ok=True)

    context = gx.get_context(mode="ephemeral")

    try:
        ds = context.data_sources.add_pandas(name="churn_runtime")
        asset = ds.add_dataframe_asset(name="churn_d")

        suite = context.add_or_update_expectation_suite(suite_name)
        validator = context.get_validator(
            batch_request=asset.build_batch_request(dataframe=df),
            expectation_suite=suite,
        )

        # Core expectations
        validator.expect_table_row_count_to_be_between(min_value=1000, max_value=100000)
        validator.expect_column_values_to_not_be_null("customerID")
        validator.expect_column_values_to_be_in_set("Churn", value_set=["Yes", "No"])
        validator.expect_column_values_to_be_between(
            "tenure", min_value=0, max_value=120
        )
        validator.expect_column_values_to_be_between(
            "MonthlyCharges", min_value=0, max_value=200
        )
        validator.save_expectation_suite(discard_failed_expectations=False)

        results = validator.validate()

        # Save results to file
        results_path = Path("artifacts/ge_results")
        results_path.mkdir(parents=True, exist_ok=True)
        results_file = results_path / f"validation_{run_id[:8]}.json"
        latest_file = results_path / "latest_validation.json"

        with open(results_file, "w") as f:
            json.dump(results.to_json_dict(), f, indent=2)

        with open(latest_file, "w") as f:
            json.dump(results.to_json_dict(), f, indent=2)

        success_rate = (
            results.statistics["successful_expectations"]
            / results.statistics["evaluated_expectations"]
        )
        logger.info(
            f"📋 GE Results: {results.statistics['successful_expectations']}/"
            f"{results.statistics['evaluated_expectations']} passed "
            f"({success_rate:.0%})"
        )
    except Exception as e:
        logger.warning(f"⚠️  GE validation failed: {e}")
        # Always create latest_validation.json for DVC
        results_path = Path("artifacts/ge_results")
        results_path.mkdir(parents=True, exist_ok=True)
        latest_file = results_path / "latest_validation.json"
        with open(latest_file, "w") as f:
            json.dump({"success": False, "error": str(e)}, f)


if __name__ == "__main__":
    df, success = validate_data()
    print(f"\nValidation complete: {len(df)} rows | Success: {success}")
