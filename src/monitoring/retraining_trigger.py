"""
Automated retraining trigger for Churn MLOps pipeline.
Checks drift reports and executes DVC repro if threshold is exceeded.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def trigger_retraining(reason: str = "Unknown"):
    """Executes DVC pipeline to retrain the model."""
    load_params()

    logger.warning(f"🚀 RETRAINING TRIGGERED: {reason}")

    # ── Log retraining event to ClickHouse ──────────────────────────────────
    try:
        from src.data.clickhouse_client import get_clickhouse_client

        ch = get_clickhouse_client()
        db = os.getenv("CLICKHOUSE_DATABASE", "churn_mlops")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        ch.client.command(f"""
            INSERT INTO {db}.model_metrics
            (run_id, model_name, metric_name, metric_value)
            VALUES ('retrain_{run_id}', 'churn_classifier', 'retraining_triggered', 1.0)
        """)
        logger.info("📊 Retraining event logged to ClickHouse")
    except Exception as e:
        logger.warning(f"⚠️  ClickHouse logging failed: {e}")

    # ── Execute DVC Repro ─────────────────────────────────────────────────────
    try:
        logger.info("⏳ Running 'dvc repro'...")
        # Use venv python if available
        # We use subprocess to run dvc repro
        # Note: In a production environment, this might be a webhook to a CI/CD runner
        result = subprocess.run(["dvc", "repro"], capture_output=True, text=True, check=True)
        logger.success("✅ DVC pipeline execution completed successfully!")
        logger.debug(f"DVC Output: {result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ DVC repro failed: {e.stderr}")
    except Exception as e:
        logger.error(f"❌ Unexpected error during retraining: {e}")


def check_and_trigger():
    """Checks the latest drift report and triggers retraining if needed."""
    params = load_params()
    latest_drift_path = Path(params["monitoring"]["report_path"]) / "latest_drift.json"

    if not latest_drift_path.exists():
        logger.info("📭 No drift reports found. Skipping trigger check.")
        return

    with open(latest_drift_path) as f:
        summary = json.load(f)

    drift_score = summary.get("drift_score", 0.0)
    threshold = params["monitoring"]["drift_threshold"]
    auto_retrain = params["monitoring"].get("auto_retrain", False)

    logger.info(f"🔍 Current drift score: {drift_score:.4f} (Threshold: {threshold})")

    if drift_score >= threshold:
        if auto_retrain:
            trigger_retraining(reason=f"Drift score {drift_score:.4f} >= threshold {threshold}")
        else:
            logger.info("ℹ️  Drift detected but auto_retrain is disabled.")
    else:
        logger.info("✅ Drift is below threshold. No retraining needed.")


if __name__ == "__main__":
    check_and_trigger()
