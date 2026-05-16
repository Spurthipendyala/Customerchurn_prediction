"""
Evidently AI drift detection and model monitoring.
Generates HTML reports and JSON metrics, exports to Prometheus and ClickHouse.
"""
import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
import yaml
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def run_drift_detection(
    current_data: Optional[pd.DataFrame] = None,
    reference_data: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run Evidently AI drift detection.
    Returns dict with drift metrics and saves HTML + JSON reports.
    """
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from evidently.metrics import (
        DatasetMissingValueCount, RowCount
    )

    params = load_params()
    run_id = str(uuid.uuid4())[:8]

    # ── Load data ─────────────────────────────────────────────────────────────
    if reference_data is None:
        ref_path = Path(params["data"]["reference_path"])
        reference_data = pd.read_csv(ref_path)
        logger.info(f"📂 Reference data loaded: {len(reference_data)} rows")

    if current_data is None:
        # Try ClickHouse first, fallback to processed CSV
        try:
            from src.data.clickhouse_client import get_clickhouse_client
            ch = get_clickhouse_client()
            database = os.getenv("CLICKHOUSE_DATABASE", "churn_mlops")
            processed_table = os.getenv("CLICKHOUSE_PROCESSED_TABLE", "churn_processed")
            current_data = ch.get_latest_processed_data(processed_table)
            logger.info(f"📂 Current data loaded from ClickHouse: {len(current_data)} rows")
        except Exception as e:
            logger.warning(f"⚠️  ClickHouse unavailable ({e}), using processed CSV")
            proc_path = Path(params["data"]["processed_path"])
            current_data = pd.read_csv(proc_path)
            # Simulate production drift by sampling a random portion
            current_data = current_data.sample(
                n=min(500, len(current_data)), random_state=42
            )

    # ── Feature columns for drift analysis ────────────────────────────────────
    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "charges_per_month", "num_services"
    ]
    categorical_features = [
        "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
        "gender_Male", "InternetService_Fiber", "InternetService_No",
        "Contract_OneYear", "Contract_TwoYear",
        "PaymentMethod_CreditCard", "PaymentMethod_ElecCheck",
        "PaymentMethod_MailedCheck", "has_internet"
    ]

    # Align columns
    common_cols = [c for c in numeric_features + categorical_features
                   if c in reference_data.columns and c in current_data.columns]
    ref_subset = reference_data[common_cols].copy()
    cur_subset = current_data[common_cols].copy()

    # ── Run Evidently Data Drift Report ───────────────────────────────────────
    logger.info("🔍 Running Evidently drift analysis...")

    drift_report = Report(metrics=[
        DataDriftPreset(),
        DatasetMissingValueCount(),
    ])
    snapshot = drift_report.run(reference_data=ref_subset, current_data=cur_subset)

    # ── Save Reports ──────────────────────────────────────────────────────────
    reports_dir = Path(params["monitoring"]["report_path"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # HTML report
    html_path = reports_dir / f"drift_report_{timestamp}.html"
    snapshot.save_html(str(html_path))
    logger.info(f"📄 Drift HTML report saved → {html_path}")

    # JSON metrics
    json_metrics = snapshot.dict()
    json_path = reports_dir / f"drift_metrics_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2, default=str)

    # Latest report
    latest_path = reports_dir / "latest_drift.json"
    with open(latest_path, "w") as f:
        json.dump(json_metrics, f, indent=2, default=str)

    # ── Extract key metrics ───────────────────────────────────────────────────
    try:
        drift_result = json_metrics.get("metrics", [])
        dataset_drift = next(
            (m for m in drift_result if "DatasetDriftMetric" in str(m.get("metric", ""))),
            {}
        )
        drift_score = dataset_drift.get("result", {}).get("drift_share", 0.0)
        n_drifted = dataset_drift.get("result", {}).get("number_of_drifted_columns", 0)
        n_features = dataset_drift.get("result", {}).get("number_of_columns", len(common_cols))
        drift_detected = drift_score >= params["monitoring"]["drift_threshold"]
    except Exception:
        drift_score = 0.0
        n_drifted = 0
        n_features = len(common_cols)
        drift_detected = False

    # ── Update Prometheus gauge ───────────────────────────────────────────────
    try:
        from prometheus_client import Gauge
        gauge = Gauge("churn_drift_score", "Latest data drift score")
        gauge.set(drift_score)
    except Exception:
        pass

    # ── Log to ClickHouse metrics table ───────────────────────────────────────
    try:
        from src.data.clickhouse_client import get_clickhouse_client
        ch = get_clickhouse_client()
        database = os.getenv("CLICKHOUSE_DATABASE", "churn_mlops")
        for metric_name, metric_value in [
            ("drift_score", drift_score),
            ("drifted_features", n_drifted),
            ("total_features", n_features),
        ]:
            ch.client.command(f"""
                INSERT INTO {database}.model_metrics
                (run_id, model_name, metric_name, metric_value)
                VALUES ('{run_id}', 'churn_classifier', '{metric_name}', {metric_value})
            """)
        logger.info("📊 Drift metrics logged to ClickHouse")
    except Exception as e:
        logger.warning(f"⚠️  ClickHouse metrics log failed: {e}")

    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "drift_score": round(drift_score, 4),
        "drift_detected": drift_detected,
        "drifted_features": n_drifted,
        "total_features": n_features,
        "drift_threshold": params["monitoring"]["drift_threshold"],
        "reference_rows": len(ref_subset),
        "current_rows": len(cur_subset),
        "html_report": str(html_path),
        "json_report": str(json_path),
    }

    # Save summary as latest_drift.json for frontend
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2)

    status = "🚨 DRIFT DETECTED" if drift_detected else "✅ No significant drift"
    logger.info(f"{status}: score={drift_score:.4f} | "
                f"{n_drifted}/{n_features} features drifted")

    return summary


if __name__ == "__main__":
    result = run_drift_detection()
    print(f"\nDrift Report:")
    for k, v in result.items():
        print(f"  {k}: {v}")
