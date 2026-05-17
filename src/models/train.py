"""
MLflow-tracked model training for Telco Churn prediction.
Trains RF, XGBoost, LightGBM with cross-validation, logs everything to MLflow,
and registers the best model in MLflow Model Registry.
"""

import json
import os
import pickle
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def prepare_data(
    params: dict,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list
]:
    """Load and split data into train/val/test sets."""
    processed_path = Path(params["data"]["processed_path"])
    df = pd.read_csv(processed_path)

    # Drop non-feature columns
    feature_cols = [
        c for c in df.columns if c not in ["customerID", "Churn", "tenure_group"]
    ]
    X = df[feature_cols]
    y = df["Churn"]

    # Train / temp split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=params["data"]["val_size"],
        random_state=params["data"]["random_state"],
        stratify=y_train,
    )

    logger.info("📊 Dataset splits:")
    logger.info(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logger.info(
        f"   Churn rate → Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    """Apply SMOTE to balance the training set."""
    smote = SMOTE(random_state=random_state, sampling_strategy=0.7)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        f"🔄 SMOTE applied: {len(X_train)} → {len(X_res)} samples | "
        f"Churn rate: {y_train.mean():.2%} → {y_res.mean():.2%}"
    )
    return X_res, y_res


def get_models(params: dict) -> Dict[str, Any]:
    """Initialize all models with params from params.yaml."""
    rf_params = params["train"]["random_forest"]
    xgb_params = params["train"]["xgboost"]
    lgbm_params = params["train"]["lightgbm"]

    return {
        "random_forest": RandomForestClassifier(**rf_params),
        "xgboost": XGBClassifier(
            **xgb_params, use_label_encoder=False, eval_metric="logloss", verbosity=0
        ),
        "lightgbm": LGBMClassifier(**lgbm_params, verbose=-1),
    }


def evaluate_model(
    model, X: pd.DataFrame, y: pd.Series, prefix: str = ""
) -> Dict[str, float]:
    """Compute all classification metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    prefix = f"{prefix}_" if prefix else ""
    return {
        f"{prefix}roc_auc": round(roc_auc_score(y, y_proba), 4),
        f"{prefix}accuracy": round(accuracy_score(y, y_pred), 4),
        f"{prefix}f1": round(f1_score(y, y_pred), 4),
        f"{prefix}precision": round(precision_score(y, y_pred), 4),
        f"{prefix}recall": round(recall_score(y, y_pred), 4),
    }


def train_and_log_model(
    model_name: str,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    params: dict,
    experiment_id: str,
) -> Tuple[float, str]:
    """Train a single model and log everything to MLflow. Returns (val_auc, run_id)."""

    with mlflow.start_run(
        run_name=f"{model_name}_{uuid.uuid4().hex[:6]}",
        experiment_id=experiment_id,
    ) as run:
        run_id = run.info.run_id
        logger.info(f"🚀 Training {model_name} | run_id={run_id}")

        # Log parameters
        model_params = params["train"].get(model_name.replace("-", "_"), {})
        mlflow.log_params({f"model/{k}": v for k, v in model_params.items()})
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("cv_folds", params["train"]["cv_folds"])
        mlflow.log_param("dataset_size", len(X_train) + len(X_val) + len(X_test))

        # Cross-validation
        cv = StratifiedKFold(
            n_splits=params["train"]["cv_folds"], shuffle=True, random_state=42
        )
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        mlflow.log_metric("cv_auc_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_auc_std", round(cv_scores.std(), 4))
        logger.info(f"   CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final training
        model.fit(X_train, y_train)

        # Evaluate on all splits
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        val_metrics = evaluate_model(model, X_val, y_val, "val")
        test_metrics = evaluate_model(model, X_test, y_test, "test")

        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)

        logger.info(
            f"   Train AUC: {train_metrics['train_roc_auc']:.4f} | "
            f"Val AUC: {val_metrics['val_roc_auc']:.4f} | "
            f"Test AUC: {test_metrics['test_roc_auc']:.4f}"
        )

        # Log model artifact
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif model_name == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Log feature importances
        if hasattr(model, "feature_importances_"):
            fi_path = Path(f"artifacts/plots/feature_importance_{model_name}.json")
            fi_path.parent.mkdir(parents=True, exist_ok=True)
            fi = dict(
                zip(X_train.columns, model.feature_importances_.round(6).tolist())
            )
            fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20])
            with open(fi_path, "w") as f:
                json.dump(fi_sorted, f, indent=2)
            mlflow.log_artifact(str(fi_path))

        # Classification report
        report = classification_report(
            y_test,
            model.predict(X_test),
            target_names=["No Churn", "Churn"],
            output_dict=True,
        )
        report_path = Path(f"artifacts/reports/classification_{model_name}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))

        # Tags
        mlflow.set_tags(
            {
                "model_family": model_name,
                "dataset": "telco_churn",
                "task": "binary_classification",
                "pipeline_stage": "training",
            }
        )

        return val_metrics["val_roc_auc"], run_id


def train() -> None:
    """Main training orchestration function."""
    params = load_params()

    # ── Setup MLflow ──────────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = params["mlflow"]["experiment_name"]
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=f"./artifacts/mlflow/{experiment_name}",
            tags={"project": "telco_churn", "team": "mlops"},
        )
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    logger.info(f"🔬 MLflow experiment: '{experiment_name}' (id={experiment_id})")

    # ── Prepare data ──────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_data(params)
    X_train_res, y_train_res = apply_smote(
        X_train, y_train, params["data"]["random_state"]
    )

    # ── Train all models ──────────────────────────────────────────────────────
    models = get_models(params)
    results: Dict[str, Tuple[float, str, Any]] = {}

    for model_name, model in models.items():
        logger.info(f"\n{'─' * 50}")
        logger.info(f"🏋️  Training: {model_name.upper()}")
        val_auc, run_id = train_and_log_model(
            model_name=model_name,
            model=model,
            X_train=X_train_res,
            y_train=y_train_res,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            params=params,
            experiment_id=experiment_id,
        )
        results[model_name] = (val_auc, run_id, model)

    # ── Ensemble model ────────────────────────────────────────────────────────
    logger.info(f"\n{'─' * 50}")
    logger.info("🏋️  Training: VOTING ENSEMBLE")
    ensemble = VotingClassifier(
        estimators=[
            ("r", results["random_forest"][2]),
            ("xgb", results["xgboost"][2]),
            ("lgbm", results["lightgbm"][2]),
        ],
        voting="soft",
        n_jobs=-1,
    )
    ens_auc, ens_run_id = train_and_log_model(
        model_name="voting_ensemble",
        model=ensemble,
        X_train=X_train_res,
        y_train=y_train_res,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        params=params,
        experiment_id=experiment_id,
    )
    results["voting_ensemble"] = (ens_auc, ens_run_id, ensemble)

    # ── Select best model ─────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k][0])
    best_auc, best_run_id, best_model = results[best_name]

    logger.success(f"\n🏆 BEST MODEL: {best_name.upper()}")
    logger.success(f"   Val AUC: {best_auc:.4f} | Run ID: {best_run_id}")

    # ── Save best model locally ───────────────────────────────────────────────
    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    logger.success("✅ Best model saved → artifacts/models/best_model.pkl")

    # ── Save feature columns ──────────────────────────────────────────────────
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # ── Register best model in MLflow Model Registry ──────────────────────────
    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = params["mlflow"]["registered_model_name"]

    mv = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    # Transition to Staging
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.success(f"📦 Model '{registered_model_name}' v{mv.version} → Staging")

    # ── Save training metrics ──────────────────────────────────────────────────
    metrics_dir = Path("artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "best_model": best_name,
        "best_val_auc": best_auc,
        "best_run_id": best_run_id,
        "all_results": {k: v[0] for k, v in results.items()},
        "model_registry": {
            "name": registered_model_name,
            "version": mv.version,
            "stage": "Staging",
        },
    }
    with open(metrics_dir / "train_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.success("🎉 Training pipeline complete!")
    logger.info("\n📊 All model AUCs:")
    for name, (auc, _, _) in sorted(
        results.items(), key=lambda x: x[1][0], reverse=True
    ):
        marker = "⭐" if name == best_name else "  "
        logger.info(f"  {marker} {name:20s}: {auc:.4f}")


if __name__ == "__main__":
    train()
