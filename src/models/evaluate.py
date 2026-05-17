"""
Model evaluation module.
Evaluates the trained model on validation data and logs metrics.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def evaluate_model():
    params = load_params()

    # Load processed data
    data_path = Path(params["data"]["processed_path"])
    df = pd.read_csv(data_path)

    # Simple train-test split logic based on random_state to reproduce validation set
    from sklearn.model_selection import train_test_split

    params["data"]["target_column"]

    # Drop non-feature columns
    feature_cols = [
        c for c in df.columns if c not in ["customerID", "Churn", "tenure_group"]
    ]
    X = df[feature_cols]
    y = df["Churn"]

    # Reproduce test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )

    # Load best model
    model_path = Path("artifacts/models/best_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"Loaded model from {model_path}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    logger.info(f"Evaluation metrics: {metrics}")

    # Save metrics
    metrics_path = Path("artifacts/metrics/eval_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate curves
    plots_dir = Path("artifacts/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        plots_dir / "roc_curve.csv", index=False
    )

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
        plots_dir / "pr_curve.csv", index=False
    )

    logger.success("✅ Evaluation complete. Metrics and plots saved.")


if __name__ == "__main__":
    evaluate_model()
