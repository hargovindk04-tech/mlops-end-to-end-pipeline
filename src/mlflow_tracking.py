"""
mlflow_tracking.py

MLflow experiment setup and logging for model selection.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from config.config import CLASS_LIST, MODEL_SELECTION_EXPERIMENT
from config.paths import get_mlflow_tracking_uri

SKOPS_TRUSTED_TYPES = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "collections.OrderedDict",
]


def configure_mlflow(experiment_name: str) -> None:
    """Set tracking URI and active experiment."""
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(experiment_name)


def log_evaluation_metrics(metrics: dict[str, float]) -> None:
    """Log classification metrics using MLflow metric names."""
    mlflow.log_metric("macro_f1", metrics["Macro F1"])
    mlflow.log_metric("weighted_f1", metrics["Weighted F1"])
    mlflow.log_metric("accuracy", metrics["Accuracy"])
    for class_label in CLASS_LIST:
        mlflow.log_metric(
            f"f1_class_{class_label}",
            metrics[f"F1_Class_{class_label}"],
        )


def log_sklearn_model(
    model: ClassifierMixin | Pipeline,
    X_val: pd.DataFrame,
    *,
    registered_model_name: str | None = None,
):
    """Log a fitted sklearn-compatible model to MLflow."""
    return mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=X_val.iloc[:5],
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
        registered_model_name=registered_model_name,
    )


def setup_model_selection_experiment() -> None:
    """Configure MLflow for the model selection phase."""
    configure_mlflow(MODEL_SELECTION_EXPERIMENT)
