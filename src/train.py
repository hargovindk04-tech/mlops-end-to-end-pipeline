"""
train.py

Train and compare baseline classifiers for failure-type prediction.
"""

from __future__ import annotations

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.config import N_ESTIMATORS, RANDOM_STATE
from config.paths import BEST_MODEL_PATH, MODEL_DIR
from src.evaluation import (
    build_comparison_table,
    evaluate_classifier,
    get_best_model_name,
    print_comparison_table,
)


def get_candidate_models() -> dict[str, ClassifierMixin | Pipeline]:
    """Return the four models used in notebook model selection."""
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        ),
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[pd.DataFrame, str, dict[str, ClassifierMixin | Pipeline]]:
    """
    Train each candidate model on SMOTE-balanced training data and evaluate on validation.
    """
    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, ClassifierMixin | Pipeline] = {}

    for model_name, model in get_candidate_models().items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[model_name] = evaluate_classifier(y_val, y_pred)
        fitted_models[model_name] = model

    results_df = build_comparison_table(results)
    print_comparison_table(results_df)

    best_model_name, best_model_score = get_best_model_name(results_df)
    print(
        f"\nThe best model is {best_model_name} "
        f"with a Macro F1 score of {best_model_score:.4f}"
    )

    return results_df, best_model_name, fitted_models


def save_model(model: ClassifierMixin | Pipeline, path=None) -> None:
    """Persist a trained model to disk."""
    save_path = BEST_MODEL_PATH if path is None else path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Saved model to {save_path}")
