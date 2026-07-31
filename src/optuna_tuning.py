"""
optuna_tuning.py

Hyperparameter tuning with Optuna and MLflow Model Registry integration.
"""

import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import mlflow

from config.config import (
    OPTUNA_EXPERIMENT,
    OPTUNA_N_TRIALS,
    PRODUCTION_MODEL_ALIAS,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
)
from src.mlflow_tracking import configure_mlflow, log_sklearn_model

def _suggest_xgboost_params(trial: optuna.Trial) -> dict:
    """Optuna search space for XGBoost (notebook Section 2.3)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "random_state": RANDOM_STATE,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict, float]:
    """Run Optuna study maximizing macro F1 on the validation set."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgboost_params(trial)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average="macro")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    print(f"Running Optuna optimization ({OPTUNA_N_TRIALS} trials)...")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)
    print(f"Optuna tuning complete. Best macro F1: {study.best_value:.4f}")
    return study.best_params, study.best_value


def train_final_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> XGBClassifier:
    """Train the final XGBoost model with Optuna's best hyperparameters."""
    params = {
        **best_params,
        "random_state": RANDOM_STATE,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }
    print("Training final XGBoost with optimal parameters...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def register_tuned_model(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: dict,
) -> float:
    """Log tuned model to MLflow, register it, and promote to production alias."""
    configure_mlflow(OPTUNA_EXPERIMENT)

    final_macro_f1 = f1_score(y_val, model.predict(X_val), average="macro")

    print("Logging to MLflow and Model Registry...")
    with mlflow.start_run(run_name="XGBoost_Optuna_Best"):
        mlflow.log_params(best_params)
        mlflow.log_metric("final_macro_f1", final_macro_f1)

        model_info = log_sklearn_model(
            model,
            X_val,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        client = MlflowClient()
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=PRODUCTION_MODEL_ALIAS,
            version=model_info.registered_model_version,
        )
        print(
            f"Model registered as '{REGISTERED_MODEL_NAME}' "
            f"and promoted to '{PRODUCTION_MODEL_ALIAS}' alias."
        )

    return final_macro_f1


def print_tuning_improvement(baseline_macro_f1: float, tuned_macro_f1: float) -> None:
    """Print macro F1 improvement over baseline XGBoost."""
    improvement = tuned_macro_f1 - baseline_macro_f1
    print("\n--- Final Results ---")
    print(f"Baseline XGBoost Macro F1: {baseline_macro_f1:.4f}")
    print(f"Tuned XGBoost Macro F1:    {tuned_macro_f1:.4f}")
    print(f"Improvement:               {improvement:+.4f}")


def tune_register_and_save(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    baseline_xgb_macro_f1: float,
) -> XGBClassifier:
    """End-to-end Optuna tuning, registry promotion, and return final model."""
    best_params, _ = tune_xgboost(X_train, y_train, X_val, y_val)
    final_model = train_final_xgboost(X_train, y_train, best_params)
    tuned_macro_f1 = register_tuned_model(final_model, X_val, y_val, best_params)
    print_tuning_improvement(baseline_xgb_macro_f1, tuned_macro_f1)
    return final_model
