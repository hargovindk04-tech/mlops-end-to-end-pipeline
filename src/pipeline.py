"""
pipeline.py

End-to-end MLOps orchestration: one command runs the full workflow or selected stages.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd

from config.paths import BEST_MODEL_PATH
from src.data_loader import (
    load_current_data,
    load_raw_data,
    load_stress_data,
    load_train_data,
)
from src.drift_detection import run_current_drift_report, run_stress_drift_report
from src.feature_engineering import engineer_features, print_feature_summary
from src.optuna_tuning import tune_register_and_save
from src.preprocessing import encode_type, prepare_training_data
from src.schema_validation import validate_dataframe, validate_stress
from src.shap_analysis import run_shap_analysis
from src.train import save_model, train_and_evaluate_models


@dataclass
class PipelineContext:
    """In-memory artifacts passed between pipeline stages."""

    train_df: pd.DataFrame
    current_df: pd.DataFrame
    stress_df: pd.DataFrame | None
    training_data: dict[str, Any]
    raw_df: pd.DataFrame | None = None


def _print_stage(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def run_data_stage() -> PipelineContext:
    """Load, validate, engineer features, and prepare training splits."""
    _print_stage("Stage 1: Data loading, validation, and preprocessing")

    train_df = validate_dataframe(load_train_data(), name="train")
    current_df = validate_dataframe(load_current_data(), name="current")
    stress_df = validate_stress(load_stress_data())
    raw_df = load_raw_data()

    print("\n--- Data Drift Hint Check ---")
    print(f"Mean Rotational speed (current): {current_df['Rotational speed'].mean():.2f}")
    if stress_df is not None:
        print(f"Mean Rotational speed (stress):  {stress_df['Rotational speed'].mean():.2f}")

    train_df = engineer_features(train_df)
    current_df = engineer_features(current_df)
    if stress_df is not None:
        stress_df = engineer_features(stress_df)

    print("\nEngineered features 'Power_W' and 'Temp_diff' added to datasets.")
    print_feature_summary(train_df, "Train Data")
    print_feature_summary(current_df, "Current Data")
    if stress_df is not None:
        print_feature_summary(stress_df, "Stress Data")

    training_data = prepare_training_data(train_df)

    train_df, _ = encode_type(train_df, training_data["encoder"])
    current_df, _ = encode_type(current_df, training_data["encoder"])
    if stress_df is not None:
        stress_df, _ = encode_type(stress_df, training_data["encoder"])

    print("\nPreprocessing complete.")
    print(f"Training features shape (after SMOTE): {training_data['X_train_sm'].shape}")
    print(f"Validation features shape: {training_data['X_val'].shape}")

    return PipelineContext(
        train_df=train_df,
        current_df=current_df,
        stress_df=stress_df,
        training_data=training_data,
        raw_df=raw_df,
    )


def run_training_stage(
    ctx: PipelineContext,
    *,
    use_mlflow: bool = True,
    use_optuna: bool = True,
) -> str:
    """Train candidate models, optionally tune with Optuna, and save production model."""
    _print_stage("Stage 2: Model training and registration")

    training_data = ctx.training_data
    results_df, best_model_name, fitted_models = train_and_evaluate_models(
        training_data["X_train_sm"],
        training_data["y_train_sm"],
        training_data["X_val"],
        training_data["y_val"],
        use_mlflow=use_mlflow,
    )

    if use_optuna:
        baseline_xgb_macro_f1 = results_df.loc["XGBoost", "Macro F1"]
        final_model = tune_register_and_save(
            training_data["X_train_sm"],
            training_data["y_train_sm"],
            training_data["X_val"],
            training_data["y_val"],
            baseline_xgb_macro_f1,
        )
    else:
        final_model = fitted_models[best_model_name]
        print(f"\nSkipping Optuna. Saving model selection winner: {best_model_name}")

    save_model(final_model)
    return best_model_name


def run_monitoring_stage(ctx: PipelineContext) -> None:
    """Generate Evidently drift reports for current and stress batches."""
    _print_stage("Stage 3: Drift detection and monitoring")

    run_current_drift_report(ctx.train_df, ctx.current_df)
    if ctx.stress_df is not None:
        run_stress_drift_report(ctx.train_df, ctx.stress_df)
    else:
        print("Stress dataset unavailable; skipping stress drift report.")


def run_explainability_stage(ctx: PipelineContext) -> None:
    """SHAP analysis using the saved production model."""
    _print_stage("Stage 4: Model explainability (SHAP)")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. Run training stage first "
            "(`--stage train` or `--stage all`)."
        )
    run_shap_analysis(ctx.train_df)


def run_pipeline(
    stage: str = "all",
    *,
    use_mlflow: bool = True,
    use_optuna: bool = True,
) -> int:
    """
    Execute pipeline stages.

    Returns:
        Process exit code (0 success, 1 failure).
    """
    try:
        ctx = run_data_stage()

        if stage == "data":
            print("\nPipeline stopped after data stage (--stage data).\n")
            return 0

        best_model_name = None
        if stage in ("all", "train"):
            best_model_name = run_training_stage(
                ctx,
                use_mlflow=use_mlflow,
                use_optuna=use_optuna,
            )
            if stage == "train":
                print("\nPipeline stopped after training stage (--stage train).\n")
                return 0

        if stage in ("all", "monitor"):
            run_monitoring_stage(ctx)
            if stage == "monitor":
                print("\nPipeline stopped after monitoring stage (--stage monitor).\n")
                return 0

        if stage in ("all", "explain"):
            run_explainability_stage(ctx)
            if stage == "explain":
                print("\nPipeline stopped after explainability stage (--stage explain).\n")
                return 0

        if best_model_name:
            print(
                f"\nPipeline complete. Model selection winner: {best_model_name}. "
                f"Production artifact: {BEST_MODEL_PATH}\n"
            )
        else:
            print(f"\nPipeline complete. Production artifact: {BEST_MODEL_PATH}\n")
        return 0

    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the predictive maintenance MLOps pipeline end-to-end or by stage.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "train", "monitor", "explain"],
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging during model selection.",
    )
    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip Optuna tuning; save the best model from model selection only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_pipeline(
        stage=args.stage,
        use_mlflow=not args.no_mlflow,
        use_optuna=not args.no_optuna,
    )


if __name__ == "__main__":
    raise SystemExit(main())
