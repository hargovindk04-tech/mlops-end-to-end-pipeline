"""
drift_detection.py

Evidently-based data drift monitoring reports.
"""

from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics.data_drift.column_drift_metric import ColumnDriftMetric
from evidently.legacy.metrics.data_drift.dataset_drift_metric import DatasetDriftMetric
from evidently.legacy.report import Report

from config.config import DRIFT_FEATURES
from config.paths import DRIFT_CURRENT_REPORT, DRIFT_STRESS_REPORT, REPORT_DIR


def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _print_current_drift_summary(report: Report) -> None:
    report_dict = report.as_dict()
    drift_result = report_dict["metrics"][0]["result"]
    print("\n--- Drift Summary (Current Batch vs Train) ---")
    print(f"Dataset Drift Detected: {drift_result['dataset_drift']}")
    print(
        "Number of Drifted Features: "
        f"{drift_result['number_of_drifted_columns']} out of "
        f"{drift_result['number_of_columns']}"
    )


def run_current_drift_report(
    train_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path = DRIFT_CURRENT_REPORT,
) -> Report:
    """Compare the current production batch against training data."""
    _ensure_report_dir()
    print("Running Evidently data drift report on the 'current' batch...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=train_df[DRIFT_FEATURES],
        current_data=current_df[DRIFT_FEATURES],
    )
    report.save_html(str(output_path))
    print(f"Report saved to {output_path}")
    _print_current_drift_summary(report)
    return report


def run_stress_drift_report(
    train_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    output_path: Path = DRIFT_STRESS_REPORT,
) -> pd.DataFrame:
    """Compare the stress batch against training data with per-column drift metrics."""
    _ensure_report_dir()
    print("Running Evidently column drift metrics on the 'stress' batch...")

    metrics = [DatasetDriftMetric()] + [
        ColumnDriftMetric(column_name=feature) for feature in DRIFT_FEATURES
    ]
    report = Report(metrics=metrics)
    report.run(
        reference_data=train_df[DRIFT_FEATURES],
        current_data=stress_df[DRIFT_FEATURES],
    )
    report.save_html(str(output_path))
    print(f"Report saved to {output_path}")

    report_dict = report.as_dict()
    table_rows = []
    for index, feature in enumerate(DRIFT_FEATURES):
        metric_result = report_dict["metrics"][index + 1]["result"]
        ref_mean = train_df[feature].mean()
        curr_mean = stress_df[feature].mean()
        table_rows.append(
            {
                "Feature": feature,
                "Drift Detected": metric_result["drift_detected"],
                "Wasserstein Score": round(metric_result["drift_score"], 4),
                "Ref Mean": round(ref_mean, 2),
                "Current Mean": round(curr_mean, 2),
                "Delta": round(curr_mean - ref_mean, 2),
            }
        )

    drift_table = pd.DataFrame(table_rows)
    print("\n--- Per-Feature Drift Summary (Stress Batch vs Train) ---")
    print(drift_table.to_string(index=False))
    return drift_table
