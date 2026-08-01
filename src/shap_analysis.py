"""
shap_analysis.py

SHAP-based model explainability for multiclass failure prediction.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config.config import FEATURES, SHAP_FAILURE_CLASSES
from config.paths import BEST_MODEL_PATH, REPORT_DIR, SHAP_PLOT_PATH


def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_shap_values(shap_values) -> list[np.ndarray]:
    """Support list or 3D array SHAP outputs across library versions."""
    if isinstance(shap_values, list):
        return shap_values
    return [shap_values[:, :, class_idx] for class_idx in range(shap_values.shape[2])]


def run_shap_analysis(
    train_df: pd.DataFrame,
    model_path: Path = BEST_MODEL_PATH,
    output_path: Path = SHAP_PLOT_PATH,
) -> dict[int, str]:
    """
    Compute SHAP values and save per-class mean |SHAP| bar charts.

    Returns:
        Mapping of failure class index to top driver feature name.
    """
    _ensure_report_dir()
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_df[FEATURES])
    shap_values_list = _normalize_shap_values(shap_values)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    top_drivers: dict[int, str] = {}

    print("\n--- Top Drivers per Failure Class ---")
    for subplot_index, (class_idx, class_name) in enumerate(SHAP_FAILURE_CLASSES.items()):
        mean_abs_shap = np.abs(shap_values_list[class_idx]).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        sorted_features = np.array(FEATURES)[sorted_indices]
        sorted_shap_vals = mean_abs_shap[sorted_indices]

        top_driver = sorted_features[0]
        top_drivers[class_idx] = top_driver
        short_name = class_name.split(" ")[0]
        print(
            f"{short_name} Top Driver: {top_driver} "
            f"(Mean |SHAP| = {sorted_shap_vals[0]:.4f})"
        )

        ax = axes[subplot_index]
        ax.barh(sorted_features[::-1], sorted_shap_vals[::-1], color="steelblue")
        ax.set_title(f"Mean |SHAP| - {class_name}")
        ax.set_xlabel("Mean |SHAP| impact on model output")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")
    return top_drivers
