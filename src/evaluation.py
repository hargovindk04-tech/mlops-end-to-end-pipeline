"""
evaluation.py

Classification metrics and model comparison helpers.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config.config import CLASS_LIST


def evaluate_classifier(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute accuracy, macro/weighted F1, and per-class F1 scores."""
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
    }
    for class_label, score in zip(CLASS_LIST, per_class_f1):
        metrics[f"F1_Class_{class_label}"] = score
    return metrics


def build_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a sorted model comparison table (best macro F1 first)."""
    results_df = pd.DataFrame.from_dict(results, orient="index")
    return results_df.sort_values(by="Macro F1", ascending=False)


def print_comparison_table(results_df: pd.DataFrame) -> None:
    """Print the model comparison table."""
    print("\n--- Model Comparison Table ---")
    print(results_df.round(4).to_string())


def get_best_model_name(results_df: pd.DataFrame) -> tuple[str, float]:
    """Return the top model name and its macro F1 score."""
    best_model_name = results_df.index[0]
    best_model_score = results_df.iloc[0]["Macro F1"]
    return best_model_name, best_model_score
