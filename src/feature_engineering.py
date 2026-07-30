"""
feature_engineering.py

Derived feature creation for predictive maintenance sensor data.
"""

import numpy as np
import pandas as pd

from config.config import ENGINEERED_FEATURES, TARGET_COLUMN


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mechanical power and temperature differential features.

    Power_W  = Torque * (Rotational speed * 2π / 60)
    Temp_diff = Process temperature - Air temperature
    """
    df = df.copy()
    df["Power_W"] = df["Torque"] * (df["Rotational speed"] * 2 * np.pi / 60)
    df["Temp_diff"] = df["Process temperature"] - df["Air temperature"]
    return df


def print_feature_summary(df: pd.DataFrame, name: str) -> None:
    """Print mean engineered feature values grouped by failure type."""
    summary = (
        df.groupby(TARGET_COLUMN)[ENGINEERED_FEATURES]
        .mean()
        .round(2)
    )
    print(f"\n--- Mean Feature Values by Failure Type ({name}) ---")
    print(summary.to_string())
