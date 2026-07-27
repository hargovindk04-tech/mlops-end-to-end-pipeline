"""
data_loader.py

This module is responsible for loading all CSV datasets used in the project.
It performs basic file-level validation before returning a pandas DataFrame.
"""

from pathlib import Path
import pandas as pd

from config.paths import (
    TRAIN_DATA,
    CURRENT_DATA,
    STRESS_DATA,
    RAW_DATA,
)


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Generic function to load any CSV file.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV file is empty.
    """

    # Check whether file exists
    if not file_path.exists():
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Check whether DataFrame is empty
    if df.empty:
        raise ValueError(f"❌ CSV file is empty: {file_path}")

    # Display basic information
    print("=" * 60)
    print(f"Successfully loaded : {file_path.name}")
    print(f"Rows                : {df.shape[0]}")
    print(f"Columns             : {df.shape[1]}")
    print("=" * 60)

    return df


def load_train_data() -> pd.DataFrame:
    """Load training dataset."""
    return load_csv(TRAIN_DATA)


def load_current_data() -> pd.DataFrame:
    """Load current production dataset."""
    return load_csv(CURRENT_DATA)


def load_stress_data() -> pd.DataFrame:
    """Load stress dataset."""
    return load_csv(STRESS_DATA)


def load_raw_data() -> pd.DataFrame:
    """Load raw dataset."""
    return load_csv(RAW_DATA)