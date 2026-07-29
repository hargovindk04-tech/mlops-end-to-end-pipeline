"""
schema_validation.py

Pandera schema definitions and validation helpers for sensor datasets.
"""

import os

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

SENSOR_SCHEMA = DataFrameSchema(
    columns={
        "Type": Column(dtype=str, checks=Check.isin(["L", "M", "H"])),
        "Air temperature": Column(dtype=float, checks=Check.in_range(295.0, 305.0)),
        "Process temperature": Column(dtype=float, checks=Check.in_range(305.0, 315.0)),
        "Rotational speed": Column(dtype=int, checks=Check.in_range(1000, 2900)),
        "Torque": Column(dtype=float, checks=Check.in_range(3.0, 80.0)),
        "Tool wear": Column(dtype=int, checks=Check.in_range(0, 253)),
        "Failure_Type": Column(dtype=int, checks=Check.isin([0, 1, 2, 3, 4])),
    }
)

INT_COLUMNS = ["Rotational speed", "Tool wear", "Failure_Type"]


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast integer columns before Pandera validation."""
    df = df.copy()
    for col in INT_COLUMNS:
        df[col] = df[col].astype("int64")
    return df


def validate_dataframe(
    df: pd.DataFrame,
    *,
    lazy: bool = False,
    name: str = "dataset",
) -> pd.DataFrame:
    """
    Fix dtypes, validate against SENSOR_SCHEMA, and return the validated DataFrame.

    Raises:
        pa.errors.SchemaErrors: If validation fails.
    """
    df = fix_dtypes(df)
    print(f"Validating {name}...")
    validated = SENSOR_SCHEMA.validate(df, lazy=lazy)
    print(f"Validation passed for {name}")
    return validated


def validate_stress(df: pd.DataFrame) -> pd.DataFrame | None:
    """Validate stress data with lazy=True and print a violation summary on failure."""
    df = fix_dtypes(df)
    print("Validating stress dataset (lazy=True)...")
    try:
        validated = SENSOR_SCHEMA.validate(df, lazy=True)
        print("Stress dataset passed schema validation")
        return validated
    except pa.errors.SchemaErrors as exc:
        print("Schema violations found in stress dataset:")
        print(exc)
        return None
