"""
preprocessing.py

Encoding, train-validation splitting, and class-imbalance handling.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config.config import (
    FEATURES,
    RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
    TARGET_COLUMN,
    TEST_SIZE,
    TYPE_COLUMN,
    TYPE_ENCODED_COLUMN,
)


def encode_type(
    df: pd.DataFrame,
    encoder: LabelEncoder | None = None,
    *,
    fit: bool = False,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode the categorical Type column as Type_enc."""
    df = df.copy()
    if encoder is None:
        encoder = LabelEncoder()

    if fit:
        df[TYPE_ENCODED_COLUMN] = encoder.fit_transform(df[TYPE_COLUMN])
    else:
        df[TYPE_ENCODED_COLUMN] = encoder.transform(df[TYPE_COLUMN])

    return df, encoder


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model feature columns from a prepared DataFrame."""
    return df[FEATURES]


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train-validation split."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the training split to handle class imbalance."""
    smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return (
        pd.DataFrame(X_resampled, columns=FEATURES),
        pd.Series(y_resampled, name=TARGET_COLUMN),
    )


def print_class_distribution(y: pd.Series, title: str) -> None:
    """Print sorted class counts."""
    counts = y.value_counts().sort_index()
    print(f"\n--- {title} ---")
    print(counts.to_string())


def prepare_training_data(
    train_df: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series | LabelEncoder]:
    """
    Build encoded features, split train/validation, and apply SMOTE on training only.

    SMOTE is applied only to the training split to avoid leaking synthetic samples
    into the validation set.
    """
    train_df, encoder = encode_type(train_df, fit=True)

    X = get_feature_matrix(train_df)
    y = train_df[TARGET_COLUMN]

    print("Splitting data 80/20...")
    X_train, X_val, y_train, y_val = split_train_validation(X, y)

    print("Applying SMOTE to the training split...")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    print_class_distribution(y_train_sm, "Post-SMOTE Class Distribution (Training Data Only)")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_train_sm": X_train_sm,
        "y_train_sm": y_train_sm,
        "encoder": encoder,
    }
