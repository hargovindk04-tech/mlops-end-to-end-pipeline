from src.data_loader import (
    load_train_data,
    load_current_data,
    load_stress_data,
    load_raw_data,
)
from src.schema_validation import validate_dataframe, validate_stress


def main():
    train_df = validate_dataframe(load_train_data(), name="train")
    current_df = validate_dataframe(load_current_data(), name="current")
    stress_df = validate_stress(load_stress_data())
    raw_df = load_raw_data()

    print("\n--- Data Drift Hint Check ---")
    print(f"Mean Rotational speed (current): {current_df['Rotational speed'].mean():.2f}")
    if stress_df is not None:
        print(f"Mean Rotational speed (stress):  {stress_df['Rotational speed'].mean():.2f}")

    print("\nDatasets loaded and validated successfully.\n")


if __name__ == "__main__":
    main()
