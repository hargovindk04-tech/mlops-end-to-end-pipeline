from src.data_loader import (
    load_train_data,
    load_current_data,
    load_stress_data,
    load_raw_data,
)


def main():

    train_df = load_train_data()
    current_df = load_current_data()
    stress_df = load_stress_data()
    raw_df = load_raw_data()

    print("\nDatasets loaded successfully.\n")


if __name__ == "__main__":
    main()