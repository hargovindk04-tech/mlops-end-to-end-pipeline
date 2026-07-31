from src.data_loader import (
    load_train_data,
    load_current_data,
    load_stress_data,
    load_raw_data,
)
from src.feature_engineering import engineer_features, print_feature_summary
from src.preprocessing import encode_type, prepare_training_data
from src.schema_validation import validate_dataframe, validate_stress
from src.optuna_tuning import tune_register_and_save
from src.train import save_model, train_and_evaluate_models


def main():
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

    current_df, _ = encode_type(current_df, training_data["encoder"])
    if stress_df is not None:
        stress_df, _ = encode_type(stress_df, training_data["encoder"])

    print("\nPreprocessing complete.")
    print(f"Training features shape (after SMOTE): {training_data['X_train_sm'].shape}")
    print(f"Validation features shape: {training_data['X_val'].shape}")

    results_df, best_model_name, fitted_models = train_and_evaluate_models(
        training_data["X_train_sm"],
        training_data["y_train_sm"],
        training_data["X_val"],
        training_data["y_val"],
        use_mlflow=True,
    )
    baseline_xgb_macro_f1 = results_df.loc["XGBoost", "Macro F1"]

    final_model = tune_register_and_save(
        training_data["X_train_sm"],
        training_data["y_train_sm"],
        training_data["X_val"],
        training_data["y_val"],
        baseline_xgb_macro_f1,
    )
    save_model(final_model)

    print(
        f"\nModel selection winner: {best_model_name}. "
        "Production artifact: Optuna-tuned XGBoost saved to models/best_model.pkl.\n"
    )


if __name__ == "__main__":
    main()
