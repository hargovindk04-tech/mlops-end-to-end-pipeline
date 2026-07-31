RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Failure_Type"
TYPE_COLUMN = "Type"
TYPE_ENCODED_COLUMN = "Type_enc"
MLFLOW_EXPERIMENT = "MLOps-End-to-End"
MODEL_SELECTION_EXPERIMENT = "PredMaint_ModelSelection"
OPTUNA_EXPERIMENT = "PredMaint_Optuna"
REGISTERED_MODEL_NAME = "PredMaint_XGBoost"
PRODUCTION_MODEL_ALIAS = "production"
SMOTE_K_NEIGHBORS = 3
N_ESTIMATORS = 100
OPTUNA_N_TRIALS = 30

FEATURES = [
    TYPE_ENCODED_COLUMN,
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Power_W",
    "Temp_diff",
]

ENGINEERED_FEATURES = ["Power_W", "Temp_diff"]

CLASS_NAMES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
}

CLASS_LIST = list(CLASS_NAMES.keys())