from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT/"data"/"raw"
TRAIN_DATA = DATA_DIR/"train.csv"
CURRENT_DATA = DATA_DIR/"current.csv"
STRESS_DATA = DATA_DIR/"stress.csv"
RAW_DATA = DATA_DIR/"raw.csv"
MODEL_DIR = PROJECT_ROOT/"models"
REPORT_DIR = PROJECT_ROOT/"reports"
BEST_MODEL_PATH = MODEL_DIR/"best_model.pkl"
DRIFT_CURRENT_REPORT = REPORT_DIR/"drift_current.html"
DRIFT_STRESS_REPORT = REPORT_DIR/"drift_stress.html"
SHAP_PLOT_PATH = REPORT_DIR/"shap_per_class.png"
MLFLOW_DB = PROJECT_ROOT/"mlflow.db"


def get_mlflow_tracking_uri() -> str:
    """Absolute SQLite tracking URI for MLflow."""
    return "sqlite:///" + str(MLFLOW_DB.resolve()).replace("\\", "/")