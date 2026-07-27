from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT/"data"/"raw"
TRAIN_DATA = DATA_DIR/"train.csv"
CURRENT_DATA = DATA_DIR/"current.csv"
STRESS_DATA = DATA_DIR/"stress.csv"
RAW_DATA = DATA_DIR/"raw.csv"
MODEL_DIR = PROJECT_ROOT/"models"
REPORT_DIR = PROJECT_ROOT/"reports"