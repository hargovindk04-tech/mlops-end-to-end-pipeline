# End-to-End MLOps Pipeline

Predictive maintenance (multi-class failure type) pipeline: validation, feature engineering, training, MLflow tracking, Optuna tuning, drift monitoring, and SHAP explainability.

## Project structure

```
mlops-end-to-end-pipeline/
├── config/           # Paths and hyperparameters
├── data/raw/         # train, current, stress, raw CSVs
├── models/           # best_model.pkl (generated)
├── reports/          # Evidently HTML + SHAP plots (generated)
├── notebooks/        # Original MLOps.ipynb reference
├── src/
│   ├── pipeline.py   # Orchestration CLI
│   ├── data_loader.py
│   ├── schema_validation.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluation.py
│   ├── mlflow_tracking.py
│   ├── optuna_tuning.py
│   ├── drift_detection.py
│   └── shap_analysis.py
├── requirements.txt
└── setup.py
```

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

## Run (one command)

**Full pipeline** (data → train → Optuna → drift → SHAP):

```bash
mlops-pipeline
```

Equivalent:

```bash
python -m src
python -m src.pipeline
```

### Stages

| Command | What runs |
|---------|-----------|
| `mlops-pipeline` | All stages |
| `mlops-pipeline --stage data` | Load, validate, features, SMOTE split only |
| `mlops-pipeline --stage train` | Data + model selection + Optuna + save model |
| `mlops-pipeline --stage monitor` | Data + Evidently drift reports |
| `mlops-pipeline --stage explain` | Data + SHAP (requires `models/best_model.pkl`) |

### Options

```bash
mlops-pipeline --no-mlflow    # Train without MLflow experiment logging
mlops-pipeline --no-optuna    # Skip Optuna; save best model from selection only
mlops-pipeline --help
```

## Outputs

| Artifact | Location |
|----------|----------|
| Production model | `models/best_model.pkl` |
| MLflow tracking DB | `mlflow.db` |
| Current drift report | `reports/drift_current.html` |
| Stress drift report | `reports/drift_stress.html` |
| SHAP plot | `reports/shap_per_class.png` |

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///D:/github/cursor/mlops-end-to-end-pipeline/mlflow.db
```

Adjust the path to your local `mlflow.db` file.
