# Architecture

## Pipeline stages

The orchestrator (`src/pipeline.py`) runs four logical stages. Each stage is implemented in dedicated modules so notebooks logic stays testable and reusable.

| Stage | Module(s) | Responsibility |
|-------|-----------|----------------|
| 1. Data | `data_loader`, `schema_validation`, `feature_engineering`, `preprocessing` | Load CSVs, Pandera contracts, derived features, encoding, stratified split, SMOTE |
| 2. Train | `train`, `evaluation`, `mlflow_tracking`, `optuna_tuning` | Four-model benchmark, MLflow logging, Optuna XGBoost tuning, model registry |
| 3. Monitor | `drift_detection` | Evidently reports: train vs current, train vs stress |
| 4. Explain | `shap_analysis` | Per-class SHAP drivers for failure types TWF–OSF |

## Data flow

```mermaid
flowchart TB
    subgraph inputs [Raw data]
        T[train.csv]
        C[current.csv]
        S[stress.csv]
    end

    subgraph stage1 [Stage 1 — Data]
        L[Load]
        V[Pandera validation]
        F[Feature engineering]
        P[Encode + split + SMOTE]
    end

    subgraph stage2 [Stage 2 — Train]
        M[Model selection]
        ML[MLflow PredMaint_ModelSelection]
        O[Optuna + MLflow PredMaint_Optuna]
        R[Registry PredMaint_XGBoost]
        PKL[models/best_model.pkl]
    end

    subgraph stage3 [Stage 3 — Monitor]
        E1[drift_current.html]
        E2[drift_stress.html]
    end

    subgraph stage4 [Stage 4 — Explain]
        SH[shap_per_class.png]
    end

    T --> L
    C --> L
    S --> L
    L --> V --> F --> P
    P --> M --> ML --> O --> R --> PKL
    P --> E1
    P --> E2
    PKL --> SH
    P --> SH
```

## Configuration

- **`config/paths.py`** — project root, data paths, artifact paths (`BEST_MODEL_PATH`, reports, MLflow SQLite).
- **`config/config.py`** — hyperparameters, feature lists, class names, experiment names.

## Design choices (interview talking points)

1. **Schema validation before training** — Pandera enforces sensor ranges; valid data can still drift (Evidently).
2. **SMOTE only on training split** — avoids synthetic leakage into validation.
3. **Separate model selection vs tuning experiments** — `PredMaint_ModelSelection` vs `PredMaint_Optuna`.
4. **Production alias** — tuned XGBoost registered as `PredMaint_XGBoost` with `production` alias in MLflow Model Registry.
5. **CLI stages** — `mlops-pipeline --stage …` for faster iteration without retraining.
