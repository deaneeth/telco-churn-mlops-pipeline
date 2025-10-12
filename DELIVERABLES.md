##  📋 Deliverables Checklist

This project includes all required deliverables for a production MLOps pipeline:

### ✅ Core Deliverables

- [x] **Data Pipeline**
  - [x] Raw data ingestion (`data/raw/Telco-Customer-Churn.csv`)
  - [x] Feature engineering pipeline (`src/data/preprocess.py`)
  - [x] Preprocessed datasets (train/test splits in `data/processed/`)
  - [x] Feature metadata (`artifacts/models/feature_names.json`)

- [x] **Machine Learning Models**
  - [x] Scikit-learn pipeline (`artifacts/models/sklearn_pipeline_mlflow.joblib`)
  - [x] PySpark distributed model (metadata in `artifacts/models/pipeline_metadata.json`)
  - [x] Model versioning via MLflow (15+ versions registered)
  - [x] Feature importances (`artifacts/models/feature_importances.json`)

- [x] **Experiment Tracking**
  - [x] MLflow setup and configuration
  - [x] Experiment runs logged (`mlruns/` directory with 5 experiments)
  - [x] Model registry with versioning
  - [x] Metrics tracking (`artifacts/metrics/` directory)

- [x] **Workflow Orchestration**
  - [x] Airflow DAG implementation (`dags/telco_churn_dag.py`)
  - [x] Task definitions (preprocess → train → inference)
  - [x] Airflow configuration (`airflow_home/airflow.cfg`)
  - [x] DAG execution logs

- [x] **API & Deployment**
  - [x] REST API implementation (`src/api/app.py`)
  - [x] API endpoints (`/ping`, `/predict`)
  - [x] Docker containerization (`Dockerfile`)
  - [x] Production-ready configuration

### ✅ Testing & Quality Assurance

- [x] **Comprehensive Test Suite**
  - [x] 93 passing tests across 6 test modules
  - [x] Unit tests (`tests/test_preprocessing.py`, `tests/test_training.py`)
  - [x] Integration tests (`tests/test_integration.py`)
  - [x] Data validation tests (`tests/test_data_validation.py`)
  - [x] API tests (`tests/test_inference.py`)

- [x] **Code Quality**
  - [x] Type hints and documentation
  - [x] PEP 8 compliance
  - [x] Error handling and logging
  - [x] Configuration management (`config.py`, `config.yaml`)

### ✅ Documentation

- [x] **README.md** (this file)
  - [x] Project overview and business context
  - [x] Complete installation instructions
  - [x] Step-by-step usage guide
  - [x] MLflow setup and instructions
  - [x] Airflow setup and instructions
  - [x] Troubleshooting guide
  - [x] API documentation

- [x] **Additional Documentation**
  - [x] Compliance report (`compliance_report.md`)
  - [x] License file (`LICENSE`)
  - [x] Requirements specification (`requirements.txt`)
  - [x] Setup configuration (`setup.py`)
  - [x] Jupyter notebooks (4 notebooks in `notebooks/`)

- [x] **MLflow & Airflow Screenshots**
  - [x] MLflow UI screenshots (`docs/images/mlflow_runs.png`, `docs/images/mlflow_model.png`)
  - [x] Airflow DAG screenshots (`docs/images/airflow_dags.png`, `docs/images/airflow_run.png`)
  - [x] Screenshot instructions documented

### ✅ Artifacts & Outputs

- [x] **Model Artifacts**
  - [x] Trained models (200 KB sklearn, metadata for Spark)
  - [x] Preprocessor pipeline (9 KB)
  - [x] Model performance metrics (JSON files)
  - [x] Prediction outputs (`artifacts/predictions/batch_preds.csv`)

- [x] **Validation Reports**
  - [x] Full pipeline execution summary
  - [x] Folder audit reports (before/after)
  - [x] Test coverage reports
  - [x] Compliance validation (97.5% score)

### ✅ Reproducibility

- [x] **Environment Setup**
  - [x] Requirements file with pinned versions
  - [x] Setup script for package installation
  - [x] Configuration files (Python + YAML)
  - [x] Docker image for containerized execution

- [x] **Automation**
  - [x] Makefile with common commands
  - [x] Automated testing via pytest
  - [x] CI/CD ready structure
  - [x] Automated data preprocessing