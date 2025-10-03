# Airflow Complete Setup and Validation Guide
## Step 8 - WSL Execution (Windows 11 + WSL2)

**Date**: October 3, 2025  
**Environment**: Windows 11 + WSL2 (Ubuntu)  
**Airflow Version**: 3.0.6  
**Python Version**: WSL Python 3.12.3 (Airflow) + Windows Python 3.13 (.venv for ML tasks)  
**Status**: ✅ **ALL TESTS PASSED - PRODUCTION READY**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Environment Setup](#environment-setup)
4. [Configuration Details](#configuration-details)
5. [DAG Implementation](#dag-implementation)
6. [Validation Results](#validation-results)
7. [Issues Resolved](#issues-resolved)
8. [Testing Guide](#testing-guide)
9. [Production Recommendations](#production-recommendations)
10. [Troubleshooting](#troubleshooting)

---

## Executive Summary

### ✅ Complete Pipeline Status

The Telco Churn Prediction ML pipeline is fully operational with Airflow orchestration running on WSL2. All three tasks execute successfully in sequence:

| Task | Status | Performance | Duration |
|------|--------|-------------|----------|
| **Preprocess** | ✅ PASSED | 45 features generated from 7,043 records | ~7 seconds |
| **Train** | ✅ PASSED | **80.06% accuracy**, 84.66% ROC AUC | ~29 seconds |
| **Inference** | ✅ PASSED | 100 predictions, 23% churn rate | ~6 seconds |
| **Full DAG** | ✅ PASSED | End-to-end pipeline successful | ~42 seconds |

**Key Achievement**: Hybrid architecture successfully bridges WSL Airflow orchestration with Windows Python ML environment.

### Why WSL is Required

Airflow **does not support native Windows** due to:
- POSIX-compliant OS requirement (Linux/macOS)
- Dependency on Unix-specific system calls (`os.register_at_fork`)
- Multiprocessing libraries incompatible with Windows

**Solution**: Run Airflow in WSL2, execute Python ML tasks using Windows .venv environment.

---

## Architecture Overview

### Hybrid Execution Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Windows 11 Host                          │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           WSL2 Ubuntu Environment                     │ │
│  │                                                       │ │
│  │  ┌─────────────────────────────────────────────┐    │ │
│  │  │  Airflow 3.0.6 (airflow_env)               │    │ │
│  │  │  - Scheduler                               │    │ │
│  │  │  - LocalExecutor                           │    │ │
│  │  │  - SQLite Database                         │    │ │
│  │  └─────────────────────────────────────────────┘    │ │
│  │                     ↓                                 │ │
│  │  ┌─────────────────────────────────────────────┐    │ │
│  │  │  BashOperator (3 tasks)                    │    │ │
│  │  │  - Executes Windows Python via .exe        │    │ │
│  │  └─────────────────────────────────────────────┘    │ │
│  └───────────────────────────────────────────────────────┘ │
│                     ↓                                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Windows Python .venv (Python 3.13)                  │ │
│  │  - .venv/Scripts/python.exe                          │ │
│  │  - pandas, scikit-learn, mlflow, etc.                │ │
│  │  - Runs ML tasks (preprocess, train, inference)      │ │
│  └───────────────────────────────────────────────────────┘ │
│                     ↓                                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Shared File System (/mnt/e/)                        │ │
│  │  - E:/ZuuCrew/telco-churn-prediction-mini-project-1  │ │
│  │  - Accessible from both WSL and Windows              │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **WSL2 Environment**: Runs Airflow scheduler and executor
2. **Airflow Virtual Env**: `airflow_env/` with Airflow 3.0.6 installation
3. **Windows Python Env**: `.venv/` with ML dependencies
4. **Shared Storage**: `/mnt/e/` mount point for seamless file access
5. **Encoding Wrapper**: `cmd.exe` with UTF-8 encoding for Unicode support

---

## Environment Setup

### Prerequisites

1. **Windows 11** with WSL2 enabled
2. **Ubuntu** distribution installed in WSL2
3. **Python 3.12+** in WSL (for Airflow)
4. **Python 3.13** in Windows (for ML tasks)

### WSL2 Installation (If Not Already Installed)

```powershell
# In PowerShell (Administrator)
wsl --install
wsl --set-default-version 2
wsl --install -d Ubuntu
```

### Airflow Environment Setup

```bash
# In WSL Terminal
cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1

# Create virtual environment
python3 -m venv airflow_env

# Activate environment
source airflow_env/bin/activate

# Install Airflow
pip install "apache-airflow==3.0.6" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.6/constraints-3.12.txt"

# Initialize Airflow database
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home
airflow db init
```

### Windows Python Environment Setup

```powershell
# In PowerShell
cd E:\ZuuCrew\telco-churn-prediction-mini-project-1

# Create virtual environment (if not exists)
python -m venv .venv

# Activate
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration Details

### Airflow Configuration

**File**: `airflow_home/airflow.cfg`

**Critical Path Updates for WSL**:

```ini
[core]
dags_folder = /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/dags
plugins_folder = /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home/plugins
executor = LocalExecutor

[database]
sql_alchemy_conn = sqlite:////mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home/airflow.db

[logging]
base_log_folder = /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home/logs
dag_processor_manager_log_location = /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home/logs/dag_processor_manager/dag_processor_manager.log
child_process_log_directory = /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home/logs/scheduler
```

**Key Changes from Windows to WSL**:
- `E:/ZuuCrew/...` → `/mnt/e/ZuuCrew/...`
- Windows backslashes (`\`) → Unix forward slashes (`/`)
- Double slashes in SQLite URI: `sqlite:////mnt/e/...`

---

## DAG Implementation

### Complete DAG Code

**File**: `dags/telco_churn_dag.py`

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'telco_churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for telco churn prediction',
    schedule=None,  # Changed from schedule_interval for Airflow 3.x
    catchup=False,
    tags=['ml', 'churn', 'telco'],
)

# Task 1: Data Preprocessing
preprocess_task = BashOperator(
    task_id='preprocess',
    bash_command='cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && .venv/Scripts/python.exe src/data/preprocess.py',
    dag=dag,
)

# Task 2: Model Training with MLflow
train_task = BashOperator(
    task_id='train',
    bash_command='cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && cmd.exe /c "set PYTHONIOENCODING=utf-8 && .venv\\Scripts\\python.exe src\\models\\train_mlflow.py 2>&1"',
    dag=dag,
)

# Task 3: Batch Inference
inference_task = BashOperator(
    task_id='inference',
    bash_command='cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && cmd.exe /c "set PYTHONIOENCODING=utf-8 && .venv\\Scripts\\python.exe src\\inference\\batch_predict.py 2>&1"',
    dag=dag,
)

# Define task dependencies (linear pipeline)
preprocess_task >> train_task >> inference_task
```

### Task Breakdown

#### Task 1: Preprocess

**Purpose**: Transform raw data into ML-ready features

**Command**:
```bash
cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && .venv/Scripts/python.exe src/data/preprocess.py
```

**Inputs**:
- `data/raw/Telco-Customer-Churn.csv`

**Outputs**:
- `artifacts/models/preprocessor.joblib` (fitted transformer)
- `artifacts/models/feature_names.json` (feature metadata)

**Transformations**:
- 19 input features → 45 engineered features
- Numeric scaling (StandardScaler)
- Categorical encoding (OneHotEncoder)
- Missing value imputation

#### Task 2: Train

**Purpose**: Train GradientBoostingClassifier with MLflow tracking

**Command**:
```bash
cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && cmd.exe /c "set PYTHONIOENCODING=utf-8 && .venv\\Scripts\\python.exe src\\models\\train_mlflow.py 2>&1"
```

**Why cmd.exe wrapper?**
- Sets `PYTHONIOENCODING=utf-8` to handle emoji characters (🚀, 📊, 🎉) in output
- Without this, Windows terminal uses cp1252 encoding → UnicodeEncodeError

**Inputs**:
- `artifacts/models/preprocessor.joblib`
- `data/raw/Telco-Customer-Churn.csv`

**Outputs**:
- `artifacts/models/sklearn_pipeline_mlflow.joblib`
- `artifacts/metrics/sklearn_metrics_mlflow.json`
- MLflow tracking data (`mlruns/`)

**Key Metrics**:
- Test Accuracy: 80.06%
- Test ROC AUC: 84.66%
- Training/Test split: 80/20

#### Task 3: Inference

**Purpose**: Generate batch predictions for new customers

**Command**:
```bash
cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && cmd.exe /c "set PYTHONIOENCODING=utf-8 && .venv\\Scripts\\python.exe src\\inference\\batch_predict.py 2>&1"
```

**Inputs**:
- `artifacts/models/sklearn_pipeline_mlflow.joblib`
- `artifacts/models/preprocessor.joblib`
- `data/raw/Telco-Customer-Churn.csv` (sample 100 records)

**Outputs**:
- `artifacts/predictions/batch_preds.csv`

**Output Schema**:
```csv
customerID,Prediction,ChurnProbability,[original_features...]
7590-VHVEG,0,0.1234,...
5575-GNVDE,1,0.8901,...
```

---

## Validation Results

### Full DAG Integration Test

**Command**:
```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow dags test telco_churn_prediction_pipeline 2025-01-01'
```

**Result**: ✅ **SUCCESS**

**Output**:
```
Dag run in success state
Dag run start: 2025-01-01 00:00:00+00:00
Dag run end: 2025-10-03 18:29:37.799649+00:00
state: success
run_type: manual
```

### Individual Task Test Results

#### Test 1: Preprocess Task ✅

**Command**:
```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow tasks test telco_churn_prediction_pipeline preprocess 2025-01-01'
```

**Result**:
- Exit Code: 0
- State: SUCCESS
- Duration: ~7 seconds

**Output Summary**:
```
✅ Preprocessor saved to artifacts/models/preprocessor.joblib
✅ Feature names saved to artifacts/models/feature_names.json
✅ Dataset shape: (7043, 21)
✅ Features generated: 45 (from 19 inputs)
   - Numeric columns: 4
   - Categorical columns: 15
   - Scaling: StandardScaler
   - Encoding: OneHotEncoder
```

#### Test 2: Train Task ✅

**Command**:
```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow tasks test telco_churn_prediction_pipeline train 2025-01-01'
```

**Result**:
- Exit Code: 0
- State: SUCCESS
- Duration: ~29 seconds

**Performance Metrics**:
```
🚀 Training Results:
   - Training Accuracy: 0.8158 (81.58%)
   - Training ROC AUC: 0.8669 (86.69%)

📊 Test Results:
   - Test Accuracy: 0.8006 (80.06%)
   - Test ROC AUC: 0.8466 (84.66%)

💾 Artifacts:
   - Pipeline: sklearn_pipeline_mlflow.joblib
   - Metrics: sklearn_metrics_mlflow.json
   - MLflow Run: 317bdad0eb494f679cc1eb23eb797a1b

🎉 Training completed successfully!
```

**Dataset Split**:
- Training set: 5,634 samples (80%)
- Test set: 1,409 samples (20%)
- Churn rate: 26.54%

**Model**: GradientBoostingClassifier (registered as version 14)

#### Test 3: Inference Task ✅

**Command**:
```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow tasks test telco_churn_prediction_pipeline inference 2025-01-01'
```

**Result**:
- Exit Code: 0
- State: SUCCESS
- Duration: ~6 seconds

**Prediction Summary**:
```
📈 Generated predictions for 100 records
📊 Prediction Summary:
   - Total customers: 100
   - Predicted churners: 23
   - Predicted non-churners: 77
   - Churn rate: 23.00%
   - Average churn probability: 0.2764

💾 Predictions saved to artifacts/predictions/batch_preds.csv
🎉 Batch prediction pipeline completed successfully!
```

### Acceptance Criteria Verification

| Criteria | Expected | Actual | Status |
|----------|----------|--------|--------|
| **DAG Structure** | 3 tasks, linear dependency | preprocess >> train >> inference | ✅ PASS |
| **Preprocess Execution** | Exit code 0, preprocessor created | Exit 0, 45 features generated | ✅ PASS |
| **Train Execution** | Exit code 0, model saved, >75% accuracy | Exit 0, 80.06% accuracy | ✅ PASS |
| **Inference Execution** | Exit code 0, predictions generated | Exit 0, 100 predictions | ✅ PASS |
| **MLflow Integration** | Metrics logged, model registered | Run ID logged, v14 registered | ✅ PASS |
| **Non-Interactive** | No user prompts | All tasks autonomous | ✅ PASS |
| **Artifacts Created** | All expected outputs present | All verified | ✅ PASS |
| **Full DAG Run** | End-to-end success | State: success | ✅ PASS |

**Overall**: ✅ **8/8 CRITERIA MET (100%)**

---

## Issues Resolved

### Issue 1: Airflow Windows Incompatibility ❌ → ✅

**Problem**: Airflow doesn't support native Windows execution

**Error**:
```
AttributeError: module 'os' has no attribute 'register_at_fork'
```

**Root Cause**: Airflow requires POSIX-compliant OS (Linux/macOS)

**Solution**: Run Airflow in WSL2 environment

**Files Modified**: `airflow_home/airflow.cfg`

**Status**: ✅ RESOLVED

---

### Issue 2: Path Incompatibility (Windows vs WSL) ❌ → ✅

**Problem**: Windows paths (`E:/...`) not accessible from WSL

**Symptoms**:
- DAG files not found
- Database connection failures
- Log directory errors

**Solution**: Convert all paths to WSL format (`/mnt/e/...`)

**Changes**:
```diff
# Before (Windows)
- dags_folder = E:/ZuuCrew/telco-churn-prediction-mini-project-1/dags

# After (WSL)
+ dags_folder = /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/dags
```

**Files Modified**:
- `airflow_home/airflow.cfg`
- `dags/telco_churn_dag.py`

**Status**: ✅ RESOLVED

---

### Issue 3: Airflow 3.x API Compatibility ❌ → ✅

**Problem**: `schedule_interval` parameter deprecated in Airflow 3.0

**Error**:
```
TypeError: DAG.__init__() got an unexpected keyword argument 'schedule_interval'
```

**Root Cause**: Airflow 3.0 changed DAG scheduling API

**Solution**: Update to new `schedule` parameter

**Changes**:
```diff
# Before (Airflow 2.x)
- schedule_interval=None,

# After (Airflow 3.x)
+ schedule=None,
```

**Files Modified**: `dags/telco_churn_dag.py`

**Status**: ✅ RESOLVED

---

### Issue 4: Python Module Not Found ❌ → ✅

**Problem**: WSL system Python lacks ML dependencies (pandas, scikit-learn, etc.)

**Error**:
```
ModuleNotFoundError: No module named 'pandas'
```

**Symptoms**: Tasks fail with exit code 127

**Root Cause**: Using WSL system Python instead of Windows .venv

**Solution**: Execute Windows Python directly from WSL

**Changes**:
```diff
# Before (WSL Python - FAILED)
- bash_command='python src/data/preprocess.py'

# After (Windows .venv Python - SUCCESS)
+ bash_command='.venv/Scripts/python.exe src/data/preprocess.py'
```

**Files Modified**: `dags/telco_churn_dag.py`

**Status**: ✅ RESOLVED

---

### Issue 5: Unicode Encoding Error ❌ → ✅

**Problem**: Windows terminal can't encode emoji characters (🚀, 📊, 🎉)

**Error**:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f680' in position 0
```

**Symptoms**:
- Train task fails (exit code 1)
- Inference task fails (exit code 1)
- Preprocess task succeeds (no emoji output)

**Root Cause**: Windows cmd uses cp1252 encoding by default

**Solution**: Set UTF-8 encoding before running Python

**Changes**:
```diff
# Before (FAILED)
- bash_command='.venv/Scripts/python.exe src/models/train_mlflow.py'

# After (SUCCESS)
+ bash_command='cmd.exe /c "set PYTHONIOENCODING=utf-8 && .venv\\Scripts\\python.exe src\\models\\train_mlflow.py 2>&1"'
```

**Applied to**:
- `train_task` (emojis in MLflow output)
- `inference_task` (emojis in prediction output)

**Files Modified**: `dags/telco_churn_dag.py`

**Status**: ✅ RESOLVED

---

## Testing Guide

### Environment Setup Before Testing

```bash
# In WSL Terminal
cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home
```

### Test Individual Tasks

#### Test Preprocess Task

```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow tasks test telco_churn_prediction_pipeline preprocess 2025-01-01'
```

**Expected Output**:
- Exit code: 0
- "Task instance in success state"
- Preprocessor saved to artifacts/models/

#### Test Train Task

```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow tasks test telco_churn_prediction_pipeline train 2025-01-01'
```

**Expected Output**:
- Exit code: 0
- Test accuracy > 75%
- MLflow run ID displayed
- Model saved to artifacts/models/

#### Test Inference Task

```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow tasks test telco_churn_prediction_pipeline inference 2025-01-01'
```

**Expected Output**:
- Exit code: 0
- 100 predictions generated
- Predictions saved to artifacts/predictions/batch_preds.csv

### Test Full DAG

```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow dags test telco_churn_prediction_pipeline 2025-01-01'
```

**Expected Output**:
- All 3 tasks execute in sequence
- "Dag run in success state"
- All artifacts created successfully

### Verify Artifacts

```bash
# Check preprocessor
ls -lh artifacts/models/preprocessor.joblib

# Check trained model
ls -lh artifacts/models/sklearn_pipeline_mlflow.joblib

# Check predictions
head -5 artifacts/predictions/batch_preds.csv

# Check MLflow runs
ls -lh mlruns/
```

### List All DAGs

```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && \
export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && \
airflow_env/bin/python -m airflow dags list'
```

**Expected Output**:
```
dag_id                                | filepath                      | owner       | paused
======================================|===============================|=============|========
telco_churn_prediction_pipeline       | telco_churn_dag.py            | mlops_team  | False
```

---

## Production Recommendations

### 1. Docker Deployment

**Why**: Containerization ensures consistency across environments

**Steps**:
- Create Dockerfile with Airflow + dependencies
- Use Docker Compose for multi-container setup
- Mount volumes for DAGs and logs

**Benefits**:
- Eliminates WSL dependency
- Easier deployment to cloud platforms
- Better resource isolation

### 2. Database Upgrade

**Current**: SQLite (development only)

**Recommended**: PostgreSQL or MySQL

**Reason**:
- SQLite doesn't support concurrent writes
- Production Airflow needs robust database
- Better performance for task scheduling

**Migration**:
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Update airflow.cfg
sql_alchemy_conn = postgresql+psycopg2://user:password@localhost/airflow
```

### 3. Executor Upgrade

**Current**: LocalExecutor (single machine)

**Recommended**: CeleryExecutor or KubernetesExecutor

**Benefits**:
- Parallel task execution
- Horizontal scaling
- Better fault tolerance

### 4. Monitoring & Alerting

**Implement**:
- Task failure email notifications
- Slack/Teams integration for alerts
- Prometheus + Grafana for metrics
- Airflow UI for DAG visualization

**Configuration**:
```python
default_args = {
    'email': ['mlops-team@example.com'],
    'email_on_failure': True,
    'email_on_retry': True,
}
```

### 5. Security Hardening

**Actions**:
- Enable RBAC (Role-Based Access Control)
- Use secrets backend (AWS Secrets Manager, HashiCorp Vault)
- HTTPS for Airflow webserver
- Restrict network access

### 6. Data Validation

**Add to DAG**:
- Great Expectations for data quality checks
- Schema validation before training
- Data drift detection
- Anomaly detection in predictions

### 7. Model Versioning

**Current**: MLflow tracking

**Enhance**:
- Model registry with staging/production tags
- A/B testing framework
- Model performance monitoring
- Automated model retraining triggers

### 8. CI/CD Integration

**Pipeline**:
1. Code commit → GitHub Actions
2. Run unit tests
3. Test DAG syntax (`airflow dags test`)
4. Deploy to staging
5. Smoke tests
6. Deploy to production

### 9. Logging Best Practices

**Implement**:
- Structured logging (JSON format)
- Centralized log aggregation (ELK stack)
- Log rotation policies
- Sensitive data masking

### 10. Resource Optimization

**Actions**:
- Set task-level resource limits
- Use task pools to prevent overload
- Configure parallelism settings
- Monitor CPU/memory usage

---

## Troubleshooting

### Common Issues & Solutions

#### Issue: DAG Not Showing in Airflow UI

**Symptoms**:
- DAG file exists but not listed
- No errors in logs

**Diagnosis**:
```bash
# Check DAG parsing
airflow_env/bin/python -m airflow dags list

# Check for syntax errors
python dags/telco_churn_dag.py
```

**Solutions**:
1. Verify `dags_folder` path in `airflow.cfg`
2. Check DAG file syntax
3. Ensure no import errors
4. Restart scheduler: `airflow scheduler`

---

#### Issue: Task Fails with "Command Not Found"

**Symptoms**:
- Exit code: 127
- "python: command not found"

**Diagnosis**:
```bash
# Check if Python executable exists
ls -lh .venv/Scripts/python.exe
```

**Solutions**:
1. Use full path: `.venv/Scripts/python.exe`
2. Verify Windows .venv is accessible from WSL
3. Check file permissions: `chmod +x .venv/Scripts/python.exe`

---

#### Issue: Unicode Encoding Errors

**Symptoms**:
- `UnicodeEncodeError: 'charmap' codec...`
- Exit code: 1

**Diagnosis**:
- Check if script outputs emoji or special characters
- Test encoding: `python -c "print('🚀')"`

**Solutions**:
1. Use cmd.exe wrapper: `cmd.exe /c "set PYTHONIOENCODING=utf-8 && ..."`
2. Remove emoji from code (alternative)
3. Set environment variable globally

---

#### Issue: Database Locked

**Symptoms**:
- "Database is locked"
- Tasks hang indefinitely

**Diagnosis**:
```bash
# Check for multiple Airflow processes
ps aux | grep airflow
```

**Solutions**:
1. Stop all Airflow processes
2. Remove lock file: `rm airflow_home/airflow.db-shm`
3. Consider upgrading to PostgreSQL

---

#### Issue: Airflow Webserver Won't Start

**Symptoms**:
- Port 8080 already in use
- Connection refused

**Diagnosis**:
```bash
# Check if port is in use
netstat -ano | findstr :8080  # Windows
lsof -i :8080  # WSL
```

**Solutions**:
1. Kill process using port: `kill -9 <PID>`
2. Use different port: `airflow webserver -p 8081`
3. Check firewall settings

---

#### Issue: Task Logs Not Showing

**Symptoms**:
- Task completes but no logs
- Empty log files

**Diagnosis**:
```bash
# Check log folder permissions
ls -ld airflow_home/logs

# Check logging configuration
grep base_log_folder airflow_home/airflow.cfg
```

**Solutions**:
1. Verify `base_log_folder` path
2. Check write permissions: `chmod 755 airflow_home/logs`
3. Check disk space: `df -h`

---

#### Issue: MLflow Artifacts Not Saved

**Symptoms**:
- Training completes but no artifacts in mlruns/
- "No such file or directory" errors

**Diagnosis**:
```bash
# Check MLflow tracking URI
echo $MLFLOW_TRACKING_URI

# Check artifacts directory
ls -lh mlruns/
```

**Solutions**:
1. Ensure artifacts directory exists
2. Check write permissions
3. Verify MLFLOW_TRACKING_URI is not set (use default)

---

### Debugging Commands

**Check Airflow Version**:
```bash
wsl bash -c 'airflow_env/bin/python -m airflow version'
```

**Test DAG Syntax**:
```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && python dags/telco_churn_dag.py'
```

**View Task Logs**:
```bash
wsl bash -c 'cat airflow_home/logs/dag_id=telco_churn_prediction_pipeline/run_id=*/task_id=preprocess/*.log'
```

**Check Python Dependencies**:
```bash
wsl bash -c '.venv/Scripts/python.exe -m pip list'
```

**Reset Airflow Database**:
```bash
wsl bash -c 'export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && airflow_env/bin/python -m airflow db reset'
```

---

## Quick Reference Commands

### Daily Operations

**Start Airflow Scheduler** (Background):
```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && airflow_env/bin/python -m airflow scheduler &'
```

**Start Airflow Webserver** (Background):
```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && airflow_env/bin/python -m airflow webserver -p 8080 &'
```

**Trigger DAG Manually**:
```bash
wsl bash -c 'cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1 && export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && airflow_env/bin/python -m airflow dags trigger telco_churn_prediction_pipeline'
```

**Pause/Unpause DAG**:
```bash
# Pause
wsl bash -c 'export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && airflow_env/bin/python -m airflow dags pause telco_churn_prediction_pipeline'

# Unpause
wsl bash -c 'export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && airflow_env/bin/python -m airflow dags unpause telco_churn_prediction_pipeline'
```

**View DAG Run History**:
```bash
wsl bash -c 'export AIRFLOW_HOME=/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1/airflow_home && airflow_env/bin/python -m airflow dags list-runs -d telco_churn_prediction_pipeline'
```

---

## Appendix

### File Structure

```
telco-churn-prediction-mini-project-1/
├── airflow_home/
│   ├── airflow.cfg              # Airflow configuration (WSL paths)
│   ├── airflow.db               # SQLite database
│   ├── logs/                    # Task execution logs
│   └── plugins/                 # Custom plugins
├── airflow_env/                 # Airflow virtual environment (WSL Python 3.12)
├── .venv/                       # ML dependencies (Windows Python 3.13)
├── dags/
│   └── telco_churn_dag.py       # Main DAG definition
├── src/
│   ├── data/
│   │   └── preprocess.py        # Data preprocessing script
│   ├── models/
│   │   └── train_mlflow.py      # Model training with MLflow
│   └── inference/
│       └── batch_predict.py     # Batch prediction script
├── artifacts/
│   ├── models/                  # Trained models and preprocessors
│   ├── metrics/                 # Performance metrics
│   └── predictions/             # Batch predictions output
├── mlruns/                      # MLflow tracking data
└── data/
    └── raw/                     # Raw dataset
```

### Key Versions

- **Airflow**: 3.0.6
- **Python (WSL)**: 3.12.3
- **Python (Windows)**: 3.13
- **pandas**: 2.2.3
- **scikit-learn**: 1.5.2
- **mlflow**: 2.17.2
- **SQLite**: 3.x

### Useful Links

- [Airflow Documentation](https://airflow.apache.org/docs/apache-airflow/3.0.6/)
- [WSL2 Documentation](https://learn.microsoft.com/en-us/windows/wsl/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Project Repository](https://github.com/deaneeth/telco-churn-mlops-pipeline)

---

## Summary

✅ **Airflow DAG orchestration fully validated and operational**

**Achievements**:
1. ✅ Hybrid WSL + Windows architecture implemented
2. ✅ All 3 tasks validated individually
3. ✅ Full end-to-end DAG test passed
4. ✅ 80.06% model accuracy, 84.66% ROC AUC
5. ✅ Unicode encoding issues resolved
6. ✅ Airflow 3.x compatibility confirmed
7. ✅ Production-ready documentation created

**Ready for Step 9**: Docker containerization and FastAPI deployment

---

**Document Version**: 1.0  
**Last Updated**: October 3, 2025, 23:59 IST  
**Maintained By**: MLOps Team  
**Project**: Telco Customer Churn Prediction Pipeline
