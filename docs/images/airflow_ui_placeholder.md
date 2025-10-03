# Airflow UI Screenshot

**Purpose:** This file serves as a placeholder for Apache Airflow UI screenshots.

## How to Generate Screenshot

### Step 1: Start Airflow (WSL2/Ubuntu)

```bash
# In WSL2 terminal
wsl
source airflow_env/bin/activate
export AIRFLOW_HOME=/path/to/airflow_home

# Start services
airflow webserver --port 8080 &
airflow scheduler &
```

### Step 2: Access Airflow UI

Navigate to: `http://localhost:8080`

**Login:**
- Username: `admin`
- Password: (your admin password)

### Step 3: Capture Screenshots

**1. DAG List View:**
- File: `airflow_dags_list.png`
- Content: List of all DAGs including `telco_churn_prediction_pipeline`
- Status: Enabled (toggle ON)

**2. DAG Graph View:**
- File: `airflow_dag_graph.png`
- Content: Visual representation of task dependencies
- Tasks:
  ```
  load_data → preprocess_data → train_model → evaluate_model → batch_inference
  ```

**3. DAG Run History:**
- File: `airflow_dag_runs.png`
- Content: Recent DAG runs with status (success/failed/running)
- Show at least one successful run

**4. Task Instance Details:**
- File: `airflow_task_details.png`
- Content: Logs from a successful task execution
- Example task: `train_model` or `evaluate_model`

## Expected Content

### DAG Configuration

**DAG ID:** `telco_churn_prediction_pipeline`

**Schedule:** Daily at midnight (`0 0 * * *`)

**Tasks (5 total):**

1. **load_data**
   - Operator: `PythonOperator`
   - Function: Load raw Telco Customer Churn CSV
   - Duration: ~2 seconds

2. **preprocess_data**
   - Operator: `PythonOperator`
   - Function: Feature engineering (19 → 45 features)
   - Duration: ~3 seconds
   - Depends on: `load_data`

3. **train_model**
   - Operator: `PythonOperator`
   - Function: Train GradientBoostingClassifier with MLflow
   - Duration: ~15 seconds
   - Depends on: `preprocess_data`

4. **evaluate_model**
   - Operator: `PythonOperator`
   - Function: Calculate metrics (ROC-AUC, accuracy)
   - Duration: ~2 seconds
   - Depends on: `train_model`

5. **batch_inference**
   - Operator: `PythonOperator`
   - Function: Generate predictions on sample data
   - Duration: ~5 seconds
   - Depends on: `evaluate_model`

### DAG Metadata

- **Owner:** airflow
- **Retries:** 1
- **Retry Delay:** 5 minutes
- **Start Date:** 2025-01-01
- **Catchup:** False
- **Tags:** `['ml', 'churn', 'telco', 'production']`

### Sample Log Output (train_model task)

```
[2025-10-04 12:00:00] INFO - Starting model training...
[2025-10-04 12:00:01] INFO - Loading preprocessed data...
[2025-10-04 12:00:02] INFO - Training GradientBoostingClassifier...
[2025-10-04 12:00:15] INFO - MLflow Run ID: d165e184b3944c50851f14a65aaf12b5
[2025-10-04 12:00:15] INFO - Test Accuracy: 0.8006
[2025-10-04 12:00:15] INFO - Test ROC-AUC: 0.8466
[2025-10-04 12:00:16] INFO - Model saved to MLflow registry (version 15)
[2025-10-04 12:00:16] INFO - Task completed successfully
```

## Validation Screenshot

### DAG Syntax Validation

```bash
# Validate DAG syntax
python dags/telco_churn_dag.py
# Expected output: No errors (silent if valid)

# List DAGs
airflow dags list
# Expected: telco_churn_prediction_pipeline appears in list

# Test single task
airflow tasks test telco_churn_prediction_pipeline load_data 2025-10-04
```

---

**Note:** For submission, replace this placeholder with actual screenshots from your Airflow UI showing:
1. DAG overview with enabled status
2. Task dependency graph visualization
3. Successful run history
4. Task logs demonstrating successful execution
