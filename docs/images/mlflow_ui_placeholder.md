# MLflow UI Screenshot

**Purpose:** This file serves as a placeholder for MLflow tracking UI screenshots.

## How to Generate Screenshot

1. **Start MLflow UI:**
   ```bash
   mlflow ui --backend-store-uri file:///path/to/mlruns --port 5001
   ```

2. **Navigate to:**
   ```
   http://localhost:5001
   ```

3. **Capture Screenshots:**
   - Experiments list view
   - Run details page showing:
     - Parameters (learning_rate, n_estimators, max_depth, etc.)
     - Metrics (accuracy, roc_auc, precision, recall, f1_score)
     - Artifacts (model files, confusion matrix plots)
   - Model registry showing version 15

4. **Save as:**
   - `mlflow_experiments.png` - Experiments overview
   - `mlflow_run_details.png` - Run d165e184b3944c50851f14a65aaf12b5 details
   - `mlflow_model_registry.png` - Model registry with version 15

## Expected Content

### Experiment View
- Experiment Name: `telco-churn-prediction`
- Total Runs: 15+
- Latest Run ID: `d165e184b3944c50851f14a65aaf12b5`

### Run Details
**Parameters:**
- `learning_rate`: 0.1
- `n_estimators`: 100
- `max_depth`: 3
- `min_samples_split`: 2

**Metrics:**
- `test_accuracy`: 0.8006
- `test_roc_auc`: 0.8466
- `train_accuracy`: 0.8158
- `train_roc_auc`: 0.8669

**Artifacts:**
- sklearn_pipeline_mlflow.joblib (200 KB)
- sklearn_metrics_mlflow.json (0.4 KB)

### Model Registry
- Model Name: `telco_churn_model`
- Latest Version: 15
- Status: Production
- Run ID: `d165e184b3944c50851f14a65aaf12b5`

---

**Note:** For submission, replace this placeholder with actual screenshots from your MLflow UI.
