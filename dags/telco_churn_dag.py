"""
Telco Customer Churn Prediction DAG
====================================

This DAG orchestrates the complete ML pipeline for telco customer churn prediction:
1. Data preprocessing 
2. Model training with MLflow
3. Batch inference

Dependencies are chained: preprocess >> train >> inference
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default arguments for all tasks
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 24),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Create DAG
dag = DAG(
    'telco_churn_prediction_pipeline',
    default_args=default_args,
    description='Complete ML pipeline for telco customer churn prediction',
    schedule_interval=None,  # Manual trigger only
    max_active_runs=1,
    tags=['ml', 'churn', 'telco', 'prediction']
)

# Task 1: Data Preprocessing
preprocess_task = BashOperator(
    task_id='preprocess',
    bash_command='cd /opt/airflow && python src/data/preprocess.py',
    dag=dag,
    doc_md="""
    ## Data Preprocessing Task
    
    This task runs the data preprocessing pipeline which:
    - Loads raw telco customer data
    - Cleans and transforms features  
    - Handles missing values and data types
    - Saves processed data for training
    
    **Input**: `data/raw/telco_customer_churn.csv`  
    **Output**: `data/processed/sample.csv`
    """)

# Task 2: Model Training
train_task = BashOperator(
    task_id='train',
    bash_command='cd /opt/airflow && python src/models/train_mlflow.py',
    dag=dag,
    doc_md="""
    ## Model Training Task
    
    This task trains the machine learning model with MLflow tracking:
    - Loads processed training data
    - Trains GradientBoosting classifier with optimized hyperparameters
    - Logs metrics, parameters, and artifacts to MLflow
    - Saves trained model for inference
    
    **Input**: `data/processed/sample.csv`  
    **Output**: `artifacts/models/sklearn_pipeline_mlflow.joblib`
    """)

# Task 3: Batch Inference  
inference_task = BashOperator(
    task_id='inference',
    bash_command='cd /opt/airflow && python src/inference/batch_predict.py',
    dag=dag,
    doc_md="""
    ## Batch Inference Task
    
    This task generates predictions on batch data:
    - Loads trained model and processed data
    - Generates churn predictions and probabilities
    - Saves predictions with timestamps
    - Provides summary statistics
    
    **Input**: `artifacts/models/sklearn_pipeline_mlflow.joblib`, `data/processed/sample.csv`  
    **Output**: `artifacts/predictions/batch_predictions.csv`
    """)

# Define task dependencies - linear pipeline
preprocess_task >> train_task >> inference_task

# Optional: Add task documentation
dag.doc_md = """
# Telco Customer Churn Prediction Pipeline

This DAG implements an end-to-end machine learning pipeline for predicting customer churn
in the telecommunications industry.

## Pipeline Overview

```
Raw Data → Preprocessing → Model Training → Batch Inference → Predictions
```

## Tasks

1. **preprocess**: Cleans and transforms raw customer data
2. **train**: Trains ML model with MLflow experiment tracking  
3. **inference**: Generates batch predictions on processed data

## Usage

This DAG is configured for manual triggering (`schedule_interval=None`). 
To run the complete pipeline, trigger the DAG from the Airflow UI.

## Monitoring

- Check MLflow UI for training metrics and model artifacts
- Review task logs for detailed execution information
- Monitor prediction outputs in the artifacts directory
"""