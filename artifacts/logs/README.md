# Logs Directory

This directory contains log files generated during model training, evaluation, and inference.

## Structure

- **training/**: Training logs for model development
- **inference/**: Prediction and inference logs
- **pipeline/**: Pipeline execution logs (sklearn and spark)
- **airflow/**: Airflow DAG execution logs
- **monitoring/**: Model performance monitoring logs

## Log Files

### Training Logs
- `model_training.log`: General training process logs
- `hyperparameter_tuning.log`: Hyperparameter optimization logs
- `data_preprocessing.log`: Data preprocessing logs

### Inference Logs
- `predictions.log`: Prediction requests and results
- `batch_predictions.log`: Batch prediction job logs
- `model_loading.log`: Model loading and initialization logs

### Pipeline Logs
- `sklearn_pipeline.log`: Sklearn pipeline execution logs
- `spark_pipeline.log`: Spark pipeline execution logs
- `data_quality.log`: Data quality check logs

### Monitoring Logs
- `model_performance.log`: Model performance monitoring
- `data_drift.log`: Data drift detection logs
- `system_metrics.log`: System resource usage logs

## Log Configuration

Logs are configured using Python's logging module with:
- **Level**: INFO for general information, DEBUG for detailed debugging
- **Format**: Timestamp, level, module, message
- **Rotation**: Daily rotation with 30-day retention
- **Handlers**: File and console output

## Log Analysis

Use the following tools for log analysis:
- `grep` for searching specific patterns
- `tail -f` for real-time monitoring
- ELK stack for centralized log analysis (if configured)

## Sample Log Entry

```
2024-01-15 10:30:25,123 - INFO - model_trainer - Model training started for LogisticRegression
2024-01-15 10:30:25,124 - INFO - model_trainer - Training set size: (8000, 19)
2024-01-15 10:30:25,125 - INFO - model_trainer - Validation set size: (2000, 19)
2024-01-15 10:30:27,856 - INFO - model_trainer - Training completed. Accuracy: 0.8532
2024-01-15 10:30:27,857 - INFO - model_trainer - Model saved to artifacts/models/logistic_regression_model.pkl
```