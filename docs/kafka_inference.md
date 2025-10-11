# Kafka Inference Backend Documentation

## Overview

The `inference_backend` module provides a unified interface for loading ML models and running real-time predictions in the Kafka consumer pipeline. It abstracts model inference logic from the consumer implementation, enabling easy swapping between different ML backends.

**Supported Backends:**
- **sklearn**: Scikit-learn models (joblib format)
- **spark**: Apache Spark MLlib models (with automatic fallback to sklearn)

**Module Location:** `src/streaming/inference_backend.py`

---

## Architecture

### Backend Classes

```
ModelBackend (Abstract Base)
├── SklearnBackend
│   ├── Supports joblib pipelines
│   ├── Optional separate preprocessor
│   └── Direct pandas DataFrame inference
└── SparkBackend
    ├── Supports Spark ML pipelines
    ├── Automatic sklearn fallback
    └── Converts pandas ↔ Spark DataFrames
```

### Key Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `load_model(backend, model_path, ...)` | Load model from disk | `ModelBackend` instance |
| `predict(model_backend, features, ...)` | Run predictions on features | `(predictions, probabilities)` |
| `get_backend_info(model_backend)` | Get backend metadata | Dictionary with model info |

---

## Usage Guide

### 1. Loading Models

#### Sklearn Model

```python
from src.streaming.inference_backend import load_model

# Load sklearn pipeline (includes preprocessing)
backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')

# Load sklearn model with separate preprocessor
backend = load_model(
    'sklearn',
    'artifacts/models/model.joblib',
    preprocessor_path='artifacts/models/preprocessor.joblib'
)
```

#### Spark Model with Fallback

```python
# Load Spark model (will fallback to sklearn if Spark unavailable)
backend = load_model(
    'spark',
    'artifacts/models/spark_model',
    fallback_path='artifacts/models/sklearn_pipeline.joblib'
)

# Check if using fallback
info = get_backend_info(backend)
if info['using_fallback']:
    print("Using sklearn fallback instead of Spark")
```

### 2. Running Predictions

#### Single Record Prediction

```python
from src.streaming.inference_backend import predict

# Input as dictionary
customer = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'Yes',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 50.0,
    'TotalCharges': '600.0'
}

predictions, probabilities = predict(backend, customer)
print(f"Prediction: {predictions[0]}, Probability: {probabilities[0]:.4f}")
# Output: Prediction: No, Probability: 0.2543
```

#### Batch Prediction

```python
import pandas as pd

# Input as DataFrame
customers_df = pd.DataFrame([
    {'gender': 'Male', 'tenure': 12, ...},
    {'gender': 'Female', 'tenure': 24, ...},
    {'gender': 'Male', 'tenure': 6, ...}
])

predictions, probabilities = predict(backend, customers_df)
print(f"Batch predictions: {predictions}")
# Output: Batch predictions: ['No', 'Yes', 'No']
```

#### List of Dictionaries

```python
# Input as list of dicts
customers_list = [
    {'gender': 'Male', 'tenure': 12, ...},
    {'gender': 'Female', 'tenure': 24, ...}
]

predictions, probabilities = predict(backend, customers_list)
```

### 3. Backend Information

```python
from src.streaming.inference_backend import get_backend_info

info = get_backend_info(backend)
print(info)

# Example output for sklearn:
# {
#     'backend_type': 'sklearn',
#     'model_path': 'artifacts/models/sklearn_pipeline.joblib',
#     'model_type': 'Pipeline',
#     'has_preprocessor': False,
#     'preprocessor_path': None
# }

# Example output for spark with fallback:
# {
#     'backend_type': 'spark',
#     'model_path': 'artifacts/models/spark_model',
#     'model_type': 'Pipeline',
#     'using_fallback': True,
#     'fallback_path': 'artifacts/models/sklearn_pipeline.joblib',
#     'spark_available': False
# }
```

---

## Integration with Kafka Consumer

### Current Consumer Implementation

The consumer uses inline model loading and inference:

```python
# OLD: Direct joblib loading
model = joblib.load('artifacts/models/sklearn_pipeline.joblib')
probabilities = model.predict_proba(features)
```

### Refactored with Inference Backend

```python
# NEW: Use inference backend
from src.streaming.inference_backend import load_model, predict

# Load model once at startup
backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')

# Use in message processing loop
predictions, probabilities = predict(backend, customer_features)
```

### Benefits of Abstraction

1. **Easy Backend Switching**: Change from sklearn to Spark by modifying one line
2. **Fallback Strategy**: Automatic sklearn fallback when Spark unavailable
3. **Testability**: Mock backends easily in unit tests
4. **Consistency**: Unified interface regardless of backend
5. **Input Flexibility**: Accepts dict, list, or DataFrame

---

## Backend-Specific Details

### Sklearn Backend

**File Format:** Joblib (`.joblib` or `.pkl`)

**Model Requirements:**
- Must implement `predict_proba(X)` method
- Should accept pandas DataFrame input
- Should return numpy array with shape `(n_samples, n_classes)`

**Preprocessor (Optional):**
- If model doesn't include preprocessing, provide separate preprocessor
- Must implement `transform(X)` method
- Applied before `predict_proba`

**Example Pipeline Structure:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])
```

**Performance:**
- **Latency**: ~8-13ms per record (single-threaded)
- **Throughput**: ~100 messages/second
- **Memory**: ~500MB for loaded pipeline

### Spark Backend

**File Format:** Spark ML PipelineModel directory structure

**Fallback Strategy:**
1. Check if PySpark is installed
2. Check if Spark model path exists
3. Attempt to initialize Spark session
4. Attempt to load Spark model
5. **If any step fails → fallback to sklearn**

**Limitations:**

| Limitation | Impact | Workaround |
|------------|--------|-----------|
| **Spark Session Overhead** | 2-3 second startup time | Use long-running consumer processes |
| **Memory Usage** | 2-4GB for Spark session | Ensure adequate JVM heap space |
| **Network Latency** | Slower than sklearn for small batches | Use batch mode for efficiency |
| **Environment Dependencies** | Requires Java 8+, Spark 3.x | Provide sklearn fallback |

**When to Use Spark Backend:**
- Processing large batches (1000+ records)
- Existing Spark ML models from Mini Project 1
- Distributed inference requirements
- Consistent with upstream training pipeline

**When to Use Sklearn Fallback:**
- Real-time streaming (low latency)
- Development/testing environments
- Docker containers (smaller image size)
- Single-machine deployments

**Example Spark Configuration:**
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TelcoChurnInference") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

backend = load_model(
    'spark',
    'artifacts/models/spark_model',
    fallback_path='artifacts/models/sklearn_pipeline.joblib',
    spark_session=spark  # Reuse existing session
)
```

---

## Error Handling

### Common Errors

#### FileNotFoundError
```python
# Error: Model file doesn't exist
FileNotFoundError: Model file not found: artifacts/models/nonexistent.joblib

# Solution: Verify model path
from pathlib import Path
assert Path('artifacts/models/sklearn_pipeline.joblib').exists()
```

#### ValueError: Model not loaded
```python
# Error: Calling predict before load
ValueError: Model not loaded. Call load() first.

# Solution: Ensure load() is called
backend = SklearnBackend('path/to/model.joblib')
backend.load()  # ← Must call before predict
```

#### Spark Fallback Warnings
```python
# Warning: PySpark not available
WARNING - PySpark not available, using sklearn fallback

# Expected behavior: Will use fallback_path model
# No action needed if fallback model exists
```

#### Inference Errors
```python
# Error: Invalid input features
Exception: Cannot use median strategy with non-numeric data

# Solution: Validate input features before inference
# Check for missing values, incorrect types, etc.
```

### Error Recovery

```python
import logging
from src.streaming.inference_backend import load_model, predict

logger = logging.getLogger(__name__)

try:
    # Attempt primary backend
    backend = load_model('spark', 'artifacts/models/spark_model')
except Exception as e:
    logger.warning(f"Spark backend failed: {e}")
    # Fallback to sklearn
    backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')

try:
    predictions, probabilities = predict(backend, features)
except Exception as e:
    logger.error(f"Inference failed: {e}")
    # Route to dead letter queue or use default prediction
    predictions = ["Unknown"]
    probabilities = [0.5]
```

---

## Testing Strategy

### Unit Tests (Mocked Models)

```python
import pytest
from unittest.mock import Mock
from src.streaming.inference_backend import SklearnBackend

def test_sklearn_prediction():
    # Mock model
    mock_model = Mock()
    mock_model.predict_proba = Mock(return_value=np.array([[0.7, 0.3]]))
    
    # Create backend with mock
    backend = SklearnBackend('path/to/model.joblib')
    backend.model = mock_model
    
    # Test prediction
    predictions, probs = backend.predict(sample_df)
    
    assert predictions[0] == "No"
    assert probs[0] == 0.3
```

**Advantages:**
- No model file dependencies
- Fast execution (< 1 second)
- Test business logic separately from model

### Integration Tests (Real Models)

```python
@pytest.mark.skipif(
    not Path('artifacts/models/sklearn_pipeline.joblib').exists(),
    reason="Model file not found"
)
def test_real_sklearn_model():
    backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')
    predictions, probs = predict(backend, sample_features)
    
    assert predictions[0] in ["Yes", "No"]
    assert 0.0 <= probs[0] <= 1.0
```

**Advantages:**
- Validates actual model behavior
- Catches incompatibility issues
- End-to-end verification

### Performance Tests

```python
import time

def test_inference_latency():
    backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')
    
    start = time.time()
    for _ in range(100):
        predict(backend, sample_features)
    elapsed = time.time() - start
    
    avg_latency = elapsed / 100
    assert avg_latency < 0.020  # < 20ms per prediction
```

---

## Performance Benchmarks

### Sklearn Backend

| Metric | Value | Conditions |
|--------|-------|------------|
| Model Load Time | 1.2s | sklearn_pipeline.joblib (45MB) |
| Single Record Latency | 8-13ms | Pipeline with preprocessing |
| Batch Latency (100 records) | 0.8s | ~8ms per record |
| Throughput | 100-120 msg/s | Single-threaded consumer |
| Memory Footprint | 500MB | Loaded pipeline in RAM |

### Spark Backend (Local Mode)

| Metric | Value | Conditions |
|--------|-------|------------|
| Spark Session Startup | 2-3s | First initialization |
| Model Load Time | 1.5-2s | Spark PipelineModel |
| Single Record Latency | 50-100ms | Overhead from DataFrame conversion |
| Batch Latency (100 records) | 2-3s | ~20-30ms per record |
| Batch Latency (1000 records) | 5-8s | ~5-8ms per record (efficient) |
| Throughput | 20-50 msg/s | Single partition |
| Memory Footprint | 2-4GB | Spark session + model |

**Recommendation:** Use sklearn for streaming (< 100 records), Spark for batch (1000+ records)

---

## Migration Guide

### From Inline Model Loading

**Before:**
```python
import joblib

model = joblib.load('artifacts/models/sklearn_pipeline.joblib')
probabilities = model.predict_proba(features)
churn_prob = probabilities[0][1]
prediction = "Yes" if churn_prob >= 0.5 else "No"
```

**After:**
```python
from src.streaming.inference_backend import load_model, predict

backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')
predictions, probabilities = predict(backend, features)
# predictions = ["Yes" or "No"], probabilities = [0.0-1.0]
```

### Consumer Integration Steps

1. **Import backend module** at top of consumer.py
2. **Load model once** in main() before Kafka loop
3. **Replace inline inference** with predict() call
4. **Handle errors** with try/except around predict()
5. **Update tests** to use mocked backends

---

## Troubleshooting

### Issue: Spark session won't start

**Symptoms:**
```
Exception: Java gateway process exited before sending driver port
```

**Solutions:**
1. Verify Java 8 or 11 installed: `java -version`
2. Set JAVA_HOME environment variable
3. Use sklearn fallback for development
4. Check Spark logs in `spark-warehouse/`

### Issue: High memory usage

**Symptoms:**
- Consumer process using > 4GB RAM
- Out of memory errors

**Solutions:**
1. Reduce Spark executor memory: `.config("spark.executor.memory", "1g")`
2. Use sklearn backend instead of Spark
3. Process smaller batches
4. Implement batch prediction windowing

### Issue: Slow prediction latency

**Symptoms:**
- Prediction taking > 100ms per record

**Solutions:**
1. Use sklearn for low-latency streaming
2. Ensure model includes preprocessing (avoid separate transformer)
3. Profile with `cProfile` to identify bottlenecks
4. Consider model simplification (fewer features/estimators)

### Issue: Fallback always triggered

**Symptoms:**
```
WARNING - PySpark not available, using sklearn fallback
```

**Solutions:**
1. Install PySpark: `pip install pyspark==3.4.0`
2. Verify Spark model path exists
3. Check Spark version compatibility
4. Review Spark session configuration

---

## Best Practices

### 1. Model Loading

✅ **DO:**
- Load model once at consumer startup
- Reuse backend instance across predictions
- Provide fallback path for Spark models
- Validate model file exists before loading

❌ **DON'T:**
- Load model for every prediction
- Create new backend instances in loops
- Assume Spark is always available
- Skip error handling on model load

### 2. Prediction

✅ **DO:**
- Validate input features before prediction
- Log prediction errors to dead letter queue
- Use batch prediction for efficiency
- Monitor prediction latency metrics

❌ **DON'T:**
- Pass invalid/missing features to model
- Silently fail predictions
- Process records one-by-one in batch mode
- Ignore prediction errors

### 3. Error Handling

✅ **DO:**
- Wrap predict() in try/except
- Route failures to dead letter topic
- Log detailed error messages
- Implement graceful fallback

❌ **DON'T:**
- Let exceptions crash consumer
- Lose failed records without logging
- Use generic error messages
- Retry infinitely on persistent errors

### 4. Testing

✅ **DO:**
- Mock models in unit tests
- Test with real models in integration tests
- Benchmark prediction latency
- Test both backends (sklearn + spark)

❌ **DON'T:**
- Depend on model files in unit tests
- Skip integration tests
- Assume performance is acceptable
- Test only one backend

---

## Future Enhancements

### Planned Features

1. **ONNX Backend**: Support for ONNX Runtime models
2. **TensorFlow Backend**: TF SavedModel support
3. **Batch Windowing**: Automatic micro-batching for efficiency
4. **Model Versioning**: Track model versions in predictions
5. **A/B Testing**: Route traffic between multiple models
6. **Caching**: Cache predictions for duplicate requests
7. **Metrics**: Built-in latency/throughput monitoring

### Experimental Features

- **GPU Acceleration**: CUDA-enabled inference
- **Model Ensembling**: Combine multiple model predictions
- **Auto-fallback**: Intelligent backend selection based on batch size

---

## References

- [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Spark ML Pipeline Documentation](https://spark.apache.org/docs/latest/ml-pipeline.html)
- [Kafka Consumer Best Practices](https://kafka.apache.org/documentation/#consumerconfigs)
- Mini Project 1: Telco Churn Model Training

---

## Appendix: API Reference

### `load_model(backend, model_path, **kwargs)`

Load a model from disk for the specified backend.

**Parameters:**
- `backend` (str): Backend type ('sklearn' or 'spark')
- `model_path` (str): Path to saved model file/directory
- `preprocessor_path` (str, optional): Path to separate preprocessor (sklearn only)
- `fallback_path` (str, optional): Path to sklearn fallback (spark only)
- `spark_session` (SparkSession, optional): Existing Spark session (spark only)

**Returns:**
- `ModelBackend`: Loaded backend instance

**Raises:**
- `ValueError`: If backend type invalid
- `FileNotFoundError`: If model file doesn't exist
- `Exception`: If model loading fails

---

### `predict(model_backend, features, backend=None)`

Run prediction on input features.

**Parameters:**
- `model_backend` (ModelBackend): Loaded backend instance
- `features` (DataFrame|dict|list): Input features
- `backend` (str, optional): Backend type override

**Returns:**
- `tuple`: (predictions, probabilities)
  - `predictions` (list[str]): List of "Yes" or "No"
  - `probabilities` (list[float]): List of churn probabilities (0.0-1.0)

**Raises:**
- `ValueError`: If input format invalid
- `Exception`: If prediction fails

---

### `get_backend_info(model_backend)`

Get information about the loaded model backend.

**Parameters:**
- `model_backend` (ModelBackend): Loaded backend instance

**Returns:**
- `dict`: Backend metadata including type, path, model info

---

**Document Version:** 1.0.0  
**Last Updated:** 2025-10-11  
**Author:** Telco Churn Prediction Team
