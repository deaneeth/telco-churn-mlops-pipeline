"""
Unit Tests for Inference Backend

Tests the inference backend module functions with mocked models
to avoid heavy dependencies on actual model files.

Test Coverage:
- SklearnBackend: load, predict, with/without preprocessor
- SparkBackend: load, predict, fallback behavior
- load_model function: sklearn and spark backends
- predict function: various input formats
- get_backend_info function

Run with: pytest tests/test_inference_backend.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import joblib

from src.streaming.inference_backend import (
    load_model,
    predict,
    get_backend_info,
    SklearnBackend,
    SparkBackend,
    SPARK_AVAILABLE
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_features_dict():
    """Sample customer features as dictionary."""
    return {
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


@pytest.fixture
def sample_features_df(sample_features_dict):
    """Sample customer features as DataFrame."""
    return pd.DataFrame([sample_features_dict])


@pytest.fixture
def mock_sklearn_model():
    """Mock sklearn model with predict_proba method."""
    model = Mock()
    # Mock predict_proba to return probabilities for 2 classes
    # [[prob_no, prob_yes]] where prob_yes is churn probability
    model.predict_proba = Mock(return_value=np.array([[0.7, 0.3]]))
    return model


@pytest.fixture
def mock_sklearn_pipeline():
    """Mock sklearn pipeline."""
    pipeline = Mock()
    pipeline.__class__.__name__ = 'Pipeline'
    pipeline.predict_proba = Mock(return_value=np.array([[0.6, 0.4]]))
    return pipeline


@pytest.fixture
def mock_preprocessor():
    """Mock preprocessor with transform method."""
    preprocessor = Mock()
    preprocessor.transform = Mock(return_value=pd.DataFrame([[1, 2, 3]]))
    return preprocessor


# ============================================================================
# SklearnBackend Tests
# ============================================================================

class TestSklearnBackend:
    """Test cases for SklearnBackend class."""
    
    def test_init(self):
        """Test SklearnBackend initialization."""
        backend = SklearnBackend('path/to/model.joblib')
        assert backend.backend_type == 'sklearn'
        assert backend.model_path == 'path/to/model.joblib'
        assert backend.model is None
        assert backend.preprocessor is None
        assert backend.preprocessor_path is None
    
    def test_init_with_preprocessor(self):
        """Test SklearnBackend initialization with preprocessor path."""
        backend = SklearnBackend('path/to/model.joblib', 'path/to/preprocessor.joblib')
        assert backend.preprocessor_path == 'path/to/preprocessor.joblib'
    
    @patch('src.streaming.inference_backend.joblib.load')
    @patch('src.streaming.inference_backend.Path')
    def test_load_success(self, mock_path, mock_joblib_load, mock_sklearn_pipeline):
        """Test successful model loading."""
        # Mock file existence
        mock_path.return_value.exists.return_value = True
        mock_joblib_load.return_value = mock_sklearn_pipeline
        
        backend = SklearnBackend('path/to/model.joblib')
        model = backend.load()
        
        assert model is not None
        assert backend.model is not None
        mock_joblib_load.assert_called_once_with('path/to/model.joblib')
    
    @patch('src.streaming.inference_backend.Path')
    def test_load_file_not_found(self, mock_path):
        """Test model loading with non-existent file."""
        mock_path.return_value.exists.return_value = False
        
        backend = SklearnBackend('path/to/nonexistent.joblib')
        
        with pytest.raises(FileNotFoundError):
            backend.load()
    
    @patch('src.streaming.inference_backend.joblib.load')
    @patch('src.streaming.inference_backend.Path')
    def test_load_with_preprocessor(self, mock_path, mock_joblib_load, mock_sklearn_pipeline, mock_preprocessor):
        """Test model loading with separate preprocessor."""
        mock_path.return_value.exists.return_value = True
        # First call loads model, second loads preprocessor
        mock_joblib_load.side_effect = [mock_sklearn_pipeline, mock_preprocessor]
        
        backend = SklearnBackend('path/to/model.joblib', 'path/to/preprocessor.joblib')
        model = backend.load()
        
        assert backend.model is not None
        assert backend.preprocessor is not None
        assert mock_joblib_load.call_count == 2
    
    def test_predict_without_loading(self, sample_features_df):
        """Test prediction without loading model first."""
        backend = SklearnBackend('path/to/model.joblib')
        
        with pytest.raises(ValueError, match="Model not loaded"):
            backend.predict(sample_features_df)
    
    def test_predict_empty_dataframe(self, mock_sklearn_model):
        """Test prediction with empty DataFrame."""
        backend = SklearnBackend('path/to/model.joblib')
        backend.model = mock_sklearn_model
        
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Features DataFrame is empty"):
            backend.predict(empty_df)
    
    def test_predict_success(self, mock_sklearn_model, sample_features_df):
        """Test successful prediction without preprocessor."""
        backend = SklearnBackend('path/to/model.joblib')
        backend.model = mock_sklearn_model
        
        predictions, probabilities = backend.predict(sample_features_df)
        
        assert len(predictions) == 1
        assert len(probabilities) == 1
        assert predictions[0] == "No"  # prob = 0.3 < 0.5
        assert probabilities[0] == 0.3
        mock_sklearn_model.predict_proba.assert_called_once()
    
    def test_predict_with_preprocessor(self, mock_sklearn_model, mock_preprocessor, sample_features_df):
        """Test prediction with preprocessor."""
        backend = SklearnBackend('path/to/model.joblib')
        backend.model = mock_sklearn_model
        backend.preprocessor = mock_preprocessor
        
        predictions, probabilities = backend.predict(sample_features_df)
        
        assert len(predictions) == 1
        mock_preprocessor.transform.assert_called_once()
        mock_sklearn_model.predict_proba.assert_called_once()
    
    def test_predict_high_probability(self, sample_features_df):
        """Test prediction with high churn probability."""
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        
        backend = SklearnBackend('path/to/model.joblib')
        backend.model = mock_model
        
        predictions, probabilities = backend.predict(sample_features_df)
        
        assert predictions[0] == "Yes"  # prob = 0.7 >= 0.5
        assert probabilities[0] == 0.7
    
    def test_predict_batch(self):
        """Test batch prediction with multiple records."""
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([
            [0.8, 0.2],  # No
            [0.4, 0.6],  # Yes
            [0.9, 0.1]   # No
        ]))
        
        backend = SklearnBackend('path/to/model.joblib')
        backend.model = mock_model
        
        # Create batch of 3 records
        batch_df = pd.DataFrame([
            {'feature1': 1, 'feature2': 2},
            {'feature1': 3, 'feature2': 4},
            {'feature1': 5, 'feature2': 6}
        ])
        
        predictions, probabilities = backend.predict(batch_df)
        
        assert len(predictions) == 3
        assert len(probabilities) == 3
        assert predictions == ["No", "Yes", "No"]
        assert probabilities == [0.2, 0.6, 0.1]


# ============================================================================
# SparkBackend Tests
# ============================================================================

class TestSparkBackend:
    """Test cases for SparkBackend class."""
    
    def test_init(self):
        """Test SparkBackend initialization."""
        backend = SparkBackend('path/to/spark_model')
        assert backend.backend_type == 'spark'
        assert backend.model_path == 'path/to/spark_model'
        assert backend.fallback_path is None
        assert backend.spark_session is None
        assert backend.using_fallback is False
    
    def test_init_with_fallback(self):
        """Test SparkBackend initialization with fallback path."""
        backend = SparkBackend('path/to/spark_model', fallback_path='path/to/sklearn.joblib')
        assert backend.fallback_path == 'path/to/sklearn.joblib'
    
    @patch('src.streaming.inference_backend.SPARK_AVAILABLE', False)
    @patch('src.streaming.inference_backend.SklearnBackend')
    def test_load_spark_unavailable_with_fallback(self, mock_sklearn_backend_class, mock_sklearn_model):
        """Test loading when Spark is not available but fallback exists."""
        mock_backend_instance = Mock()
        mock_backend_instance.load.return_value = mock_sklearn_model
        mock_sklearn_backend_class.return_value = mock_backend_instance
        
        backend = SparkBackend('path/to/spark_model', fallback_path='path/to/sklearn.joblib')
        model = backend.load()
        
        assert backend.using_fallback is True
        assert backend.fallback_backend is not None
        mock_sklearn_backend_class.assert_called_once_with('path/to/sklearn.joblib')
    
    @patch('src.streaming.inference_backend.SPARK_AVAILABLE', False)
    def test_load_spark_unavailable_no_fallback(self):
        """Test loading when Spark is not available and no fallback."""
        backend = SparkBackend('path/to/spark_model')
        
        with pytest.raises(ValueError, match="no fallback path specified"):
            backend.load()
    
    @patch('src.streaming.inference_backend.Path')
    @patch('src.streaming.inference_backend.SPARK_AVAILABLE', True)
    @patch('src.streaming.inference_backend.SklearnBackend')
    def test_load_spark_model_not_found_with_fallback(self, mock_sklearn_backend_class, mock_path, mock_sklearn_model):
        """Test loading when Spark model doesn't exist but fallback does."""
        mock_path.return_value.exists.return_value = False
        mock_backend_instance = Mock()
        mock_backend_instance.load.return_value = mock_sklearn_model
        mock_sklearn_backend_class.return_value = mock_backend_instance
        
        backend = SparkBackend('path/to/nonexistent', fallback_path='path/to/sklearn.joblib')
        model = backend.load()
        
        assert backend.using_fallback is True
    
    def test_predict_using_fallback(self, sample_features_df):
        """Test prediction when using fallback sklearn model."""
        mock_fallback = Mock()
        mock_fallback.predict.return_value = (["No"], [0.3])
        
        backend = SparkBackend('path/to/spark_model')
        backend.model = Mock()  # Set to avoid "not loaded" error
        backend.using_fallback = True
        backend.fallback_backend = mock_fallback
        
        predictions, probabilities = backend.predict(sample_features_df)
        
        assert predictions == ["No"]
        assert probabilities == [0.3]
        mock_fallback.predict.assert_called_once_with(sample_features_df)


# ============================================================================
# load_model Function Tests
# ============================================================================

class TestLoadModel:
    """Test cases for load_model function."""
    
    def test_invalid_backend(self):
        """Test loading with invalid backend type."""
        with pytest.raises(ValueError, match="Invalid backend type"):
            load_model('invalid_backend', 'path/to/model')
    
    @patch('src.streaming.inference_backend.SklearnBackend')
    def test_load_sklearn_model(self, mock_sklearn_backend_class):
        """Test loading sklearn model."""
        mock_backend = Mock()
        mock_sklearn_backend_class.return_value = mock_backend
        
        result = load_model('sklearn', 'path/to/model.joblib')
        
        mock_sklearn_backend_class.assert_called_once_with('path/to/model.joblib', None)
        mock_backend.load.assert_called_once()
        assert result == mock_backend
    
    @patch('src.streaming.inference_backend.SklearnBackend')
    def test_load_sklearn_with_preprocessor(self, mock_sklearn_backend_class):
        """Test loading sklearn model with preprocessor."""
        mock_backend = Mock()
        mock_sklearn_backend_class.return_value = mock_backend
        
        result = load_model(
            'sklearn',
            'path/to/model.joblib',
            preprocessor_path='path/to/preprocessor.joblib'
        )
        
        mock_sklearn_backend_class.assert_called_once_with(
            'path/to/model.joblib',
            'path/to/preprocessor.joblib'
        )
        mock_backend.load.assert_called_once()
    
    @patch('src.streaming.inference_backend.SparkBackend')
    def test_load_spark_model(self, mock_spark_backend_class):
        """Test loading spark model."""
        mock_backend = Mock()
        mock_spark_backend_class.return_value = mock_backend
        
        result = load_model('spark', 'path/to/spark_model')
        
        mock_spark_backend_class.assert_called_once_with('path/to/spark_model', None, None)
        mock_backend.load.assert_called_once()
        assert result == mock_backend
    
    @patch('src.streaming.inference_backend.SparkBackend')
    def test_load_spark_with_fallback(self, mock_spark_backend_class):
        """Test loading spark model with fallback."""
        mock_backend = Mock()
        mock_spark_backend_class.return_value = mock_backend
        
        result = load_model(
            'spark',
            'path/to/spark_model',
            fallback_path='path/to/sklearn.joblib'
        )
        
        mock_spark_backend_class.assert_called_once_with(
            'path/to/spark_model',
            'path/to/sklearn.joblib',
            None
        )
        mock_backend.load.assert_called_once()


# ============================================================================
# predict Function Tests
# ============================================================================

class TestPredict:
    """Test cases for predict function."""
    
    def test_predict_with_dict(self, sample_features_dict):
        """Test prediction with dictionary input."""
        mock_backend = Mock()
        mock_backend.backend_type = 'sklearn'
        mock_backend.predict.return_value = (["No"], [0.3])
        
        predictions, probabilities = predict(mock_backend, sample_features_dict)
        
        assert predictions == ["No"]
        assert probabilities == [0.3]
        # Check that dict was converted to DataFrame
        call_args = mock_backend.predict.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)
        assert len(call_args) == 1
    
    def test_predict_with_list_of_dicts(self, sample_features_dict):
        """Test prediction with list of dictionaries."""
        mock_backend = Mock()
        mock_backend.backend_type = 'sklearn'
        mock_backend.predict.return_value = (["No", "Yes"], [0.3, 0.7])
        
        features_list = [sample_features_dict, sample_features_dict]
        predictions, probabilities = predict(mock_backend, features_list)
        
        assert len(predictions) == 2
        # Check that list was converted to DataFrame
        call_args = mock_backend.predict.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)
        assert len(call_args) == 2
    
    def test_predict_with_dataframe(self, sample_features_df):
        """Test prediction with DataFrame input."""
        mock_backend = Mock()
        mock_backend.backend_type = 'sklearn'
        mock_backend.predict.return_value = (["No"], [0.3])
        
        predictions, probabilities = predict(mock_backend, sample_features_df)
        
        assert predictions == ["No"]
        # DataFrame should be passed as-is
        call_args = mock_backend.predict.call_args[0][0]
        assert isinstance(call_args, pd.DataFrame)
    
    def test_predict_invalid_input_type(self):
        """Test prediction with invalid input type."""
        mock_backend = Mock()
        mock_backend.backend_type = 'sklearn'
        
        with pytest.raises(ValueError, match="Invalid features type"):
            predict(mock_backend, "invalid_input")
    
    def test_predict_with_explicit_backend(self, sample_features_df):
        """Test prediction with explicit backend parameter."""
        mock_backend = Mock()
        mock_backend.backend_type = 'sklearn'
        mock_backend.predict.return_value = (["Yes"], [0.6])
        
        predictions, probabilities = predict(mock_backend, sample_features_df, backend='sklearn')
        
        assert predictions == ["Yes"]


# ============================================================================
# get_backend_info Function Tests
# ============================================================================

class TestGetBackendInfo:
    """Test cases for get_backend_info function."""
    
    def test_sklearn_backend_info(self, mock_sklearn_pipeline):
        """Test getting info for sklearn backend."""
        backend = SklearnBackend('path/to/model.joblib', 'path/to/preprocessor.joblib')
        backend.model = mock_sklearn_pipeline
        backend.preprocessor = Mock()
        
        info = get_backend_info(backend)
        
        assert info['backend_type'] == 'sklearn'
        assert info['model_path'] == 'path/to/model.joblib'
        assert info['model_type'] == 'Pipeline'
        assert info['has_preprocessor'] is True
        assert info['preprocessor_path'] == 'path/to/preprocessor.joblib'
    
    def test_sklearn_backend_info_no_preprocessor(self, mock_sklearn_pipeline):
        """Test getting info for sklearn backend without preprocessor."""
        backend = SklearnBackend('path/to/model.joblib')
        backend.model = mock_sklearn_pipeline
        
        info = get_backend_info(backend)
        
        assert info['has_preprocessor'] is False
        assert info['preprocessor_path'] is None
    
    def test_spark_backend_info(self):
        """Test getting info for spark backend."""
        backend = SparkBackend('path/to/spark_model', fallback_path='path/to/sklearn.joblib')
        backend.model = Mock()
        backend.using_fallback = True
        
        info = get_backend_info(backend)
        
        assert info['backend_type'] == 'spark'
        assert info['model_path'] == 'path/to/spark_model'
        assert info['using_fallback'] is True
        assert info['fallback_path'] == 'path/to/sklearn.joblib'
        assert 'spark_available' in info
    
    def test_backend_info_no_model_loaded(self):
        """Test getting info when model not loaded yet."""
        backend = SklearnBackend('path/to/model.joblib')
        
        info = get_backend_info(backend)
        
        assert info['model_type'] is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests using real model files (if available)."""
    
    @pytest.mark.skipif(
        not Path('artifacts/models/sklearn_pipeline.joblib').exists(),
        reason="Sklearn model file not found"
    )
    def test_real_sklearn_model_loading(self):
        """Test loading actual sklearn model file."""
        backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')
        
        assert backend is not None
        assert backend.model is not None
        
        info = get_backend_info(backend)
        assert info['backend_type'] == 'sklearn'
    
    @pytest.mark.skipif(
        not Path('artifacts/models/sklearn_pipeline.joblib').exists(),
        reason="Sklearn model file not found"
    )
    def test_real_sklearn_model_prediction(self, sample_features_dict):
        """Test prediction with actual sklearn model."""
        backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')
        
        predictions, probabilities = predict(backend, sample_features_dict)
        
        assert len(predictions) == 1
        assert len(probabilities) == 1
        assert predictions[0] in ["Yes", "No"]
        assert 0.0 <= probabilities[0] <= 1.0
