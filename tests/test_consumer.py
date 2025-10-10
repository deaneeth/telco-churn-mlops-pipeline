"""
Unit tests for Kafka consumer.

Tests cover:
- Message validation (valid and invalid)
- Feature transformation
- Model inference (mocked)
- Prediction message composition
- Dead letter message composition  
- Full message processing pipeline
- Error handling and routing

Author: Telco Churn Prediction Team
Date: 2025-06-11
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.streaming.consumer import (
    compose_deadletter_message,
    compose_prediction_message,
    load_preprocessor,
    load_sklearn_model,
    process_message,
    run_inference,
    transform_to_features,
    validate_message,
)


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_consumer")


@pytest.fixture
def valid_customer_message():
    """Create a valid customer message."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "event_ts": "2025-10-03T04:00:00Z"
    }


@pytest.fixture
def invalid_customer_message():
    """Create an invalid customer message (missing required fields)."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        # Missing many required fields
        "event_ts": "2025-10-03T04:00:00Z"
    }


@pytest.fixture
def mock_model():
    """Create a mock sklearn model."""
    model = MagicMock()
    # Mock predict_proba to return [[0.3, 0.7]] (70% churn probability)
    model.predict_proba.return_value = [[0.3, 0.7]]
    return model


@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = MagicMock()
    return preprocessor


@pytest.fixture
def mock_schema_validator():
    """Create a mock schema validator."""
    validator = MagicMock()
    validator.validate.return_value = (True, [])  # Valid by default
    return validator


class TestValidateMessage:
    """Tests for validate_message function."""
    
    def test_validation_disabled(self, valid_customer_message, logger):
        """Test validation when validator is None (disabled)."""
        is_valid, errors = validate_message(valid_customer_message, None, logger)
        assert is_valid is True
        assert errors == []
    
    def test_valid_message(self, valid_customer_message, mock_schema_validator, logger):
        """Test validation with valid message."""
        is_valid, errors = validate_message(valid_customer_message, mock_schema_validator, logger)
        assert is_valid is True
        assert errors == []
        mock_schema_validator.validate.assert_called_once_with(valid_customer_message)
    
    def test_invalid_message(self, invalid_customer_message, mock_schema_validator, logger):
        """Test validation with invalid message."""
        mock_schema_validator.validate.return_value = (False, ["Field 'tenure' is required"])
        is_valid, errors = validate_message(invalid_customer_message, mock_schema_validator, logger)
        assert is_valid is False
        assert len(errors) == 1
        assert "Field 'tenure' is required" in errors[0]
    
    def test_validation_exception(self, valid_customer_message, mock_schema_validator, logger):
        """Test validation when validator raises exception."""
        mock_schema_validator.validate.side_effect = Exception("Validation error")
        is_valid, errors = validate_message(valid_customer_message, mock_schema_validator, logger)
        assert is_valid is False
        assert len(errors) == 1
        assert "Validation exception" in errors[0]


class TestTransformToFeatures:
    """Tests for transform_to_features function."""
    
    def test_valid_transformation(self, valid_customer_message, logger):
        """Test transformation of valid message to features."""
        df = transform_to_features(valid_customer_message, logger)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Single row
        assert "customerID" not in df.columns  # ID excluded
        assert "event_ts" not in df.columns  # Timestamp excluded
        assert "gender" in df.columns
        assert "tenure" in df.columns
        assert df["gender"].iloc[0] == "Female"
        assert df["tenure"].iloc[0] == 1
    
    def test_missing_features(self, invalid_customer_message, logger):
        """Test transformation with missing required features."""
        with pytest.raises(ValueError, match="Missing required features"):
            transform_to_features(invalid_customer_message, logger)
    
    def test_extra_fields_ignored(self, valid_customer_message, logger):
        """Test that extra fields are ignored during transformation."""
        message_with_extra = {**valid_customer_message, "extra_field": "value"}
        df = transform_to_features(message_with_extra, logger)
        
        assert "extra_field" not in df.columns
        assert len(df.columns) == 19  # 19 features expected


class TestRunInference:
    """Tests for run_inference function."""
    
    def test_inference_without_preprocessor(self, mock_model, logger):
        """Test inference without separate preprocessor."""
        features = pd.DataFrame([[1, 2, 3]])
        prediction, probability = run_inference(features, mock_model, None, logger)
        
        assert prediction == "Yes"  # 0.7 > 0.5 threshold
        assert probability == 0.7
        mock_model.predict_proba.assert_called_once()
    
    def test_inference_with_preprocessor(self, mock_model, mock_preprocessor, logger):
        """Test inference with separate preprocessor."""
        features = pd.DataFrame([[1, 2, 3]])
        prediction, probability = run_inference(features, mock_model, mock_preprocessor, logger)
        
        assert prediction == "Yes"
        assert probability == 0.7
        mock_preprocessor.transform.assert_called_once_with(features)
        mock_model.predict_proba.assert_called_once()
    
    def test_no_churn_prediction(self, mock_model, logger):
        """Test prediction when churn probability is low."""
        mock_model.predict_proba.return_value = [[0.8, 0.2]]  # 20% churn probability
        features = pd.DataFrame([[1, 2, 3]])
        prediction, probability = run_inference(features, mock_model, None, logger)
        
        assert prediction == "No"  # 0.2 < 0.5 threshold
        assert probability == 0.2
    
    def test_inference_exception(self, mock_model, logger):
        """Test inference when model raises exception."""
        mock_model.predict_proba.side_effect = Exception("Model error")
        features = pd.DataFrame([[1, 2, 3]])
        
        with pytest.raises(Exception, match="Model error"):
            run_inference(features, mock_model, None, logger)


class TestComposePredictionMessage:
    """Tests for compose_prediction_message function."""
    
    def test_basic_prediction_message(self):
        """Test composing basic prediction message."""
        message = compose_prediction_message(
            customer_id="7590-VHVEG",
            prediction="Yes",
            probability=0.75,
            event_ts="2025-10-03T04:00:00Z"
        )
        
        assert message["customerID"] == "7590-VHVEG"
        assert message["prediction"] == "Yes"
        assert message["churn_probability"] == 0.75
        assert message["event_ts"] == "2025-10-03T04:00:00Z"
        assert "processed_ts" in message
        assert "model_version" not in message  # Optional field not set
        assert "inference_latency_ms" not in message
    
    def test_prediction_message_with_optional_fields(self):
        """Test composing prediction message with optional fields."""
        message = compose_prediction_message(
            customer_id="7590-VHVEG",
            prediction="No",
            probability=0.25,
            event_ts="2025-10-03T04:00:00Z",
            inference_latency_ms=15.5,
            model_version="sklearn-1.0.0"
        )
        
        assert message["model_version"] == "sklearn-1.0.0"
        assert message["inference_latency_ms"] == 15.5
    
    def test_probability_rounding(self):
        """Test that probability is rounded to 6 decimals."""
        message = compose_prediction_message(
            customer_id="7590-VHVEG",
            prediction="Yes",
            probability=0.123456789,
            event_ts="2025-10-03T04:00:00Z"
        )
        
        assert message["churn_probability"] == 0.123457


class TestComposeDeadletterMessage:
    """Tests for compose_deadletter_message function."""
    
    def test_basic_deadletter_message(self, valid_customer_message):
        """Test composing basic deadletter message."""
        message = compose_deadletter_message(
            original_message=valid_customer_message,
            error_type="validation_error",
            error_message="Missing required field",
            source_topic="telco.raw.customers"
        )
        
        assert message["original_message"] == valid_customer_message
        assert message["error_type"] == "validation_error"
        assert message["error_message"] == "Missing required field"
        assert message["source_topic"] == "telco.raw.customers"
        assert "failed_ts" in message
        assert "validation_errors" not in message  # Optional field not set
    
    def test_deadletter_message_with_validation_errors(self, valid_customer_message):
        """Test composing deadletter message with validation errors."""
        validation_errors = [
            "Field 'tenure' is required",
            "Field 'gender' must be 'Male' or 'Female'"
        ]
        message = compose_deadletter_message(
            original_message=valid_customer_message,
            error_type="validation_error",
            error_message="Schema validation failed",
            source_topic="telco.raw.customers",
            validation_errors=validation_errors
        )
        
        assert message["validation_errors"] == validation_errors
    
    def test_deadletter_message_with_all_optional_fields(self, valid_customer_message):
        """Test composing deadletter message with all optional fields."""
        message = compose_deadletter_message(
            original_message=valid_customer_message,
            error_type="inference_error",
            error_message="Model inference failed",
            source_topic="telco.raw.customers",
            validation_errors=["Error 1"],
            consumer_group="test-consumer-group",
            retry_count=3
        )
        
        assert message["consumer_group"] == "test-consumer-group"
        assert message["retry_count"] == 3


class TestProcessMessage:
    """Tests for process_message function (full pipeline)."""
    
    def test_successful_processing(
        self, valid_customer_message, mock_model, mock_schema_validator, logger
    ):
        """Test successful message processing end-to-end."""
        prediction_msg, deadletter_msg = process_message(
            raw_message=valid_customer_message,
            model=mock_model,
            preprocessor=None,
            input_validator=mock_schema_validator,
            output_validator=mock_schema_validator,
            source_topic="telco.raw.customers",
            consumer_group="test-group",
            logger=logger
        )
        
        assert prediction_msg is not None
        assert deadletter_msg is None
        assert prediction_msg["customerID"] == "7590-VHVEG"
        assert prediction_msg["prediction"] == "Yes"
        assert prediction_msg["churn_probability"] == 0.7
        assert "inference_latency_ms" in prediction_msg
    
    def test_validation_failure(
        self, invalid_customer_message, mock_model, mock_schema_validator, logger
    ):
        """Test processing when input validation fails."""
        mock_schema_validator.validate.return_value = (False, ["Field 'tenure' is required"])
        
        prediction_msg, deadletter_msg = process_message(
            raw_message=invalid_customer_message,
            model=mock_model,
            preprocessor=None,
            input_validator=mock_schema_validator,
            output_validator=None,
            source_topic="telco.raw.customers",
            consumer_group="test-group",
            logger=logger
        )
        
        assert prediction_msg is None
        assert deadletter_msg is not None
        assert deadletter_msg["error_type"] == "validation_error"
        assert "Field 'tenure' is required" in deadletter_msg["validation_errors"]
    
    def test_feature_transformation_failure(
        self, invalid_customer_message, mock_model, mock_schema_validator, logger
    ):
        """Test processing when feature transformation fails."""
        # Validation passes but transformation fails due to missing features
        
        prediction_msg, deadletter_msg = process_message(
            raw_message=invalid_customer_message,
            model=mock_model,
            preprocessor=None,
            input_validator=None,  # Skip validation
            output_validator=None,
            source_topic="telco.raw.customers",
            consumer_group="test-group",
            logger=logger
        )
        
        assert prediction_msg is None
        assert deadletter_msg is not None
        assert deadletter_msg["error_type"] == "processing_error"
        assert "Feature transformation failed" in deadletter_msg["error_message"]
    
    def test_inference_failure(
        self, valid_customer_message, mock_model, logger
    ):
        """Test processing when model inference fails."""
        mock_model.predict_proba.side_effect = Exception("Model error")
        
        prediction_msg, deadletter_msg = process_message(
            raw_message=valid_customer_message,
            model=mock_model,
            preprocessor=None,
            input_validator=None,
            output_validator=None,
            source_topic="telco.raw.customers",
            consumer_group="test-group",
            logger=logger
        )
        
        assert prediction_msg is None
        assert deadletter_msg is not None
        assert deadletter_msg["error_type"] == "inference_error"
        assert "Model inference failed" in deadletter_msg["error_message"]
    
    def test_output_validation_failure(
        self, valid_customer_message, mock_model, mock_schema_validator, logger
    ):
        """Test processing when output validation fails (bug in code)."""
        # Input validation passes
        input_validator = MagicMock()
        input_validator.validate.return_value = (True, [])
        
        # Output validation fails (simulating a bug in our code)
        output_validator = MagicMock()
        output_validator.validate.return_value = (False, ["Field 'prediction' is invalid"])
        
        prediction_msg, deadletter_msg = process_message(
            raw_message=valid_customer_message,
            model=mock_model,
            preprocessor=None,
            input_validator=input_validator,
            output_validator=output_validator,
            source_topic="telco.raw.customers",
            consumer_group="test-group",
            logger=logger
        )
        
        assert prediction_msg is None
        assert deadletter_msg is not None
        assert deadletter_msg["error_type"] == "processing_error"
        assert "internal error" in deadletter_msg["error_message"]
    
    def test_unexpected_error(
        self, valid_customer_message, mock_model, logger
    ):
        """Test processing when unexpected error occurs."""
        # Make model raise unexpected error
        mock_model.predict_proba.side_effect = RuntimeError("Unexpected error")
        
        prediction_msg, deadletter_msg = process_message(
            raw_message=valid_customer_message,
            model=mock_model,
            preprocessor=None,
            input_validator=None,
            output_validator=None,
            source_topic="telco.raw.customers",
            consumer_group="test-group",
            logger=logger
        )
        
        assert prediction_msg is None
        assert deadletter_msg is not None
        # Should be caught as inference_error or unknown_error


class TestLoadSklearnModel:
    """Tests for load_sklearn_model function."""
    
    def test_load_existing_model(self, logger):
        """Test loading existing sklearn model."""
        model_path = "artifacts/models/sklearn_pipeline.joblib"
        if not Path(model_path).exists():
            pytest.skip("Model file not found (expected in CI environment)")
        
        model = load_sklearn_model(model_path, logger)
        assert model is not None
        assert hasattr(model, "predict_proba")
    
    def test_load_nonexistent_model(self, logger):
        """Test loading non-existent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_sklearn_model("nonexistent/model.joblib", logger)


class TestLoadPreprocessor:
    """Tests for load_preprocessor function."""
    
    def test_no_preprocessor(self, logger):
        """Test when preprocessor path is None."""
        preprocessor = load_preprocessor(None, logger)
        assert preprocessor is None
    
    def test_load_nonexistent_preprocessor(self, logger):
        """Test loading non-existent preprocessor raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_preprocessor("nonexistent/preprocessor.joblib", logger)


class TestIntegration:
    """Integration tests using real model (if available)."""
    
    def test_end_to_end_with_real_model(self, valid_customer_message, logger):
        """Test end-to-end processing with real model."""
        model_path = "artifacts/models/sklearn_pipeline.joblib"
        if not Path(model_path).exists():
            pytest.skip("Model file not found (expected in CI environment)")
        
        model = load_sklearn_model(model_path, logger)
        
        prediction_msg, deadletter_msg = process_message(
            raw_message=valid_customer_message,
            model=model,
            preprocessor=None,
            input_validator=None,
            output_validator=None,
            source_topic="telco.raw.customers",
            consumer_group="test-group",
            logger=logger
        )
        
        assert prediction_msg is not None
        assert deadletter_msg is None
        assert prediction_msg["customerID"] == "7590-VHVEG"
        assert prediction_msg["prediction"] in ["Yes", "No"]
        assert 0.0 <= prediction_msg["churn_probability"] <= 1.0
