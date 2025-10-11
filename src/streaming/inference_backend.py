"""
Inference Backend for Telco Churn Prediction

This module provides a unified interface for loading models and running predictions
across different ML backends (sklearn, Spark MLlib). It abstracts the inference
logic from the Kafka consumer implementation.

Supported Backends:
    - sklearn: Scikit-learn models (joblib format)
    - spark: Apache Spark MLlib models (with fallback to sklearn if unavailable)

Usage:
    # Load a sklearn model
    model = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')
    
    # Run prediction
    predictions, probabilities = predict(model, df, backend='sklearn')
    
    # Load a spark model (with fallback)
    model = load_model('spark', 'artifacts/models/spark_model', 
                       fallback_path='artifacts/models/sklearn_pipeline.joblib')

Author: Telco Churn Prediction Team
Date: 2025-10-11
Version: 1.0.0
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd
import numpy as np

# Spark imports (optional, will fallback to sklearn if not available)
SPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
    SPARK_AVAILABLE = True
except ImportError:
    SparkSession = None
    PipelineModel = None
    LogisticRegressionModel = None
    RandomForestClassificationModel = None


# Module-level logger
logger = logging.getLogger(__name__)


class ModelBackend:
    """
    Base class for ML model backends.
    
    This provides a common interface for different model types,
    enabling easy swapping between sklearn and Spark models.
    """
    
    def __init__(self, backend_type: str, model_path: str):
        """
        Initialize the model backend.
        
        Args:
            backend_type: Type of backend ('sklearn' or 'spark')
            model_path: Path to the saved model
        """
        self.backend_type = backend_type
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
    
    def load(self) -> Any:
        """Load the model from disk. To be implemented by subclasses."""
        raise NotImplementedError
    
    def predict(self, features: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """Run prediction on features. To be implemented by subclasses."""
        raise NotImplementedError


class SklearnBackend(ModelBackend):
    """
    Scikit-learn model backend.
    
    Supports sklearn pipelines saved in joblib format. The pipeline
    may include preprocessing steps or use a separate preprocessor.
    """
    
    def __init__(self, model_path: str, preprocessor_path: Optional[str] = None):
        """
        Initialize sklearn backend.
        
        Args:
            model_path: Path to saved sklearn model (joblib format)
            preprocessor_path: Optional path to separate preprocessor
        """
        super().__init__('sklearn', model_path)
        self.preprocessor_path = preprocessor_path
    
    def load(self) -> Any:
        """
        Load sklearn model from disk.
        
        Returns:
            Loaded sklearn model object
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading sklearn model from {self.model_path}...")
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✓ Successfully loaded sklearn model: {type(self.model).__name__}")
        except Exception as e:
            logger.error(f"✗ Failed to load sklearn model: {e}")
            raise
        
        # Load preprocessor if specified
        if self.preprocessor_path:
            preprocessor_file = Path(self.preprocessor_path)
            if not preprocessor_file.exists():
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
            
            logger.info(f"Loading preprocessor from {self.preprocessor_path}...")
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info(f"✓ Successfully loaded preprocessor: {type(self.preprocessor).__name__}")
            except Exception as e:
                logger.error(f"✗ Failed to load preprocessor: {e}")
                raise
        else:
            logger.info("No separate preprocessor specified (assuming model includes preprocessing)")
        
        return self.model
    
    def predict(self, features: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """
        Run prediction on features using sklearn model.
        
        Args:
            features: DataFrame with model input features
        
        Returns:
            Tuple of (predictions, probabilities)
            - predictions: List of "Yes" or "No" values
            - probabilities: List of churn probabilities (0.0 to 1.0)
        
        Raises:
            ValueError: If model not loaded or features invalid
            Exception: If prediction fails
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        try:
            # Apply preprocessing if separate preprocessor exists
            processed_features = features
            if self.preprocessor is not None:
                processed_features = self.preprocessor.transform(features)
            
            # Run prediction (sklearn models return class probabilities)
            # Assuming binary classification with classes [0, 1] = ["No", "Yes"]
            probabilities = self.model.predict_proba(processed_features)
            
            # Extract churn probabilities (class 1) and predictions
            churn_probs = probabilities[:, 1].tolist()
            predictions = ["Yes" if prob >= 0.5 else "No" for prob in churn_probs]
            
            logger.debug(f"Sklearn inference: {len(predictions)} predictions generated")
            
            return predictions, churn_probs
        except Exception as e:
            logger.error(f"Sklearn inference error: {e}")
            raise


class SparkBackend(ModelBackend):
    """
    Apache Spark MLlib model backend.
    
    Supports Spark ML pipelines and models. If Spark is not available
    or configured, can fallback to sklearn model.
    """
    
    def __init__(
        self,
        model_path: str,
        fallback_path: Optional[str] = None,
        spark_session: Optional['SparkSession'] = None
    ):
        """
        Initialize Spark backend.
        
        Args:
            model_path: Path to saved Spark model
            fallback_path: Optional path to sklearn fallback model
            spark_session: Optional existing Spark session (will create if None)
        """
        super().__init__('spark', model_path)
        self.fallback_path = fallback_path
        self.spark_session = spark_session
        self.fallback_backend = None
        self.using_fallback = False
    
    def load(self) -> Any:
        """
        Load Spark model from disk (or fallback to sklearn).
        
        Returns:
            Loaded Spark model or sklearn fallback model
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails and no fallback available
        """
        # Check if Spark is available
        if not SPARK_AVAILABLE:
            logger.warning("PySpark not available, using sklearn fallback")
            return self._load_fallback()
        
        # Check if model path exists
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            logger.warning(f"Spark model not found: {self.model_path}")
            return self._load_fallback()
        
        try:
            # Initialize Spark session if not provided
            if self.spark_session is None:
                logger.info("Creating Spark session for model inference...")
                self.spark_session = SparkSession.builder \
                    .appName("TelcoChurnInference") \
                    .config("spark.driver.memory", "2g") \
                    .config("spark.executor.memory", "2g") \
                    .getOrCreate()
                logger.info("✓ Spark session created")
            
            # Load Spark model
            logger.info(f"Loading Spark model from {self.model_path}...")
            self.model = PipelineModel.load(self.model_path)
            logger.info(f"✓ Successfully loaded Spark model")
            
            return self.model
        except Exception as e:
            logger.error(f"✗ Failed to load Spark model: {e}")
            logger.warning("Falling back to sklearn model...")
            return self._load_fallback()
    
    def _load_fallback(self) -> Any:
        """
        Load sklearn fallback model.
        
        Returns:
            Loaded sklearn model
        
        Raises:
            ValueError: If no fallback path specified
            Exception: If fallback model loading fails
        """
        if not self.fallback_path:
            raise ValueError(
                "Spark model unavailable and no fallback path specified. "
                "Please provide fallback_path parameter."
            )
        
        logger.info(f"Loading sklearn fallback model from {self.fallback_path}...")
        self.fallback_backend = SklearnBackend(self.fallback_path)
        self.model = self.fallback_backend.load()
        self.using_fallback = True
        logger.info("✓ Fallback model loaded successfully")
        
        return self.model
    
    def predict(self, features: pd.DataFrame) -> Tuple[List[str], List[float]]:
        """
        Run prediction on features using Spark model (or fallback).
        
        Args:
            features: DataFrame with model input features
        
        Returns:
            Tuple of (predictions, probabilities)
            - predictions: List of "Yes" or "No" values
            - probabilities: List of churn probabilities (0.0 to 1.0)
        
        Raises:
            ValueError: If model not loaded or features invalid
            Exception: If prediction fails
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # If using fallback, delegate to sklearn backend
        if self.using_fallback:
            return self.fallback_backend.predict(features)
        
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        try:
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark_session.createDataFrame(features)
            
            # Run prediction
            predictions_df = self.model.transform(spark_df)
            
            # Extract predictions and probabilities
            # Spark ML models typically output 'prediction' and 'probability' columns
            result_pandas = predictions_df.select('prediction', 'probability').toPandas()
            
            # Convert predictions to Yes/No
            predictions = ["Yes" if pred == 1.0 else "No" for pred in result_pandas['prediction']]
            
            # Extract churn probability (probability of class 1)
            churn_probs = [
                float(prob[1]) if isinstance(prob, (list, np.ndarray)) else float(prob)
                for prob in result_pandas['probability']
            ]
            
            logger.debug(f"Spark inference: {len(predictions)} predictions generated")
            
            return predictions, churn_probs
        except Exception as e:
            logger.error(f"Spark inference error: {e}")
            raise


def load_model(
    backend: str,
    model_path: str,
    preprocessor_path: Optional[str] = None,
    fallback_path: Optional[str] = None,
    spark_session: Optional['SparkSession'] = None
) -> ModelBackend:
    """
    Load a model from disk for the specified backend.
    
    This is the main entry point for loading models. It creates
    the appropriate backend instance and loads the model.
    
    Args:
        backend: Backend type ('sklearn' or 'spark')
        model_path: Path to the saved model file/directory
        preprocessor_path: Optional path to separate preprocessor (sklearn only)
        fallback_path: Optional path to sklearn fallback (spark only)
        spark_session: Optional existing Spark session (spark only)
    
    Returns:
        Loaded ModelBackend instance
    
    Raises:
        ValueError: If backend type is invalid
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    
    Examples:
        >>> # Load sklearn model
        >>> backend = load_model('sklearn', 'artifacts/models/sklearn_pipeline.joblib')
        >>> 
        >>> # Load spark model with fallback
        >>> backend = load_model(
        ...     'spark',
        ...     'artifacts/models/spark_model',
        ...     fallback_path='artifacts/models/sklearn_pipeline.joblib'
        ... )
    """
    if backend not in ['sklearn', 'spark']:
        raise ValueError(f"Invalid backend type: {backend}. Must be 'sklearn' or 'spark'")
    
    logger.info(f"Loading model with backend: {backend}")
    
    if backend == 'sklearn':
        model_backend = SklearnBackend(model_path, preprocessor_path)
        model_backend.load()
        return model_backend
    
    else:  # backend == 'spark'
        model_backend = SparkBackend(model_path, fallback_path, spark_session)
        model_backend.load()
        return model_backend


def predict(
    model_backend: ModelBackend,
    features: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    backend: Optional[str] = None
) -> Tuple[List[str], List[float]]:
    """
    Run prediction on input features.
    
    This is the main entry point for running predictions. It handles
    conversion of various input types to DataFrame and delegates to
    the backend-specific prediction logic.
    
    Args:
        model_backend: Loaded ModelBackend instance
        features: Input features (DataFrame, dict, or list of dicts)
        backend: Optional backend type (will use model_backend.backend_type if None)
    
    Returns:
        Tuple of (predictions, probabilities)
        - predictions: List of "Yes" or "No" values
        - probabilities: List of churn probabilities (0.0 to 1.0)
    
    Raises:
        ValueError: If input format is invalid
        Exception: If prediction fails
    
    Examples:
        >>> # Single record prediction
        >>> predictions, probs = predict(backend, {'tenure': 12, 'MonthlyCharges': 50.0, ...})
        >>> 
        >>> # Batch prediction
        >>> predictions, probs = predict(backend, df)
    """
    # Convert features to DataFrame if needed
    if isinstance(features, dict):
        features = pd.DataFrame([features])
    elif isinstance(features, list):
        features = pd.DataFrame(features)
    elif not isinstance(features, pd.DataFrame):
        raise ValueError(f"Invalid features type: {type(features)}. Expected DataFrame, dict, or list")
    
    # Use backend from model_backend if not specified
    if backend is None:
        backend = model_backend.backend_type
    
    logger.debug(f"Running prediction with backend: {backend}, features shape: {features.shape}")
    
    # Delegate to backend-specific prediction
    return model_backend.predict(features)


def get_backend_info(model_backend: ModelBackend) -> Dict[str, Any]:
    """
    Get information about the loaded model backend.
    
    Args:
        model_backend: Loaded ModelBackend instance
    
    Returns:
        Dictionary with backend information
    
    Examples:
        >>> info = get_backend_info(backend)
        >>> print(info)
        {
            'backend_type': 'sklearn',
            'model_path': 'artifacts/models/sklearn_pipeline.joblib',
            'model_type': 'Pipeline',
            'has_preprocessor': False,
            'using_fallback': False
        }
    """
    info = {
        'backend_type': model_backend.backend_type,
        'model_path': model_backend.model_path,
        'model_type': type(model_backend.model).__name__ if model_backend.model else None,
    }
    
    # Add sklearn-specific info
    if isinstance(model_backend, SklearnBackend):
        info['has_preprocessor'] = model_backend.preprocessor is not None
        info['preprocessor_path'] = model_backend.preprocessor_path
    
    # Add spark-specific info
    elif isinstance(model_backend, SparkBackend):
        info['using_fallback'] = model_backend.using_fallback
        info['fallback_path'] = model_backend.fallback_path
        info['spark_available'] = SPARK_AVAILABLE
    
    return info
