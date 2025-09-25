"""
Configuration module for Telco Churn Prediction project.

This module provides configuration management for the ML pipeline,
loading settings from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()


@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_telco_dataset: str = "data/raw/Telco-Customer-Churn.csv"
    processed_sample_data: str = "data/processed/sample.csv"
    processed_columns_config: str = "data/processed/columns.json"
    
    def get_raw_data_path(self) -> Path:
        return PROJECT_ROOT / self.raw_telco_dataset
    
    def get_processed_sample_path(self) -> Path:
        return PROJECT_ROOT / self.processed_sample_data
    
    def get_columns_config_path(self) -> Path:
        return PROJECT_ROOT / self.processed_columns_config


@dataclass
class ArtifactsConfig:
    """Artifacts and model output configuration."""
    base_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    metrics_dir: str = "artifacts/metrics"
    predictions_dir: str = "artifacts/predictions"
    logs_dir: str = "artifacts/logs"
    
    # Model files
    sklearn_pipeline: str = "artifacts/models/sklearn_pipeline_mlflow.joblib"
    preprocessor: str = "artifacts/models/preprocessor.joblib"
    feature_names: str = "artifacts/models/feature_names.json"
    
    # Metrics files
    sklearn_metrics: str = "artifacts/metrics/sklearn_metrics_mlflow.json"
    
    # Prediction files
    batch_predictions: str = "artifacts/predictions/batch_preds.csv"
    
    def get_model_path(self) -> Path:
        return PROJECT_ROOT / self.sklearn_pipeline
    
    def get_preprocessor_path(self) -> Path:
        return PROJECT_ROOT / self.preprocessor
    
    def get_feature_names_path(self) -> Path:
        return PROJECT_ROOT / self.feature_names
    
    def get_metrics_path(self) -> Path:
        return PROJECT_ROOT / self.sklearn_metrics
    
    def get_predictions_path(self) -> Path:
        return PROJECT_ROOT / self.batch_predictions
    
    def ensure_directories(self):
        """Create all artifact directories if they don't exist."""
        dirs = [self.base_dir, self.models_dir, self.metrics_dir, 
                self.predictions_dir, self.logs_dir]
        for dir_path in dirs:
            (PROJECT_ROOT / dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class MLFlowConfig:
    """MLflow tracking configuration."""
    experiment_name: str = "telco_churn_prediction"
    tracking_uri: str = "file:./mlruns"
    run_name_prefix: str = "sklearn_pipeline"
    
    def get_tracking_uri(self) -> str:
        return str(PROJECT_ROOT / self.tracking_uri.replace("file:./", ""))


@dataclass
class ModelConfig:
    """Model training configuration."""
    algorithm: str = "gradient_boosting"
    target_column: str = "Churn"
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    
    # Model parameters
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42
    })


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    numerical_features: list = field(default_factory=lambda: [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ])
    
    categorical_features: list = field(default_factory=lambda: [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ])
    
    # Preprocessing settings
    numerical_imputation: str = "median"
    categorical_imputation: str = "constant"
    categorical_imputation_value: str = "Unknown"
    scaling: str = "standard"
    encoding: str = "onehot"


@dataclass
class EvaluationConfig:
    """Model evaluation configuration."""
    primary_metric: str = "roc_auc"
    metrics: list = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "roc_auc"
    ])
    threshold: float = 0.5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True


@dataclass
class AirflowConfig:
    """Airflow DAG configuration."""
    dag_id: str = "telco_churn_dag"
    schedule_interval: str = "@daily"
    start_date: str = "2024-01-01"
    email_on_failure: bool = False
    email_on_retry: bool = False
    retries: int = 1


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    batch_size: int = 1000
    prediction_threshold: float = 0.5
    model_monitoring: bool = True
    data_drift_detection: bool = False
    performance_monitoring: bool = True


@dataclass
class Config:
    """Main configuration class that holds all configuration sections."""
    project_name: str = "telco-churn-prediction"
    project_version: str = "0.1.0"
    project_author: str = "Dean Hettiarachchi"
    
    # Configuration sections
    data: DataConfig = field(default_factory=DataConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    mlflow: MLFlowConfig = field(default_factory=MLFlowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    airflow: AirflowConfig = field(default_factory=AirflowConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    
    def __post_init__(self):
        """Ensure artifact directories exist after initialization."""
        self.artifacts.ensure_directories()


class ConfigManager:
    """Configuration manager for loading and managing configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(PROJECT_ROOT / "config.yaml")
        self._config = None
    
    def load_config(self) -> Config:
        """Load configuration from YAML file and environment variables."""
        if self._config is None:
            self._config = self._load_from_yaml()
            self._override_from_env()
        return self._config
    
    def _load_from_yaml(self) -> Config:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return Config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            return self._dict_to_config(yaml_config)
        
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            print("Using default configuration.")
            return Config()
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary from YAML to Config object."""
        config = Config()
        
        # Update project info
        project_info = config_dict.get('project', {})
        config.project_name = project_info.get('name', config.project_name)
        config.project_version = project_info.get('version', config.project_version)
        config.project_author = project_info.get('author', config.project_author)
        
        # Update data config
        data_config = config_dict.get('data', {})
        if data_config:
            raw_data = data_config.get('raw', {})
            processed_data = data_config.get('processed', {})
            
            config.data.raw_telco_dataset = raw_data.get('telco_dataset', config.data.raw_telco_dataset)
            config.data.processed_sample_data = processed_data.get('sample_data', config.data.processed_sample_data)
            config.data.processed_columns_config = processed_data.get('columns_config', config.data.processed_columns_config)
        
        # Update artifacts config
        artifacts_config = config_dict.get('artifacts', {})
        if artifacts_config:
            models_config = artifacts_config.get('models', {})
            metrics_config = artifacts_config.get('metrics', {})
            predictions_config = artifacts_config.get('predictions', {})
            
            config.artifacts.base_dir = artifacts_config.get('base_dir', config.artifacts.base_dir)
            config.artifacts.sklearn_pipeline = models_config.get('sklearn_pipeline', config.artifacts.sklearn_pipeline)
            config.artifacts.preprocessor = models_config.get('preprocessor', config.artifacts.preprocessor)
            config.artifacts.feature_names = models_config.get('feature_names', config.artifacts.feature_names)
            config.artifacts.sklearn_metrics = metrics_config.get('sklearn_metrics', config.artifacts.sklearn_metrics)
            config.artifacts.batch_predictions = predictions_config.get('batch_predictions', config.artifacts.batch_predictions)
        
        # Update MLflow config
        mlflow_config = config_dict.get('mlflow', {})
        if mlflow_config:
            config.mlflow.experiment_name = mlflow_config.get('experiment_name', config.mlflow.experiment_name)
            config.mlflow.tracking_uri = mlflow_config.get('tracking_uri', config.mlflow.tracking_uri)
            config.mlflow.run_name_prefix = mlflow_config.get('run_name_prefix', config.mlflow.run_name_prefix)
        
        # Update model config
        model_config = config_dict.get('model', {})
        if model_config:
            config.model.algorithm = model_config.get('algorithm', config.model.algorithm)
            config.model.target_column = model_config.get('target_column', config.model.target_column)
            config.model.test_size = model_config.get('test_size', config.model.test_size)
            config.model.cv_folds = model_config.get('cv_folds', config.model.cv_folds)
            config.model.random_state = model_config.get('random_state', config.model.random_state)
            config.model.parameters = model_config.get('parameters', config.model.parameters)
        
        # Update features config
        features_config = config_dict.get('features', {})
        if features_config:
            config.features.numerical_features = features_config.get('numerical_features', config.features.numerical_features)
            config.features.categorical_features = features_config.get('categorical_features', config.features.categorical_features)
            
            preprocessing = features_config.get('preprocessing', {})
            if preprocessing:
                config.features.numerical_imputation = preprocessing.get('numerical_imputation', config.features.numerical_imputation)
                config.features.categorical_imputation = preprocessing.get('categorical_imputation', config.features.categorical_imputation)
                config.features.categorical_imputation_value = preprocessing.get('categorical_imputation_value', config.features.categorical_imputation_value)
                config.features.scaling = preprocessing.get('scaling', config.features.scaling)
                config.features.encoding = preprocessing.get('encoding', config.features.encoding)
        
        return config
    
    def _override_from_env(self):
        """Override configuration with environment variables."""
        # Allow environment variables to override key settings
        env_overrides = {
            'MLFLOW_TRACKING_URI': ('mlflow', 'tracking_uri'),
            'MLFLOW_EXPERIMENT_NAME': ('mlflow', 'experiment_name'),
            'MODEL_ALGORITHM': ('model', 'algorithm'),
            'RANDOM_STATE': ('model', 'random_state'),
            'LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, (section, key) in env_overrides.items():
            env_value = os.getenv(env_var)
            if env_value:
                config_section = getattr(self._config, section)
                if hasattr(config_section, key):
                    # Convert to appropriate type
                    current_value = getattr(config_section, key)
                    if isinstance(current_value, int):
                        env_value = int(env_value)
                    elif isinstance(current_value, float):
                        env_value = float(env_value)
                    elif isinstance(current_value, bool):
                        env_value = env_value.lower() in ('true', '1', 'yes')
                    
                    setattr(config_section, key, env_value)
    
    def get_config(self) -> Config:
        """Get the current configuration."""
        return self.load_config()
    
    def save_config(self, config: Config, output_path: Optional[str] = None):
        """Save configuration to YAML file."""
        output_path = output_path or self.config_path
        
        # Convert config to dictionary
        config_dict = {
            'project': {
                'name': config.project_name,
                'version': config.project_version,
                'author': config.project_author
            },
            'data': {
                'raw': {'telco_dataset': config.data.raw_telco_dataset},
                'processed': {
                    'sample_data': config.data.processed_sample_data,
                    'columns_config': config.data.processed_columns_config
                }
            },
            'artifacts': {
                'base_dir': config.artifacts.base_dir,
                'models': {
                    'sklearn_pipeline': config.artifacts.sklearn_pipeline,
                    'preprocessor': config.artifacts.preprocessor,
                    'feature_names': config.artifacts.feature_names
                },
                'metrics': {'sklearn_metrics': config.artifacts.sklearn_metrics},
                'predictions': {'batch_predictions': config.artifacts.batch_predictions}
            },
            'mlflow': {
                'experiment_name': config.mlflow.experiment_name,
                'tracking_uri': config.mlflow.tracking_uri,
                'run_name_prefix': config.mlflow.run_name_prefix
            },
            'model': {
                'algorithm': config.model.algorithm,
                'target_column': config.model.target_column,
                'test_size': config.model.test_size,
                'cv_folds': config.model.cv_folds,
                'random_state': config.model.random_state,
                'parameters': config.model.parameters
            },
            'features': {
                'numerical_features': config.features.numerical_features,
                'categorical_features': config.features.categorical_features,
                'preprocessing': {
                    'numerical_imputation': config.features.numerical_imputation,
                    'categorical_imputation': config.features.categorical_imputation,
                    'categorical_imputation_value': config.features.categorical_imputation_value,
                    'scaling': config.features.scaling,
                    'encoding': config.features.encoding
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to {output_path}")


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    return _config_manager.get_config()


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager.get_config()


# Convenience functions for common configuration access
def get_data_paths() -> DataConfig:
    """Get data configuration."""
    return get_config().data


def get_model_paths() -> ArtifactsConfig:
    """Get model artifacts configuration."""
    return get_config().artifacts


def get_mlflow_config() -> MLFlowConfig:
    """Get MLflow configuration."""
    return get_config().mlflow


def get_feature_config() -> FeatureConfig:
    """Get feature engineering configuration."""
    return get_config().features


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Project: {config.project_name} v{config.project_version}")
    print(f"Model path: {config.artifacts.get_model_path()}")
    print(f"Data path: {config.data.get_raw_data_path()}")
    print(f"MLflow experiment: {config.mlflow.experiment_name}")