"""
Logging utility module for telco churn prediction project.

This module provides centralized logging configuration and utilities
for consistent logging across the entire project.
"""

import logging
import logging.config
import os
from datetime import datetime
from pathlib import Path
import json


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_dir: str = "../../artifacts/logs",
    config_file: str = None
):
    """
    Setup logging configuration for the project.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str): Name of the log file (optional)
        log_dir (str): Directory to store log files
        config_file (str): Path to logging configuration file (optional)
    """
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"telco_churn_{timestamp}.log"
    
    log_file_path = log_path / log_file
    
    # Use custom config file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        # Default logging configuration
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'simple': {
                    'format': '%(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': log_level,
                    'formatter': 'simple',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'level': log_level,
                    'formatter': 'detailed',
                    'filename': str(log_file_path),
                    'mode': 'a'
                }
            },
            'loggers': {
                '': {  # root logger
                    'level': log_level,
                    'handlers': ['console', 'file'],
                    'propagate': False
                }
            }
        }
        
        logging.config.dictConfig(logging_config)
    
    # Log the setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup completed. Log file: {log_file_path}")
    logger.info(f"Log level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): Name of the logger (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


class ModelLogger:
    """
    Specialized logger for ML model operations.
    """
    
    def __init__(self, model_name: str, log_dir: str = "../../artifacts/logs"):
        """
        Initialize the ModelLogger.
        
        Args:
            model_name (str): Name of the model
            log_dir (str): Directory to store log files
        """
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model-specific logger
        self.logger = logging.getLogger(f"model.{model_name}")
        
        # Create model-specific log file
        log_file = self.log_dir / f"{model_name}_model.log"
        
        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_training_start(self, dataset_info: dict):
        """Log the start of model training."""
        self.logger.info(f"Starting training for {self.model_name}")
        self.logger.info(f"Dataset info: {dataset_info}")
    
    def log_training_complete(self, metrics: dict, duration: float):
        """Log the completion of model training."""
        self.logger.info(f"Training completed for {self.model_name}")
        self.logger.info(f"Training duration: {duration:.2f} seconds")
        self.logger.info(f"Training metrics: {metrics}")
    
    def log_prediction(self, input_shape: tuple, prediction_count: int):
        """Log prediction operations."""
        self.logger.info(f"Prediction made by {self.model_name}")
        self.logger.info(f"Input shape: {input_shape}, Predictions: {prediction_count}")
    
    def log_model_save(self, file_path: str):
        """Log model saving."""
        self.logger.info(f"Model {self.model_name} saved to {file_path}")
    
    def log_model_load(self, file_path: str):
        """Log model loading."""
        self.logger.info(f"Model {self.model_name} loaded from {file_path}")
    
    def log_error(self, error_message: str, exception: Exception = None):
        """Log errors."""
        self.logger.error(f"Error in {self.model_name}: {error_message}")
        if exception:
            self.logger.exception(f"Exception details: {str(exception)}")


class ExperimentLogger:
    """
    Logger for tracking ML experiments.
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "../../artifacts/logs"):
        """
        Initialize the ExperimentLogger.
        
        Args:
            experiment_name (str): Name of the experiment
            log_dir (str): Directory to store log files
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific logger
        self.logger = logging.getLogger(f"experiment.{experiment_name}")
        
        # Create experiment log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"experiment_{experiment_name}_{timestamp}.log"
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.experiment_start_time = datetime.now()
        self.logger.info(f"Experiment '{experiment_name}' started")
    
    def log_hyperparameters(self, hyperparams: dict):
        """Log hyperparameters for the experiment."""
        self.logger.info(f"Hyperparameters: {hyperparams}")
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics for the experiment."""
        step_info = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metrics{step_info}: {metrics}")
    
    def log_experiment_end(self, final_metrics: dict = None):
        """Log the end of the experiment."""
        duration = datetime.now() - self.experiment_start_time
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
        self.logger.info(f"Total duration: {duration}")
        if final_metrics:
            self.logger.info(f"Final metrics: {final_metrics}")


def log_function_call(func):
    """
    Decorator to log function calls and execution time.
    
    Args:
        func: Function to be decorated
        
    Returns:
        Wrapped function with logging
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        
        logger.debug(f"Calling function: {func.__name__}")
        logger.debug(f"Args: {args}")
        logger.debug(f"Kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Function {func.__name__} completed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Function {func.__name__} failed after {execution_time:.3f} seconds")
            logger.exception(f"Exception in {func.__name__}: {str(e)}")
            raise
    
    return wrapper


def create_log_analysis_report(log_file: str) -> dict:
    """
    Analyze log file and create a summary report.
    
    Args:
        log_file (str): Path to the log file
        
    Returns:
        dict: Log analysis report
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    log_levels = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
    total_lines = 0
    errors = []
    warnings = []
    
    with open(log_file, 'r') as f:
        for line in f:
            total_lines += 1
            
            # Count log levels
            for level in log_levels.keys():
                if f" {level} " in line:
                    log_levels[level] += 1
                    
                    # Collect errors and warnings
                    if level == 'ERROR':
                        errors.append(line.strip())
                    elif level == 'WARNING':
                        warnings.append(line.strip())
                    break
    
    report = {
        'log_file': log_file,
        'total_lines': total_lines,
        'log_level_counts': log_levels,
        'error_count': len(errors),
        'warning_count': len(warnings),
        'recent_errors': errors[-5:] if errors else [],  # Last 5 errors
        'recent_warnings': warnings[-5:] if warnings else [],  # Last 5 warnings
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Logger utility module loaded successfully!")
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Get a logger and test it
    logger = get_logger(__name__)
    logger.info("Logger setup completed successfully")
    
    # Test ModelLogger
    model_logger = ModelLogger("test_model")
    model_logger.log_training_start({"train_size": 1000, "test_size": 200})
    
    # Test ExperimentLogger
    exp_logger = ExperimentLogger("test_experiment")
    exp_logger.log_hyperparameters({"learning_rate": 0.01, "batch_size": 32})
    exp_logger.log_metrics({"accuracy": 0.85, "loss": 0.3})
    
    print("All logger components tested successfully!")