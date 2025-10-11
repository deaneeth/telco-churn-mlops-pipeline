"""
Kafka Consumer for Telco Churn Prediction

This module implements a Kafka consumer that:
1. Reads customer messages from telco.raw.customers topic
2. Validates messages against the telco customer schema
3. Transforms validated messages to model features
4. Runs churn prediction inference using sklearn or spark models
5. Publishes predictions to telco.churn.predictions topic
6. Routes invalid/failed messages to telco.deadletter topic

Usage:
    # Streaming mode (continuous consumption)
    python -m src.streaming.consumer --mode streaming --validate
    
    # Batch mode (bounded processing)
    python -m src.streaming.consumer --mode batch --max-messages 1000
    
    # Dry-run mode (no actual Kafka interaction)
    python -m src.streaming.consumer --mode streaming --dry-run

Author: Telco Churn Prediction Team
Date: 2025-06-11
"""

import argparse
import json
import logging
import logging.handlers
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from threading import Thread

import joblib
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# Prometheus client for metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, Info, generate_latest, REGISTRY
    from prometheus_client.core import CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Info = None

# Dual import strategy for schema validator (supports both module and script execution)
try:
    from .schema_validator import SchemaValidator
except ImportError:
    try:
        from src.streaming.schema_validator import SchemaValidator
    except ImportError:
        SchemaValidator = None  # Will check for None if --validate flag is used

# Constants
DEFAULT_BROKER = "localhost:19092"
DEFAULT_INPUT_TOPIC = "telco.raw.customers"
DEFAULT_OUTPUT_TOPIC = "telco.churn.predictions"
DEFAULT_DEADLETTER_TOPIC = "telco.deadletter"
DEFAULT_CONSUMER_GROUP = "telco-churn-consumer"
DEFAULT_MODEL_BACKEND = "sklearn"
DEFAULT_MODEL_PATH = "artifacts/models/sklearn_pipeline.joblib"
DEFAULT_PREPROCESSOR_PATH = None  # Pipeline includes preprocessing

# Expected feature columns for model input (21 features from dataset)
EXPECTED_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Global flag for graceful shutdown
shutdown_requested = False

# Prometheus Metrics (initialized if prometheus_client available)
if PROMETHEUS_AVAILABLE:
    # Counters for message processing
    MESSAGES_PROCESSED = Counter(
        'kafka_messages_processed_total',
        'Total number of messages processed',
        ['status', 'topic']  # status: success or failed
    )
    
    MESSAGES_FAILED = Counter(
        'kafka_messages_failed_total',
        'Total number of failed messages',
        ['error_type', 'topic']  # error_type: validation_error, inference_error, etc.
    )
    
    # Histogram for processing latency
    PROCESSING_LATENCY = Histogram(
        'kafka_processing_latency_seconds',
        'Message processing latency in seconds',
        ['operation']  # operation: validation, transformation, inference, total
    )
    
    # Gauge for current state
    CONSUMER_LAG = Gauge(
        'kafka_consumer_lag',
        'Current consumer lag',
        ['topic', 'partition']
    )
    
    MODEL_LOADED = Gauge(
        'kafka_consumer_model_loaded',
        'Whether the ML model is loaded (1=loaded, 0=not loaded)'
    )
    
    BROKER_CONNECTED = Gauge(
        'kafka_consumer_broker_connected',
        'Whether consumer is connected to broker (1=connected, 0=disconnected)'
    )
    
    # Info metric for metadata
    CONSUMER_INFO = Info(
        'kafka_consumer',
        'Consumer metadata and configuration'
    )
else:
    # Dummy metrics if Prometheus not available
    class DummyMetric:
        def labels(self, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass
    
    MESSAGES_PROCESSED = DummyMetric()
    MESSAGES_FAILED = DummyMetric()
    PROCESSING_LATENCY = DummyMetric()
    CONSUMER_LAG = DummyMetric()
    MODEL_LOADED = DummyMetric()
    BROKER_CONNECTED = DummyMetric()
    CONSUMER_INFO = DummyMetric()


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Outputs log records as JSON objects with standard fields plus custom extras.
    """
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'message_id'):
            log_data['message_id'] = record.message_id
        if hasattr(record, 'topic'):
            log_data['topic'] = record.topic
        if hasattr(record, 'partition'):
            log_data['partition'] = record.partition
        if hasattr(record, 'offset'):
            log_data['offset'] = record.offset
        if hasattr(record, 'latency_ms'):
            log_data['latency_ms'] = record.latency_ms
        if hasattr(record, 'event_type'):
            log_data['event_type'] = record.event_type
        if hasattr(record, 'error_type'):
            log_data['error_type'] = record.error_type
        if hasattr(record, 'prediction'):
            log_data['prediction'] = record.prediction
        if hasattr(record, 'churn_probability'):
            log_data['churn_probability'] = record.churn_probability
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the consumer with dual output:
    - Console: Human-readable format
    - File (structured JSON): Machine-readable format for log aggregation
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create artifacts/logs for timestamped files
    artifact_log_dir = Path("artifacts/logs")
    artifact_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("kafka_consumer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with simple, human-readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Structured JSON file handler for machine-readable logs
    json_handler = logging.handlers.RotatingFileHandler(
        log_dir / "kafka_consumer.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    json_handler.setLevel(logging.DEBUG)
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)
    
    # Traditional timestamped file handler for human-readable logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        artifact_log_dir / f"consumer_{timestamp}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def signal_handler(signum, frame):
    """
    Handle shutdown signals (SIGINT, SIGTERM) for graceful shutdown.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    logger = logging.getLogger("kafka_consumer")
    logger.info(f"Received {signal_name} signal. Initiating graceful shutdown...")
    shutdown_requested = True


def start_metrics_server(port: int = 8000) -> Optional[Thread]:
    """
    Start Prometheus metrics HTTP server in a background thread.
    
    Args:
        port: Port number for metrics endpoint (default: 8000)
    
    Returns:
        Thread object if Prometheus available, None otherwise
    """
    if not PROMETHEUS_AVAILABLE:
        logger = logging.getLogger("kafka_consumer")
        logger.warning("Prometheus client not available. Metrics server not started.")
        return None
    
    try:
        # Start HTTP server in a daemon thread
        start_http_server(port)
        logger = logging.getLogger("kafka_consumer")
        logger.info(f"Metrics server started on http://localhost:{port}/metrics")
        return None  # start_http_server runs in its own thread
    except OSError as e:
        logger = logging.getLogger("kafka_consumer")
        logger.error(f"Failed to start metrics server on port {port}: {e}")
        return None


def get_health_status(consumer: Optional[KafkaConsumer] = None, 
                     model_loaded: bool = False) -> dict:
    """
    Get current health status of the consumer.
    
    Args:
        consumer: KafkaConsumer instance (None if not initialized)
        model_loaded: Whether ML model is loaded
    
    Returns:
        Dictionary with health status
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "checks": {
            "broker_connection": "unknown",
            "model_loaded": "pass" if model_loaded else "fail"
        }
    }
    
    # Check broker connection
    if consumer is not None:
        try:
            # Try to get topics - this will fail if broker is unreachable
            topics = consumer.topics()
            health["checks"]["broker_connection"] = "pass"
            BROKER_CONNECTED.set(1)
        except Exception as e:
            health["checks"]["broker_connection"] = "fail"
            health["status"] = "unhealthy"
            health["error"] = str(e)
            BROKER_CONNECTED.set(0)
    else:
        health["checks"]["broker_connection"] = "not_initialized"
    
    # Overall health based on checks
    if any(v == "fail" for v in health["checks"].values()):
        health["status"] = "unhealthy"
    
    return health


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Kafka Consumer for Telco Churn Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Streaming mode with validation
  python -m src.streaming.consumer --mode streaming --validate
  
  # Batch mode processing 1000 messages
  python -m src.streaming.consumer --mode batch --max-messages 1000
  
  # Dry-run mode (simulation without Kafka)
  python -m src.streaming.consumer --mode streaming --dry-run
  
  # Custom broker and topics
  python -m src.streaming.consumer --broker kafka.example.com:9092 \\
      --input-topic customers --output-topic predictions
  
  # Use Spark model backend
  python -m src.streaming.consumer --model-backend spark \\
      --model-path artifacts/models/spark_model
        """
    )
    
    # Mode settings
    parser.add_argument(
        '--mode',
        type=str,
        choices=['streaming', 'batch'],
        default='streaming',
        help='Consumer mode: streaming (continuous) or batch (bounded)'
    )
    
    # Kafka settings
    parser.add_argument(
        '--broker',
        type=str,
        default=DEFAULT_BROKER,
        help=f'Kafka bootstrap server (default: {DEFAULT_BROKER})'
    )
    parser.add_argument(
        '--input-topic',
        type=str,
        default=DEFAULT_INPUT_TOPIC,
        help=f'Input topic for customer messages (default: {DEFAULT_INPUT_TOPIC})'
    )
    parser.add_argument(
        '--output-topic',
        type=str,
        default=DEFAULT_OUTPUT_TOPIC,
        help=f'Output topic for predictions (default: {DEFAULT_OUTPUT_TOPIC})'
    )
    parser.add_argument(
        '--deadletter-topic',
        type=str,
        default=DEFAULT_DEADLETTER_TOPIC,
        help=f'Dead letter topic for failed messages (default: {DEFAULT_DEADLETTER_TOPIC})'
    )
    parser.add_argument(
        '--consumer-group',
        type=str,
        default=DEFAULT_CONSUMER_GROUP,
        help=f'Consumer group ID (default: {DEFAULT_CONSUMER_GROUP})'
    )
    
    # Model settings
    parser.add_argument(
        '--model-backend',
        type=str,
        choices=['sklearn', 'spark'],
        default=DEFAULT_MODEL_BACKEND,
        help=f'Model backend type (default: {DEFAULT_MODEL_BACKEND})'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to saved model file (default: {DEFAULT_MODEL_PATH})'
    )
    parser.add_argument(
        '--preprocessor-path',
        type=str,
        default=DEFAULT_PREPROCESSOR_PATH,
        help='Path to saved preprocessor (optional, if separate from model)'
    )
    
    # Batch mode settings
    parser.add_argument(
        '--max-messages',
        type=int,
        default=None,
        help='Maximum messages to process in batch mode (default: unbounded)'
    )
    parser.add_argument(
        '--timeout-ms',
        type=int,
        default=10000,
        help='Consumer poll timeout in milliseconds (default: 10000)'
    )
    
    # Validation settings
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Enable schema validation for input messages'
    )
    parser.add_argument(
        '--customer-schema',
        type=str,
        default='schemas/telco_customer_schema.json',
        help='Path to customer message schema (default: schemas/telco_customer_schema.json)'
    )
    parser.add_argument(
        '--prediction-schema',
        type=str,
        default='schemas/churn_prediction_schema.json',
        help='Path to prediction message schema (default: schemas/churn_prediction_schema.json)'
    )
    
    # Operational settings
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry-run mode: simulate processing without Kafka'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--metrics-interval',
        type=int,
        default=100,
        help='Log metrics every N messages (default: 100)'
    )
    
    return parser.parse_args()


def load_sklearn_model(model_path: str, logger: logging.Logger) -> Any:
    """
    Load a scikit-learn model from disk.
    
    Args:
        model_path: Path to the saved model file (joblib format)
        logger: Logger instance
    
    Returns:
        Loaded model object
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading sklearn model from {model_path}...")
    try:
        model = joblib.load(model_path)
        logger.info(f"✓ Successfully loaded sklearn model: {type(model).__name__}")
        return model
    except Exception as e:
        logger.error(f"✗ Failed to load sklearn model: {e}")
        raise


def load_preprocessor(preprocessor_path: Optional[str], logger: logging.Logger) -> Optional[Any]:
    """
    Load a preprocessor from disk (if separate from model).
    
    Args:
        preprocessor_path: Path to the saved preprocessor file (optional)
        logger: Logger instance
    
    Returns:
        Loaded preprocessor object or None if not provided
    
    Raises:
        FileNotFoundError: If preprocessor file doesn't exist
        Exception: If preprocessor loading fails
    """
    if preprocessor_path is None:
        logger.info("No separate preprocessor specified (assuming model includes preprocessing)")
        return None
    
    preprocessor_file = Path(preprocessor_path)
    if not preprocessor_file.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    logger.info(f"Loading preprocessor from {preprocessor_path}...")
    try:
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"✓ Successfully loaded preprocessor: {type(preprocessor).__name__}")
        return preprocessor
    except Exception as e:
        logger.error(f"✗ Failed to load preprocessor: {e}")
        raise


def validate_message(
    message: Dict[str, Any],
    validator: Optional['SchemaValidator'],
    logger: logging.Logger
) -> Tuple[bool, List[str]]:
    """
    Validate a message against the input schema.
    
    Args:
        message: Message dictionary to validate
        validator: Schema validator instance (None if validation disabled)
        logger: Logger instance
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    if validator is None:
        # Validation disabled, assume valid
        return True, []
    
    try:
        is_valid, errors = validator.validate(message)
        return is_valid, errors
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, [f"Validation exception: {str(e)}"]


def transform_to_features(message: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Transform a message to model input features.
    
    Args:
        message: Customer message dictionary
        logger: Logger instance
    
    Returns:
        DataFrame with model features (single row)
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    try:
        # Extract only the feature columns (exclude customerID and event_ts)
        feature_dict = {k: v for k, v in message.items() if k in EXPECTED_FEATURES}
        
        # Verify all expected features are present
        missing_features = set(EXPECTED_FEATURES) - set(feature_dict.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Create DataFrame with single row (model expects DataFrame)
        df = pd.DataFrame([feature_dict])
        
        # Log feature summary at debug level
        logger.debug(f"Transformed message to features: {list(feature_dict.keys())}")
        
        return df
    except Exception as e:
        logger.error(f"Feature transformation error: {e}")
        raise


def run_inference(
    features: pd.DataFrame,
    model: Any,
    preprocessor: Optional[Any],
    logger: logging.Logger
) -> Tuple[str, float]:
    """
    Run churn prediction inference on features.
    
    Args:
        features: DataFrame with model input features
        model: Loaded model object
        preprocessor: Optional preprocessor (None if model includes preprocessing)
        logger: Logger instance
    
    Returns:
        Tuple of (prediction, probability)
        - prediction: "Yes" or "No"
        - probability: Churn probability (0.0 to 1.0)
    
    Raises:
        Exception: If inference fails
    """
    try:
        # Apply preprocessing if separate preprocessor exists
        if preprocessor is not None:
            features = preprocessor.transform(features)
        
        # Run prediction (sklearn models return class probabilities)
        # Assuming binary classification with classes [0, 1] = ["No", "Yes"]
        probabilities = model.predict_proba(features)
        
        # Get probability of churn (class 1)
        churn_probability = float(probabilities[0][1])
        
        # Get prediction (Yes/No)
        prediction = "Yes" if churn_probability >= 0.5 else "No"
        
        logger.debug(f"Inference result: {prediction} (probability: {churn_probability:.4f})")
        
        return prediction, churn_probability
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise


def compose_prediction_message(
    customer_id: str,
    prediction: str,
    probability: float,
    event_ts: str,
    inference_latency_ms: Optional[float] = None,
    model_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compose a prediction result message.
    
    Args:
        customer_id: Customer ID
        prediction: Churn prediction ("Yes" or "No")
        probability: Churn probability (0.0 to 1.0)
        event_ts: Original event timestamp from input message
        inference_latency_ms: Inference latency in milliseconds (optional)
        model_version: Model version string (optional)
    
    Returns:
        Prediction message dictionary conforming to churn_prediction_schema.json
    """
    # Format timestamp as ISO 8601 with Z suffix (UTC)
    processed_ts = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    message = {
        "customerID": customer_id,
        "churn_probability": round(probability, 6),  # Round to 6 decimals
        "prediction": prediction,
        "event_ts": event_ts,
        "processed_ts": processed_ts
    }
    
    # Add optional fields if provided
    if model_version is not None:
        message["model_version"] = model_version
    if inference_latency_ms is not None:
        message["inference_latency_ms"] = round(inference_latency_ms, 2)
    
    return message


def compose_deadletter_message(
    original_message: Dict[str, Any],
    error_type: str,
    error_message: str,
    source_topic: str,
    validation_errors: Optional[List[str]] = None,
    consumer_group: Optional[str] = None,
    retry_count: int = 0
) -> Dict[str, Any]:
    """
    Compose a dead letter queue message for failed processing.
    
    Args:
        original_message: Original message that failed
        error_type: Type of error (validation_error, inference_error, processing_error, unknown_error)
        error_message: Human-readable error message
        source_topic: Source topic where message came from
        validation_errors: List of validation error messages (optional)
        consumer_group: Consumer group ID (optional)
        retry_count: Number of retry attempts (optional)
    
    Returns:
        Dead letter message dictionary conforming to deadletter_schema.json
    """
    # Format timestamp as ISO 8601 with Z suffix (UTC)
    failed_ts = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    message = {
        "original_message": original_message,
        "error_type": error_type,
        "error_message": error_message,
        "source_topic": source_topic,
        "failed_ts": failed_ts
    }
    
    # Add optional fields if provided
    if validation_errors is not None:
        message["validation_errors"] = validation_errors
    if consumer_group is not None:
        message["consumer_group"] = consumer_group
    if retry_count > 0:
        message["retry_count"] = retry_count
    
    return message


def process_message(
    raw_message: Dict[str, Any],
    model: Any,
    preprocessor: Optional[Any],
    input_validator: Optional['SchemaValidator'],
    output_validator: Optional['SchemaValidator'],
    source_topic: str,
    consumer_group: str,
    logger: logging.Logger
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a single message through the full pipeline.
    
    Pipeline:
    1. Validate input message (if validator enabled)
    2. Transform to model features
    3. Run inference
    4. Compose prediction message
    5. Validate output message (if validator enabled)
    
    Args:
        raw_message: Raw input message from Kafka
        model: Loaded model object
        preprocessor: Optional preprocessor
        input_validator: Input schema validator (None if disabled)
        output_validator: Output schema validator (None if disabled)
        source_topic: Source topic name
        consumer_group: Consumer group ID
        logger: Logger instance
    
    Returns:
        Tuple of (prediction_message, deadletter_message)
        - prediction_message: Prediction result (None if processing failed)
        - deadletter_message: Dead letter message (None if processing succeeded)
    """
    start_time = time.time()
    
    try:
        # Step 1: Validate input message
        is_valid, validation_errors = validate_message(raw_message, input_validator, logger)
        if not is_valid:
            logger.warning(f"Message validation failed: {validation_errors[:3]}...")  # Show first 3 errors
            deadletter = compose_deadletter_message(
                original_message=raw_message,
                error_type="validation_error",
                error_message="Input message failed schema validation",
                source_topic=source_topic,
                validation_errors=validation_errors,
                consumer_group=consumer_group
            )
            return None, deadletter
        
        # Step 2: Transform to features
        try:
            features = transform_to_features(raw_message, logger)
        except ValueError as e:
            logger.error(f"Feature transformation failed: {e}")
            deadletter = compose_deadletter_message(
                original_message=raw_message,
                error_type="processing_error",
                error_message=f"Feature transformation failed: {str(e)}",
                source_topic=source_topic,
                consumer_group=consumer_group
            )
            return None, deadletter
        
        # Step 3: Run inference
        try:
            prediction, probability = run_inference(features, model, preprocessor, logger)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            deadletter = compose_deadletter_message(
                original_message=raw_message,
                error_type="inference_error",
                error_message=f"Model inference failed: {str(e)}",
                source_topic=source_topic,
                consumer_group=consumer_group
            )
            return None, deadletter
        
        # Step 4: Compose prediction message
        inference_latency_ms = (time.time() - start_time) * 1000
        prediction_msg = compose_prediction_message(
            customer_id=raw_message.get("customerID", "UNKNOWN"),
            prediction=prediction,
            probability=probability,
            event_ts=raw_message.get("event_ts", datetime.now(timezone.utc).isoformat()),
            inference_latency_ms=inference_latency_ms
        )
        
        # Step 5: Validate output message (optional)
        if output_validator is not None:
            is_valid, validation_errors = validate_message(prediction_msg, output_validator, logger)
            if not is_valid:
                logger.error(f"Output validation failed (BUG): {validation_errors}")
                # This is a bug in our code, not the input data
                deadletter = compose_deadletter_message(
                    original_message=raw_message,
                    error_type="processing_error",
                    error_message="Output message failed schema validation (internal error)",
                    source_topic=source_topic,
                    validation_errors=validation_errors,
                    consumer_group=consumer_group
                )
                return None, deadletter
        
        return prediction_msg, None
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error processing message: {e}")
        deadletter = compose_deadletter_message(
            original_message=raw_message,
            error_type="unknown_error",
            error_message=f"Unexpected processing error: {str(e)}",
            source_topic=source_topic,
            consumer_group=consumer_group
        )
        return None, deadletter


def streaming_mode(
    args: argparse.Namespace,
    model: Any,
    preprocessor: Optional[Any],
    input_validator: Optional['SchemaValidator'],
    output_validator: Optional['SchemaValidator'],
    logger: logging.Logger
):
    """
    Run consumer in streaming mode (continuous consumption).
    
    Args:
        args: Parsed command-line arguments
        model: Loaded model object
        preprocessor: Optional preprocessor
        input_validator: Input schema validator (None if disabled)
        output_validator: Output schema validator (None if disabled)
        logger: Logger instance
    """
    global shutdown_requested
    
    logger.info("Starting streaming mode...")
    
    # Initialize Kafka consumer
    logger.info(f"Connecting to Kafka broker: {args.broker}")
    consumer = KafkaConsumer(
        args.input_topic,
        bootstrap_servers=args.broker,
        group_id=args.consumer_group,
        auto_offset_reset='earliest',  # Start from earliest if no offset
        enable_auto_commit=True,       # Auto-commit offsets
        auto_commit_interval_ms=5000,  # Commit every 5 seconds
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        key_deserializer=lambda m: m.decode('utf-8') if m else None
    )
    
    # Initialize Kafka producer for predictions and deadletter
    producer = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda m: json.dumps(m).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None
    )
    
    logger.info(f"✓ Connected to Kafka (subscribed to {args.input_topic})")
    
    # Metrics
    messages_processed = 0
    messages_succeeded = 0
    messages_failed = 0
    validation_errors_count = 0
    inference_errors_count = 0
    total_latency_ms = 0.0
    
    try:
        logger.info("Starting message consumption (press Ctrl+C to stop)...")
        
        for message in consumer:
            if shutdown_requested:
                logger.info("Shutdown requested, stopping consumption...")
                break
            
            start_time = time.time()
            
            try:
                raw_message = message.value
                customer_id = raw_message.get("customerID", "UNKNOWN")
                
                # Process message
                prediction_msg, deadletter_msg = process_message(
                    raw_message=raw_message,
                    model=model,
                    preprocessor=preprocessor,
                    input_validator=input_validator,
                    output_validator=output_validator,
                    source_topic=args.input_topic,
                    consumer_group=args.consumer_group,
                    logger=logger
                )
                
                # Publish result
                if prediction_msg is not None:
                    # Success: publish prediction
                    producer.send(
                        args.output_topic,
                        key=customer_id,
                        value=prediction_msg
                    )
                    messages_succeeded += 1
                    
                    # Update latency metrics
                    latency_ms = (time.time() - start_time) * 1000
                    total_latency_ms += latency_ms
                
                else:
                    # Failure: publish to deadletter
                    producer.send(
                        args.deadletter_topic,
                        key=customer_id,
                        value=deadletter_msg
                    )
                    messages_failed += 1
                    
                    # Track error type
                    error_type = deadletter_msg.get("error_type", "unknown")
                    if error_type == "validation_error":
                        validation_errors_count += 1
                    elif error_type == "inference_error":
                        inference_errors_count += 1
                
                messages_processed += 1
                
                # Log metrics periodically
                if messages_processed % args.metrics_interval == 0:
                    avg_latency = total_latency_ms / messages_succeeded if messages_succeeded > 0 else 0
                    success_rate = (messages_succeeded / messages_processed * 100) if messages_processed > 0 else 0
                    logger.info(
                        f"📊 Metrics: processed={messages_processed}, "
                        f"succeeded={messages_succeeded}, failed={messages_failed}, "
                        f"success_rate={success_rate:.2f}%, avg_latency={avg_latency:.2f}ms"
                    )
            
            except Exception as e:
                logger.exception(f"Error processing message: {e}")
                messages_failed += 1
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    
    finally:
        # Flush producer and close connections
        logger.info("Flushing producer...")
        producer.flush(timeout=10)
        producer.close()
        
        logger.info("Closing consumer...")
        consumer.close()
        
        # Final metrics
        avg_latency = total_latency_ms / messages_succeeded if messages_succeeded > 0 else 0
        success_rate = (messages_succeeded / messages_processed * 100) if messages_processed > 0 else 0
        logger.info("=" * 80)
        logger.info("FINAL METRICS")
        logger.info("=" * 80)
        logger.info(f"Total messages processed: {messages_processed}")
        logger.info(f"Successful predictions: {messages_succeeded}")
        logger.info(f"Failed messages: {messages_failed}")
        logger.info(f"  - Validation errors: {validation_errors_count}")
        logger.info(f"  - Inference errors: {inference_errors_count}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Average latency: {avg_latency:.2f}ms")
        logger.info("=" * 80)


def batch_mode(
    args: argparse.Namespace,
    model: Any,
    preprocessor: Optional[Any],
    input_validator: Optional['SchemaValidator'],
    output_validator: Optional['SchemaValidator'],
    logger: logging.Logger
):
    """
    Run consumer in batch mode (bounded processing).
    
    Args:
        args: Parsed command-line arguments
        model: Loaded model object
        preprocessor: Optional preprocessor
        input_validator: Input schema validator (None if disabled)
        output_validator: Output schema validator (None if disabled)
        logger: Logger instance
    """
    global shutdown_requested
    
    max_messages = args.max_messages or float('inf')
    logger.info(f"Starting batch mode (max_messages={max_messages})...")
    
    # Initialize Kafka consumer
    logger.info(f"Connecting to Kafka broker: {args.broker}")
    consumer = KafkaConsumer(
        args.input_topic,
        bootstrap_servers=args.broker,
        group_id=args.consumer_group,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        auto_commit_interval_ms=5000,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        key_deserializer=lambda m: m.decode('utf-8') if m else None,
        consumer_timeout_ms=args.timeout_ms  # Stop after timeout if no messages
    )
    
    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda m: json.dumps(m).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None
    )
    
    logger.info(f"✓ Connected to Kafka (subscribed to {args.input_topic})")
    
    # Metrics
    messages_processed = 0
    messages_succeeded = 0
    messages_failed = 0
    validation_errors_count = 0
    inference_errors_count = 0
    total_latency_ms = 0.0
    
    try:
        logger.info(f"Processing up to {max_messages} messages (timeout={args.timeout_ms}ms)...")
        
        for message in consumer:
            if shutdown_requested:
                logger.info("Shutdown requested, stopping batch processing...")
                break
            
            if messages_processed >= max_messages:
                logger.info(f"Reached max_messages limit ({max_messages}), stopping...")
                break
            
            start_time = time.time()
            
            try:
                raw_message = message.value
                customer_id = raw_message.get("customerID", "UNKNOWN")
                
                # Process message
                prediction_msg, deadletter_msg = process_message(
                    raw_message=raw_message,
                    model=model,
                    preprocessor=preprocessor,
                    input_validator=input_validator,
                    output_validator=output_validator,
                    source_topic=args.input_topic,
                    consumer_group=args.consumer_group,
                    logger=logger
                )
                
                # Publish result
                if prediction_msg is not None:
                    producer.send(args.output_topic, key=customer_id, value=prediction_msg)
                    messages_succeeded += 1
                    latency_ms = (time.time() - start_time) * 1000
                    total_latency_ms += latency_ms
                else:
                    producer.send(args.deadletter_topic, key=customer_id, value=deadletter_msg)
                    messages_failed += 1
                    error_type = deadletter_msg.get("error_type", "unknown")
                    if error_type == "validation_error":
                        validation_errors_count += 1
                    elif error_type == "inference_error":
                        inference_errors_count += 1
                
                messages_processed += 1
                
                # Log metrics periodically
                if messages_processed % args.metrics_interval == 0:
                    avg_latency = total_latency_ms / messages_succeeded if messages_succeeded > 0 else 0
                    success_rate = (messages_succeeded / messages_processed * 100) if messages_processed > 0 else 0
                    logger.info(
                        f"📊 Progress: {messages_processed}/{max_messages if max_messages != float('inf') else '∞'}, "
                        f"succeeded={messages_succeeded}, failed={messages_failed}, "
                        f"success_rate={success_rate:.2f}%, avg_latency={avg_latency:.2f}ms"
                    )
            
            except Exception as e:
                logger.exception(f"Error processing message: {e}")
                messages_failed += 1
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    
    finally:
        # Flush producer and close connections
        logger.info("Flushing producer...")
        producer.flush(timeout=10)
        producer.close()
        
        logger.info("Closing consumer...")
        consumer.close()
        
        # Final metrics
        avg_latency = total_latency_ms / messages_succeeded if messages_succeeded > 0 else 0
        success_rate = (messages_succeeded / messages_processed * 100) if messages_processed > 0 else 0
        logger.info("=" * 80)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total messages processed: {messages_processed}")
        logger.info(f"Successful predictions: {messages_succeeded}")
        logger.info(f"Failed messages: {messages_failed}")
        logger.info(f"  - Validation errors: {validation_errors_count}")
        logger.info(f"  - Inference errors: {inference_errors_count}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Average latency: {avg_latency:.2f}ms")
        logger.info("=" * 80)


def main():
    """
    Main entry point for the consumer.
    """
    global shutdown_requested
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("=" * 80)
    logger.info("Telco Churn Prediction Consumer")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Broker: {args.broker}")
    logger.info(f"Input Topic: {args.input_topic}")
    logger.info(f"Output Topic: {args.output_topic}")
    logger.info(f"Dead Letter Topic: {args.deadletter_topic}")
    logger.info(f"Consumer Group: {args.consumer_group}")
    logger.info(f"Model Backend: {args.model_backend}")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Validation Enabled: {args.validate}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("=" * 80)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start Prometheus metrics server
    start_metrics_server(port=8000)
    
    # Set consumer info metadata
    if PROMETHEUS_AVAILABLE:
        CONSUMER_INFO.info({
            'version': '1.0.0',
            'mode': args.mode,
            'broker': args.broker,
            'consumer_group': args.consumer_group,
            'model_backend': args.model_backend
        })
    
    # Check for schema validator if validation is enabled
    if args.validate and SchemaValidator is None:
        logger.error("✗ Schema validation requested but SchemaValidator not available")
        logger.error("  Make sure jsonschema is installed: pip install jsonschema")
        sys.exit(1)
    
    try:
        # Load model
        if args.model_backend == 'sklearn':
            model = load_sklearn_model(args.model_path, logger)
            MODEL_LOADED.set(1)  # Mark model as loaded
        else:
            logger.error(f"✗ Model backend '{args.model_backend}' not yet implemented")
            sys.exit(1)
        
        # Load preprocessor (if separate)
        preprocessor = load_preprocessor(args.preprocessor_path, logger)
        
        # Initialize schema validators if validation is enabled
        input_validator = None
        output_validator = None
        if args.validate:
            logger.info(f"Loading customer schema from {args.customer_schema}...")
            input_validator = SchemaValidator(Path(args.customer_schema))
            logger.info("✓ Customer schema loaded")
            
            logger.info(f"Loading prediction schema from {args.prediction_schema}...")
            output_validator = SchemaValidator(Path(args.prediction_schema))
            logger.info("✓ Prediction schema loaded")
        
        # Dry-run mode: simulate processing
        if args.dry_run:
            logger.info("🔧 DRY-RUN MODE: Simulating consumer without Kafka")
            logger.info("✓ Model loaded successfully")
            if input_validator:
                logger.info("✓ Input schema validator initialized")
            if output_validator:
                logger.info("✓ Output schema validator initialized")
            logger.info("✓ Consumer setup complete (dry-run)")
            logger.info("In production mode, consumer would:")
            logger.info(f"  1. Connect to Kafka broker: {args.broker}")
            logger.info(f"  2. Subscribe to topic: {args.input_topic}")
            logger.info(f"  3. Validate messages with schema (if --validate enabled)")
            logger.info(f"  4. Run inference using {args.model_backend} model")
            logger.info(f"  5. Publish predictions to: {args.output_topic}")
            logger.info(f"  6. Route errors to: {args.deadletter_topic}")
            logger.info("✓ Dry-run complete")
            return
        
        # Run in selected mode
        if args.mode == 'streaming':
            streaming_mode(
                args=args,
                model=model,
                preprocessor=preprocessor,
                input_validator=input_validator,
                output_validator=output_validator,
                logger=logger
            )
        elif args.mode == 'batch':
            batch_mode(
                args=args,
                model=model,
                preprocessor=preprocessor,
                input_validator=input_validator,
                output_validator=output_validator,
                logger=logger
            )
    
    except FileNotFoundError as e:
        logger.error(f"✗ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}")
        logger.exception("Exception details:")
        sys.exit(1)
    finally:
        logger.info("Consumer shutdown complete")


if __name__ == "__main__":
    main()
