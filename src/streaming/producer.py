#!/usr/bin/env python3
"""
Kafka Producer for Telco Customer Churn Prediction
===================================================

This script produces customer data messages to Kafka topics for real-time
churn prediction. Supports both streaming (continuous random sampling) and
batch (sequential CSV processing) modes.

Features:
- Streaming mode: Random customer sampling at configurable rate
- Batch mode: Chunked CSV processing with checkpoint resume
- Dry-run mode: Message generation without Kafka publishing
- Structured logging with metrics
- Graceful shutdown with signal handling

Author: Telco Churn MLOps Team
Version: 1.0.0
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Generator
import random

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# Import schema validator (optional, only if validation is enabled)
try:
    from .schema_validator import SchemaValidator
    VALIDATION_AVAILABLE = True
except ImportError as e:
    try:
        # Fallback to absolute import
        from src.streaming.schema_validator import SchemaValidator
        VALIDATION_AVAILABLE = True
    except ImportError:
        VALIDATION_AVAILABLE = False
        SchemaValidator = None


# Configuration
DEFAULT_BROKER = "localhost:19092"
DEFAULT_TOPIC = "telco.raw.customers"
DEFAULT_DATASET = "data/raw/Telco-Customer-Churn.csv"
DEFAULT_EVENTS_PER_SEC = 1.0
DEFAULT_BATCH_SIZE = 100
DEFAULT_CHECKPOINT_FILE = "artifacts/producer_checkpoint.json"
LOG_FILE = "logs/kafka_producer.log"

# Global flag for graceful shutdown
shutdown_requested = False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured logging for the producer.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger("kafka_producer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logging.getLogger("kafka_producer").info(
        f"Shutdown signal received ({signum}). Gracefully stopping..."
    )


def load_dataset(dataset_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load customer dataset from CSV.
    
    Args:
        dataset_path: Path to CSV file
        logger: Logger instance
        
    Returns:
        DataFrame with customer data
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        pd.errors.ParserError: If CSV parsing fails
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} customer records with {len(df.columns)} columns")
    
    return df


def create_kafka_producer(
    bootstrap_servers: str,
    logger: logging.Logger,
    dry_run: bool = False
) -> Optional[KafkaProducer]:
    """Create and configure Kafka producer.
    
    Args:
        bootstrap_servers: Kafka broker address(es)
        logger: Logger instance
        dry_run: If True, return None (no actual producer)
        
    Returns:
        KafkaProducer instance or None in dry-run mode
        
    Raises:
        NoBrokersAvailable: If connection to Kafka fails
    """
    if dry_run:
        logger.info("DRY-RUN mode: Kafka producer not initialized")
        return None
    
    logger.info(f"Connecting to Kafka broker: {bootstrap_servers}")
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas to acknowledge
            retries=3,   # Retry failed sends
            max_in_flight_requests_per_connection=5,
            compression_type='gzip',
            linger_ms=10,  # Small batching delay
        )
        
        logger.info("Kafka producer initialized successfully")
        return producer
        
    except NoBrokersAvailable as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        logger.error(f"Make sure Kafka is running at {bootstrap_servers}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating producer: {e}")
        raise


def customer_to_message(customer: pd.Series, add_timestamp: bool = True) -> Dict[str, Any]:
    """Convert customer record to Kafka message format.
    
    Args:
        customer: Customer data as pandas Series
        add_timestamp: Whether to add event_ts field
        
    Returns:
        Dictionary with customer data in JSON-serializable format
    """
    message = customer.to_dict()
    
    # Convert any NaN values to None for JSON serialization
    message = {k: (None if pd.isna(v) else v) for k, v in message.items()}
    
    # Add event timestamp
    if add_timestamp:
        # Random timestamp within last 24 hours for more realistic simulation
        now = datetime.now(timezone.utc)
        random_offset = timedelta(seconds=random.randint(0, 86400))
        event_time = now - random_offset
        message['event_ts'] = event_time.isoformat().replace('+00:00', 'Z')
    
    return message


def publish_message(
    producer: Optional[KafkaProducer],
    topic: str,
    key: str,
    message: Dict[str, Any],
    logger: logging.Logger,
    dry_run: bool = False,
    validator: Any = None
) -> tuple[bool, Optional[list[str]]]:
    """Publish message to Kafka topic with optional validation.
    
    Args:
        producer: KafkaProducer instance (or None in dry-run)
        topic: Topic name
        key: Message key (typically customerID)
        message: Message payload
        logger: Logger instance
        dry_run: If True, only log without publishing
        validator: Optional SchemaValidator for message validation
        
    Returns:
        Tuple of (success: bool, validation_errors: Optional[list])
        - success: True if message was published (or would be in dry-run)
        - validation_errors: List of validation errors, or None if valid/not validated
    """
    # Validate message if validator is provided
    if validator is not None:
        is_valid, errors = validator.validate(message)
        if not is_valid:
            logger.warning(
                f"Message validation failed for key {key}: {len(errors)} error(s)"
            )
            for error in errors[:3]:  # Log first 3 errors
                logger.debug(f"  - {error}")
            if len(errors) > 3:
                logger.debug(f"  ... and {len(errors) - 3} more errors")
            return False, errors
    
    if dry_run:
        logger.debug(f"[DRY-RUN] Would publish to {topic} | Key: {key} | Message: {json.dumps(message)[:100]}...")
        return True, None
    
    try:
        future = producer.send(topic, key=key, value=message)
        # Block until message is sent or timeout
        record_metadata = future.get(timeout=10)
        
        logger.debug(
            f"Published to {topic} | Partition: {record_metadata.partition} | "
            f"Offset: {record_metadata.offset} | Key: {key}"
        )
        return True, None
        
    except KafkaError as e:
        logger.error(f"Kafka error publishing message: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error publishing message: {e}")
        return False, None


def load_checkpoint(checkpoint_file: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load checkpoint data for batch processing resume.
    
    Args:
        checkpoint_file: Path to checkpoint JSON file
        logger: Logger instance
        
    Returns:
        Dictionary with checkpoint data (last_row, last_offset, timestamp)
    """
    if not os.path.exists(checkpoint_file):
        logger.info("No checkpoint file found, starting from beginning")
        return {"last_row": 0, "last_offset": 0, "timestamp": None}
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint: row {checkpoint.get('last_row', 0)}")
        return checkpoint
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}. Starting from beginning.")
        return {"last_row": 0, "last_offset": 0, "timestamp": None}


def save_checkpoint(
    checkpoint_file: str,
    last_row: int,
    last_offset: int,
    logger: logging.Logger
) -> None:
    """Save checkpoint data for batch processing resume.
    
    Args:
        checkpoint_file: Path to checkpoint JSON file
        last_row: Last processed row index
        last_offset: Last Kafka offset
        logger: Logger instance
    """
    checkpoint_data = {
        "last_row": last_row,
        "last_offset": last_offset,
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }
    
    # Create directory if it doesn't exist
    Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.debug(f"Checkpoint saved: row {last_row}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def streaming_mode(
    producer: Optional[KafkaProducer],
    df: pd.DataFrame,
    topic: str,
    events_per_sec: float,
    logger: logging.Logger,
    dry_run: bool = False,
    validator: Any = None
) -> None:
    """Run producer in streaming mode (continuous random sampling).
    
    Args:
        producer: KafkaProducer instance
        df: Customer dataset
        topic: Kafka topic name
        events_per_sec: Target event rate (events/second)
        logger: Logger instance
        dry_run: If True, simulate without publishing
        validator: Optional SchemaValidator for message validation
    """
    logger.info(f"Starting STREAMING mode: {events_per_sec} events/sec to topic '{topic}'")
    logger.info(f"Dataset size: {len(df)} customers")
    logger.info(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    logger.info(f"Validation: {'ENABLED' if validator else 'DISABLED'}")
    logger.info("Press Ctrl+C to stop gracefully...")
    
    interval = 1.0 / events_per_sec
    total_sent = 0
    total_failed = 0
    validation_failed = 0
    start_time = time.time()
    
    try:
        while not shutdown_requested:
            # Sample random customer
            customer = df.sample(n=1).iloc[0]
            customer_id = str(customer['customerID'])
            
            # Convert to message
            message = customer_to_message(customer, add_timestamp=True)
            
            # Publish with validation
            success, errors = publish_message(producer, topic, customer_id, message, logger, dry_run, validator)
            
            if success:
                total_sent += 1
            else:
                total_failed += 1
                if errors is not None:  # Validation failure
                    validation_failed += 1
            
            # Log progress every 100 messages
            if total_sent % 100 == 0:
                elapsed = time.time() - start_time
                actual_rate = total_sent / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {total_sent} sent, {total_failed} failed "
                    f"(validation: {validation_failed}) | "
                    f"Actual rate: {actual_rate:.2f} events/sec"
                )
            
            # Sleep to maintain rate
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("STREAMING MODE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total messages sent: {total_sent}")
        logger.info(f"Total failures: {total_failed}")
        if validator:
            logger.info(f"Validation failures: {validation_failed}")
        logger.info(f"Duration: {elapsed:.2f} seconds")
        logger.info(f"Average rate: {total_sent / elapsed:.2f} events/sec")
        logger.info("=" * 60)


def batch_mode(
    producer: Optional[KafkaProducer],
    df: pd.DataFrame,
    topic: str,
    batch_size: int,
    checkpoint_file: str,
    logger: logging.Logger,
    dry_run: bool = False,
    validator: Any = None
) -> None:
    """Run producer in batch mode (sequential CSV processing).
    
    Args:
        producer: KafkaProducer instance
        df: Customer dataset
        topic: Kafka topic name
        batch_size: Number of records per batch
        checkpoint_file: Path to checkpoint file for resume
        logger: Logger instance
        dry_run: If True, simulate without publishing
        validator: Optional SchemaValidator for message validation
    """
    logger.info(f"Starting BATCH mode: {batch_size} records/batch to topic '{topic}'")
    logger.info(f"Dataset size: {len(df)} customers")
    logger.info(f"Mode: {'DRY-RUN' if dry_run else 'LIVE'}")
    logger.info(f"Validation: {'ENABLED' if validator else 'DISABLED'}")
    logger.info(f"Checkpoint file: {checkpoint_file}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file, logger)
    start_row = checkpoint.get('last_row', 0)
    
    if start_row > 0:
        logger.info(f"Resuming from row {start_row}")
    
    total_sent = 0
    total_failed = 0
    validation_failed = 0
    start_time = time.time()
    last_offset = checkpoint.get('last_offset', 0)
    
    try:
        # Process in chunks
        for chunk_start in range(start_row, len(df), batch_size):
            if shutdown_requested:
                break
            
            chunk_end = min(chunk_start + batch_size, len(df))
            chunk = df.iloc[chunk_start:chunk_end]
            
            logger.info(f"Processing batch: rows {chunk_start} to {chunk_end-1} ({len(chunk)} records)")
            
            # Process each record in chunk
            for idx, customer in chunk.iterrows():
                if shutdown_requested:
                    break
                
                customer_id = str(customer['customerID'])
                message = customer_to_message(customer, add_timestamp=True)
                
                success, errors = publish_message(producer, topic, customer_id, message, logger, dry_run, validator)
                
                if success:
                    total_sent += 1
                else:
                    total_failed += 1
                    if errors is not None:
                        validation_failed += 1
            
            # Save checkpoint after each batch
            save_checkpoint(checkpoint_file, chunk_end, last_offset + total_sent, logger)
            
            logger.info(
                f"Batch complete: {total_sent} sent, {total_failed} failed "
                f"(validation: {validation_failed})"
            )
            
            # Small delay between batches
            time.sleep(0.1)
        
        if total_sent == len(df):
            logger.info("All records processed successfully!")
            # Reset checkpoint
            save_checkpoint(checkpoint_file, 0, 0, logger)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("BATCH MODE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total messages sent: {total_sent}")
        logger.info(f"Total failures: {total_failed}")
        if validator:
            logger.info(f"Validation failures: {validation_failed}")
        logger.info(f"Duration: {elapsed:.2f} seconds")
        logger.info(f"Average rate: {total_sent / elapsed:.2f} records/sec")
        logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Kafka Producer for Telco Customer Churn Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Streaming mode (dry-run)
  python src/kafka/producer.py --mode streaming --events-per-sec 5 --dry-run
  
  # Streaming mode (live)
  python src/kafka/producer.py --mode streaming --events-per-sec 1 --broker localhost:19092
  
  # Batch mode with checkpointing
  python src/kafka/producer.py --mode batch --batch-size 100 --checkpoint-file artifacts/checkpoint.json
  
  # Custom dataset and topic
  python src/kafka/producer.py --mode streaming --dataset-path data/custom.csv --topic my.topic
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['streaming', 'batch'],
        help='Producer mode: streaming (continuous random sampling) or batch (sequential processing)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--broker',
        type=str,
        default=DEFAULT_BROKER,
        help=f'Kafka bootstrap server address (default: {DEFAULT_BROKER})'
    )
    
    parser.add_argument(
        '--topic',
        type=str,
        default=DEFAULT_TOPIC,
        help=f'Kafka topic name (default: {DEFAULT_TOPIC})'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=DEFAULT_DATASET,
        help=f'Path to customer dataset CSV (default: {DEFAULT_DATASET})'
    )
    
    # Streaming mode arguments
    parser.add_argument(
        '--events-per-sec',
        type=float,
        default=DEFAULT_EVENTS_PER_SEC,
        help=f'Events per second for streaming mode (default: {DEFAULT_EVENTS_PER_SEC})'
    )
    
    # Batch mode arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for batch mode (default: {DEFAULT_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--checkpoint-file',
        type=str,
        default=DEFAULT_CHECKPOINT_FILE,
        help=f'Checkpoint file for batch mode resume (default: {DEFAULT_CHECKPOINT_FILE})'
    )
    
    # Testing and debugging
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry-run mode: generate messages without publishing to Kafka'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Enable message validation against JSON schema before publishing'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the producer."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 60)
    logger.info("KAFKA PRODUCER - Telco Customer Churn Prediction")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Topic: {args.topic}")
    logger.info(f"Broker: {args.broker}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Dry-run: {args.dry_run}")
    logger.info(f"Validation: {args.validate}")
    logger.info("=" * 60)
    
    producer = None
    validator = None
    
    try:
        # Initialize validator if requested
        if args.validate:
            if not VALIDATION_AVAILABLE:
                logger.error("Validation requested but jsonschema library not available")
                logger.error("Install with: pip install jsonschema>=4.0.0")
                sys.exit(1)
            try:
                validator = SchemaValidator()
                logger.info("Schema validator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize schema validator: {e}")
                sys.exit(1)
        
        # Load dataset
        df = load_dataset(args.dataset_path, logger)
        
        # Create Kafka producer
        producer = create_kafka_producer(args.broker, logger, args.dry_run)
        
        # Run appropriate mode
        if args.mode == 'streaming':
            streaming_mode(
                producer=producer,
                df=df,
                topic=args.topic,
                events_per_sec=args.events_per_sec,
                logger=logger,
                dry_run=args.dry_run,
                validator=validator
            )
        elif args.mode == 'batch':
            batch_mode(
                producer=producer,
                df=df,
                topic=args.topic,
                batch_size=args.batch_size,
                checkpoint_file=args.checkpoint_file,
                logger=logger,
                dry_run=args.dry_run,
                validator=validator
            )
        
    except FileNotFoundError as e:
        logger.error(f"Dataset error: {e}")
        sys.exit(1)
    except NoBrokersAvailable:
        logger.error("Cannot connect to Kafka. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        if producer is not None:
            logger.info("Flushing and closing Kafka producer...")
            producer.flush()
            producer.close()
            logger.info("Producer closed successfully")
        
        logger.info("Producer shutdown complete")


if __name__ == '__main__':
    main()
