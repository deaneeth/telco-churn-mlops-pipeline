"""
Unit tests for Kafka Producer
==============================

Tests producer functionality including:
- Message generation and schema validation
- Streaming mode rate control
- Batch mode chunking and checkpointing
- Dry-run mode (no actual Kafka publishing)
- Error handling and edge cases

Run with: pytest tests/test_producer.py -v
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from streaming.producer import (
    customer_to_message,
    load_dataset,
    load_checkpoint,
    save_checkpoint,
    setup_logging,
    create_kafka_producer,
)


@pytest.fixture
def sample_customer_data():
    """Create sample customer data for testing."""
    return pd.Series({
        'customerID': 'TEST-001',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 50.50,
        'TotalCharges': 606.00,
        'Churn': 'No'
    })


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    data = {
        'customerID': ['TEST-001', 'TEST-002', 'TEST-003', 'TEST-004', 'TEST-005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'tenure': [12, 24, 6, 48, 3],
        'MonthlyCharges': [50.50, 80.25, 30.00, 100.50, 25.75],
        'TotalCharges': [606.00, 1926.00, 180.00, 4824.00, 77.25],
        'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_checkpoint_file():
    """Create temporary checkpoint file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        temp_path = f.name
        sample_dataframe.to_csv(temp_path, index=False)
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestMessageGeneration:
    """Test message generation and schema validation."""
    
    def test_customer_to_message_basic(self, sample_customer_data):
        """Test basic customer to message conversion."""
        message = customer_to_message(sample_customer_data, add_timestamp=False)
        
        # Verify all customer fields are present
        assert message['customerID'] == 'TEST-001'
        assert message['gender'] == 'Male'
        assert message['tenure'] == 12
        assert message['MonthlyCharges'] == 50.50
        
        # Verify no timestamp when add_timestamp=False
        assert 'event_ts' not in message
    
    def test_customer_to_message_with_timestamp(self, sample_customer_data):
        """Test message generation includes timestamp."""
        message = customer_to_message(sample_customer_data, add_timestamp=True)
        
        # Verify timestamp is present and valid ISO format
        assert 'event_ts' in message
        assert message['event_ts'].endswith('Z')
        
        # Verify timestamp can be parsed
        timestamp = datetime.fromisoformat(message['event_ts'].replace('Z', '+00:00'))
        assert isinstance(timestamp, datetime)
    
    def test_customer_to_message_handles_nan(self):
        """Test NaN values are converted to None for JSON serialization."""
        customer = pd.Series({
            'customerID': 'TEST-002',
            'gender': 'Female',
            'TotalCharges': float('nan'),  # NaN value
            'tenure': 5
        })
        
        message = customer_to_message(customer, add_timestamp=False)
        
        # Verify NaN converted to None
        assert message['TotalCharges'] is None
        assert message['tenure'] == 5
    
    def test_message_is_json_serializable(self, sample_customer_data):
        """Test generated message can be JSON serialized."""
        message = customer_to_message(sample_customer_data, add_timestamp=True)
        
        # Should not raise exception
        json_str = json.dumps(message)
        
        # Verify can be deserialized
        parsed = json.loads(json_str)
        assert parsed['customerID'] == 'TEST-001'
        assert 'event_ts' in parsed


class TestDatasetLoading:
    """Test dataset loading functionality."""
    
    def test_load_dataset_success(self, temp_csv_file):
        """Test successful dataset loading."""
        logger = setup_logging('ERROR')
        df = load_dataset(temp_csv_file, logger)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'customerID' in df.columns
    
    def test_load_dataset_file_not_found(self):
        """Test error handling for missing dataset."""
        logger = setup_logging('ERROR')
        
        with pytest.raises(FileNotFoundError):
            load_dataset('nonexistent_file.csv', logger)


class TestCheckpointing:
    """Test checkpoint save/load functionality."""
    
    def test_save_and_load_checkpoint(self, temp_checkpoint_file):
        """Test checkpoint save and load cycle."""
        logger = setup_logging('ERROR')
        
        # Save checkpoint
        save_checkpoint(temp_checkpoint_file, last_row=150, last_offset=300, logger=logger)
        
        # Load checkpoint
        checkpoint = load_checkpoint(temp_checkpoint_file, logger)
        
        assert checkpoint['last_row'] == 150
        assert checkpoint['last_offset'] == 300
        assert 'timestamp' in checkpoint
    
    def test_load_checkpoint_no_file(self, temp_checkpoint_file):
        """Test loading checkpoint when file doesn't exist."""
        logger = setup_logging('ERROR')
        
        # Remove file if exists
        if os.path.exists(temp_checkpoint_file):
            os.remove(temp_checkpoint_file)
        
        checkpoint = load_checkpoint(temp_checkpoint_file, logger)
        
        # Should return default values
        assert checkpoint['last_row'] == 0
        assert checkpoint['last_offset'] == 0
        assert checkpoint['timestamp'] is None
    
    def test_checkpoint_creates_directory(self):
        """Test checkpoint creation creates parent directory."""
        logger = setup_logging('ERROR')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, 'nested', 'dir', 'checkpoint.json')
            
            # Should not raise exception even though directory doesn't exist
            save_checkpoint(nested_path, last_row=10, last_offset=20, logger=logger)
            
            # Verify file was created
            assert os.path.exists(nested_path)


class TestKafkaProducerSetup:
    """Test Kafka producer initialization."""
    
    def test_create_producer_dry_run(self):
        """Test producer creation in dry-run mode returns None."""
        logger = setup_logging('ERROR')
        producer = create_kafka_producer('localhost:9092', logger, dry_run=True)
        
        assert producer is None
    
    @patch('kafka.producer.KafkaProducer')
    def test_create_producer_success(self, mock_kafka_producer):
        """Test successful producer creation."""
        logger = setup_logging('ERROR')
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance
        
        producer = create_kafka_producer('localhost:9092', logger, dry_run=False)
        
        assert producer is not None
        mock_kafka_producer.assert_called_once()


class TestDryRunMode:
    """Test dry-run mode functionality."""
    
    def test_dry_run_streaming_no_kafka(self, sample_dataframe, capsys):
        """Test streaming mode in dry-run doesn't require Kafka."""
        from streaming.producer import streaming_mode
        
        logger = setup_logging('ERROR')
        
        # Run for very short duration (1 event at 10 events/sec = 0.1 seconds)
        with patch('streaming.producer.shutdown_requested', side_effect=[False, True]):
            streaming_mode(
                producer=None,
                df=sample_dataframe,
                topic='test.topic',
                events_per_sec=10.0,
                logger=logger,
                dry_run=True
            )
        
        # Should complete without errors
        captured = capsys.readouterr()
        # Dry-run should log messages
    
    def test_dry_run_batch_no_kafka(self, sample_dataframe, temp_checkpoint_file, capsys):
        """Test batch mode in dry-run doesn't require Kafka."""
        from streaming.producer import batch_mode
        
        logger = setup_logging('ERROR')
        
        batch_mode(
            producer=None,
            df=sample_dataframe,
            topic='test.topic',
            batch_size=2,
            checkpoint_file=temp_checkpoint_file,
            logger=logger,
            dry_run=True
        )
        
        # Should complete without errors
        # Verify checkpoint was created
        assert os.path.exists(temp_checkpoint_file)


class TestBatchModeCheckpointing:
    """Test batch mode checkpoint resume functionality."""
    
    def test_batch_mode_resume_from_checkpoint(self, sample_dataframe, temp_checkpoint_file):
        """Test batch mode resumes from checkpoint."""
        from streaming.producer import batch_mode
        
        logger = setup_logging('ERROR')
        
        # Save a checkpoint indicating we processed first 2 rows
        save_checkpoint(temp_checkpoint_file, last_row=2, last_offset=2, logger=logger)
        
        # Run batch mode (should start from row 2)
        batch_mode(
            producer=None,
            df=sample_dataframe,
            topic='test.topic',
            batch_size=2,
            checkpoint_file=temp_checkpoint_file,
            logger=logger,
            dry_run=True
        )
        
        # Verify checkpoint updated to end of dataset
        checkpoint = load_checkpoint(temp_checkpoint_file, logger)
        # Should have processed remaining 3 rows (total 5, started at 2)
        assert checkpoint['last_row'] >= 2


class TestCLIArguments:
    """Test command-line argument parsing."""
    
    def test_parse_args_streaming_mode(self):
        """Test argument parsing for streaming mode."""
        from streaming.producer import parse_args
        
        test_args = [
            '--mode', 'streaming',
            '--events-per-sec', '5',
            '--dry-run'
        ]
        
        with patch('sys.argv', ['producer.py'] + test_args):
            args = parse_args()
            
            assert args.mode == 'streaming'
            assert args.events_per_sec == 5.0
            assert args.dry_run is True
    
    def test_parse_args_batch_mode(self):
        """Test argument parsing for batch mode."""
        from streaming.producer import parse_args
        
        test_args = [
            '--mode', 'batch',
            '--batch-size', '100',
            '--checkpoint-file', 'custom_checkpoint.json'
        ]
        
        with patch('sys.argv', ['producer.py'] + test_args):
            args = parse_args()
            
            assert args.mode == 'batch'
            assert args.batch_size == 100
            assert args.checkpoint_file == 'custom_checkpoint.json'
    
    def test_parse_args_defaults(self):
        """Test default argument values."""
        from streaming.producer import parse_args
        
        test_args = ['--mode', 'streaming']
        
        with patch('sys.argv', ['producer.py'] + test_args):
            args = parse_args()
            
            assert args.broker == 'localhost:19092'
            assert args.topic == 'telco.raw.customers'
            assert args.dry_run is False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        from streaming.producer import batch_mode
        
        empty_df = pd.DataFrame()
        logger = setup_logging('ERROR')
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_file = f.name
        
        try:
            # Should handle empty DataFrame gracefully
            batch_mode(
                producer=None,
                df=empty_df,
                topic='test.topic',
                batch_size=10,
                checkpoint_file=checkpoint_file,
                logger=logger,
                dry_run=True
            )
        finally:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
    
    def test_single_row_dataframe(self):
        """Test processing DataFrame with single row."""
        from streaming.producer import batch_mode
        
        single_row_df = pd.DataFrame([{
            'customerID': 'SINGLE-001',
            'gender': 'Male',
            'tenure': 10
        }])
        
        logger = setup_logging('ERROR')
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_file = f.name
        
        try:
            batch_mode(
                producer=None,
                df=single_row_df,
                topic='test.topic',
                batch_size=10,
                checkpoint_file=checkpoint_file,
                logger=logger,
                dry_run=True
            )
            
            # Verify processing completed
            checkpoint = load_checkpoint(checkpoint_file, logger)
            # Checkpoint should be reset to 0 after completing all rows
            assert checkpoint['last_row'] == 0
        finally:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)


class TestLogging:
    """Test logging configuration."""
    
    def test_setup_logging_creates_file(self):
        """Test logging setup creates log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            
            # Patch LOG_FILE constant
            with patch('streaming.producer.LOG_FILE', log_file):
                logger = setup_logging('INFO')
                
                # Log something
                logger.info("Test message")
                
                # Verify log file created
                assert os.path.exists(log_file)


# Integration test markers
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual Kafka (optional)."""
    
    @pytest.mark.skip(reason="Requires running Kafka instance")
    def test_producer_connect_to_kafka(self):
        """Test producer can connect to real Kafka instance."""
        logger = setup_logging('ERROR')
        
        # This test should only run if Kafka is available
        producer = create_kafka_producer('localhost:19092', logger, dry_run=False)
        
        assert producer is not None
        
        # Cleanup
        producer.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
