"""
Comprehensive pytest fixtures and configuration for Telco Churn Prediction project

This module provides shared fixtures, test utilities, and configuration
for the entire test suite. It ensures consistent test setup and teardown
across all test modules.

Author: AI Assistant
Created: 2024-12-28
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global test configuration
TEST_DATA_SIZE = 1000
RANDOM_SEED = 42
TEST_TIMEOUT = 300  # 5 minutes

@pytest.fixture(scope="session")
def test_config():
    """Session-level test configuration"""
    return {
        'data_size': TEST_DATA_SIZE,
        'random_seed': RANDOM_SEED,
        'timeout': TEST_TIMEOUT,
        'test_data_dir': os.path.join(os.path.dirname(__file__), 'test_data'),
        'artifacts_dir': os.path.join(os.path.dirname(__file__), 'test_artifacts')
    }

@pytest.fixture(scope="session")
def test_data_directory(test_config):
    """Create and manage test data directory"""
    test_dir = test_config['test_data_dir']
    
    # Create test directory
    os.makedirs(test_dir, exist_ok=True)
    
    yield test_dir
    
    # Cleanup after all tests
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.fixture(scope="session")
def sample_telco_data():
    """Generate realistic sample telco churn data for testing"""
    np.random.seed(RANDOM_SEED)
    
    n_samples = TEST_DATA_SIZE
    
    # Generate base features
    data = {
        'customerID': [f'CUST_{i:06d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.4, 0.3]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.2, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples, p=[0.3, 0.2, 0.25, 0.25]),
    }
    
    # Generate correlated numerical features
    monthly_charges = np.random.normal(65, 20, n_samples)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)  # Realistic bounds
    
    total_charges = monthly_charges * data['tenure'] + np.random.normal(0, 100, n_samples)
    total_charges = np.maximum(total_charges, 0)  # No negative charges
    
    data['MonthlyCharges'] = monthly_charges
    data['TotalCharges'] = total_charges
    
    # Generate churn target with realistic correlations
    churn_prob = (
        0.1 +  # Base churn rate
        0.3 * (data['Contract'] == 'Month-to-month') +  # Higher for month-to-month
        0.2 * (monthly_charges > 80) +  # Higher for expensive plans
        0.15 * (data['tenure'] < 12) +  # Higher for new customers
        0.1 * (data['SeniorCitizen'] == 1) +  # Slightly higher for seniors
        0.05 * (data['PaymentMethod'] == 'Electronic check') -  # Electronic check issues
        0.15 * (data['Contract'] == 'Two year') -  # Lower for long contracts
        0.1 * (data['Partner'] == 'Yes')  # Lower for partnered customers
    )
    
    churn_prob = np.clip(churn_prob, 0, 1)
    data['Churn'] = np.random.binomial(1, churn_prob, n_samples)
    
    return pd.DataFrame(data)

@pytest.fixture
def processed_telco_data(sample_telco_data):
    """Preprocessed telco data ready for model training"""
    df = sample_telco_data.copy()
    
    # Convert TotalCharges to numeric (handle spaces like real data)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Create dummy variables for categorical features
    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    return df_encoded

@pytest.fixture
def train_test_data(processed_telco_data):
    """Split processed data into train/test sets"""
    df = processed_telco_data.copy()
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['customerID', 'Churn']]
    X = df[feature_cols]
    y = df['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_cols
    }

@pytest.fixture
def mock_trained_model():
    """Mock trained model for testing"""
    model = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
    
    # Create simple training data
    X_mock = np.random.rand(100, 19)  # 19 features
    y_mock = np.random.choice([0, 1], 100)
    
    model.fit(X_mock, y_mock)
    
    return model

@pytest.fixture
def mock_preprocessor():
    """Mock data preprocessor for testing"""
    scaler = StandardScaler()
    
    # Fit on mock data
    X_mock = np.random.rand(100, 19)
    scaler.fit(X_mock)
    
    return scaler

@pytest.fixture
def temp_model_directory():
    """Temporary directory for model artifacts"""
    temp_dir = tempfile.mkdtemp(prefix='test_models_')
    
    yield temp_dir
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def temp_data_directory():
    """Temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix='test_data_')
    
    yield temp_dir
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def sample_prediction_features():
    """Sample features for prediction testing"""
    return {
        "SeniorCitizen": 0,
        "MonthlyCharges": 65.0,
        "TotalCharges": 1200.0,
        "gender_Male": 1,
        "Partner_Yes": 0,
        "Dependents_Yes": 0,
        "PhoneService_Yes": 1,
        "MultipleLines_Yes": 0,
        "InternetService_DSL": 1,
        "InternetService_Fiber optic": 0,
        "OnlineSecurity_Yes": 0,
        "OnlineBackup_Yes": 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_Yes": 0,
        "StreamingTV_Yes": 0,
        "StreamingMovies_Yes": 0,
        "Contract_One year": 0,
        "Contract_Two year": 0,
        "PaymentMethod_Credit card (automatic)": 0,
        "PaymentMethod_Electronic check": 1
    }

@pytest.fixture
def sample_batch_features(sample_prediction_features):
    """Sample batch of features for batch prediction testing"""
    batch = []
    
    for i in range(4):
        features = sample_prediction_features.copy()
        
        # Vary some features for diversity
        if i == 1:
            features['MonthlyCharges'] = 85.0
            features['SeniorCitizen'] = 1
        elif i == 2:
            features['Contract_One year'] = 1
            features['Contract_Two year'] = 0
        elif i == 3:
            features['PaymentMethod_Electronic check'] = 0
            features['PaymentMethod_Credit card (automatic)'] = 1
        
        batch.append(features)
    
    return batch

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Auto-use fixture to set up test environment for each test"""
    
    # Set environment variables
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    
    yield
    
    # Cleanup after each test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

@pytest.fixture
def mock_file_system():
    """Mock file system operations for testing"""
    with patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs, \
         patch('joblib.dump') as mock_dump, \
         patch('joblib.load') as mock_load:
        
        mock_exists.return_value = True
        
        yield {
            'exists': mock_exists,
            'makedirs': mock_makedirs,
            'dump': mock_dump,
            'load': mock_load
        }

@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing"""
    import time
    
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return PerformanceTimer()

# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom settings"""
    logger.info("Configuring pytest for Telco Churn Prediction tests...")
    
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "api: API integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test file names
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        else:
            item.add_marker(pytest.mark.unit)

def pytest_runtest_setup(item):
    """Setup for each test run"""
    logger.info(f"Setting up test: {item.name}")

def pytest_runtest_teardown(item, nextitem):
    """Teardown after each test run"""
    logger.info(f"Tearing down test: {item.name}")

def pytest_sessionstart(session):
    """Called after the Session object has been created"""
    logger.info("=" * 60)
    logger.info("🚀 STARTING TELCO CHURN PREDICTION TEST SUITE")
    logger.info("=" * 60)

def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished"""
    logger.info("=" * 60)
    if exitstatus == 0:
        logger.info("✅ ALL TESTS PASSED SUCCESSFULLY")
    else:
        logger.info(f"❌ TEST SUITE FAILED (exit code: {exitstatus})")
    logger.info("=" * 60)

# Custom pytest markers for test categorization
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning")
]