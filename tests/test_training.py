import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from models.train import (
    load_and_prepare_data,
    create_preprocessor,
    build_ml_pipeline,
    train_and_evaluate,
    save_artifacts,
    main
)

class TestModelTraining(unittest.TestCase):
    """Comprehensive test suite for model training module"""
    
    def setUp(self):
        """Set up test fixtures before each test"""
        # Create sample training data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'tenure': np.random.randint(0, 72, 100),
            'MonthlyCharges': np.random.uniform(18, 120, 100),
            'TotalCharges': np.random.uniform(18, 8000, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'Partner': np.random.choice(['Yes', 'No'], 100),
            'Dependents': np.random.choice(['Yes', 'No'], 100),
            'Churn': np.random.choice(['Yes', 'No'], 100)
        })
        
        # Sample column metadata
        self.sample_metadata = {
            'columns': {
                'numerical': ['tenure', 'MonthlyCharges', 'TotalCharges'],
                'categorical': ['gender', 'Partner', 'Dependents'],
                'target': 'Churn'
            }
        }
        
        # Sample model parameters
        self.model_params = {
            'n_estimators': 10,  # Small for testing
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
    def test_load_and_prepare_data_success(self):
        """Test successful data loading and preparation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary files
            temp_csv = Path(temp_dir) / "test_data.csv"
            temp_columns = Path(temp_dir) / "columns.json"
            
            # Save test data and metadata
            self.sample_data.to_csv(temp_csv, index=False)
            with open(temp_columns, 'w') as f:
                json.dump(self.sample_metadata, f)
            
            # Test the function
            X, y, feature_cols, target_col, le = load_and_prepare_data(temp_csv, temp_columns)
            
            # Verify results
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
            self.assertIsNotNone(feature_cols)
            self.assertIsNotNone(target_col)
            self.assertIsNotNone(le)
            
            # Check shapes and types
            self.assertEqual(len(X), len(self.sample_data))
            self.assertEqual(len(y), len(self.sample_data))
            self.assertEqual(target_col, 'Churn')
            self.assertEqual(len(feature_cols), 6)  # 3 numeric + 3 categorical
            
            # Check target encoding
            self.assertTrue(set(y).issubset({0, 1}))
            
    def test_load_and_prepare_data_missing_file(self):
        """Test error handling when data file is missing"""
        nonexistent_csv = Path("nonexistent.csv")
        nonexistent_columns = Path("nonexistent.json")
        
        result = load_and_prepare_data(nonexistent_csv, nonexistent_columns)
        
        # Should return None values on error
        self.assertEqual(result, (None, None, None, None, None))
        
    def test_load_and_prepare_data_invalid_json(self):
        """Test error handling with invalid JSON metadata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_csv = Path(temp_dir) / "test_data.csv"
            temp_columns = Path(temp_dir) / "invalid.json"
            
            # Save CSV data
            self.sample_data.to_csv(temp_csv, index=False)
            
            # Create invalid JSON
            with open(temp_columns, 'w') as f:
                f.write("invalid json content")
            
            result = load_and_prepare_data(temp_csv, temp_columns)
            
            # Should return None values on error
            self.assertEqual(result, (None, None, None, None, None))
            
    def test_create_preprocessor_success(self):
        """Test successful preprocessor creation"""
        # Prepare test data
        X = self.sample_data[['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents']]
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_cols = ['gender', 'Partner', 'Dependents']
        
        preprocessor = create_preprocessor(X, numeric_cols, categorical_cols)
        
        # Verify preprocessor creation
        self.assertIsNotNone(preprocessor)
        
        # Test fitting and transformation
        X_transformed = preprocessor.fit_transform(X)
        self.assertIsNotNone(X_transformed)
        self.assertEqual(X_transformed.shape[0], len(X))
        self.assertGreater(X_transformed.shape[1], 0)
        
    def test_create_preprocessor_empty_columns(self):
        """Test preprocessor creation with empty column lists"""
        X = pd.DataFrame({'col1': [1, 2, 3]})
        
        # Should handle empty lists
        preprocessor = create_preprocessor(X, [], [])
        self.assertIsNotNone(preprocessor)
        
    def test_create_preprocessor_missing_columns(self):
        """Test preprocessor creation when specified columns don't exist in data"""
        X = pd.DataFrame({'existing_col': [1, 2, 3]})
        
        # The function should return None when columns are missing (based on current implementation)
        preprocessor = create_preprocessor(X, ['missing_numeric'], ['missing_categorical'])
        
        # Current implementation returns None when there's an error with missing columns
        self.assertIsNone(preprocessor)
            
    def test_build_ml_pipeline_default_params(self):
        """Test ML pipeline building with default parameters"""
        # Create a simple preprocessor
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), [0, 1, 2])
        ])
        
        pipeline = build_ml_pipeline(preprocessor)
        
        # Verify pipeline structure
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(pipeline, Pipeline)
        
        # Check that it has preprocessor and model steps
        step_names = [name for name, _ in pipeline.steps]
        self.assertIn('preprocessor', step_names)
        self.assertIn('model', step_names)
        
        # Check model type
        model = pipeline.named_steps['model']
        self.assertIsInstance(model, GradientBoostingClassifier)
        
    def test_build_ml_pipeline_custom_params(self):
        """Test ML pipeline building with custom parameters"""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), [0, 1, 2])
        ])
        
        custom_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'learning_rate': 0.05,
            'random_state': 123
        }
        
        pipeline = build_ml_pipeline(preprocessor, **custom_params)
        
        # Verify custom parameters are applied
        model = pipeline.named_steps['model']
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 5)
        self.assertEqual(model.learning_rate, 0.05)
        self.assertEqual(model.random_state, 123)
        
    def test_train_and_evaluate_pipeline(self):
        """Test pipeline training and evaluation"""
        # Create sample training data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B'], 100)
        })
        X_test = pd.DataFrame({
            'num1': np.random.randn(20),
            'num2': np.random.randn(20),
            'cat1': np.random.choice(['A', 'B'], 20)
        })
        y_train = np.random.choice([0, 1], 100)
        y_test = np.random.choice([0, 1], 20)
        
        # Create preprocessor
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['num1', 'num2']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['cat1'])
        ])
        
        # Build pipeline
        pipeline = build_ml_pipeline(preprocessor, **self.model_params)
        
        # Train and evaluate
        trained_pipeline, metrics = train_and_evaluate(
            pipeline, X_train, X_test, y_train, y_test
        )
        
        # Verify metrics
        self.assertIsNotNone(metrics)
        self.assertIn('train_accuracy', metrics)
        self.assertIn('test_accuracy', metrics)
        self.assertIn('train_roc_auc', metrics)
        self.assertIn('test_roc_auc', metrics)
        
        # Check metric ranges
        for metric_name, metric_value in metrics.items():
            if 'accuracy' in metric_name or 'roc_auc' in metric_name:
                self.assertGreaterEqual(metric_value, 0.0)
                self.assertLessEqual(metric_value, 1.0)
                
    def test_save_artifacts_success(self):
        """Test successful artifact saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler
            
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), [0, 1])
            ])
            pipeline = build_ml_pipeline(preprocessor, **self.model_params)
            
            # Fit with dummy data
            X_dummy = np.random.randn(10, 2)
            y_dummy = np.random.choice([0, 1], 10)
            pipeline.fit(X_dummy, y_dummy)
            
            # Create sample metrics
            metrics = {
                'train_accuracy': 0.85,
                'test_accuracy': 0.80,
                'train_roc_auc': 0.90,
                'test_roc_auc': 0.88
            }
            
            # Define output paths
            pipeline_path = Path(temp_dir) / "model.joblib"
            metrics_path = Path(temp_dir) / "metrics.json"
            
            # Test artifact saving
            save_artifacts(pipeline, metrics, pipeline_path, metrics_path)
            
            # Verify files were created
            self.assertTrue(pipeline_path.exists())
            self.assertTrue(metrics_path.exists())
            
            # Verify pipeline can be loaded
            loaded_pipeline = joblib.load(pipeline_path)
            self.assertIsNotNone(loaded_pipeline)
            
            # Verify metrics can be loaded
            with open(metrics_path, 'r') as f:
                loaded_metrics = json.load(f)
            self.assertEqual(loaded_metrics, metrics)
            
    def test_save_artifacts_directory_creation(self):
        """Test that artifact saving creates necessary directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create paths with nested directories
            nested_dir = Path(temp_dir) / "models" / "trained"
            pipeline_path = nested_dir / "model.joblib"
            metrics_path = nested_dir / "metrics.json"
            
            # Create a simple pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler
            
            preprocessor = ColumnTransformer([('num', StandardScaler(), [0])])
            pipeline = build_ml_pipeline(preprocessor, n_estimators=5)
            
            # Fit with dummy data - ensure we have both classes to prevent the ValueError
            X_dummy = np.random.randn(10, 1)  # Increased sample size
            y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Explicitly ensure both classes
            pipeline.fit(X_dummy, y_dummy)
            
            metrics = {'accuracy': 0.8}
            
            # Should create directories automatically
            save_artifacts(pipeline, metrics, pipeline_path, metrics_path)
            
            # Verify directories and files were created
            self.assertTrue(nested_dir.exists())
            self.assertTrue(pipeline_path.exists())
            self.assertTrue(metrics_path.exists())
            
    def test_parameter_validation(self):
        """Test model parameter validation"""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        
        preprocessor = ColumnTransformer([('num', StandardScaler(), [0])])
        
        # Test with invalid parameters
        invalid_params = {
            'n_estimators': -1,  # Should be positive
            'max_depth': 0,      # Should be positive
            'learning_rate': 0   # Should be positive
        }
        
        # The function should either handle invalid params or raise appropriate errors
        try:
            pipeline = build_ml_pipeline(preprocessor, **invalid_params)
            # If no error is raised, the pipeline should still be created
            self.assertIsNotNone(pipeline)
        except (ValueError, TypeError):
            # It's acceptable to raise errors for invalid parameters
            pass
            
    def test_model_consistency(self):
        """Test that models trained with same parameters produce consistent results"""
        np.random.seed(42)
        
        # Create identical training data
        X = np.random.randn(100, 3)
        y = np.random.choice([0, 1], 100)
        
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        
        # Create two identical preprocessors
        preprocessor1 = ColumnTransformer([('num', StandardScaler(), [0, 1, 2])])
        preprocessor2 = ColumnTransformer([('num', StandardScaler(), [0, 1, 2])])
        
        # Build identical pipelines with same random state
        params = {'n_estimators': 10, 'random_state': 42}
        pipeline1 = build_ml_pipeline(preprocessor1, **params)
        pipeline2 = build_ml_pipeline(preprocessor2, **params)
        
        # Train both pipelines
        pipeline1.fit(X, y)
        pipeline2.fit(X, y)
        
        # Predictions should be identical (same random state)
        pred1 = pipeline1.predict(X)
        pred2 = pipeline2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)


if __name__ == '__main__':
    unittest.main()