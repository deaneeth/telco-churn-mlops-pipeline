import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from data.preprocess import (
    build_preprocessor, 
    fit_save_preprocessor, 
    load_preprocessor,
    transform_data,
    main
)

class TestDataPreprocessing(unittest.TestCase):
    """Comprehensive test suite for data preprocessing module"""
    
    def setUp(self):
        """Set up test fixtures before each test"""
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'tenure': [1, 34, 2, 45, np.nan],
            'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
            'TotalCharges': ['29.85', '1889.5', '108.15', '1840.75', ' '],  # Mix of strings and spaces
            'gender': ['Female', 'Male', 'Male', 'Male', 'Female'],
            'Partner': ['Yes', 'No', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'No', 'No', 'No', 'No'],
            'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
        })
        
        # Sample column metadata
        self.sample_metadata = {
            'columns': {
                'numerical': ['tenure', 'MonthlyCharges', 'TotalCharges'],
                'categorical': ['gender', 'Partner', 'Dependents'],
                'target': 'Churn'
            }
        }
        
    def test_build_preprocessor_valid_input(self):
        """Test preprocessor building with valid column lists"""
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_cols = ['gender', 'Partner', 'Dependents']
        
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        
        # Check that preprocessor is created successfully
        self.assertIsNotNone(preprocessor)
        self.assertEqual(len(preprocessor.transformers), 2)  # numeric + categorical
        
        # Verify transformer names
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        self.assertIn('numeric', transformer_names)
        self.assertIn('categorical', transformer_names)
        
    def test_build_preprocessor_empty_columns(self):
        """Test preprocessor building with empty column lists"""
        # Should handle empty lists gracefully
        preprocessor = build_preprocessor([], [])
        self.assertIsNotNone(preprocessor)
        
    def test_build_preprocessor_single_column_type(self):
        """Test preprocessor with only numeric or only categorical columns"""
        # Only numeric columns
        preprocessor_numeric = build_preprocessor(['tenure'], [])
        self.assertIsNotNone(preprocessor_numeric)
        
        # Only categorical columns  
        preprocessor_categorical = build_preprocessor([], ['gender'])
        self.assertIsNotNone(preprocessor_categorical)
        
    def test_preprocessor_fit_transform(self):
        """Test that preprocessor can fit and transform data correctly"""
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_cols = ['gender', 'Partner', 'Dependents']
        
        # Prepare data (handle TotalCharges conversion)
        test_data = self.sample_data.copy()
        test_data['TotalCharges'] = pd.to_numeric(test_data['TotalCharges'], errors='coerce')
        
        X = test_data[numeric_cols + categorical_cols]
        
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        X_transformed = preprocessor.fit_transform(X)
        
        # Check output shape
        self.assertEqual(X_transformed.shape[0], len(test_data))
        self.assertGreater(X_transformed.shape[1], 0)  # Should have features
        
        # Check that output is numeric
        self.assertTrue(np.isreal(X_transformed).all())
        
    def test_missing_data_handling(self):
        """Test that preprocessor handles missing data correctly"""
        # Create data with various missing patterns
        data_with_missing = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', None, 'A', 'C']
        })
        
        preprocessor = build_preprocessor(['numeric_col'], ['categorical_col'])
        X_transformed = preprocessor.fit_transform(data_with_missing)
        
        # Should not have any NaN values after preprocessing
        self.assertFalse(np.isnan(X_transformed).any())
        
    def test_fit_save_preprocessor_success(self):
        """Test successful preprocessor fitting and saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary files
            temp_csv = Path(temp_dir) / "test_data.csv"
            temp_columns = Path(temp_dir) / "columns.json"
            temp_preprocessor = Path(temp_dir) / "preprocessor.joblib"
            
            # Save test data and metadata
            self.sample_data.to_csv(temp_csv, index=False)
            with open(temp_columns, 'w') as f:
                json.dump(self.sample_metadata, f)
            
            # Test the function with explicit columns path
            preprocessor, feature_names, X_transformed = fit_save_preprocessor(
                temp_csv, temp_preprocessor, temp_columns
            )
            
            # Verify results
            self.assertIsNotNone(preprocessor)
            self.assertIsNotNone(feature_names)
            self.assertIsNotNone(X_transformed)
            self.assertEqual(X_transformed.shape[0], len(self.sample_data))
                
    def test_fit_save_preprocessor_missing_file(self):
        """Test error handling when input file is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_csv = Path(temp_dir) / "nonexistent.csv"
            temp_preprocessor = Path(temp_dir) / "preprocessor.joblib"
            
            # Function returns None tuple on error instead of raising
            result = fit_save_preprocessor(nonexistent_csv, temp_preprocessor)
            self.assertEqual(result, (None, None, None))
                
    def test_fit_save_preprocessor_missing_metadata(self):
        """Test error handling when columns metadata is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_csv = Path(temp_dir) / "test_data.csv"
            temp_preprocessor = Path(temp_dir) / "preprocessor.joblib"
            nonexistent_columns = Path(temp_dir) / "nonexistent_columns.json"
            
            # Save only the CSV, not the metadata
            self.sample_data.to_csv(temp_csv, index=False)
            
            # Function returns None tuple on error instead of raising
            result = fit_save_preprocessor(temp_csv, temp_preprocessor, nonexistent_columns)
            self.assertEqual(result, (None, None, None))
                
    def test_totalcharges_conversion(self):
        """Test proper handling of TotalCharges data quality issue"""
        # Test data with string TotalCharges including spaces
        test_data = pd.DataFrame({
            'TotalCharges': ['29.85', '1889.5', ' ', '1840.75', ''],
            'tenure': [1, 34, 2, 45, 12],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male']
        })
        
        # Convert TotalCharges as the preprocessor would
        test_data['TotalCharges'] = pd.to_numeric(test_data['TotalCharges'], errors='coerce')
        
        # Check that spaces and empty strings become NaN
        self.assertTrue(pd.isna(test_data.loc[2, 'TotalCharges']))
        self.assertTrue(pd.isna(test_data.loc[4, 'TotalCharges']))
        
        # Check that valid strings are converted to float
        self.assertEqual(test_data.loc[0, 'TotalCharges'], 29.85)
        self.assertEqual(test_data.loc[1, 'TotalCharges'], 1889.5)
        
    def test_create_sample_data_function(self):
        """Test the create_sample_data function if it exists"""
        try:
            # Try to import the function dynamically
            from data.preprocess import create_sample_data
            sample_df = create_sample_data()
            self.assertIsInstance(sample_df, pd.DataFrame)
            self.assertGreater(len(sample_df), 0)
        except (ImportError, AttributeError, NameError):
            # Function doesn't exist, which is fine
            self.skipTest("create_sample_data function not implemented")
            
    def test_column_type_validation(self):
        """Test that preprocessor handles different column types correctly"""
        # Test with mixed data types
        mixed_data = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['A', 'B', 'C', 'D', 'E'],
            'bool_col': [True, False, True, False, True]
        })
        
        numeric_cols = ['int_col', 'float_col']
        categorical_cols = ['str_col', 'bool_col']
        
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        X_transformed = preprocessor.fit_transform(mixed_data)
        
        # Should handle all data types without error
        self.assertIsNotNone(X_transformed)
        self.assertEqual(X_transformed.shape[0], len(mixed_data))
        
    def test_feature_names_generation(self):
        """Test that feature names are generated correctly after transformation"""
        numeric_cols = ['tenure', 'MonthlyCharges']
        categorical_cols = ['gender', 'Partner']
        
        # Prepare clean test data
        test_data = pd.DataFrame({
            'tenure': [1, 2, 3],
            'MonthlyCharges': [30.0, 40.0, 50.0],
            'gender': ['Male', 'Female', 'Male'],
            'Partner': ['Yes', 'No', 'Yes']
        })
        
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        X_transformed = preprocessor.fit_transform(test_data)
        
        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
            self.assertIsInstance(feature_names, (list, np.ndarray))
            self.assertEqual(len(feature_names), X_transformed.shape[1])
        except AttributeError:
            # Older sklearn versions might not have this method
            self.skipTest("get_feature_names_out not available in this sklearn version")


if __name__ == '__main__':
    unittest.main()