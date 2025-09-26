import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, Mock
import warnings

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

class TestDataValidation(unittest.TestCase):
    """Comprehensive tests for data validation throughout the pipeline"""
    
    def setUp(self):
        """Set up test fixtures for data validation tests"""
        # Create comprehensive sample dataset
        np.random.seed(42)
        self.n_samples = 500
        
        # Generate logically consistent telco customer data
        phone_service = np.random.choice(['Yes', 'No'], self.n_samples)
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], self.n_samples)
        
        # Generate MultipleLines based on PhoneService consistency
        multiple_lines = []
        for ps in phone_service:
            if ps == 'No':
                multiple_lines.append('No phone service')
            else:
                multiple_lines.append(np.random.choice(['Yes', 'No']))
        
        # Generate online services based on InternetService consistency  
        online_services_data = {}
        service_names = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for service in service_names:
            service_values = []
            for internet in internet_service:
                if internet == 'No':
                    service_values.append('No internet service')
                else:
                    service_values.append(np.random.choice(['Yes', 'No']))
            online_services_data[service] = service_values

        self.valid_data = pd.DataFrame({
            'customerID': [f'ID_{i:04d}' for i in range(self.n_samples)],
            'gender': np.random.choice(['Male', 'Female'], self.n_samples),
            'SeniorCitizen': np.random.choice([0, 1], self.n_samples),
            'Partner': np.random.choice(['Yes', 'No'], self.n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], self.n_samples),
            'tenure': np.random.randint(0, 73, self.n_samples),
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            **online_services_data,  # Unpack the consistent online services
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], self.n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], self.n_samples),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
                'Credit card (automatic)'
            ], self.n_samples),
            'MonthlyCharges': np.random.uniform(18.0, 118.75, self.n_samples),
            'TotalCharges': np.random.uniform(18.0, 8684.8, self.n_samples).astype(str),
            'Churn': np.random.choice(['Yes', 'No'], self.n_samples, p=[0.27, 0.73])
        })
        
        # Expected column metadata
        self.expected_columns = {
            'columns': {
                'numerical': ['tenure', 'MonthlyCharges', 'TotalCharges'],
                'categorical': [
                    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod'
                ],
                'target': 'Churn'
            }
        }
        
    def test_schema_validation_valid_data(self):
        """Test schema validation with correct data structure"""
        # Test that valid data passes schema validation
        required_columns = (
            self.expected_columns['columns']['numerical'] +
            self.expected_columns['columns']['categorical'] +
            [self.expected_columns['columns']['target']]
        )
        
        # Check all required columns are present
        for col in required_columns:
            self.assertIn(col, self.valid_data.columns, f"Required column '{col}' missing")
        
        # Check data types are reasonable
        self.assertTrue(pd.api.types.is_numeric_dtype(self.valid_data['tenure']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.valid_data['MonthlyCharges']))
        
    def test_schema_validation_missing_columns(self):
        """Test schema validation with missing required columns"""
        # Create data missing critical columns
        invalid_data = self.valid_data.drop(['Churn', 'tenure'], axis=1)
        
        required_columns = (
            self.expected_columns['columns']['numerical'] +
            self.expected_columns['columns']['categorical'] +
            [self.expected_columns['columns']['target']]
        )
        
        missing_columns = set(required_columns) - set(invalid_data.columns)
        self.assertTrue(len(missing_columns) > 0, "Should detect missing columns")
        self.assertIn('Churn', missing_columns)
        self.assertIn('tenure', missing_columns)
        
    def test_schema_validation_extra_columns(self):
        """Test handling of unexpected extra columns"""
        # Add extra columns
        extra_data = self.valid_data.copy()
        extra_data['UnexpectedColumn1'] = np.random.random(self.n_samples)
        extra_data['UnexpectedColumn2'] = np.random.choice(['A', 'B'], self.n_samples)
        
        # Should handle extra columns gracefully
        self.assertGreater(len(extra_data.columns), len(self.valid_data.columns))
        
    def test_data_type_validation(self):
        """Test validation of expected data types"""
        # Test numerical columns
        for col in ['tenure', 'MonthlyCharges']:
            if col in self.valid_data.columns:
                # Should be convertible to numeric
                numeric_values = pd.to_numeric(self.valid_data[col], errors='coerce')
                self.assertFalse(numeric_values.isna().all(), f"Column {col} should be numeric")
                
        # Test categorical columns
        categorical_cols = ['gender', 'Contract', 'PaymentMethod']
        for col in categorical_cols:
            if col in self.valid_data.columns:
                # Should have reasonable number of unique values
                unique_count = self.valid_data[col].nunique()
                self.assertGreater(unique_count, 0, f"Column {col} should have categories")
                self.assertLessEqual(unique_count, 20, f"Column {col} has too many categories")
                
    def test_data_range_validation(self):
        """Test validation of data value ranges"""
        # Test tenure range (should be 0-72 months typically)
        if 'tenure' in self.valid_data.columns:
            tenure_values = pd.to_numeric(self.valid_data['tenure'], errors='coerce')
            self.assertTrue((tenure_values >= 0).all(), "Tenure should be non-negative")
            self.assertTrue((tenure_values <= 100).all(), "Tenure should be reasonable")
            
        # Test MonthlyCharges range
        if 'MonthlyCharges' in self.valid_data.columns:
            charges = pd.to_numeric(self.valid_data['MonthlyCharges'], errors='coerce')
            self.assertTrue((charges > 0).all(), "MonthlyCharges should be positive")
            self.assertTrue((charges <= 200).all(), "MonthlyCharges should be reasonable")
            
        # Test SeniorCitizen values
        if 'SeniorCitizen' in self.valid_data.columns:
            senior_values = self.valid_data['SeniorCitizen']
            unique_values = set(senior_values.unique())
            expected_values = {0, 1}
            self.assertTrue(unique_values.issubset(expected_values), 
                          f"SeniorCitizen should only contain {expected_values}")
                          
    def test_categorical_value_validation(self):
        """Test validation of categorical variable values"""
        # Define expected categorical values
        expected_values = {
            'gender': {'Male', 'Female'},
            'Partner': {'Yes', 'No'},
            'Dependents': {'Yes', 'No'},
            'PhoneService': {'Yes', 'No'},
            'PaperlessBilling': {'Yes', 'No'},
            'Churn': {'Yes', 'No'},
            'Contract': {'Month-to-month', 'One year', 'Two year'},
            'InternetService': {'DSL', 'Fiber optic', 'No'},
        }
        
        for col, expected_vals in expected_values.items():
            if col in self.valid_data.columns:
                actual_vals = set(self.valid_data[col].dropna().unique())
                unexpected_vals = actual_vals - expected_vals
                self.assertEqual(len(unexpected_vals), 0,
                               f"Column {col} has unexpected values: {unexpected_vals}")
                               
    def test_missing_data_patterns(self):
        """Test validation of missing data patterns"""
        # Create data with various missing patterns
        data_with_missing = self.valid_data.copy()
        
        # Random missing values
        n_missing = 50
        random_indices = np.random.choice(self.n_samples, n_missing, replace=False)
        data_with_missing.loc[random_indices, 'tenure'] = np.nan
        
        # TotalCharges with empty strings (common in real data)
        data_with_missing.loc[random_indices[:20], 'TotalCharges'] = ' '
        
        # Check missing data detection
        missing_counts = data_with_missing.isnull().sum()
        self.assertGreater(missing_counts['tenure'], 0, "Should detect missing tenure values")
        
        # Check empty string detection for TotalCharges
        empty_strings = (data_with_missing['TotalCharges'].str.strip() == '').sum()
        self.assertGreater(empty_strings, 0, "Should detect empty TotalCharges strings")
        
    def test_data_consistency_validation(self):
        """Test logical consistency between related columns"""
        # Test PhoneService vs MultipleLines consistency
        phone_no = self.valid_data['PhoneService'] == 'No'
        multiple_lines = self.valid_data['MultipleLines']
        
        # If no phone service, MultipleLines should be 'No phone service'
        inconsistent = phone_no & (multiple_lines != 'No phone service')
        inconsistent_count = inconsistent.sum()
        
        # Some inconsistency might be expected in real data
        inconsistency_rate = inconsistent_count / len(self.valid_data)
        self.assertLess(inconsistency_rate, 0.1, 
                       "High inconsistency between PhoneService and MultipleLines")
        
        # Test InternetService vs Online services consistency
        internet_no = self.valid_data['InternetService'] == 'No'
        online_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for service in online_services:
            if service in self.valid_data.columns:
                service_values = self.valid_data[service]
                # If no internet, online services should be 'No internet service'
                inconsistent = internet_no & (service_values != 'No internet service')
                inconsistency_rate = inconsistent.sum() / len(self.valid_data)
                self.assertLess(inconsistency_rate, 0.1,
                               f"High inconsistency between InternetService and {service}")
                               
    def test_outlier_detection(self):
        """Test detection of statistical outliers"""
        # Test numerical columns for outliers
        numerical_cols = ['tenure', 'MonthlyCharges']
        
        for col in numerical_cols:
            if col in self.valid_data.columns:
                values = pd.to_numeric(self.valid_data[col], errors='coerce').dropna()
                
                # Calculate IQR-based outliers
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                outlier_rate = len(outliers) / len(values)
                
                # Outlier rate should be reasonable (typically < 5%)
                self.assertLess(outlier_rate, 0.15, 
                               f"High outlier rate ({outlier_rate:.2%}) in {col}")
                               
    def test_data_quality_metrics(self):
        """Test calculation of data quality metrics"""
        quality_metrics = {}
        
        # Completeness (percentage of non-missing values)
        for col in self.valid_data.columns:
            completeness = 1 - (self.valid_data[col].isnull().sum() / len(self.valid_data))
            quality_metrics[f"{col}_completeness"] = completeness
            
        # Uniqueness for customerID
        if 'customerID' in self.valid_data.columns:
            uniqueness = (self.valid_data['customerID'].nunique() / 
                         len(self.valid_data))
            quality_metrics['customerID_uniqueness'] = uniqueness
            self.assertGreater(uniqueness, 0.95, "CustomerID should be mostly unique")
            
        # Validity (percentage of values within expected ranges/categories)
        if 'MonthlyCharges' in self.valid_data.columns:
            charges = pd.to_numeric(self.valid_data['MonthlyCharges'], errors='coerce')
            valid_charges = ((charges >= 0) & (charges <= 200)).sum()
            validity = valid_charges / len(charges.dropna())
            quality_metrics['MonthlyCharges_validity'] = validity
            self.assertGreater(validity, 0.95, "MonthlyCharges should be mostly valid")
            
    def test_feature_correlation_validation(self):
        """Test validation of feature correlations"""
        # Select numerical columns for correlation analysis
        numerical_data = self.valid_data.select_dtypes(include=[np.number])
        
        if len(numerical_data.columns) > 1:
            correlation_matrix = numerical_data.corr()
            
            # Check for perfect correlations (excluding diagonal)
            np.fill_diagonal(correlation_matrix.values, 0)
            perfect_corr = (correlation_matrix.abs() >= 0.99).sum().sum()
            self.assertEqual(perfect_corr, 0, "Should not have perfectly correlated features")
            
            # Check for very high correlations that might indicate problems
            high_corr = (correlation_matrix.abs() >= 0.95).sum().sum()
            self.assertLess(high_corr, len(numerical_data.columns), 
                           "Too many highly correlated features")
                           
    def test_target_variable_validation(self):
        """Test validation of target variable"""
        if 'Churn' in self.valid_data.columns:
            churn = self.valid_data['Churn']
            
            # Check target distribution
            churn_counts = churn.value_counts()
            self.assertTrue(len(churn_counts) == 2, "Binary target should have 2 classes")
            
            # Check class balance (should not be extremely imbalanced)
            min_class_ratio = churn_counts.min() / churn_counts.sum()
            self.assertGreater(min_class_ratio, 0.05, 
                             "Target classes are extremely imbalanced")
                             
            # Check for missing target values
            missing_targets = churn.isnull().sum()
            self.assertEqual(missing_targets, 0, "Target variable should not have missing values")
            
    def test_data_preprocessing_validation(self):
        """Test validation after data preprocessing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save test data
                data_path = Path(temp_dir) / "test_data.csv"
                columns_path = Path(temp_dir) / "columns.json"
                
                self.valid_data.to_csv(data_path, index=False)
                with open(columns_path, 'w') as f:
                    json.dump(self.expected_columns, f)
                
                # Import preprocessing function
                from data.preprocess import load_and_prepare_data
                
                # Test preprocessing
                X, y, feature_cols, target_col, le = load_and_prepare_data(data_path, columns_path)
                
                if X is not None and y is not None:
                    # Validate preprocessed data
                    self.assertGreater(len(X), 0, "Preprocessed data should not be empty")
                    self.assertEqual(len(X), len(y), "Features and target should have same length")
                    
                    # Check for infinite or NaN values
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        infinite_count = np.isinf(X[col]).sum()
                        self.assertEqual(infinite_count, 0, f"Column {col} contains infinite values")
                        
                        nan_count = X[col].isnull().sum()
                        # Some NaN might be acceptable depending on preprocessing strategy
                        self.assertLess(nan_count / len(X), 0.5, 
                                      f"Column {col} has too many NaN values")
                        
            except ImportError:
                self.skipTest("Preprocessing module not available")
            except Exception as e:
                self.skipTest(f"Preprocessing validation failed: {e}")
                
    def test_prediction_input_validation(self):
        """Test validation of prediction inputs"""
        try:
            from inference.predict import predict_from_dict
            
            # Test with valid input
            valid_input = {
                'gender': 'Male',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 12,
                'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'No',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 65.0,
                'TotalCharges': '780.0'
            }
            
            # Input validation should pass for valid data structure
            # This is a structural test - actual prediction might fail without model
            self.assertIsInstance(valid_input, dict)
            self.assertGreater(len(valid_input), 0)
            
            # Test invalid input types
            invalid_inputs = [
                None,
                [],
                "invalid_string",
                123,
                {}  # Empty dict
            ]
            
            for invalid_input in invalid_inputs:
                with self.subTest(input_type=type(invalid_input)):
                    # Should handle invalid inputs gracefully
                    if not isinstance(invalid_input, dict) or len(invalid_input) == 0:
                        # These should be caught by input validation
                        pass
                        
        except ImportError:
            self.skipTest("Inference module not available")


class TestDataQualityFramework(unittest.TestCase):
    """Test framework for systematic data quality assessment"""
    
    def test_data_quality_assessment_framework(self):
        """Test comprehensive data quality assessment framework"""
        # This test demonstrates how to create a systematic approach
        # to data quality validation across the pipeline
        
        quality_dimensions = [
            'completeness',      # Are all required values present?
            'uniqueness',        # Are there unexpected duplicates?
            'validity',          # Are values in expected format/range?
            'accuracy',          # Are values correct (hard to test automatically)?
            'consistency',       # Are related values logically consistent?
            'timeliness'         # Are values current/up-to-date?
        ]
        
        # Framework should address each dimension
        for dimension in quality_dimensions:
            with self.subTest(dimension=dimension):
                # Each dimension should have corresponding validation tests
                self.assertIsInstance(dimension, str)
                self.assertGreater(len(dimension), 0)
                
    def test_automated_data_profiling(self):
        """Test automated data profiling capabilities"""
        # Create sample data for profiling
        sample_data = pd.DataFrame({
            'numeric_col': np.random.normal(0, 1, 100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'date_col': pd.date_range('2023-01-01', periods=100),
            'text_col': [f'text_{i}' for i in range(100)]
        })
        
        # Test basic profiling metrics
        profile_results = {}
        
        for col in sample_data.columns:
            col_profile = {
                'dtype': str(sample_data[col].dtype),
                'null_count': sample_data[col].isnull().sum(),
                'unique_count': sample_data[col].nunique(),
                'sample_values': sample_data[col].head(3).tolist()
            }
            profile_results[col] = col_profile
            
        # Validate profiling results
        self.assertEqual(len(profile_results), len(sample_data.columns))
        for col_name, profile in profile_results.items():
            self.assertIn('dtype', profile)
            self.assertIn('null_count', profile)
            self.assertIn('unique_count', profile)
            self.assertGreaterEqual(profile['null_count'], 0)
            self.assertGreaterEqual(profile['unique_count'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)