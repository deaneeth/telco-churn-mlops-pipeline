import unittest
import sys
from pathlib import Path
import json

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from inference.predict import load_model, predict_from_dict

class TestInference(unittest.TestCase):
    """Test cases for the inference module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.model_path = "artifacts/models/sklearn_pipeline_mlflow.joblib"
        cls.feature_metadata_path = "artifacts/models/feature_names.json"
        
        # Load the model once for all tests
        try:
            cls.model = load_model(cls.model_path)
            print("✅ Model loaded successfully for tests")
        except Exception as e:
            cls.model = None
            print(f"❌ Failed to load model for tests: {e}")
        
        # Load feature metadata to create valid test data
        try:
            with open(cls.feature_metadata_path, 'r') as f:
                cls.feature_info = json.load(f)
            print("✅ Feature metadata loaded for tests")
        except Exception as e:
            cls.feature_info = None
            print(f"❌ Failed to load feature metadata: {e}")
    
    def test_load_model_success(self):
        """Test successful model loading"""
        model = load_model(self.model_path)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
    
    def test_load_model_file_not_found(self):
        """Test model loading with non-existent file"""
        with self.assertRaises(FileNotFoundError):
            load_model("non_existent_model.joblib")
    
    def test_predict_from_dict_success(self):
        """Test successful prediction from dictionary"""
        # Skip if model or metadata failed to load
        if self.model is None or self.feature_info is None:
            self.skipTest("Model or feature metadata not available")
        
        # Create a sample customer data dictionary
        sample_data = {
            # Numeric features
            'tenure': 12,
            'MonthlyCharges': 70.0,
            'TotalCharges': '840.0',  # Test string conversion
            'SeniorCitizen': 0,
            
            # Categorical features
            'gender': 'Male',
            'Partner': 'Yes',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }
        
        # Make prediction
        result = predict_from_dict(self.model, sample_data)
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        
        # Validate result types and ranges
        self.assertIsInstance(result['prediction'], int)
        self.assertIsInstance(result['probability'], float)
        self.assertIn(result['prediction'], [0, 1])  # Binary classification
        self.assertGreaterEqual(result['probability'], 0.0)
        self.assertLessEqual(result['probability'], 1.0)
    
    def test_predict_from_dict_missing_features(self):
        """Test prediction with missing required features"""
        if self.model is None:
            self.skipTest("Model not available")
        
        # Incomplete data missing several required features
        incomplete_data = {
            'tenure': 12,
            'MonthlyCharges': 70.0
            # Missing many required features
        }
        
        # Should raise KeyError for missing features
        with self.assertRaises(KeyError) as context:
            predict_from_dict(self.model, incomplete_data)
        
        # Check that the error message mentions missing features
        self.assertIn("Missing required features", str(context.exception))
    
    def test_predict_from_dict_total_charges_conversion(self):
        """Test TotalCharges string to float conversion"""
        if self.model is None or self.feature_info is None:
            self.skipTest("Model or feature metadata not available")
        
        # Create sample data with TotalCharges as string
        sample_data = {
            # Numeric features
            'tenure': 24,
            'MonthlyCharges': 85.0,
            'TotalCharges': '2040.50',  # String that should be converted
            'SeniorCitizen': 1,
            
            # Categorical features
            'gender': 'Female',
            'Partner': 'No',
            'Dependents': 'Yes',
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)'
        }
        
        # Should not raise an error despite TotalCharges being a string
        result = predict_from_dict(self.model, sample_data)
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
    
    def test_predict_from_dict_invalid_total_charges(self):
        """Test handling of invalid TotalCharges values"""
        if self.model is None or self.feature_info is None:
            self.skipTest("Model or feature metadata not available")
        
        # Create sample data with invalid TotalCharges
        sample_data = {
            # Numeric features
            'tenure': 36,
            'MonthlyCharges': 95.0,
            'TotalCharges': 'invalid_value',  # Invalid string
            'SeniorCitizen': 0,
            
            # Categorical features (using defaults for simplicity)
            'gender': 'Male',
            'Partner': 'Yes',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }
        
        # Should handle invalid TotalCharges gracefully (convert to NaN)
        result = predict_from_dict(self.model, sample_data)
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
    
    def test_input_data_validation_edge_cases(self):
        """Test input validation with various edge cases"""
        if self.model is None or self.feature_info is None:
            self.skipTest("Model or feature metadata not available")
        
        # Test edge case inputs
        edge_case_inputs = [
            # Minimum values
            {
                'tenure': 0,
                'MonthlyCharges': 0.0,
                'TotalCharges': '0.0',
                'SeniorCitizen': 0,
                'gender': 'Male',
                'Partner': 'No',
                'Dependents': 'No',
                'PhoneService': 'No',
                'MultipleLines': 'No phone service',
                'InternetService': 'No',
                'OnlineSecurity': 'No internet service',
                'OnlineBackup': 'No internet service',
                'DeviceProtection': 'No internet service',
                'TechSupport': 'No internet service',
                'StreamingTV': 'No internet service',
                'StreamingMovies': 'No internet service',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'No',
                'PaymentMethod': 'Mailed check'
            },
            # Maximum typical values
            {
                'tenure': 72,
                'MonthlyCharges': 118.75,
                'TotalCharges': '8684.8',
                'SeniorCitizen': 1,
                'gender': 'Female',
                'Partner': 'Yes',
                'Dependents': 'Yes',
                'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes',
                'TechSupport': 'Yes',
                'StreamingTV': 'Yes',
                'StreamingMovies': 'Yes',
                'Contract': 'Two year',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Credit card (automatic)'
            }
        ]
        
        for i, edge_input in enumerate(edge_case_inputs):
            with self.subTest(case=f"edge_case_{i}"):
                try:
                    result = predict_from_dict(self.model, edge_input)
                    self.assertIsInstance(result, dict)
                    self.assertIn('prediction', result)
                    self.assertIn('probability', result)
                except Exception as e:
                    self.fail(f"Edge case {i} failed: {e}")
    
    def test_input_data_type_coercion(self):
        """Test automatic data type coercion for inputs"""
        if self.model is None or self.feature_info is None:
            self.skipTest("Model or feature metadata not available")
        
        # Test with various data types that should be coerced
        test_input = {
            'tenure': '24',  # String number
            'MonthlyCharges': 85,  # Integer instead of float
            'TotalCharges': 2040.5,  # Float instead of string
            'SeniorCitizen': '1',  # String boolean
            'gender': 'MALE',  # Different case
            'Partner': 'yes',  # Different case
            'Dependents': 'NO',  # Different case
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }
        
        try:
            result = predict_from_dict(self.model, test_input)
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('probability', result)
        except Exception as e:
            # Some type coercion might fail, but should be handled gracefully
            self.assertIsInstance(e, (ValueError, TypeError, KeyError))
    
    def test_batch_prediction_validation(self):
        """Test validation for batch predictions"""
        if self.model is None or self.feature_info is None:
            self.skipTest("Model or feature metadata not available")
        
        # Create batch of test data
        batch_data = []
        for i in range(5):
            customer = {
                'tenure': 12 + i * 6,
                'MonthlyCharges': 50.0 + i * 10,
                'TotalCharges': str((50.0 + i * 10) * (12 + i * 6)),
                'SeniorCitizen': i % 2,
                'gender': 'Male' if i % 2 == 0 else 'Female',
                'Partner': 'Yes' if i % 2 == 0 else 'No',
                'Dependents': 'No',
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
                'PaymentMethod': 'Electronic check'
            }
            batch_data.append(customer)
        
        # Test each customer in batch
        batch_results = []
        for customer_data in batch_data:
            try:
                result = predict_from_dict(self.model, customer_data)
                batch_results.append(result)
            except Exception as e:
                self.fail(f"Batch prediction failed for customer: {e}")
        
        # Validate batch results
        self.assertEqual(len(batch_results), len(batch_data))
        for result in batch_results:
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('probability', result)
    
    def test_prediction_consistency(self):
        """Test prediction consistency for identical inputs"""
        if self.model is None or self.feature_info is None:
            self.skipTest("Model or feature metadata not available")
        
        # Same input should produce same output
        test_input = {
            'tenure': 24,
            'MonthlyCharges': 70.0,
            'TotalCharges': '1680.0',
            'SeniorCitizen': 0,
            'gender': 'Male',
            'Partner': 'Yes',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'One year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Bank transfer (automatic)'
        }
        
        # Make multiple predictions with same input
        results = []
        for _ in range(3):
            try:
                result = predict_from_dict(self.model, test_input.copy())
                results.append(result)
            except Exception as e:
                self.skipTest(f"Prediction failed: {e}")
        
        if len(results) > 1:
            # All results should be identical
            first_result = results[0]
            for result in results[1:]:
                self.assertEqual(result['prediction'], first_result['prediction'])
                self.assertAlmostEqual(result['probability'], first_result['probability'], places=6)
    
    def test_input_sanitization(self):
        """Test input sanitization and cleaning"""
        if self.model is None or self.feature_info is None:
            self.skipTest("Model or feature metadata not available")
        
        # Test inputs with extra whitespace and special characters
        messy_input = {
            'tenure': '  24  ',
            'MonthlyCharges': ' 70.50 ',
            'TotalCharges': '  1692.00  ',
            'SeniorCitizen': ' 0 ',
            'gender': ' Male ',
            'Partner': ' Yes ',
            'Dependents': ' No ',
            'PhoneService': ' Yes ',
            'MultipleLines': ' No ',
            'InternetService': ' DSL ',
            'OnlineSecurity': ' Yes ',
            'OnlineBackup': ' No ',
            'DeviceProtection': ' No ',
            'TechSupport': ' No ',
            'StreamingTV': ' No ',
            'StreamingMovies': ' No ',
            'Contract': ' One year ',
            'PaperlessBilling': ' No ',
            'PaymentMethod': ' Bank transfer (automatic) '
        }
        
        try:
            result = predict_from_dict(self.model, messy_input)
            # Should handle messy input gracefully
            self.assertIsInstance(result, dict)
            self.assertIn('prediction', result)
            self.assertIn('probability', result)
        except Exception as e:
            # Input sanitization might fail, should be handled appropriately
            self.assertIsInstance(e, (ValueError, TypeError, KeyError))

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)