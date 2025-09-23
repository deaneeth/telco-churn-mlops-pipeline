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
        cls.model_path = "artifacts/models/sklearn_pipeline.joblib"
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

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)