import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Union, List, Optional

def load_model(model_path: str):
    """
    Load a trained machine learning pipeline from disk with version compatibility handling.
    
    Args:
        model_path (str): Path to the saved joblib model file
    
    Returns:
        sklearn.pipeline.Pipeline: Loaded ML pipeline
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If there's an error loading the model or version incompatibility
    """
    import warnings
    import sklearn
    
    try:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Suppress sklearn version warnings during loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = joblib.load(model_path_obj)
        
        print(f"[OK] Model loaded successfully from {model_path}")
        print(f"[INFO] Current sklearn version: {sklearn.__version__}")
        return model
        
    except AttributeError as e:
        if "_RemainderColsList" in str(e) or "sklearn.compose" in str(e):
            error_msg = (
                f"[ERROR] Sklearn version compatibility issue: {e}\n"
                f"Current sklearn version: {sklearn.__version__}\n"
                f"Model was likely trained with a different sklearn version.\n"
                f"Please retrain the model or ensure sklearn version compatibility."
            )
            print(error_msg)
            raise Exception(error_msg)
        else:
            print(f"[ERROR] Failed to load model - {e}")
            raise
        
    except Exception as e:
        print(f"[ERROR] Failed to load model - {e}")
        raise

def predict_from_dict(model, input_dict: Dict[str, Union[str, int, float]], threshold: float = 0.35) -> Dict[str, Union[int, float]]:
    """
    Make a prediction from a dictionary of input features.
    
    Args:
        model: Trained sklearn pipeline
        input_dict (dict): Dictionary containing feature values
        threshold (float): Decision threshold for classification (default: 0.35, optimized for recall)
    
    Returns:
        dict: Dictionary with 'prediction' (int), 'probability' (float), and 'threshold_used' (float)
    
    Raises:
        KeyError: If required features are missing from input_dict
        Exception: If there's an error during prediction
    """
    try:
        # Load feature metadata to know the expected feature order
        # Use absolute path relative to the project root
        project_root = Path(__file__).parent.parent.parent
        feature_metadata_path = project_root / "artifacts/models/feature_names.json"
        if not feature_metadata_path.exists():
            raise FileNotFoundError(f"Feature metadata not found: {feature_metadata_path}")
        
        with open(feature_metadata_path, 'r') as f:
            feature_info = json.load(f)
        
        # Get expected feature columns
        expected_features = feature_info['numeric_features'] + feature_info['categorical_features']
        
        # Validate that all required features are present
        missing_features = [feat for feat in expected_features if feat not in input_dict]
        if missing_features:
            raise KeyError(f"Missing required features: {missing_features}")
        
        # Create DataFrame with the input data
        # Ensure TotalCharges is numeric if present
        if 'TotalCharges' in input_dict:
            try:
                input_dict['TotalCharges'] = float(input_dict['TotalCharges'])
            except (ValueError, TypeError):
                input_dict['TotalCharges'] = np.nan
        
        # Create DataFrame from input dict
        input_df = pd.DataFrame([input_dict])
        
        # Select only the expected features in the correct order
        X = input_df[expected_features]
        
        # Get prediction probability
        probability = model.predict_proba(X)[0, 1]  # Probability of positive class (churn)
        
        # Apply custom threshold for prediction (optimized for recall)
        prediction = 1 if probability >= threshold else 0
        
        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "threshold_used": float(threshold)
        }
        
        print(f"[INFO] Prediction: {result['prediction']}, Probability: {result['probability']:.4f}, Threshold: {threshold}")
        return result
        
    except Exception as e:
        print(f"[ERROR] Prediction failed - {e}")
        raise


class ChurnPredictor:
    """
    ChurnPredictor class for API integration and testing.
    Provides a class-based interface for churn prediction models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ChurnPredictor.
        
        Args:
            model_path (str, optional): Path to the trained model file
        """
        self.model = None
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = load_model(model_path)
        self.model_path = model_path
    
    def predict(self, input_data: Union[Dict, pd.DataFrame], threshold: float = 0.35) -> Dict[str, Union[int, float]]:
        """
        Make a prediction using the loaded model.
        
        Args:
            input_data: Input data as dictionary or DataFrame
            threshold: Decision threshold for classification (default: 0.35, optimized for recall)
            
        Returns:
            Dict containing prediction, probability, and threshold_used
            
        Raises:
            ValueError: If no model is loaded
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if isinstance(input_data, dict):
            return predict_from_dict(self.model, input_data, threshold=threshold)
        elif isinstance(input_data, pd.DataFrame):
            # Convert DataFrame to dict format for compatibility
            if len(input_data) > 1:
                raise ValueError("DataFrame should contain exactly one row for single prediction")
            input_dict = input_data.iloc[0].to_dict()
            return predict_from_dict(self.model, input_dict, threshold=threshold)
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")
    
    def predict_batch(self, input_data: pd.DataFrame) -> List[Dict[str, Union[int, float]]]:
        """
        Make batch predictions.
        
        Args:
            input_data: DataFrame with multiple rows for prediction
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        results = []
        for _, row in input_data.iterrows():
            input_dict = row.to_dict()
            result = predict_from_dict(self.model, input_dict)
            results.append(result)
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None