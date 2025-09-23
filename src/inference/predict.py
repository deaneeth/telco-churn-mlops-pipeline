import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Union

def load_model(model_path: str):
    """
    Load a trained machine learning pipeline from disk.
    
    Args:
        model_path (str): Path to the saved joblib model file
    
    Returns:
        sklearn.pipeline.Pipeline: Loaded ML pipeline
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If there's an error loading the model
    """
    try:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path_obj)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load model - {e}")
        raise

def predict_from_dict(model, input_dict: Dict[str, Union[str, int, float]]) -> Dict[str, Union[int, float]]:
    """
    Make a prediction from a dictionary of input features.
    
    Args:
        model: Trained sklearn pipeline
        input_dict (dict): Dictionary containing feature values
    
    Returns:
        dict: Dictionary with 'prediction' (int) and 'probability' (float)
    
    Raises:
        KeyError: If required features are missing from input_dict
        Exception: If there's an error during prediction
    """
    try:
        # Load feature metadata to know the expected feature order
        feature_metadata_path = Path("artifacts/models/feature_names.json")
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
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]  # Probability of positive class (churn)
        
        result = {
            "prediction": int(prediction),
            "probability": float(probability)
        }
        
        print(f"📊 Prediction: {result['prediction']}, Probability: {result['probability']:.4f}")
        return result
        
    except Exception as e:
        print(f"❌ ERROR: Prediction failed - {e}")
        raise