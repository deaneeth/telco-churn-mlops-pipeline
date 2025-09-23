#!/usr/bin/env python3
"""
Batch prediction script for Telco Customer Churn prediction.
Loads processed data and trained model to generate predictions.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(data_path: str = "data/processed/sample.csv") -> pd.DataFrame:
    """
    Load processed data for batch prediction.
    
    Args:
        data_path (str): Path to processed data file
        
    Returns:
        pd.DataFrame: Processed data ready for prediction
    """
    try:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"✅ Loaded {len(df)} records from {data_path}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise

def load_trained_model(model_path: str = "artifacts/models/sklearn_pipeline.joblib") -> object:
    """
    Load the trained ML model.
    
    Args:
        model_path (str): Path to the trained model
        
    Returns:
        object: Loaded ML pipeline
    """
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def generate_batch_predictions(model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for batch data.
    
    Args:
        model: Trained ML pipeline
        data (pd.DataFrame): Input data for prediction
        
    Returns:
        pd.DataFrame: Data with predictions and probabilities
    """
    try:
        # Load feature metadata
        feature_metadata_path = Path("artifacts/models/feature_names.json")
        if feature_metadata_path.exists():
            with open(feature_metadata_path, 'r') as f:
                feature_info = json.load(f)
            expected_features = feature_info['numeric_features'] + feature_info['categorical_features']
        else:
            # Fallback: use all columns except target-related ones
            expected_features = [col for col in data.columns 
                               if col not in ['Churn', 'ChurnLabel', 'customerID']]
        
        # Prepare features for prediction
        X = data[expected_features]
        
        # Generate predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of churn
        
        # Add predictions to dataframe
        result_df = data.copy()
        result_df['predicted_churn'] = predictions
        result_df['churn_probability'] = probabilities
        result_df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"✅ Generated predictions for {len(result_df)} records")
        logger.info(f"📊 Predicted churn rate: {predictions.mean():.2%}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")
        raise

def save_predictions(predictions_df: pd.DataFrame, output_path: str = "artifacts/predictions/batch_preds.csv"):
    """
    Save batch predictions to file.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions
        output_path (str): Path to save predictions
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"✅ Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save predictions: {e}")
        raise

def main():
    """
    Main function to run batch prediction pipeline.
    """
    try:
        logger.info("🚀 Starting batch prediction pipeline...")
        
        # Load processed data
        data = load_processed_data()
        
        # Load trained model
        model = load_trained_model()
        
        # Generate predictions
        predictions_df = generate_batch_predictions(model, data)
        
        # Save predictions
        save_predictions(predictions_df)
        
        logger.info("🎉 Batch prediction pipeline completed successfully!")
        
        # Print summary statistics
        churn_count = predictions_df['predicted_churn'].sum()
        total_count = len(predictions_df)
        avg_probability = predictions_df['churn_probability'].mean()
        
        print(f"\n📈 Prediction Summary:")
        print(f"   Total customers: {total_count}")
        print(f"   Predicted churners: {churn_count}")
        print(f"   Churn rate: {churn_count/total_count:.2%}")
        print(f"   Average churn probability: {avg_probability:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Batch prediction pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()