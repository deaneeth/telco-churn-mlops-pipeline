import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sys
import os
import mlflow
import mlflow.sklearn

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_and_prepare_data(data_path, columns_path):
    """
    Load data and prepare it for training.
    
    Args:
        data_path (str or Path): Path to the raw data CSV file
        columns_path (str or Path): Path to the columns metadata JSON
    
    Returns:
        tuple: (X, y, feature_columns, target_column)
    """
    try:
        print(f"📂 Loading data from {data_path}")
        
        # Load the data
        df = pd.read_csv(data_path)
        print(f"   Dataset shape: {df.shape}")
        
        # Load column metadata
        with open(columns_path, 'r') as f:
            metadata = json.load(f)
        
        # Convert TotalCharges to numeric (handle known data quality issue)
        print(f"🔧 Converting TotalCharges to numeric...")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        na_count = df['TotalCharges'].isna().sum()
        if na_count > 0:
            print(f"   ⚠️  Found {na_count} non-numeric TotalCharges values (converted to NaN)")
        
        # Get feature columns and target
        numeric_cols = metadata['columns']['numeric']
        categorical_cols = metadata['columns']['categorical']
        feature_columns = numeric_cols + categorical_cols
        target_column = metadata['columns']['target']
        
        print(f"   Features: {len(feature_columns)} columns")
        print(f"   Target: {target_column}")
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Encode target variable to binary (0/1)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Show target distribution
        value_counts = pd.Series(y_encoded).value_counts()
        churn_rate = (y_encoded == 1).mean() * 100
        print(f"   Target distribution: {dict(value_counts)}")
        print(f"   Churn rate: {churn_rate:.2f}%")
        
        print(f"✅ Data preparation completed")
        return X, y_encoded, feature_columns, target_column, label_encoder
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load and prepare data - {e}")
        return None, None, None, None, None

def load_preprocessor(preprocessor_path):
    """
    Load the fitted preprocessor from disk.
    
    Args:
        preprocessor_path (str): Path to the saved preprocessor
    
    Returns:
        ColumnTransformer: Fitted preprocessor
    """
    try:
        print(f"📦 Loading preprocessor from {preprocessor_path}")
        
        if not Path(preprocessor_path).exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        preprocessor = joblib.load(preprocessor_path)
        print(f"✅ Preprocessor loaded successfully")
        
        return preprocessor
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load preprocessor - {e}")
        return None

def build_ml_pipeline(preprocessor, random_state=42, n_estimators=100, max_depth=10):
    """
    Build the complete ML pipeline.
    
    Args:
        preprocessor: Fitted ColumnTransformer for preprocessing
        random_state (int): Random state for reproducibility
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the trees
    
    Returns:
        Pipeline: Complete ML pipeline
    """
    print(f"🔧 Building ML pipeline...")
    
    # Create RandomForest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_classifier)
    ])
    
    print(f"   ✅ Pipeline created with RandomForestClassifier")
    print(f"   Parameters: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}")
    
    return pipeline

def train_and_evaluate_with_mlflow(pipeline, X_train, X_test, y_train, y_test, model_params):
    """
    Train the pipeline and evaluate performance with MLflow tracking.
    
    Args:
        pipeline: ML pipeline to train
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        model_params: Dictionary of model parameters for logging
    
    Returns:
        tuple: (trained_pipeline, metrics_dict)
    """
    try:
        print(f"🚀 Training pipeline with MLflow tracking...")
        print(f"   Training set size: {X_train.shape}")
        print(f"   Test set size: {X_test.shape}")
        
        # Train the pipeline
        pipeline.fit(X_train, y_train)
        print(f"   ✅ Training completed")
        
        # Make predictions
        print(f"📊 Evaluating performance...")
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Get prediction probabilities for ROC AUC
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = float(accuracy_score(y_train, y_train_pred))
        test_accuracy = float(accuracy_score(y_test, y_test_pred))
        train_roc_auc = float(roc_auc_score(y_train, y_train_proba))
        test_roc_auc = float(roc_auc_score(y_test, y_test_proba))
        
        # Log parameters to MLflow
        print(f"📝 Logging parameters to MLflow...")
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", model_params["n_estimators"])
        mlflow.log_param("max_depth", model_params["max_depth"])
        mlflow.log_param("min_samples_split", model_params["min_samples_split"])
        mlflow.log_param("min_samples_leaf", model_params["min_samples_leaf"])
        mlflow.log_param("random_state", model_params["random_state"])
        mlflow.log_param("data_train_size", X_train.shape[0])
        mlflow.log_param("data_test_size", X_test.shape[0])
        mlflow.log_param("feature_count", X_train.shape[1])
        
        # Log metrics to MLflow
        print(f"📊 Logging metrics to MLflow...")
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("accuracy", test_accuracy)  # Primary metric
        mlflow.log_metric("roc_auc", test_roc_auc)    # Primary metric
        
        # Log model to MLflow
        print(f"🔧 Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="telco_churn_rf_model"
        )
        
        # Create metrics dictionary for local saving
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_roc_auc': train_roc_auc,
            'test_roc_auc': test_roc_auc,
            'model_params': model_params,
            'feature_count': X_train.shape[1],
            'train_size': int(X_train.shape[0]),
            'test_size': int(X_test.shape[0])
        }
        
        # Print results
        print(f"   Training Results:")
        print(f"      Accuracy: {train_accuracy:.4f}")
        print(f"      ROC AUC:  {train_roc_auc:.4f}")
        print(f"   Test Results:")
        print(f"      Accuracy: {test_accuracy:.4f}")
        print(f"      ROC AUC:  {test_roc_auc:.4f}")
        
        # Feature importance (top 10)
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            feature_importance = pipeline.named_steps['classifier'].feature_importances_
            
            # Load feature names
            try:
                with open('artifacts/models/feature_names.json', 'r') as f:
                    feature_info = json.load(f)
                feature_names = feature_info['feature_names']
                
                # Get top 10 features
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                top_features = importance_df.head(10)
                print(f"   Top 10 Feature Importances:")
                for idx, row in top_features.iterrows():
                    print(f"      {row['feature']}: {row['importance']:.4f}")
                
                # Log feature importance as artifact
                importance_dict = dict(zip(feature_names, feature_importance.tolist()))
                mlflow.log_dict(importance_dict, "feature_importance.json")
                
                # Add to metrics
                metrics['feature_importance'] = importance_dict
                
            except Exception as e:
                print(f"   ⚠️  Could not log feature importance: {e}")
        
        return pipeline, metrics
        
    except Exception as e:
        print(f"❌ ERROR: Training failed - {e}")
        return None, None

def save_artifacts(pipeline, metrics, pipeline_output_path, metrics_output_path):
    """
    Save the trained pipeline and metrics to disk.
    
    Args:
        pipeline: Trained ML pipeline
        metrics (dict): Performance metrics
        pipeline_output_path (str): Path to save the pipeline
        metrics_output_path (str): Path to save metrics JSON
    """
    try:
        print(f"💾 Saving artifacts...")
        
        # Create output directories
        Path(pipeline_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        joblib.dump(pipeline, pipeline_output_path)
        print(f"   ✅ Pipeline saved to {pipeline_output_path}")
        
        # Save metrics
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   ✅ Metrics saved to {metrics_output_path}")
        
        # Also log artifacts to MLflow
        mlflow.log_artifact(pipeline_output_path, "local_artifacts")
        mlflow.log_artifact(metrics_output_path, "local_artifacts")
        
        print(f"   ✅ Artifacts logged to MLflow")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to save artifacts - {e}")

def main():
    """
    Main training function with MLflow tracking.
    """
    # Define paths
    data_path = "data/raw/Telco-Customer-Churn.csv"
    columns_path = "data/processed/columns.json"
    preprocessor_path = "artifacts/models/preprocessor.joblib"
    pipeline_output_path = "artifacts/models/sklearn_pipeline_mlflow.joblib"
    metrics_output_path = "artifacts/metrics/sklearn_metrics_mlflow.json"
    
    # MLflow configuration
    mlflow.set_experiment("telco-churn-prediction")
    
    # Start MLflow run
    with mlflow.start_run():
        print("🚀 Starting ML training pipeline with MLflow tracking...")
        print(f"📋 MLflow run ID: {mlflow.active_run().info.run_id}")
        
        # Load and prepare data
        X, y, feature_columns, target_column, label_encoder = load_and_prepare_data(data_path, columns_path)
        if X is None:
            return
        
        # Split data
        print(f"✂️  Splitting data (stratified by {target_column})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        train_churn_rate = y_train.mean() * 100
        test_churn_rate = y_test.mean() * 100
        print(f"   Train set: {X_train.shape[0]} samples, churn rate: {train_churn_rate:.2f}%")
        print(f"   Test set:  {X_test.shape[0]} samples, churn rate: {test_churn_rate:.2f}%")
        
        # Log data split info
        mlflow.log_param("data_test_size_ratio", 0.2)
        mlflow.log_param("data_stratify", True)
        mlflow.log_metric("train_churn_rate", train_churn_rate)
        mlflow.log_metric("test_churn_rate", test_churn_rate)
        
        # Load preprocessor
        preprocessor = load_preprocessor(preprocessor_path)
        if preprocessor is None:
            return
        
        # Model parameters
        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42
        }
        
        # Build pipeline
        pipeline = build_ml_pipeline(
            preprocessor, 
            random_state=model_params["random_state"],
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"]
        )
        
        # Train and evaluate with MLflow tracking
        trained_pipeline, metrics = train_and_evaluate_with_mlflow(
            pipeline, X_train, X_test, y_train, y_test, model_params
        )
        if trained_pipeline is None:
            return
        
        # Save artifacts locally
        save_artifacts(trained_pipeline, metrics, pipeline_output_path, metrics_output_path)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Final Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"   Final Test ROC AUC:  {metrics['test_roc_auc']:.4f}")
        print(f"   MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()