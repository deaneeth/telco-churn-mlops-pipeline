import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sys
import os

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
        numeric_cols = metadata['columns']['numerical']
        categorical_cols = metadata['columns']['categorical']
        feature_columns = numeric_cols + categorical_cols
        target_column = metadata['columns']['target']
        
        print(f"📊 Features: {len(feature_columns)} ({len(numeric_cols)} numeric + {len(categorical_cols)} categorical)")
        print(f"   Target: {target_column}")
        
        # Prepare features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Encode target variable (Yes/No -> 1/0)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"   Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return X, y_encoded, feature_columns, target_column, le
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load and prepare data - {e}")
        return None, None, None, None, None

def create_preprocessor(X, numeric_cols, categorical_cols):
    """
    Create and fit a fresh preprocessor for the data.
    
    Args:
        X: Training data
        numeric_cols: List of numeric column names  
        categorical_cols: List of categorical column names
    
    Returns:
        ColumnTransformer: Fitted preprocessor
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        
        print(f"🔧 Creating fresh preprocessor...")
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Fit the preprocessor
        preprocessor.fit(X)
        print(f"   ✅ Preprocessor created and fitted successfully")
        return preprocessor
        
    except Exception as e:
        print(f"❌ ERROR: Failed to create preprocessor - {e}")
        return None

def build_ml_pipeline(preprocessor, random_state=42):
    """
    Build the machine learning pipeline.
    
    Args:
        preprocessor: Fitted ColumnTransformer for preprocessing
        random_state (int): Random state for reproducibility
    
    Returns:
        Pipeline: Complete ML pipeline
    """
    print(f"🔧 Building ML pipeline...")
    
    # Create GradientBoosting classifier with optimized parameters from experiments
    gb_classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=random_state
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', gb_classifier)
    ])
    
    print(f"   ✅ Pipeline created with GradientBoostingClassifier")
    print(f"   Parameters: n_estimators=100, learning_rate=0.05, max_depth=3, random_state={random_state}")
    
    return pipeline

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """
    Train the pipeline and evaluate performance.
    
    Args:
        pipeline: ML pipeline to train
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
    
    Returns:
        tuple: (trained_pipeline, metrics_dict)
    """
    try:
        print(f"🚀 Training pipeline...")
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
        metrics = {
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
            'train_roc_auc': float(roc_auc_score(y_train, y_train_proba)),
            'test_roc_auc': float(roc_auc_score(y_test, y_test_proba)),
            'model_params': {
                'n_estimators': pipeline.named_steps['classifier'].n_estimators,
                'learning_rate': pipeline.named_steps['classifier'].learning_rate,
                'max_depth': pipeline.named_steps['classifier'].max_depth,
                'min_samples_split': pipeline.named_steps['classifier'].min_samples_split,
                'min_samples_leaf': pipeline.named_steps['classifier'].min_samples_leaf,
                'subsample': pipeline.named_steps['classifier'].subsample
            },
            'feature_count': X_train.shape[1],
            'train_size': int(X_train.shape[0]),
            'test_size': int(X_test.shape[0])
        }
        
        # Print results
        print(f"   Training Results:")
        print(f"      Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"      ROC AUC:  {metrics['train_roc_auc']:.4f}")
        print(f"   Test Results:")
        print(f"      Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"      ROC AUC:  {metrics['test_roc_auc']:.4f}")
        
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
                metrics['top_10_features'] = [
                    {'feature': row['feature'], 'importance': float(row['importance'])}
                    for _, row in top_features.iterrows()
                ]
                
                print(f"   Top 5 Features:")
                for i, (_, row) in enumerate(top_features.head(5).iterrows()):
                    print(f"      {i+1}. {row['feature']}: {row['importance']:.4f}")
                    
            except Exception as e:
                print(f"   ⚠️  Could not load feature names for importance analysis: {e}")
        
        return pipeline, metrics
        
    except Exception as e:
        print(f"❌ ERROR: Training and evaluation failed - {e}")
        return None, None

def save_artifacts(pipeline, metrics, pipeline_path, metrics_path):
    """
    Save the trained pipeline and metrics.
    
    Args:
        pipeline: Trained ML pipeline
        metrics (dict): Performance metrics
        pipeline_path (str or Path): Path to save the pipeline
        metrics_path (str or Path): Path to save the metrics
    """
    try:
        # Create directories if they don't exist
        Path(pipeline_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        print(f"💾 Saving artifacts...")
        joblib.dump(pipeline, pipeline_path)
        print(f"   Pipeline saved to: {pipeline_path}")
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   Metrics saved to: {metrics_path}")
        
        print(f"   ✅ All artifacts saved successfully")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to save artifacts - {e}")

def main():
    """
    Main training function.
    """
    # Define paths
    data_path = "data/raw/Telco-Customer-Churn.csv"
    columns_path = "data/processed/columns.json"
    preprocessor_path = "artifacts/models/preprocessor.joblib"
    pipeline_output_path = "artifacts/models/sklearn_pipeline.joblib"
    metrics_output_path = "artifacts/metrics/sklearn_metrics.json"
    
    print("🚀 Starting ML training pipeline...")
    
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
    
    # Create fresh preprocessor (avoid sklearn version compatibility issues)
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in feature_columns if col not in numeric_cols]
    preprocessor = create_preprocessor(X_train, numeric_cols, categorical_cols)
    if preprocessor is None:
        return
    
    # Build pipeline
    pipeline = build_ml_pipeline(preprocessor)
    
    # Train and evaluate
    trained_pipeline, metrics = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
    if trained_pipeline is None:
        return
    
    # Save artifacts
    save_artifacts(trained_pipeline, metrics, pipeline_output_path, metrics_output_path)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"   Final Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   Final Test ROC AUC:  {metrics['test_roc_auc']:.4f}")

if __name__ == "__main__":
    main()