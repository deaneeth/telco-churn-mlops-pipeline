import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocessor(numeric_cols, categorical_cols):
    """
    Build a preprocessing pipeline using ColumnTransformer.
    
    Args:
        numeric_cols (list): List of numeric column names
        categorical_cols (list): List of categorical column names
    
    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    # Numeric preprocessing pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both pipelines using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_cols),
        ('categorical', categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor

def fit_save_preprocessor(input_csv, out_path):
    """
    Load data, fit preprocessor, and save it to specified path.
    
    Args:
        input_csv (str or Path): Path to input CSV file
        out_path (str or Path): Path to save the fitted preprocessor
    
    Returns:
        tuple: (fitted_preprocessor, feature_names, X_transformed)
    """
    try:
        # Convert paths to Path objects
        input_csv = Path(input_csv)
        out_path = Path(out_path)
        columns_path = Path("data/processed/columns.json")
        
        print(f"[INFO] Loading data from {input_csv}")
        
        # Check if input file exists
        if not input_csv.exists():
            raise FileNotFoundError(f"Input file not found: {input_csv}")
        
        # Check if columns metadata exists
        if not columns_path.exists():
            raise FileNotFoundError(f"Columns metadata not found: {columns_path}")
        
        # Load the data
        df = pd.read_csv(input_csv)
        print(f"   Dataset shape: {df.shape}")
        
        # Load column metadata
        with open(columns_path, 'r') as f:
            metadata = json.load(f)
        
        numeric_cols = metadata['columns']['numerical']
        categorical_cols = metadata['columns']['categorical']
        
        print(f"[INFO] Column Information:")
        print(f"   Numeric columns ({len(numeric_cols)}): {numeric_cols}")
        print(f"   Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        
        # Convert TotalCharges to numeric if needed (handle the known data quality issue)
        if 'TotalCharges' in numeric_cols:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Prepare feature matrix (exclude target and identifier columns)
        feature_cols = numeric_cols + categorical_cols
        X = df[feature_cols]
        
        print(f"[INFO] Building and fitting preprocessor...")
        
        # Build preprocessor
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        
        # Fit the preprocessor
        X_transformed = preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        feature_names = []
        
        # Add numeric feature names (unchanged)
        feature_names.extend(numeric_cols)
        
        # Add categorical feature names (after one-hot encoding)
        categorical_transformer = preprocessor.named_transformers_['categorical']
        if hasattr(categorical_transformer, 'named_steps'):
            encoder = categorical_transformer.named_steps['encoder']
            if hasattr(encoder, 'get_feature_names_out'):
                # For newer sklearn versions
                cat_feature_names = encoder.get_feature_names_out(categorical_cols)
            else:
                # For older sklearn versions
                cat_feature_names = encoder.get_feature_names(categorical_cols)
            feature_names.extend(cat_feature_names)
        
        print(f"   Original features: {len(feature_cols)}")
        print(f"   Transformed features: {X_transformed.shape[1]}")
        print(f"   Feature names generated: {len(feature_names)}")
        
        # Create output directory if it doesn't exist
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the fitted preprocessor
        joblib.dump(preprocessor, out_path)
        print(f"[SUCCESS] Saved preprocessor to {out_path}")
        
        # Save feature names for reference
        feature_names_path = out_path.parent / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump({
                'feature_names': list(feature_names),
                'numeric_features': numeric_cols,
                'categorical_features': categorical_cols,
                'total_features': len(feature_names)
            }, f, indent=2)
        print(f"[SUCCESS] Saved feature names to {feature_names_path}")
        
        return preprocessor, feature_names, X_transformed
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return None, None, None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred - {e}")
        return None, None, None

def load_preprocessor(preprocessor_path):
    """
    Load a saved preprocessor from disk.
    
    Args:
        preprocessor_path (str or Path): Path to the saved preprocessor
    
    Returns:
        ColumnTransformer: Loaded preprocessor
    """
    try:
        preprocessor_path = Path(preprocessor_path)
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        preprocessor = joblib.load(preprocessor_path)
        print(f"[SUCCESS] Loaded preprocessor from {preprocessor_path}")
        return preprocessor
        
    except Exception as e:
        print(f"[ERROR] Failed to load preprocessor - {e}")
        return None

def transform_data(preprocessor, data, feature_cols=None):
    """
    Transform new data using a fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        data: DataFrame or array to transform
        feature_cols: List of feature column names (if data is DataFrame)
    
    Returns:
        np.ndarray: Transformed data
    """
    try:
        if isinstance(data, pd.DataFrame) and feature_cols:
            X = data[feature_cols]
        else:
            X = data
        
        X_transformed = preprocessor.transform(X)
        print(f"[INFO] Transformed data shape: {X_transformed.shape}")
        return X_transformed
        
    except Exception as e:
        print(f"[ERROR] Failed to transform data - {e}")
        return None

def main():
    """
    Main function to demonstrate preprocessing pipeline.
    """
    # Define paths
    input_file = "data/raw/Telco-Customer-Churn.csv"
    output_file = "artifacts/models/preprocessor.joblib"
    
    print("[INFO] Starting preprocessing pipeline...")
    
    # Fit and save preprocessor
    preprocessor, feature_names, X_transformed = fit_save_preprocessor(input_file, output_file)
    
    if preprocessor is not None:
        print(f"\n[SUCCESS] Preprocessing completed successfully!")
        print(f"   Preprocessor saved to: {output_file}")
        print(f"   Transformed data shape: {X_transformed.shape}")
        print(f"   Total features after transformation: {len(feature_names)}")
        
        # Test loading the preprocessor
        print(f"\n[TEST] Testing preprocessor loading...")
        loaded_preprocessor = load_preprocessor(output_file)
        if loaded_preprocessor is not None:
            print(f"   [SUCCESS] Preprocessor loaded successfully!")
    else:
        print(f"[ERROR] Preprocessing failed!")

if __name__ == "__main__":
    main()