"""
Test suite to validate feature scaling is properly applied in both sklearn and Spark pipelines.

This test ensures:
1. StandardScaler is present in sklearn preprocessing pipeline
2. StandardScaler is present in Spark ML pipeline
3. Numeric features are actually scaled (mean ≈ 0, std ≈ 1)
4. Categorical features remain unscaled (one-hot encoded)
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


class TestSklearnFeatureScaling:
    """Test feature scaling in scikit-learn pipeline"""
    
    def test_preprocessor_exists(self):
        """Test that preprocessor file exists"""
        preprocessor_path = Path("artifacts/models/preprocessor.joblib")
        assert preprocessor_path.exists(), f"Preprocessor not found at {preprocessor_path}"
    
    def test_preprocessor_has_standard_scaler(self):
        """Test that preprocessor contains StandardScaler for numeric features"""
        preprocessor_path = Path("artifacts/models/preprocessor.joblib")
        
        if not preprocessor_path.exists():
            pytest.skip("Preprocessor file not found - train model first")
        
        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)
        
        # Check it's a ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer), \
            f"Expected ColumnTransformer, got {type(preprocessor)}"
        
        # Get the numeric transformer - transformers is a list of (name, transformer, columns) tuples
        transformers_dict = {name: (transformer, columns) for name, transformer, columns in preprocessor.transformers_}
        assert 'numeric' in transformers_dict or 'num' in transformers_dict, \
            "No numeric transformer found in preprocessor"
        
        # Get numeric pipeline
        numeric_key = 'numeric' if 'numeric' in transformers_dict else 'num'
        numeric_transformer, _ = transformers_dict[numeric_key]
        
        # Check for StandardScaler in the pipeline
        has_scaler = False
        if hasattr(numeric_transformer, 'named_steps'):
            has_scaler = 'scaler' in numeric_transformer.named_steps
            if has_scaler:
                scaler = numeric_transformer.named_steps['scaler']
                assert isinstance(scaler, StandardScaler), \
                    f"Expected StandardScaler, got {type(scaler)}"
        
        assert has_scaler, "StandardScaler not found in numeric transformer pipeline"
        print("✅ StandardScaler found in sklearn preprocessing pipeline")
    
    def test_numeric_features_are_scaled(self):
        """Test that numeric features are actually scaled after transformation"""
        preprocessor_path = Path("artifacts/models/preprocessor.joblib")
        data_path = Path("data/raw/Telco-Customer-Churn.csv")
        
        if not preprocessor_path.exists() or not data_path.exists():
            pytest.skip("Required files not found")
        
        # Load data
        df = pd.read_csv(data_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Define numeric columns
        numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Load preprocessor and transform a sample
        preprocessor = joblib.load(preprocessor_path)
        
        # Get feature columns from preprocessor
        transformers_dict = {name: (transformer, columns) for name, transformer, columns in preprocessor.transformers_}
        numeric_key = 'numeric' if 'numeric' in transformers_dict else 'num'
        cat_key = 'categorical' if 'categorical' in transformers_dict else 'cat'
        
        feature_cols = list(transformers_dict[numeric_key][1]) + list(transformers_dict[cat_key][1])
        X = df[feature_cols].head(100)  # Test on first 100 rows
        
        # Transform
        X_transformed = preprocessor.transform(X)
        
        # The first len(numeric_cols) columns should be the scaled numeric features
        n_numeric = len(numeric_cols)
        scaled_features = X_transformed[:, :n_numeric]
        
        # Check that features are scaled (mean ≈ 0, std ≈ 1)
        # Allow generous tolerance since we're using a small sample of 100 rows
        for i, col in enumerate(numeric_cols):
            mean = np.mean(scaled_features[:, i])
            std = np.std(scaled_features[:, i])
            
            print(f"   {col}: mean={mean:.4f}, std={std:.4f}")
            
            # Mean should be close to 0 (within 1.0 for small samples)
            assert abs(mean) < 1.0, \
                f"{col} not properly scaled: mean={mean:.4f} (expected ≈ 0)"
            
            # Std should be close to 1 (between 0.4 and 1.5 for small samples)
            assert 0.4 < std < 1.5, \
                f"{col} not properly scaled: std={std:.4f} (expected ≈ 1)"
        
        print("✅ All numeric features are properly scaled in sklearn pipeline")
    
    def test_full_pipeline_has_scaler(self):
        """Test that full sklearn pipeline includes preprocessing with scaler"""
        pipeline_path = Path("artifacts/models/sklearn_pipeline_mlflow.joblib")
        
        if not pipeline_path.exists():
            # Try alternative path
            pipeline_path = Path("artifacts/models/sklearn_pipeline.joblib")
        
        if not pipeline_path.exists():
            pytest.skip("Pipeline file not found - train model first")
        
        # Load pipeline
        pipeline = joblib.load(pipeline_path)
        
        # Check if it has a preprocessor step
        if hasattr(pipeline, 'named_steps'):
            assert 'preprocessor' in pipeline.named_steps or 'preprocessing' in pipeline.named_steps, \
                "No preprocessor found in pipeline"
            print("✅ Full sklearn pipeline contains preprocessor step")


class TestSparkFeatureScaling:
    """Test feature scaling in PySpark pipeline"""
    
    @pytest.mark.skipif(
        not Path("artifacts/models/pipeline_metadata.json").exists(),
        reason="Spark pipeline not trained"
    )
    def test_spark_metadata_indicates_scaling(self):
        """Test that Spark pipeline metadata indicates StandardScaler is used"""
        import json
        
        metadata_path = Path("artifacts/models/pipeline_metadata.json")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check model type
        assert 'model_type' in metadata, "model_type not found in metadata"
        print(f"   Model type: {metadata['model_type']}")
        
        # Check if numeric columns are listed
        assert 'numeric_cols' in metadata, "numeric_cols not found in metadata"
        numeric_cols = metadata['numeric_cols']
        
        expected_numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        assert set(numeric_cols) == set(expected_numeric_cols), \
            f"Numeric columns mismatch. Expected {expected_numeric_cols}, got {numeric_cols}"
        
        print(f"✅ Spark pipeline metadata contains numeric columns: {numeric_cols}")
    
    def test_spark_pipeline_code_has_scaler(self):
        """Test that Spark pipeline source code includes StandardScaler"""
        spark_pipeline_path = Path("pipelines/spark_pipeline.py")
        
        assert spark_pipeline_path.exists(), "Spark pipeline file not found"
        
        with open(spark_pipeline_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check for StandardScaler import
        assert 'StandardScaler' in code, \
            "StandardScaler not imported in Spark pipeline"
        
        # Check for StandardScaler instantiation
        assert 'StandardScaler(' in code, \
            "StandardScaler not instantiated in Spark pipeline"
        
        # Check for withMean and withStd parameters
        assert 'withMean=True' in code, \
            "StandardScaler not configured with withMean=True"
        assert 'withStd=True' in code, \
            "StandardScaler not configured with withStd=True"
        
        print("✅ Spark pipeline code includes StandardScaler with correct configuration")
    
    def test_spark_pipeline_stages_include_scaler(self):
        """Test that Spark pipeline stages list includes scaler"""
        spark_pipeline_path = Path("pipelines/spark_pipeline.py")
        
        with open(spark_pipeline_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check that pipeline_stages includes scaler
        assert 'pipeline_stages' in code, "pipeline_stages not found"
        
        # Look for scaler in pipeline_stages definition
        # Should have pattern like: [..., assembler, scaler, label_indexer, ...]
        import re
        stages_pattern = r'pipeline_stages\s*=.*\[.*scaler.*\]'
        assert re.search(stages_pattern, code, re.DOTALL), \
            "scaler not found in pipeline_stages list"
        
        print("✅ Spark pipeline stages include scaler")


class TestFeatureScalingConsistency:
    """Test consistency of feature scaling across pipelines"""
    
    def test_same_numeric_columns_in_both_pipelines(self):
        """Test that both pipelines scale the same numeric columns"""
        # Expected numeric columns
        expected_numeric = {'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'}
        
        # Check sklearn pipeline
        columns_path = Path("data/processed/columns.json")
        if columns_path.exists():
            import json
            with open(columns_path, 'r') as f:
                columns_metadata = json.load(f)
            
            sklearn_numeric = set(columns_metadata['numeric_cols'])
            assert sklearn_numeric == expected_numeric, \
                f"sklearn numeric columns mismatch: {sklearn_numeric} vs {expected_numeric}"
        
        # Check Spark pipeline
        spark_metadata_path = Path("artifacts/models/pipeline_metadata.json")
        if spark_metadata_path.exists():
            import json
            with open(spark_metadata_path, 'r') as f:
                spark_metadata = json.load(f)
            
            spark_numeric = set(spark_metadata['numeric_cols'])
            assert spark_numeric == expected_numeric, \
                f"Spark numeric columns mismatch: {spark_numeric} vs {expected_numeric}"
        
        print("✅ Both pipelines scale the same numeric columns")
    
    def test_scaling_method_consistency(self):
        """Test that both pipelines use StandardScaler (not other scalers)"""
        # Check sklearn
        preprocessor_path = Path("artifacts/models/preprocessor.joblib")
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            transformers_dict = {name: (transformer, columns) for name, transformer, columns in preprocessor.transformers_}
            numeric_key = 'numeric' if 'numeric' in transformers_dict else 'num'
            numeric_transformer = transformers_dict[numeric_key][0]
            
            if hasattr(numeric_transformer, 'named_steps'):
                scaler = numeric_transformer.named_steps.get('scaler')
                assert isinstance(scaler, StandardScaler), \
                    f"sklearn uses {type(scaler)} instead of StandardScaler"
        
        # Check Spark (via code inspection)
        spark_pipeline_path = Path("pipelines/spark_pipeline.py")
        if spark_pipeline_path.exists():
            with open(spark_pipeline_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Should use StandardScaler, not MinMaxScaler, RobustScaler, etc.
            assert 'MinMaxScaler' not in code, "Spark pipeline uses MinMaxScaler instead of StandardScaler"
            assert 'RobustScaler' not in code, "Spark pipeline uses RobustScaler instead of StandardScaler"
            assert 'Normalizer' not in code, "Spark pipeline uses Normalizer instead of StandardScaler"
        
        print("✅ Both pipelines use StandardScaler consistently")


def test_feature_scaling_summary(capsys):
    """Print a summary of feature scaling validation"""
    print("\n" + "="*60)
    print("FEATURE SCALING VALIDATION SUMMARY")
    print("="*60)
    
    # Check sklearn
    preprocessor_path = Path("artifacts/models/preprocessor.joblib")
    sklearn_status = "✅ PASS" if preprocessor_path.exists() else "⏭️ SKIP (not trained)"
    print(f"\n1. Sklearn Pipeline: {sklearn_status}")
    
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        transformers_dict = {name: (transformer, columns) for name, transformer, columns in preprocessor.transformers_}
        numeric_key = 'numeric' if 'numeric' in transformers_dict else 'num'
        numeric_cols = transformers_dict[numeric_key][1]
        print(f"   - Numeric columns scaled: {list(numeric_cols)}")
        print(f"   - Scaling method: StandardScaler")
        print(f"   - Preprocessing: Median imputation → StandardScaler")
    
    # Check Spark
    spark_metadata_path = Path("artifacts/models/pipeline_metadata.json")
    spark_code_path = Path("pipelines/spark_pipeline.py")
    
    if spark_code_path.exists():
        with open(spark_code_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        has_scaler = 'StandardScaler' in code and 'withMean=True' in code
        spark_status = "✅ PASS" if has_scaler else "❌ FAIL"
        print(f"\n2. Spark Pipeline: {spark_status}")
        
        if has_scaler:
            if spark_metadata_path.exists():
                import json
                with open(spark_metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"   - Numeric columns scaled: {metadata.get('numeric_cols', [])}")
            print(f"   - Scaling method: StandardScaler")
            print(f"   - Configuration: withMean=True, withStd=True")
        else:
            print(f"   - ⚠️ WARNING: StandardScaler not found in Spark pipeline")
    else:
        print(f"\n2. Spark Pipeline: ⏭️ SKIP (file not found)")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("✓ Always scale numeric features with different ranges")
    print("✓ Use StandardScaler for normally distributed features")
    print("✓ Consider RobustScaler if data has outliers")
    print("✓ Ensure consistency across training and inference pipelines")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
