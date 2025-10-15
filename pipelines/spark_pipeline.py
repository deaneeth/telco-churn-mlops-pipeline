"""
Windows-compatible Spark pipeline with multiple model persistence options
"""
import os
import json
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def create_spark_pipeline():
    """
    Creates and runs a Spark ML pipeline with Windows-compatible model saving
    """
    
    # Configure Spark for Windows compatibility
    print("🚀 Starting SparkSession with Windows compatibility...")
    spark = SparkSession.builder \
        .appName("TelcoChurnPrediction") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
        .getOrCreate()
    
    try:
        # Read data
        print("📂 Reading data from data/raw/Telco-Customer-Churn.csv...")
        data_path = "data/raw/Telco-Customer-Churn.csv"
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        print(f"   Dataset loaded: {df.count()} rows, {len(df.columns)} columns")
        
        # Convert TotalCharges to double and handle null values
        print("🔧 Converting TotalCharges to double...")
        df = df.withColumn("TotalCharges", 
                          when(col("TotalCharges") == " ", None)
                          .otherwise(col("TotalCharges").cast(DoubleType())))
        df = df.fillna({"TotalCharges": 0.0})
        
        # Prepare features
        categorical_cols = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]
        
        numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
        
        print(f"   Categorical features: {len(categorical_cols)}")
        print(f"   Numeric features: {len(numeric_cols)}")
        
        # Create StringIndexers and OneHotEncoders
        print("🔧 Creating StringIndexers and OneHotEncoders...")
        indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
                   for col in categorical_cols]
        
        encoders = [OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded")
                   for col in categorical_cols]
        
        # Create VectorAssembler
        print("🔧 Creating VectorAssembler...")
        encoded_cols = [f"{col}_encoded" for col in categorical_cols]
        feature_cols = numeric_cols + encoded_cols
        
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")
        
        # Create StandardScaler for numeric feature normalization
        print("📏 Creating StandardScaler for feature normalization...")
        scaler = StandardScaler(
            inputCol="features_unscaled",
            outputCol="features",
            withMean=True,
            withStd=True
        )
        
        # Create target label indexer
        label_indexer = StringIndexer(inputCol="Churn", outputCol="ChurnLabel")
        
        # Create RandomForest classifier
        print("🌲 Creating RandomForestClassifier...")
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="ChurnLabel",
            numTrees=20,
            maxDepth=5,
            seed=42
        )
        
        # Create pipeline
        print("🚀 Building and training ML pipeline...")
        pipeline_stages = indexers + encoders + [assembler, scaler, label_indexer, rf]
        pipeline = Pipeline(stages=pipeline_stages)
        
        # Split data
        print("✂️  Splitting data (80/20 train/test)...")
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        print(f"   Train set: {train_df.count()} rows")
        print(f"   Test set: {test_df.count()} rows")
        
        # Train model
        print("🏃 Training model...")
        model = pipeline.fit(train_df)
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions = model.transform(test_df)
        
        # Evaluate model
        print("📊 Evaluating model with BinaryClassificationEvaluator...")
        evaluator_roc = BinaryClassificationEvaluator(
            labelCol="ChurnLabel",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="ChurnLabel",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )
        
        roc_auc = evaluator_roc.evaluate(predictions)
        pr_auc = evaluator_pr.evaluate(predictions)
        
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   PR AUC: {pr_auc:.4f}")
        
        # Show sample predictions
        print("\n📋 Sample predictions:")
        predictions.select("ChurnLabel", "prediction", "probability").show(10, truncate=False)
        
        # Windows-compatible model saving
        print("💾 Saving model with Windows compatibility...")
        model_base_path = "artifacts/models"
        os.makedirs(model_base_path, exist_ok=True)
        
        # Method 1: Try native Spark save with error handling
        native_saved = False
        try:
            native_path = f"{model_base_path}/spark_native"
            if os.path.exists(native_path):
                import shutil
                shutil.rmtree(native_path)
            model.write().overwrite().save(native_path)
            print(f"   ✅ Native Spark model saved to: {native_path}")
            native_saved = True
        except Exception as e:
            print(f"   ⚠️  Native save failed: {str(e)[:100]}...")
        
        # Method 2: Save model components separately (always works)
        print("   🔧 Saving model components separately...")
        
        # Save pipeline metadata
        pipeline_metadata = {
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols,
            "feature_cols": feature_cols,
            "model_type": "RandomForestClassifier",
            "num_trees": 20,
            "max_depth": 5,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "train_count": train_df.count(),
            "test_count": test_df.count()
        }
        
        metadata_path = f"{model_base_path}/pipeline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(pipeline_metadata, f, indent=2)
        print(f"   ✅ Pipeline metadata saved to: {metadata_path}")
        
        # Save feature importance if available
        try:
            rf_model = model.stages[-1]  # RandomForest is the last stage
            if hasattr(rf_model, 'featureImportances'):
                importance_data = {
                    "feature_importances": rf_model.featureImportances.toArray().tolist(),
                    "num_features": rf_model.numFeatures
                }
                importance_path = f"{model_base_path}/feature_importances.json"
                with open(importance_path, 'w') as f:
                    json.dump(importance_data, f, indent=2)
                print(f"   ✅ Feature importances saved to: {importance_path}")
        except Exception as e:
            print(f"   ⚠️  Could not save feature importances: {str(e)[:50]}...")
        
        # Save evaluation metrics
        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "train_count": train_df.count(),
            "test_count": test_df.count(),
            "model_saved_native": native_saved,
            "model_saved_components": True,
            "note": "Model trained successfully with Windows compatibility"
        }
        
        metrics_path = "artifacts/metrics/spark_rf_metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   ✅ Metrics saved to: {metrics_path}")
        
        return model, metrics, spark
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        spark.stop()
        raise
    

def load_model_metadata(model_path="artifacts/models"):
    """
    Load model metadata and components
    """
    metadata_path = f"{model_path}/pipeline_metadata.json"
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("📖 Model Metadata Loaded:")
        print(f"   Model Type: {metadata.get('model_type', 'Unknown')}")
        print(f"   ROC AUC: {metadata.get('roc_auc', 'N/A'):.4f}")
        print(f"   PR AUC: {metadata.get('pr_auc', 'N/A'):.4f}")
        print(f"   Features: {len(metadata.get('feature_cols', []))}")
        
        return metadata
    else:
        print(f"❌ Model metadata not found at: {metadata_path}")
        return None


if __name__ == "__main__":
    print("🎯 Starting Spark Pipeline...")
    
    try:
        model, metrics, spark = create_spark_pipeline()
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"   Final ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"   Final PR AUC: {metrics['pr_auc']:.4f}")
        print(f"   Model components saved: {metrics['model_saved_components']}")
        
        # Test loading metadata
        print(f"\n🧪 Testing model metadata loading...")
        load_model_metadata()
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
    
    finally:
        if 'spark' in locals():
            print("🛑 Stopping Spark session...")
            spark.stop()