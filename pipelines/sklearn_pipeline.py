"""
Scikit-learn ML Pipeline for Telco Churn Prediction.

This module implements a complete machine learning pipeline using scikit-learn
for predicting customer churn in a telecommunications company.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelcoChurnPipeline:
    """
    Complete ML pipeline for telco churn prediction using scikit-learn.
    """
    
    def __init__(self, experiment_name: str = "telco_churn_sklearn"):
        """
        Initialize the pipeline.
        
        Args:
            experiment_name (str): Name for MLflow experiment
        """
        self.experiment_name = experiment_name
        self.pipeline = None
        self.preprocessor = None
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        logger.info(f"Pipeline initialized with experiment: {experiment_name}")
    
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if present
        if 'Churn' in categorical_features:
            categorical_features.remove('Churn')
        
        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        # Create preprocessing steps
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
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def create_pipeline(self, model_type: str = 'gradient_boosting') -> Pipeline:
        """
        Create complete ML pipeline with preprocessing and model.
        
        Args:
            model_type (str): Type of model to use
            
        Returns:
            Pipeline: Complete ML pipeline
        """
        # Model selection
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=1,
                subsample=0.8,
                random_state=42
            ),
            'svm': SVC(
                probability=True, 
                random_state=42
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not supported. Choose from: {list(models.keys())}")
        
        model = models[model_type]
        self.model = model
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])
        
        self.pipeline = pipeline
        logger.info(f"Pipeline created with model: {model_type}")
        return pipeline
    
    def prepare_data(self, file_path: str, target_column: str = 'Churn') -> tuple:
        """
        Load and prepare data for training.
        
        Args:
            file_path (str): Path to the data file
            target_column (str): Name of the target column
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Load data
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        
        # Handle data cleaning
        df = self._clean_data(df)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        return X_train, X_test, y_train, y_test
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset.
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric if it's string
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Handle missing values
        if df_clean.isnull().sum().sum() > 0:
            logger.info("Handling missing values...")
            # Fill missing values will be handled by the preprocessing pipeline
        
        # Remove customerID if present (not useful for prediction)
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['customerID'])
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray, model_type: str = 'random_forest') -> Pipeline:
        """
        Train the pipeline.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training target
            model_type (str): Type of model to train
            
        Returns:
            Pipeline: Trained pipeline
        """
        with mlflow.start_run(run_name=f"{model_type}_pipeline"):
            start_time = datetime.now()
            
            # Create preprocessor and pipeline
            self.create_preprocessor(X_train)
            self.create_pipeline(model_type)
            
            # Train the pipeline
            logger.info(f"Training {model_type} pipeline...")
            self.pipeline.fit(X_train, y_train)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("training_samples", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("training_time_seconds", training_time)
            
            # Log model parameters
            model_params = self.pipeline.named_steps['classifier'].get_params()
            for param, value in model_params.items():
                mlflow.log_param(f"model_{param}", value)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Log the pipeline
            mlflow.sklearn.log_model(self.pipeline, "sklearn_pipeline")
            
        return self.pipeline
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
        """
        Evaluate the trained pipeline.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not trained yet. Call train() first.")
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics to MLflow
        with mlflow.start_run(nested=True):
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
        
        # Print detailed results
        print("\n=== Model Evaluation Results ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        logger.info("Model evaluation completed")
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: np.ndarray, cv: int = 5) -> dict:
        """
        Perform cross-validation.
        
        Args:
            X (pd.DataFrame): Features
            y (np.ndarray): Target
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not created yet.")
        
        # Perform cross-validation
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for score in scoring:
            scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring=score, n_jobs=-1)
            cv_results[f'{score}_mean'] = scores.mean()
            cv_results[f'{score}_std'] = scores.std()
        
        logger.info(f"Cross-validation completed with {cv} folds")
        return cv_results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                            model_type: str = 'random_forest') -> Pipeline:
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training target
            model_type (str): Type of model to tune
            
        Returns:
            Pipeline: Best pipeline from grid search
        """
        # Create base pipeline
        self.create_preprocessor(X_train)
        self.create_pipeline(model_type)
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'logistic_regression': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'lbfgs']
            },
            'gradient_boosting': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
        }
        
        if model_type not in param_grids:
            logger.warning(f"No parameter grid defined for {model_type}. Using default parameters.")
            return self.pipeline
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.pipeline,
            param_grids[model_type],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        grid_search.fit(X_train, y_train)
        
        # Update pipeline with best estimator
        self.pipeline = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self.pipeline
    
    def save_pipeline(self, file_path: str):
        """
        Save the trained pipeline to disk.
        
        Args:
            file_path (str): Path to save the pipeline
        """
        if self.pipeline is None:
            raise ValueError("No pipeline to save. Train a pipeline first.")
        
        # Save pipeline and label encoder
        pipeline_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(pipeline_data, file_path)
        logger.info(f"Pipeline saved to {file_path}")
    
    def load_pipeline(self, file_path: str):
        """
        Load a trained pipeline from disk.
        
        Args:
            file_path (str): Path to the saved pipeline
        """
        pipeline_data = joblib.load(file_path)
        
        self.pipeline = pipeline_data['pipeline']
        self.label_encoder = pipeline_data['label_encoder']
        self.feature_names = pipeline_data.get('feature_names', None)
        
        logger.info(f"Pipeline loaded from {file_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained pipeline.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if self.pipeline is None:
            raise ValueError("No trained pipeline available. Train or load a pipeline first.")
        
        predictions = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.pipeline is None:
            raise ValueError("No trained pipeline available. Train or load a pipeline first.")
        
        return self.pipeline.predict_proba(X)


def run_complete_pipeline(data_path: str, model_type: str = 'random_forest', 
                         tune_hyperparams: bool = False) -> TelcoChurnPipeline:
    """
    Run the complete pipeline from data loading to model evaluation.
    
    Args:
        data_path (str): Path to the dataset
        model_type (str): Type of model to train
        tune_hyperparams (bool): Whether to perform hyperparameter tuning
        
    Returns:
        TelcoChurnPipeline: Trained pipeline
    """
    # Initialize pipeline
    pipeline = TelcoChurnPipeline()
    
    # Prepare data
    X_train, X_test, y_train, y_test = pipeline.prepare_data(data_path)
    
    # Train model
    if tune_hyperparams:
        pipeline.hyperparameter_tuning(X_train, y_train, model_type)
    else:
        pipeline.train(X_train, y_train, model_type)
    
    # Evaluate model
    metrics = pipeline.evaluate(X_test, y_test)
    
    # Perform cross-validation
    cv_results = pipeline.cross_validate(
        pd.concat([X_train, X_test]), 
        np.concatenate([y_train, y_test])
    )
    
    print("\n=== Cross-Validation Results ===")
    for metric, value in cv_results.items():
        print(f"{metric}: {value:.4f}")
    
    return pipeline


if __name__ == "__main__":
    # Example usage
    print("Scikit-learn Pipeline for Telco Churn Prediction")
    print("=" * 50)
    
    # Example with sample data path
    data_path = "data/raw/Telco-Customer-Churn.csv"
    
    try:
        # Run pipeline with Gradient Boosting (default optimized model)
        pipeline = run_complete_pipeline(
            data_path=data_path,
            model_type='gradient_boosting',
            tune_hyperparams=False
        )
        
        # Save the trained pipeline
        pipeline.save_pipeline("../artifacts/models/sklearn_churn_pipeline.pkl")
        
        print("\nPipeline execution completed successfully!")
        
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please ensure the dataset is available at the specified location.")
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        logger.exception("Pipeline execution failed")