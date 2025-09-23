telco-churn-prediction-mini-project-1/
├── .gitignore                           # Git ignore rules
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
├── setup.py                            # Python packaging
├── file_structure.md                   # Project structure documentation
├── Mini-Project-1-Productionising-Telco-Churn-Prediction.pdf  # Project specification
│
├── data/                               # Datasets
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv   # Original dataset (7,043 records)
│   └── processed/
│       ├── columns.json                # Column metadata
│       └── sample.csv                  # Processed sample data (100 records)
│
├── notebooks/                          # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_dev_experiments.ipynb
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                     # FastAPI application
│   ├── data/
│   │   ├── __init__.py
│   │   ├── eda.py                     # Exploratory data analysis
│   │   ├── load_data.py               # Data loading utilities
│   │   └── preprocess.py              # Data preprocessing pipeline
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── batch_predict.py           # Batch prediction script
│   │   └── predict.py                 # Single prediction utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluate.py                # Model evaluation
│   │   ├── train.py                   # Basic training script
│   │   └── train_mlflow.py            # MLflow-enabled training
│   └── utils/
│       ├── __init__.py
│       └── logger.py                  # Logging utilities
│
├── pipelines/                         # ML Pipelines
│   ├── sklearn_pipeline.py            # Scikit-learn pipeline
│   └── spark_pipeline.py              # Apache Spark pipeline
│
├── dags/                              # Airflow DAGs
│   └── telco_churn_dag.py            # Complete ML pipeline DAG
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   └── test_inference.py              # Inference testing
│
├── artifacts/                         # Generated artifacts
│   ├── logs/
│   │   └── README.md                  # Log directory placeholder
│   ├── models/                        # Trained models
│   │   ├── feature_importances.json
│   │   ├── feature_names.json
│   │   ├── pipeline_metadata.json
│   │   ├── preprocessor.joblib
│   │   ├── sklearn_pipeline.joblib
│   │   ├── sklearn_pipeline_mlflow.joblib
│   │   └── spark_native/              # Spark model files
│   ├── metrics/                       # Model metrics
│   │   ├── sklearn_metrics.json
│   │   ├── sklearn_metrics_mlflow.json
│   │   └── spark_rf_metrics.json
│   └── predictions/                   # Prediction outputs
│       └── batch_preds.csv            # Batch prediction results
│
└── mlruns/                            # MLflow experiment tracking
    ├── .trash/
    ├── [experiment-ids]/              # MLflow experiments
    └── models/                        # MLflow model registry