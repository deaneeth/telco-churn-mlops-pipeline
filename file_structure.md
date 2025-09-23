# 📂 Project Structure - Telco Churn Prediction (Mini Project 1)

Telco-Churn-Prediction/
│── README.md                # Project overview, setup, and instructions
│── requirements.txt         # Python dependencies
│── setup.py                 # (Optional) for packaging if needed
│── .gitignore               # Ignore unnecessary files
│
├── data/                    # Datasets
│   ├── raw/                 # Original raw dataset(s)
│   ├── processed/           # Cleaned & feature-engineered data
│
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_dev_experiments.ipynb
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── data/                # Data loading & preprocessing
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── models/              # Model training & evaluation
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── inference/           # Prediction pipeline
│   │   ├── __init__.py
│   │   └── predict.py
│   └── utils/               # Helper functions
│       ├── __init__.py
│       └── logger.py
│
├── pipelines/               # ML pipelines
│   ├── sklearn_pipeline.py  # Scikit-learn pipeline
│   └── spark_pipeline.py    # PySpark MLlib pipeline
│
├── mlruns/                  # MLflow tracking artifacts (auto-generated)
│
├── dags/                    # Airflow DAGs for orchestration
│   └── telco_churn_dag.py
│
├── tests/                   # Unit & integration tests
│   ├── __init__.py
│   └── test_inference.py
│
└── artifacts/               # Saved models & logs
    ├── models/              # Pickle/Joblib models
    └── logs/                # Training & evaluation logs
