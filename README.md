# Telco Churn Prediction - Mini Project 1

## Project Overview
This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. The project implements both traditional scikit-learn and PySpark MLlib pipelines for scalable machine learning.

## Features
- Data preprocessing and feature engineering
- Model training and evaluation
- Real-time inference pipeline
- MLflow experiment tracking
- Airflow orchestration
- Comprehensive testing

## Setup Instructions

### Prerequisites
- Python 3.8+
- Apache Spark (for PySpark pipeline)
- Apache Airflow (for orchestration)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the setup:
   ```bash
   python setup.py install
   ```

## Usage

### Data Exploration
Start with the Jupyter notebooks in the `notebooks/` directory:
1. `01_data_exploration.ipynb` - Initial data analysis
2. `02_feature_engineering.ipynb` - Feature creation and selection
3. `03_model_dev_experiments.ipynb` - Model development and experiments

### Training
Run the training pipeline:
```bash
python src/models/train.py
```

### Inference
Make predictions:
```bash
python src/inference/predict.py
```

## Project Structure
See `file_structure.md` for detailed project organization.

## License
This project is licensed under the MIT License.