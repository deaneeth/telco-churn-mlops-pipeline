# 🚀 Quick Start Guide

Get the Telco Churn MLOps Pipeline running in **10 minutes**!

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.12 or higher installed
- [ ] Git installed
- [ ] 8GB+ RAM available
- [ ] Internet connection (for downloading data)

---

## Step 1: Clone the Repository (1 minute)

```bash
# Clone the repository
git clone https://github.com/deaneeth/telco-churn-mlops-pipeline.git

# Navigate to project directory
cd telco-churn-mlops-pipeline

# Verify you're on the main branch
git branch
```

**Expected output**: `* main`

---

## Step 2: Set Up Python Environment (2 minutes)

### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Verify activation
which python  # Should point to .venv
```

### Option B: Using conda

```bash
# Create conda environment
conda create -n telco-churn python=3.12 -y

# Activate environment
conda activate telco-churn
```

---

## Step 3: Install Dependencies (3 minutes)

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Verify installation
pip list | grep -E "scikit-learn|mlflow|flask|pyspark"
```

**Expected packages**:
- scikit-learn 1.6.1
- mlflow 2.17.2
- flask 3.1.0
- pyspark 4.0.0

---

## Step 4: Download Dataset (1 minute)

```bash
# Create data directory if it doesn't exist
mkdir -p data/raw

# Download dataset (if not already present)
# Option 1: Manual download
# Visit: https://www.kaggle.com/blastchar/telco-customer-churn
# Place CSV in: data/raw/Telco-Customer-Churn.csv

# Option 2: Using kaggle CLI (if configured)
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```

**Verify**: Check that `data/raw/Telco-Customer-Churn.csv` exists

---

## Step 5: Run Your First Test (1 minute)

```bash
# Run a quick smoke test
pytest tests/test_data_validation.py -v

# Expected output:
# ✅ 10+ tests passed
```

---

## Step 6: Start the API Server (1 minute)

```bash
# Start Flask API
python src/api/app.py

# You should see:
# [OK] Model loaded successfully
# [OK] Flask app initialized
# [OK] Starting production server on http://0.0.0.0:5000
```

**Keep this terminal open!**

---

## Step 7: Test the API (1 minute)

Open a **new terminal** and test the endpoints:

### Test Health Check

```bash
# Windows PowerShell:
Invoke-WebRequest -Uri "http://localhost:5000/ping" -Method GET -UseBasicParsing | Select-Object -ExpandProperty Content

# macOS/Linux:
curl http://localhost:5000/ping

# Expected output: "pong"
```

### Test Prediction

```bash
# Create test customer data
$testCustomer = @{
    gender = "Female"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "No"
    tenure = 12
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "DSL"
    OnlineSecurity = "Yes"
    OnlineBackup = "No"
    DeviceProtection = "Yes"
    TechSupport = "No"
    StreamingTV = "Yes"
    StreamingMovies = "No"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 65.50
    TotalCharges = 786.0
} | ConvertTo-Json

# Make prediction (PowerShell)
Invoke-WebRequest -Uri "http://localhost:5000/predict" -Method POST -Body $testCustomer -ContentType "application/json" | Select-Object -ExpandProperty Content

# Expected output (JSON):
# {
#   "message": "Prediction successful",
#   "prediction": 0 or 1,
#   "probability": 0.XX,
#   "success": true
# }
```

---

## 🎉 Success! You're Running!

If you see the prediction response, **congratulations!** You've successfully:

✅ Set up the environment  
✅ Installed dependencies  
✅ Loaded the dataset  
✅ Started the API server  
✅ Made your first prediction  

---

## What's Next?

### Learn More About the System

1. **[Model Architecture](Model-Architecture)** - Understand the ML model
2. **[API Endpoints](API-Endpoints)** - Full API documentation
3. **[Training a New Model](Training-a-New-Model)** - Retrain with your data

### Explore MLOps Features

4. **[MLflow Setup](MLflow-Setup)** - Start experiment tracking
5. **[Airflow Orchestration](Airflow-Orchestration)** - Automate workflows
6. **[Docker Deployment](Docker-Deployment)** - Containerize the app

### Run the Full Test Suite

```bash
# Run all 93 tests
pytest -v

# Expected: 93 passed, 4 skipped, 10 warnings
```

---

## Common Issues

### Issue 1: Port 5000 Already in Use

**Solution**:
```bash
# Find process using port 5000 (Windows)
netstat -ano | findstr :5000

# Kill the process
taskkill /PID <PID> /F

# Or change port in config.yaml
```

### Issue 2: Module Not Found

**Solution**:
```bash
# Ensure virtual environment is activated
# Check Python path
which python

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Issue 3: Model File Not Found

**Solution**:
```bash
# Train a new model
python src/models/train.py

# Or download pre-trained model
# Check artifacts/models/ directory
```

---

## Quick Reference Card

| Command | Purpose |
|---------|---------|
| `python src/api/app.py` | Start API server |
| `pytest -v` | Run all tests |
| `python src/models/train.py` | Train new model |
| `mlflow ui` | Start MLflow UI |
| `docker build -t telco-churn .` | Build Docker image |

---

## 📞 Need Help?

- 📖 **Full Documentation**: [Installation Guide](Installation)
- 🐛 **Issues**: [GitHub Issues](https://github.com/deaneeth/telco-churn-mlops-pipeline/issues)
- 💬 **Questions**: [Discussions](https://github.com/deaneeth/telco-churn-mlops-pipeline/discussions)

---

**Estimated Time**: 10 minutes  
**Difficulty**: Beginner  
**Last Updated**: January 2025

---

[← Back to Home](Home) | [Next: Full Installation →](Installation)
