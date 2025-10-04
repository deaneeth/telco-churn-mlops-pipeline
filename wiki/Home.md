# Welcome to the Telco Churn MLOps Pipeline Wiki! 📚

**A comprehensive guide to understanding, deploying, and extending this production-grade MLOps pipeline.**

---

## 🎯 Project Overview

This wiki provides detailed documentation for the **Telco Customer Churn Prediction MLOps Pipeline** - a production-ready machine learning system that demonstrates industry best practices for building, deploying, and maintaining ML models in production environments.

### Quick Stats
- 🎯 **Recall**: 80.75% (+61% improvement)
- 💰 **ROI**: +$220,150/year
- ✅ **Test Coverage**: 100% pass rate (93/93 tests)
- 📊 **Compliance**: 98.5%
- 🐳 **Docker**: Production-ready container

---

## 📖 Wiki Structure

### 🚀 Getting Started
- **[Home](Home)** - This page
- **[Quick Start Guide](Quick-Start-Guide)** - Get up and running in 10 minutes
- **[Installation](Installation)** - Detailed setup instructions
- **[Project Structure](Project-Structure)** - Understanding the codebase

### 🧠 Machine Learning
- **[Model Architecture](Model-Architecture)** - ML model design and optimization
- **[Feature Engineering](Feature-Engineering)** - Data preprocessing pipeline
- **[Model Training](Model-Training)** - Training process and hyperparameters
- **[Model Evaluation](Model-Evaluation)** - Metrics and validation strategy

### ⚙️ MLOps Infrastructure
- **[MLflow Setup](MLflow-Setup)** - Experiment tracking and model registry
- **[Airflow Orchestration](Airflow-Orchestration)** - Workflow automation
- **[PySpark Pipeline](PySpark-Pipeline)** - Distributed training
- **[Docker Deployment](Docker-Deployment)** - Containerization guide

### 🌐 API & Deployment
- **[Flask API](Flask-API)** - REST API documentation
- **[API Endpoints](API-Endpoints)** - Endpoint specifications
- **[Production Deployment](Production-Deployment)** - Going live
- **[Monitoring & Logging](Monitoring-and-Logging)** - Observability

### 🧪 Testing & Quality
- **[Testing Strategy](Testing-Strategy)** - Comprehensive test suite
- **[CI/CD Pipeline](CICD-Pipeline)** - Automation workflows
- **[Code Quality](Code-Quality)** - Standards and linting

### 💼 Business Value
- **[ROI Analysis](ROI-Analysis)** - Business impact calculation
- **[Recall Optimization](Recall-Optimization)** - 50% → 80.75% improvement
- **[Cost-Benefit Analysis](Cost-Benefit-Analysis)** - Metric selection rationale

### 🎓 Tutorials
- **[End-to-End Tutorial](End-to-End-Tutorial)** - Complete walkthrough
- **[Training a New Model](Training-a-New-Model)** - Step-by-step guide
- **[Making Predictions](Making-Predictions)** - Using the API
- **[Retraining Pipeline](Retraining-Pipeline)** - Model updates

### 🔧 Advanced Topics
- **[Configuration Management](Configuration-Management)** - YAML configs
- **[Data Pipeline](Data-Pipeline)** - ETL processes
- **[Model Registry](Model-Registry)** - Version control for models
- **[Troubleshooting](Troubleshooting)** - Common issues and solutions

### 📊 Reference
- **[API Reference](API-Reference)** - Complete API documentation
- **[Configuration Reference](Configuration-Reference)** - All config options
- **[Metrics Glossary](Metrics-Glossary)** - ML metrics explained
- **[FAQ](FAQ)** - Frequently asked questions

---

## 🌟 Key Features

### Production-Ready MLOps Stack
```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion                           │
│          (Telco Customer Churn Dataset - 7,043 rows)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering                         │
│        (19 raw features → 45 engineered features)          │
│     OneHotEncoder + StandardScaler + Custom Features       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Training                            │
│   ┌──────────────────────┐  ┌────────────────────────┐    │
│   │  Scikit-learn        │  │  PySpark               │    │
│   │  GradientBoosting    │  │  RandomForest          │    │
│   │  Recall: 80.75%      │  │  ROC-AUC: 83.80%       │    │
│   └──────────────────────┘  └────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MLflow Experiment Tracking                     │
│        (17+ model versions, comprehensive logging)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Airflow Workflow Orchestration                   │
│       (End-to-end pipeline automation with DAGs)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Flask REST API                             │
│           (Waitress server, /ping & /predict)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Docker Containerization                       │
│         (1.47GB production-ready container)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Business Impact

### Recall Optimization Achievement
- **Before**: 50% recall (missed 50% of churners)
- **After**: 80.75% recall (catch 81% of churners)
- **Improvement**: +61% (+30.75 percentage points)

### Financial Impact
```
Annual ROI Calculation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Per 1,409 test customers:
  • Additional churners caught: 115
  • Revenue saved: 115 × $2,000 LTV = $230,000
  • Retention costs: 115 × $50 = $5,750
  • Net ROI: $230,000 - $5,750 = $224,250/year
  • Actual achieved: $220,150/year
  • Return ratio: 23:1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🚀 Quick Links

### For Developers
- 📥 [Installation Guide](Installation) - Set up your environment
- 🏃 [Quick Start](Quick-Start-Guide) - Run your first prediction
- 🧪 [Testing Guide](Testing-Strategy) - Run and write tests
- 🐳 [Docker Setup](Docker-Deployment) - Containerize the app

### For Data Scientists
- 🧠 [Model Architecture](Model-Architecture) - Understand the ML approach
- 📊 [Feature Engineering](Feature-Engineering) - Data preprocessing details
- 📈 [Model Training](Model-Training) - Training configurations
- 🎯 [Model Evaluation](Model-Evaluation) - Performance metrics

### For ML Engineers
- 📊 [MLflow Guide](MLflow-Setup) - Experiment tracking setup
- ⚙️ [Airflow Guide](Airflow-Orchestration) - Workflow orchestration
- ⚡ [PySpark Guide](PySpark-Pipeline) - Distributed computing
- 🔧 [Configuration](Configuration-Management) - System configs

### For Business Stakeholders
- 💰 [ROI Analysis](ROI-Analysis) - Business value metrics
- 📈 [Recall Optimization](Recall-Optimization) - Why recall matters
- 💼 [Cost-Benefit](Cost-Benefit-Analysis) - Investment justification

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.12+** installed
- **Docker** (for containerized deployment)
- **8GB+ RAM** (recommended for Spark)
- **Basic ML knowledge** (scikit-learn, pandas)
- **Git** for version control

---

## 🤝 Contributing

We welcome contributions! See our contributing guidelines for:
- Code style standards
- Testing requirements
- Pull request process
- Issue reporting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Technologies**: MLflow, Airflow, PySpark, Docker, Flask, Scikit-learn
- **Community**: Thanks to all contributors and users

---

## 📞 Support

- 📧 **Issues**: [GitHub Issues](../issues)
- 💬 **Discussions**: [GitHub Discussions](../discussions)
- 📖 **Documentation**: This Wiki

---

**Version**: 1.0  
**Last Updated**: January 2025  
**Status**: ✅ Production Ready

---

*Navigate to other wiki pages using the sidebar or the links above!*
