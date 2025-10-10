# Makefile for Telco Churn Prediction ML Project
# Author: Dean Hettiarachchi
# Description: Comprehensive build automation for the ML pipeline

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
PROJECT_NAME := telco-churn-prediction
SRC_DIR := src
TESTS_DIR := tests
DATA_DIR := data
ARTIFACTS_DIR := artifacts
NOTEBOOKS_DIR := notebooks
REQUIREMENTS_FILE := requirements.txt
CONFIG_FILE := config.yaml

# Python environment
VENV := .venv
PYTHON_VERSION := 3.8

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

##@ Help
help: ## Display this help message
	@echo ""
	@echo "$(BLUE)Telco Churn Prediction ML Project$(NC)"
	@echo "$(BLUE)====================================$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(YELLOW)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Environment Setup
install: ## Install all dependencies
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	$(PIP) install -r $(REQUIREMENTS_FILE)
	$(PIP) install jupyter notebook ipykernel black flake8 isort mypy pre-commit
	@echo "$(GREEN)Development dependencies installed successfully!$(NC)"

setup-env: ## Create virtual environment and install dependencies
	@echo "$(YELLOW)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(YELLOW)Activating virtual environment and installing dependencies...$(NC)"
	./$(VENV)/Scripts/python.exe -m pip install --upgrade pip
	./$(VENV)/Scripts/python.exe -m pip install -r $(REQUIREMENTS_FILE)
	@echo "$(GREEN)Virtual environment created and configured!$(NC)"
	@echo "$(BLUE)To activate: $(VENV)\\Scripts\\activate$(NC)"

freeze: ## Generate requirements.txt from current environment
	$(PIP) freeze > requirements-freeze.txt
	@echo "$(GREEN)Requirements frozen to requirements-freeze.txt$(NC)"

##@ Project Setup
init-project: ## Initialize project structure and directories
	@echo "$(YELLOW)Initializing project structure...$(NC)"
	@mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed
	@mkdir -p $(ARTIFACTS_DIR)/models $(ARTIFACTS_DIR)/metrics $(ARTIFACTS_DIR)/predictions $(ARTIFACTS_DIR)/logs
	@mkdir -p $(TESTS_DIR)
	@touch $(DATA_DIR)/raw/README.md $(DATA_DIR)/processed/README.md
	@touch $(ARTIFACTS_DIR)/logs/README.md $(ARTIFACTS_DIR)/models/README.md
	@echo "$(GREEN)Project structure initialized!$(NC)"

check-config: ## Validate configuration files
	@echo "$(YELLOW)Checking configuration...$(NC)"
	@if [ -f $(CONFIG_FILE) ]; then \
		echo "$(GREEN)✓$(NC) config.yaml found"; \
		$(PYTHON) -c "import yaml; yaml.safe_load(open('$(CONFIG_FILE)'))" && echo "$(GREEN)✓$(NC) config.yaml is valid YAML"; \
	else \
		echo "$(RED)✗$(NC) config.yaml not found"; \
	fi
	@if [ -f config.py ]; then \
		echo "$(GREEN)✓$(NC) config.py found"; \
		$(PYTHON) -c "from config import get_config; get_config()" && echo "$(GREEN)✓$(NC) config.py loads successfully"; \
	else \
		echo "$(RED)✗$(NC) config.py not found"; \
	fi

##@ Code Quality
lint: ## Run linting (flake8)
	@echo "$(YELLOW)Running flake8 linting...$(NC)"
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 $(SRC_DIR) $(TESTS_DIR) --max-line-length=100 --ignore=E203,W503; \
		echo "$(GREEN)Linting completed!$(NC)"; \
	else \
		echo "$(RED)flake8 not installed. Run 'make install-dev' first.$(NC)"; \
	fi

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code with black and isort...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black $(SRC_DIR) $(TESTS_DIR) --line-length=100; \
		echo "$(GREEN)✓$(NC) black formatting completed"; \
	else \
		echo "$(RED)black not installed. Run 'make install-dev' first.$(NC)"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		isort $(SRC_DIR) $(TESTS_DIR) --profile black; \
		echo "$(GREEN)✓$(NC) isort import sorting completed"; \
	else \
		echo "$(RED)isort not installed. Run 'make install-dev' first.$(NC)"; \
	fi

type-check: ## Run type checking with mypy
	@echo "$(YELLOW)Running type checking with mypy...$(NC)"
	@if command -v mypy >/dev/null 2>&1; then \
		mypy $(SRC_DIR) --ignore-missing-imports; \
		echo "$(GREEN)Type checking completed!$(NC)"; \
	else \
		echo "$(RED)mypy not installed. Run 'make install-dev' first.$(NC)"; \
	fi

check: lint type-check ## Run all code quality checks

##@ Testing
test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	$(PYTEST) $(TESTS_DIR) -v --tb=short
	@echo "$(GREEN)All tests completed!$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	@if command -v pytest-cov >/dev/null 2>&1; then \
		$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term; \
	else \
		echo "$(RED)pytest-cov not installed. Installing...$(NC)"; \
		$(PIP) install pytest-cov; \
		$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term; \
	fi
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-inference: ## Run inference pipeline tests
	@echo "$(YELLOW)Testing inference pipeline...$(NC)"
	$(PYTEST) $(TESTS_DIR)/test_inference.py -v
	@echo "$(GREEN)Inference tests completed!$(NC)"

##@ Data Pipeline
load-data: ## Load and validate raw data
	@echo "$(YELLOW)Loading raw data...$(NC)"
	$(PYTHON) $(SRC_DIR)/data/load_data.py
	@echo "$(GREEN)Data loading completed!$(NC)"

preprocess: ## Run data preprocessing
	@echo "$(YELLOW)Running data preprocessing...$(NC)"
	$(PYTHON) $(SRC_DIR)/data/preprocess.py
	@echo "$(GREEN)Data preprocessing completed!$(NC)"

explore: ## Run data exploration notebook
	@echo "$(YELLOW)Starting Jupyter for data exploration...$(NC)"
	@if command -v jupyter >/dev/null 2>&1; then \
		jupyter notebook $(NOTEBOOKS_DIR)/01_data_exploration.ipynb; \
	else \
		echo "$(RED)Jupyter not installed. Run 'make install-dev' first.$(NC)"; \
	fi

##@ Model Training
train: ## Train the machine learning model
	@echo "$(YELLOW)Training ML model...$(NC)"
	$(PYTHON) $(SRC_DIR)/models/train_mlflow.py
	@echo "$(GREEN)Model training completed!$(NC)"

train-basic: ## Train basic model (without MLflow)
	@echo "$(YELLOW)Training basic ML model...$(NC)"
	$(PYTHON) $(SRC_DIR)/models/train.py
	@echo "$(GREEN)Basic model training completed!$(NC)"

evaluate: ## Evaluate trained model
	@echo "$(YELLOW)Evaluating model...$(NC)"
	$(PYTHON) $(SRC_DIR)/models/evaluate.py
	@echo "$(GREEN)Model evaluation completed!$(NC)"

##@ Model Inference
predict: ## Run batch predictions
	@echo "$(YELLOW)Running batch predictions...$(NC)"
	$(PYTHON) $(SRC_DIR)/inference/batch_predict.py
	@echo "$(GREEN)Batch predictions completed!$(NC)"

predict-single: ## Run single prediction example
	@echo "$(YELLOW)Running single prediction example...$(NC)"
	$(PYTHON) $(SRC_DIR)/inference/predict.py
	@echo "$(GREEN)Single prediction completed!$(NC)"

##@ MLflow Operations
mlflow-ui: ## Start MLflow UI server
	@echo "$(YELLOW)Starting MLflow UI...$(NC)"
	@echo "$(BLUE)Open http://localhost:5000 in your browser$(NC)"
	mlflow ui --backend-store-uri file:./mlruns --port 5000

mlflow-experiments: ## List MLflow experiments
	@echo "$(YELLOW)Listing MLflow experiments...$(NC)"
	mlflow experiments list

mlflow-runs: ## List recent MLflow runs
	@echo "$(YELLOW)Listing recent MLflow runs...$(NC)"
	mlflow runs list --experiment-name telco_churn_prediction

##@ Pipeline Orchestration
airflow-init: ## Initialize Airflow database
	@echo "$(YELLOW)Initializing Airflow database...$(NC)"
	airflow db init
	@echo "$(GREEN)Airflow database initialized!$(NC)"

airflow-webserver: ## Start Airflow webserver
	@echo "$(YELLOW)Starting Airflow webserver...$(NC)"
	@echo "$(BLUE)Open http://localhost:8080 in your browser$(NC)"
	airflow webserver --port 8080

airflow-scheduler: ## Start Airflow scheduler
	@echo "$(YELLOW)Starting Airflow scheduler...$(NC)"
	airflow scheduler

##@ Full Pipeline
pipeline: load-data preprocess train evaluate predict ## Run complete ML pipeline
	@echo "$(GREEN)Complete ML pipeline executed successfully!$(NC)"

pipeline-test: ## Run pipeline with testing
	@$(MAKE) load-data
	@$(MAKE) test-inference
	@$(MAKE) preprocess
	@$(MAKE) train
	@$(MAKE) test
	@$(MAKE) predict
	@echo "$(GREEN)Complete pipeline with testing executed successfully!$(NC)"

##@ Deployment
build: ## Build the project (install, lint, test)
	@$(MAKE) install
	@$(MAKE) lint
	@$(MAKE) test
	@echo "$(GREEN)Project built successfully!$(NC)"

package: ## Package the project for deployment
	@echo "$(YELLOW)Packaging project...$(NC)"
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)Project packaged in dist/ directory!$(NC)"

docker-build: ## Build Docker image (if Dockerfile exists)
	@if [ -f Dockerfile ]; then \
		echo "$(YELLOW)Building Docker image...$(NC)"; \
		docker build -t $(PROJECT_NAME):latest .; \
		echo "$(GREEN)Docker image built successfully!$(NC)"; \
	else \
		echo "$(RED)Dockerfile not found. Create one for Docker deployment.$(NC)"; \
	fi

##@ Utilities
clean: ## Clean up generated files and caches
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-artifacts: ## Clean up model artifacts and outputs
	@echo "$(YELLOW)Cleaning up artifacts...$(NC)"
	@rm -rf $(ARTIFACTS_DIR)/models/* $(ARTIFACTS_DIR)/metrics/* $(ARTIFACTS_DIR)/predictions/*
	@echo "$(GREEN)Artifacts cleaned!$(NC)"

clean-data: ## Clean up processed data (keeps raw data)
	@echo "$(YELLOW)Cleaning up processed data...$(NC)"
	@rm -rf $(DATA_DIR)/processed/*
	@echo "$(GREEN)Processed data cleaned!$(NC)"

reset: clean clean-artifacts clean-data ## Full reset - clean everything except raw data
	@echo "$(GREEN)Project reset completed!$(NC)"

status: ## Show project status and key information
	@echo "$(BLUE)Project Status$(NC)"
	@echo "=============="
	@echo "$(YELLOW)Python version:$(NC) $$($(PYTHON) --version)"
	@echo "$(YELLOW)Working directory:$(NC) $$(pwd)"
	@echo "$(YELLOW)Git branch:$(NC) $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo ""
	@echo "$(YELLOW)Key Files:$(NC)"
	@echo "  Config file: $$([ -f $(CONFIG_FILE) ] && echo '$(GREEN)✓$(NC)' || echo '$(RED)✗$(NC)') $(CONFIG_FILE)"
	@echo "  Requirements: $$([ -f $(REQUIREMENTS_FILE) ] && echo '$(GREEN)✓$(NC)' || echo '$(RED)✗$(NC)') $(REQUIREMENTS_FILE)"
	@echo "  Setup script: $$([ -f setup.py ] && echo '$(GREEN)✓$(NC)' || echo '$(RED)✗$(NC)') setup.py"
	@echo ""
	@echo "$(YELLOW)Directories:$(NC)"
	@echo "  Source: $$([ -d $(SRC_DIR) ] && echo '$(GREEN)✓$(NC)' || echo '$(RED)✗$(NC)') $(SRC_DIR)/"
	@echo "  Tests: $$([ -d $(TESTS_DIR) ] && echo '$(GREEN)✓$(NC)' || echo '$(RED)✗$(NC)') $(TESTS_DIR)/"
	@echo "  Data: $$([ -d $(DATA_DIR) ] && echo '$(GREEN)✓$(NC)' || echo '$(RED)✗$(NC)') $(DATA_DIR)/"
	@echo "  Artifacts: $$([ -d $(ARTIFACTS_DIR) ] && echo '$(GREEN)✓$(NC)' || echo '$(RED)✗$(NC)') $(ARTIFACTS_DIR)/"
	@echo "  Notebooks: $$([ -d $(NOTEBOOKS_DIR) ] && echo '$(GREEN)✓$(NC)' || echo '$(RED)✗$(NC)') $(NOTEBOOKS_DIR)/"

info: status ## Alias for status

##@ Documentation
docs: ## Generate documentation (if available)
	@echo "$(YELLOW)Generating documentation...$(NC)"
	@if [ -f docs/Makefile ]; then \
		cd docs && make html; \
		echo "$(GREEN)Documentation generated in docs/_build/html/$(NC)"; \
	else \
		echo "$(RED)No documentation setup found.$(NC)"; \
	fi

notebook-docs: ## Convert notebooks to documentation
	@echo "$(YELLOW)Converting notebooks to documentation...$(NC)"
	@if command -v jupyter >/dev/null 2>&1; then \
		jupyter nbconvert $(NOTEBOOKS_DIR)/*.ipynb --to html --output-dir docs/notebooks/; \
		echo "$(GREEN)Notebook documentation generated in docs/notebooks/$(NC)"; \
	else \
		echo "$(RED)Jupyter not installed. Run 'make install-dev' first.$(NC)"; \
	fi

##@ Kafka Infrastructure (Mini Project 2)
kafka-up: ## Start Kafka/Redpanda services (Redpanda + Console)
	@echo "$(YELLOW)Starting Kafka infrastructure...$(NC)"
	docker compose -f docker-compose.kafka.yml up -d
	@echo "$(GREEN)Kafka services started!$(NC)"
	@echo "$(BLUE)Redpanda Console:$(NC) http://localhost:8080"
	@echo "$(BLUE)Kafka Broker:$(NC) localhost:19092"

kafka-down: ## Stop Kafka services
	@echo "$(YELLOW)Stopping Kafka services...$(NC)"
	docker compose -f docker-compose.kafka.yml down
	@echo "$(GREEN)Kafka services stopped.$(NC)"

kafka-down-volumes: ## Stop Kafka and remove all data volumes (DESTRUCTIVE)
	@echo "$(RED)WARNING: This will delete all Kafka topics and messages!$(NC)"
	@echo "$(YELLOW)Stopping Kafka and removing volumes...$(NC)"
	docker compose -f docker-compose.kafka.yml down -v
	@echo "$(GREEN)Kafka services stopped and volumes removed.$(NC)"

kafka-status: ## Check Kafka service health
	@echo "$(YELLOW)Checking Kafka service status...$(NC)"
	@docker compose -f docker-compose.kafka.yml ps
	@echo ""
	@echo "$(YELLOW)Redpanda cluster health:$(NC)"
	@docker exec telco-redpanda rpk cluster health 2>/dev/null || echo "$(RED)Redpanda not running$(NC)"

kafka-topics: ## Create Kafka topics (telco.raw.customers, telco.churn.predictions, telco.deadletter)
	@echo "$(YELLOW)Creating Kafka topics...$(NC)"
	@if [ -x scripts/kafka_create_topics.sh ]; then \
		bash scripts/kafka_create_topics.sh; \
	else \
		chmod +x scripts/kafka_create_topics.sh && bash scripts/kafka_create_topics.sh; \
	fi

kafka-list: ## List all Kafka topics
	@echo "$(YELLOW)Listing Kafka topics...$(NC)"
	@docker exec telco-redpanda rpk topic list

kafka-describe: ## Describe a Kafka topic (usage: make kafka-describe TOPIC=telco.raw.customers)
	@if [ -z "$(TOPIC)" ]; then \
		echo "$(RED)Error: TOPIC variable required$(NC)"; \
		echo "Usage: make kafka-describe TOPIC=telco.raw.customers"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Describing topic: $(TOPIC)$(NC)"
	@docker exec telco-redpanda rpk topic describe $(TOPIC)

kafka-console: ## Open Redpanda Console in browser
	@echo "$(BLUE)Opening Redpanda Console...$(NC)"
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:8080; \
	elif command -v open >/dev/null 2>&1; then \
		open http://localhost:8080; \
	elif command -v start >/dev/null 2>&1; then \
		start http://localhost:8080; \
	else \
		echo "$(YELLOW)Please open http://localhost:8080 in your browser$(NC)"; \
	fi

kafka-consume: ## Consume messages from a topic (usage: make kafka-consume TOPIC=telco.raw.customers)
	@if [ -z "$(TOPIC)" ]; then \
		echo "$(RED)Error: TOPIC variable required$(NC)"; \
		echo "Usage: make kafka-consume TOPIC=telco.raw.customers"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Consuming from topic: $(TOPIC)$(NC)"
	@echo "$(BLUE)Press Ctrl+C to stop$(NC)"
	@docker exec -it telco-redpanda rpk topic consume $(TOPIC) --from-beginning

kafka-produce: ## Produce test message to a topic (usage: make kafka-produce TOPIC=telco.raw.customers MSG='{"test":"data"}')
	@if [ -z "$(TOPIC)" ]; then \
		echo "$(RED)Error: TOPIC variable required$(NC)"; \
		echo "Usage: make kafka-produce TOPIC=telco.raw.customers MSG='{\"test\":\"data\"}'"; \
		exit 1; \
	fi
	@if [ -z "$(MSG)" ]; then \
		echo "$(RED)Error: MSG variable required$(NC)"; \
		echo "Usage: make kafka-produce TOPIC=telco.raw.customers MSG='{\"test\":\"data\"}'"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Producing message to topic: $(TOPIC)$(NC)"
	@echo "$(MSG)" | docker exec -i telco-redpanda rpk topic produce $(TOPIC)
	@echo "$(GREEN)Message sent successfully!$(NC)"

kafka-logs: ## Show Kafka/Redpanda logs
	@echo "$(YELLOW)Showing Redpanda logs (last 50 lines)...$(NC)"
	@docker compose -f docker-compose.kafka.yml logs --tail=50 redpanda

kafka-logs-follow: ## Follow Kafka logs in real-time
	@echo "$(YELLOW)Following Kafka logs (Ctrl+C to stop)...$(NC)"
	@docker compose -f docker-compose.kafka.yml logs -f

kafka-reset: ## Full reset - stop, remove volumes, restart, recreate topics
	@echo "$(RED)WARNING: This will delete all Kafka data!$(NC)"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "$(YELLOW)Performing full Kafka reset...$(NC)"
	$(MAKE) kafka-down-volumes
	@sleep 2
	$(MAKE) kafka-up
	@echo "$(YELLOW)Waiting for Redpanda to be ready...$(NC)"
	@sleep 10
	$(MAKE) kafka-topics
	@echo "$(GREEN)Kafka reset complete!$(NC)"

kafka-test: ## Test Kafka setup (send and receive test message)
	@echo "$(YELLOW)Testing Kafka setup...$(NC)"
	@echo "$(BLUE)1. Sending test message...$(NC)"
	@echo '{"customerID":"TEST-$(shell date +%s)","test":true,"timestamp":"$(shell date -u +%Y-%m-%dT%H:%M:%SZ)"}' | \
		docker exec -i telco-redpanda rpk topic produce telco.raw.customers
	@echo "$(GREEN)✓ Message sent$(NC)"
	@sleep 1
	@echo "$(BLUE)2. Consuming last message...$(NC)"
	@docker exec telco-redpanda rpk topic consume telco.raw.customers --num 1 --offset -1
	@echo "$(GREEN)✓ Kafka test successful!$(NC)"

# Phony targets (targets that don't create files)
.PHONY: help install install-dev setup-env freeze init-project check-config
.PHONY: lint format type-check check test test-coverage test-inference
.PHONY: load-data preprocess explore train train-basic evaluate predict predict-single
.PHONY: mlflow-ui mlflow-experiments mlflow-runs airflow-init airflow-webserver airflow-scheduler
.PHONY: pipeline pipeline-test build package docker-build
.PHONY: clean clean-artifacts clean-data reset status info docs notebook-docs
.PHONY: kafka-up kafka-down kafka-down-volumes kafka-status kafka-topics kafka-list kafka-describe
.PHONY: kafka-console kafka-consume kafka-produce kafka-logs kafka-logs-follow kafka-reset kafka-test
