# Kafka Integration Testing Guide

Comprehensive guide for running end-to-end integration tests for the Kafka streaming pipeline.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Test Architecture](#test-architecture)
- [Running Tests](#running-tests)
- [Test Scenarios](#test-scenarios)
- [Troubleshooting](#troubleshooting)
- [CI/CD Integration](#cicd-integration)

---

## Overview

The integration test suite validates the complete message flow:

```
Producer → Kafka Topic → Consumer → Predictions Topic
```

### What's Tested

✅ **Broker Connectivity**: Kafka/Redpanda availability  
✅ **Topic Management**: Topic creation and configuration  
✅ **Producer Publishing**: Batch message publishing  
✅ **Consumer Processing**: End-to-end message processing  
✅ **Schema Validation**: Prediction message schema  
✅ **Dead Letter Queue**: Invalid message handling  
✅ **Consumer Resilience**: Offset management and restart recovery  

### Test Environment

- **Broker**: Redpanda (Kafka-compatible) on port `19093`
- **Topics**: Test-specific topics with `.test` suffix
- **Storage**: Ephemeral (tmpfs) for fast teardown
- **Isolation**: Separate from development environment

---

## Prerequisites

### Required Software

1. **Docker Desktop** (or Docker Engine + Docker Compose)
   - Version: 20.10+
   - Download: https://www.docker.com/products/docker-desktop

2. **Python 3.8+**
   - Verify: `python --version`

3. **pytest and dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### System Requirements

- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 2GB free space
- **Ports**: `19093`, `18084`, `18083`, `9645` must be available

### Verify Prerequisites

**Linux/macOS:**
```bash
bash scripts/run_kafka_integration_tests.sh --help
```

**Windows (PowerShell):**
```powershell
.\scripts\run_kafka_integration_tests.ps1 -Help
```

---

## Quick Start

### Linux/macOS

```bash
# Run all integration tests
bash scripts/run_kafka_integration_tests.sh

# With verbose output
bash scripts/run_kafka_integration_tests.sh --verbose

# Keep containers running for debugging
bash scripts/run_kafka_integration_tests.sh --keep-containers
```

### Windows (PowerShell)

```powershell
# Run all integration tests
.\scripts\run_kafka_integration_tests.ps1

# With verbose output
.\scripts\run_kafka_integration_tests.ps1 -VerboseOutput

# Keep containers running for debugging
.\scripts\run_kafka_integration_tests.ps1 -KeepContainers
```

### Expected Output

```
[INFO] =========================================
[INFO] Kafka Integration Test Runner
[INFO] =========================================
[INFO] Checking prerequisites...
[SUCCESS] All prerequisites satisfied
[INFO] STEP 1: Starting Kafka test environment...
[SUCCESS] Docker Compose started
[INFO] STEP 2: Waiting for broker to be ready...
[SUCCESS] Kafka broker is ready
[INFO] STEP 3: Running integration tests...
tests/test_kafka_integration.py::test_kafka_broker_connectivity PASSED
tests/test_kafka_integration.py::test_topic_creation PASSED
tests/test_kafka_integration.py::test_producer_publish_batch PASSED
tests/test_kafka_integration.py::test_end_to_end_message_flow PASSED
tests/test_kafka_integration.py::test_deadletter_handling PASSED
tests/test_kafka_integration.py::test_consumer_resilience PASSED
[SUCCESS] All integration tests PASSED ✓
```

---

## Test Architecture

### Test Structure

```
tests/
├── test_kafka_integration.py       # Main integration test suite
├── conftest.py                      # Shared fixtures
└── test_data/                       # Test data (auto-generated)

scripts/
├── run_kafka_integration_tests.sh  # Linux/macOS test runner
└── run_kafka_integration_tests.ps1 # Windows test runner

docker-compose.test.yml              # Test Kafka environment
```

### Key Components

#### 1. Docker Compose Test Environment

**File**: `docker-compose.test.yml`

```yaml
services:
  redpanda-test:
    image: docker.redpanda.com/redpandadata/redpanda:v24.2.4
    ports:
      - "19093:19093"  # Kafka API (test port)
    tmpfs:
      - /var/lib/redpanda/data  # Ephemeral storage
```

**Features**:
- Single broker (dev-container mode)
- 512MB memory limit (CI-friendly)
- Fast startup (~10 seconds)
- No persistent volumes

#### 2. Test Fixtures

**File**: `tests/test_kafka_integration.py`

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `kafka_broker_ready` | module | Wait for broker availability |
| `kafka_topics` | module | Create and cleanup test topics |
| `test_model_path` | module | Generate minimal test model |
| `test_data_batch` | module | 50 fixed test messages |
| `consumer_process` | function | Background consumer subprocess |

#### 3. Test Harness Scripts

**Purpose**: Orchestrate full test workflow

**Workflow**:
1. Check prerequisites (Docker, Python, pytest)
2. Start Docker Compose
3. Wait for broker health check
4. Run pytest with markers
5. Collect logs to `reports/`
6. Tear down environment

---

## Running Tests

### Using Test Runner Scripts (Recommended)

**Advantages**:
- Automatic environment setup/teardown
- Log collection
- Error handling
- CI-ready

**Linux/macOS:**
```bash
bash scripts/run_kafka_integration_tests.sh [OPTIONS]
```

**Windows:**
```powershell
.\scripts\run_kafka_integration_tests.ps1 [OPTIONS]
```

**Options**:
- `--keep-containers` / `-KeepContainers`: Don't tear down after tests
- `--verbose` / `-VerboseOutput`: Show detailed output

### Manual Test Execution

**1. Start test environment:**
```bash
docker-compose -f docker-compose.test.yml up -d
```

**2. Wait for broker:**
```bash
# Check broker health
docker exec telco-redpanda-test rpk cluster health
```

**3. Run tests:**
```bash
pytest tests/test_kafka_integration.py -v -m "integration and kafka"
```

**4. Cleanup:**
```bash
docker-compose -f docker-compose.test.yml down -v
```

### Running Specific Tests

**Single test:**
```bash
pytest tests/test_kafka_integration.py::test_end_to_end_message_flow -v
```

**By marker:**
```bash
# All integration tests
pytest -m integration -v

# Only Kafka tests
pytest -m kafka -v

# Slow tests only
pytest -m slow -v
```

### Debugging Failed Tests

**1. Keep containers running:**
```bash
bash scripts/run_kafka_integration_tests.sh --keep-containers --verbose
```

**2. Inspect broker:**
```bash
# Check broker logs
docker-compose -f docker-compose.test.yml logs redpanda-test

# List topics
docker exec telco-redpanda-test rpk topic list

# Consume messages manually
docker exec telco-redpanda-test rpk topic consume telco.raw.customers.test
```

**3. Check test logs:**
```bash
cat reports/kafka_integration.log
cat reports/redpanda.log
```

---

## Test Scenarios

### 1. Broker Connectivity Test

**Test**: `test_kafka_broker_connectivity`

**Validates**:
- Kafka broker is accessible
- Admin API works
- Cluster metadata available

**Duration**: ~1 second

---

### 2. Topic Creation Test

**Test**: `test_topic_creation`

**Validates**:
- Topics created successfully
- Correct partition count (3 for input/output, 1 for deadletter)
- Topic configuration applied

**Duration**: ~2 seconds

---

### 3. Producer Batch Publish Test

**Test**: `test_producer_publish_batch`

**Validates**:
- Producer can publish 50 messages
- All messages acknowledged
- No publish failures

**Duration**: ~3 seconds

**Expected Results**:
- 50/50 messages published successfully
- Success rate: 100%

---

### 4. End-to-End Message Flow Test ⭐

**Test**: `test_end_to_end_message_flow`

**This is the primary integration test.**

**Flow**:
1. Start consumer in background
2. Publish 50 test messages to input topic
3. Wait 30 seconds for processing
4. Consume predictions from output topic
5. Validate schema and content

**Validates**:
- Complete pipeline functionality
- Message processing (>80% success rate expected)
- Prediction schema compliance
- Customer ID matching

**Duration**: ~60 seconds

**Expected Results**:
```json
{
  "messages_sent": 50,
  "predictions_received": 45,  // 90% success rate
  "success_rate": 90.0,
  "customer_id_overlap": 45,
  "status": "PASSED"
}
```

**Assertions**:
- ✅ At least 40 predictions received (80% threshold)
- ✅ Each prediction has required fields
- ✅ `prediction` is 0 or 1
- ✅ `churn_probability` is between 0 and 1
- ✅ Customer IDs match input messages

---

### 5. Dead Letter Queue Test

**Test**: `test_deadletter_handling`

**Validates**:
- Invalid messages routed to DLQ
- Dead letter schema compliance
- Error information captured

**Flow**:
1. Publish intentionally invalid messages
2. Wait for processing
3. Consume from dead letter topic
4. Validate error metadata

**Expected Results**:
- Invalid messages appear in DLQ
- `error_type` field present
- `original_message` preserved

**Duration**: ~15 seconds

---

### 6. Consumer Resilience Test

**Test**: `test_consumer_resilience`

**Validates**:
- Offset commit mechanism
- Consumer restart recovery
- No message reprocessing

**Flow**:
1. Publish 25 messages
2. Consume first 10 messages
3. Close consumer (commits offsets)
4. Restart consumer with same group ID
5. Verify it resumes from message 11

**Expected Results**:
- First batch: 10 messages
- Second batch: 15 messages
- No overlap (no reprocessing)

**Duration**: ~10 seconds

---

## Troubleshooting

### Issue: "Docker is not installed"

**Symptoms**:
```
[ERROR] Docker is not installed. Please install Docker first.
```

**Solution**:
Install Docker Desktop:
- Windows: https://docs.docker.com/desktop/install/windows-install/
- macOS: https://docs.docker.com/desktop/install/mac-install/
- Linux: https://docs.docker.com/engine/install/

---

### Issue: "Kafka broker failed to become ready"

**Symptoms**:
```
[ERROR] Kafka broker failed to become ready within 60s
```

**Diagnosis**:
```bash
# Check container status
docker ps -a | grep redpanda-test

# Check container logs
docker logs telco-redpanda-test
```

**Solutions**:
1. **Port conflict**: Port 19093 already in use
   ```bash
   # Windows
   netstat -ano | findstr :19093
   
   # Linux/macOS
   lsof -i :19093
   ```
   Stop the conflicting process or change test port.

2. **Insufficient memory**: Increase Docker memory to 4GB+

3. **Docker daemon not running**: Start Docker Desktop

---

### Issue: "Consumer process failed to start"

**Symptoms**:
```
pytest.fail: Consumer process failed to start
```

**Diagnosis**:
```bash
# Check consumer log
cat /tmp/pytest-*/logs/consumer.log
```

**Common Causes**:
1. **Model not found**: Check `test_model_path` fixture
2. **Import error**: Missing dependencies
3. **Port conflict**: Metrics port 8000 in use

**Solution**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run consumer manually to debug
python -m src.streaming.consumer --help
```

---

### Issue: "Test timeout"

**Symptoms**:
```
FAILED tests/test_kafka_integration.py::test_end_to_end_message_flow - Timeout
```

**Diagnosis**:
- Consumer is slow or hung
- Messages not being produced
- Network issues

**Solution**:
```bash
# Increase timeout in pytest.ini
timeout = 600  # 10 minutes

# Or run with increased timeout
pytest --timeout=600 tests/test_kafka_integration.py
```

---

### Issue: "Less than 80% predictions received"

**Symptoms**:
```
AssertionError: Should receive at least 40 predictions, got 25
```

**Diagnosis**:
Check dead letter queue for errors:
```bash
docker exec telco-redpanda-test rpk topic consume telco.deadletter.test
```

**Common Causes**:
1. **Schema validation failures**: Disable validation with `--no-validate`
2. **Model inference errors**: Check test model compatibility
3. **Feature transformation errors**: Invalid test data

**Solution**:
1. Review consumer logs
2. Check dead letter messages for error types
3. Adjust test data to match expected format

---

### Issue: "Topics already exist"

**Symptoms**:
```
kafka.errors.TopicAlreadyExistsError
```

**Solution**:
```bash
# Delete test topics manually
docker exec telco-redpanda-test rpk topic delete telco.raw.customers.test
docker exec telco-redpanda-test rpk topic delete telco.churn.predictions.test
docker exec telco-redpanda-test rpk topic delete telco.deadletter.test

# Or tear down and restart
docker-compose -f docker-compose.test.yml down -v
docker-compose -f docker-compose.test.yml up -d
```

---

## CI/CD Integration

### GitHub Actions

**File**: `.github/workflows/integration-tests.yml`

```yaml
name: Kafka Integration Tests

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run integration tests
        run: bash scripts/run_kafka_integration_tests.sh
      
      - name: Upload test reports
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: integration-test-reports
          path: reports/
```

### GitLab CI

**File**: `.gitlab-ci.yml`

```yaml
integration-tests:
  stage: test
  image: python:3.11
  services:
    - docker:dind
  script:
    - apt-get update && apt-get install -y docker-compose
    - pip install -r requirements.txt
    - bash scripts/run_kafka_integration_tests.sh
  artifacts:
    when: always
    paths:
      - reports/
    expire_in: 7 days
```

### Jenkins

**Jenkinsfile**:

```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Integration Tests') {
            steps {
                sh 'bash scripts/run_kafka_integration_tests.sh'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'reports/**/*', allowEmptyArchive: true
            junit 'test-results.xml'
        }
    }
}
```

### Azure Pipelines

**azure-pipelines.yml**:

```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'
  
  - script: pip install -r requirements.txt
    displayName: 'Install dependencies'
  
  - script: bash scripts/run_kafka_integration_tests.sh
    displayName: 'Run integration tests'
  
  - task: PublishTestResults@2
    condition: always()
    inputs:
      testResultsFiles: 'test-results.xml'
      testRunTitle: 'Kafka Integration Tests'
  
  - task: PublishPipelineArtifact@1
    condition: always()
    inputs:
      targetPath: 'reports'
      artifact: 'test-reports'
```

---

## Best Practices

### Test Data Management

✅ **Use fixed test data** for reproducibility  
✅ **Small batch sizes** (50 messages) for fast execution  
✅ **Diverse test cases** (valid, invalid, edge cases)  

### Test Isolation

✅ **Separate test topics** (`.test` suffix)  
✅ **Unique consumer groups** per test  
✅ **Ephemeral storage** (no persistent volumes)  

### Performance Optimization

✅ **Module-scoped fixtures** for shared setup  
✅ **Parallel test execution** where possible  
✅ **Fast broker startup** (dev-container mode)  

### Debugging

✅ **Use `--keep-containers`** to inspect state  
✅ **Check `reports/` directory** for logs  
✅ **Manual topic inspection** with `rpk topic consume`  

---

## Summary

The integration test suite provides comprehensive validation of the Kafka streaming pipeline with:

✅ **6 test scenarios** covering all critical paths  
✅ **Automated environment setup** with Docker Compose  
✅ **CI/CD ready** with test harness scripts  
✅ **Detailed logging** and error reporting  
✅ **Fast execution** (~2 minutes total)  

**Next Steps**:
1. Run tests locally: `bash scripts/run_kafka_integration_tests.sh`
2. Review test reports in `reports/` directory
3. Integrate into CI/CD pipeline
4. Add custom test scenarios as needed

For questions or issues, refer to the [Troubleshooting](#troubleshooting) section.
