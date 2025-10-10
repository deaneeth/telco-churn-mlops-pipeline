# 📋 Mini Project 2 - Kafka Integration Gap Analysis

**Analysis Date:** October 10, 2025  
**Project:** Telco Customer Churn Prediction - Kafka Integration Extension  
**Base:** Mini Project 1 - Production MLOps Pipeline

---

## 🎯 Executive Summary

This gap analysis maps **Mini Project 2: Kafka Integration** requirements against the existing Mini Project 1 codebase to identify what needs to be built.

### Compliance Overview

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ **Covered** | 5 | 20% |
| ⚠️ **Partial** | 3 | 12% |
| ❌ **Missing** | 17 | 68% |
| **TOTAL** | **25** | **100%** |

**Key Insight:** While MP1 provides excellent foundation (trained model, preprocessing, Airflow, Docker, testing), **Kafka streaming infrastructure is entirely new** and requires significant development.

---

## 📊 Requirements Breakdown by Category

### 1️⃣ Kafka Infrastructure (Critical) ❌

**Score: 0/4 requirements covered**

| ID | Requirement | Status | Gap |
|----|-------------|--------|-----|
| INFRA-01 | Kafka broker setup (local dev) | ❌ Missing | No docker-compose.yml with Kafka |
| INFRA-02 | Topic: `telco.raw.customers` | ❌ Missing | Need topic creation |
| INFRA-03 | Topic: `telco.churn.predictions` | ❌ Missing | Need topic creation |
| INFRA-04 | Topic: `telco.deadletter` (optional) | ❌ Missing | Need DLQ topic |

#### 📝 Action Items

**INFRA-01: Docker Compose Setup**
- **Priority:** 🔴 Critical
- **Effort:** Medium (4 hours)
- **Tasks:**
  ```yaml
  # Create docker-compose.yml with:
  - Zookeeper service
  - Kafka broker (exposed on localhost:9092)
  - Kafka UI (optional, for monitoring)
  - Network configuration
  ```
- **Files to Create:**
  - `docker-compose.yml` (root directory)
  - `kafka/docker-compose.kafka.yml` (optional modular approach)
- **Notes:** MP1 has Dockerfile for Flask API, but no docker-compose. Need multi-service orchestration.

**INFRA-02, 03, 04: Topic Creation**
- **Priority:** 🔴 Critical
- **Effort:** Low (1 hour)
- **Tasks:**
  ```bash
  # Option 1: Auto-create in docker-compose
  KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
  
  # Option 2: Init script
  kafka/init-topics.sh with kafka-topics commands
  ```
- **Files to Create:**
  - `kafka/init-topics.sh`
  - Update `docker-compose.yml` with init container
- **Topic Configs:**
  - `telco.raw.customers` (partitions: 3, replication: 1)
  - `telco.churn.predictions` (partitions: 3, replication: 1)
  - `telco.deadletter` (partitions: 1, replication: 1)

---

### 2️⃣ Producer Implementation (20 marks) ❌

**Score: 1/5 requirements covered (partial)**

| ID | Requirement | Status | Current State | Gap |
|----|-------------|--------|---------------|-----|
| PROD-01 | `producer.py` streaming mode | ❌ Missing | No Kafka producer | Need continuous event generation |
| PROD-02 | `producer.py` batch mode | ❌ Missing | Has `load_data.py` only | Need Kafka publishing |
| PROD-03 | Argparse CLI (modes, rates) | ❌ Missing | No CLI args | Need arg parsing |
| PROD-04 | Checkpoint/resume mechanism | ❌ Missing | No state management | Need checkpoint file |
| PROD-05 | Message schema (JSON + key) | ⚠️ Partial | Schema in `config.py` | Need `event_ts` field |

#### 📝 Action Items

**PROD-01, 02: Create Producer Script**
- **Priority:** 🔴 Critical
- **Effort:** High (8 hours)
- **Tasks:**
  1. Create `src/streaming/producer.py`
  2. Implement streaming mode:
     - Continuous loop sampling from CSV
     - Random row selection with replacement
     - Add synthetic `event_ts` (current timestamp)
     - Configurable rate: `--events-per-sec` (default: 10)
     - Publish to `telco.raw.customers`
  3. Implement batch mode:
     - Read full CSV in chunks
     - Configurable `--batch-size` (default: 1000)
     - Send chunks to Kafka
  4. Add `KafkaProducer` from `kafka-python`
  
- **Code Template:**
  ```python
  # src/streaming/producer.py
  import argparse
  from kafka import KafkaProducer
  import pandas as pd
  import json
  from datetime import datetime
  import time
  
  def produce_streaming(args):
      producer = KafkaProducer(
          bootstrap_servers=args.broker,
          value_serializer=lambda v: json.dumps(v).encode('utf-8'),
          key_serializer=lambda k: k.encode('utf-8')
      )
      df = pd.read_csv(args.data_path)
      
      while True:
          row = df.sample(n=1).iloc[0].to_dict()
          row['event_ts'] = datetime.utcnow().isoformat() + 'Z'
          
          producer.send(
              args.topic,
              key=row['customerID'],
              value=row
          )
          time.sleep(1 / args.events_per_sec)
  
  def produce_batch(args):
      # Similar but read chunks
      pass
  
  if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--mode', choices=['streaming', 'batch'])
      # ... more args
  ```

**PROD-03: CLI Configuration**
- **Priority:** 🔴 Critical
- **Effort:** Low (1 hour)
- **Arguments Needed:**
  ```python
  --mode {streaming|batch}      # Required
  --broker localhost:9092       # Kafka broker
  --topic telco.raw.customers   # Output topic
  --data-path data/raw/Telco-Customer-Churn.csv
  --events-per-sec 10           # Streaming mode
  --batch-size 1000             # Batch mode
  --checkpoint-file checkpoints/producer.json
  ```

**PROD-04: Checkpoint Mechanism**
- **Priority:** 🟡 Medium
- **Effort:** Medium (2 hours)
- **Implementation:**
  ```python
  # checkpoints/producer_checkpoint.json
  {
    "last_processed_index": 1234,
    "last_event_ts": "2025-10-10T10:30:00Z",
    "total_sent": 5000
  }
  
  # On startup: read checkpoint, resume from last index
  # On each batch: update checkpoint
  # Handle SIGINT/SIGTERM gracefully
  ```

**PROD-05: Message Schema**
- **Priority:** 🔴 Critical
- **Effort:** Low (30 minutes)
- **Current:** `config.py` has all 19 input features
- **Gap:** Need to add `event_ts` field
- **Schema:**
  ```json
  {
    "customerID": "7590-VHVEG",      // KEY
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "Churn": "No",
    "event_ts": "2025-10-03T04:00:00Z"  // NEW
  }
  ```

---

### 3️⃣ Consumer Implementation (40 marks) ⚠️

**Score: 3/6 requirements covered (2 full + 1 partial)**

| ID | Requirement | Status | Current State | Gap |
|----|-------------|--------|---------------|-----|
| CONS-01 | `consumer.py` streaming mode | ❌ Missing | Has `predict.py` only | Need Kafka consumer |
| CONS-02 | `consumer.py` batch mode | ⚠️ Partial | Has `batch_predict.py` | Need Kafka integration |
| CONS-03 | Model integration | ✅ Covered | `sklearn_pipeline_mlflow.joblib` | None |
| CONS-04 | Preprocessing pipeline | ✅ Covered | Embedded in model | None |
| CONS-05 | Output schema | ⚠️ Partial | Returns prediction + prob | Need `processed_ts` |
| CONS-06 | Batch analytics | ❌ Missing | No aggregation | Need summary stats |

#### 📝 Action Items

**CONS-01, 02: Create Consumer Script**
- **Priority:** 🔴 Critical
- **Effort:** High (10 hours)
- **Tasks:**
  1. Create `src/streaming/consumer.py`
  2. Implement streaming mode:
     - `KafkaConsumer` reading from `telco.raw.customers`
     - Deserialize JSON messages
     - Load model: `joblib.load('artifacts/models/sklearn_pipeline_mlflow.joblib')`
     - Run prediction (model already has preprocessing)
     - Publish to `telco.churn.predictions`
     - Error handling → DLQ
  3. Implement batch mode:
     - Consume fixed window (e.g., 1000 messages)
     - Batch predictions
     - Generate summary analytics
     - Publish results

- **Code Template:**
  ```python
  # src/streaming/consumer.py
  import argparse
  from kafka import KafkaConsumer, KafkaProducer
  import json
  import joblib
  from datetime import datetime
  
  def consume_streaming(args):
      # Load model once
      model = joblib.load(args.model_path)
      
      consumer = KafkaConsumer(
          args.input_topic,
          bootstrap_servers=args.broker,
          value_deserializer=lambda m: json.loads(m.decode('utf-8')),
          group_id='churn-prediction-group',
          auto_offset_reset='earliest'
      )
      
      producer = KafkaProducer(
          bootstrap_servers=args.broker,
          value_serializer=lambda v: json.dumps(v).encode('utf-8')
      )
      
      for message in consumer:
          try:
              customer_data = message.value
              
              # Predict (model handles preprocessing)
              prediction = model.predict([customer_data])[0]
              probability = model.predict_proba([customer_data])[0][1]
              
              # Output message
              output = {
                  "customerID": customer_data['customerID'],
                  "churn_probability": float(probability),
                  "prediction": "Yes" if prediction == 1 else "No",
                  "event_ts": customer_data['event_ts'],
                  "processed_ts": datetime.utcnow().isoformat() + 'Z'
              }
              
              producer.send(args.output_topic, value=output)
              
          except Exception as e:
              # Send to dead letter queue
              producer.send(args.dlq_topic, value={
                  "original_message": customer_data,
                  "error": str(e),
                  "timestamp": datetime.utcnow().isoformat()
              })
  
  def consume_batch(args):
      # Similar but with windowing and analytics
      pass
  ```

**CONS-05: Output Message Schema**
- **Priority:** 🔴 Critical
- **Effort:** Low (30 minutes)
- **Current:** `predict.py` returns `{"prediction": 1, "probability": 0.82}`
- **Gap:** Need to add `customerID`, `event_ts`, `processed_ts`
- **Target Schema:**
  ```json
  {
    "customerID": "7590-VHVEG",
    "churn_probability": 0.82,
    "prediction": "Yes",
    "event_ts": "2025-10-03T04:00:00Z",
    "processed_ts": "2025-10-03T04:00:01Z"
  }
  ```

**CONS-06: Batch Analytics**
- **Priority:** 🟡 Medium
- **Effort:** Medium (3 hours)
- **Tasks:**
  1. After batch processing, calculate:
     - Overall churn rate
     - Churn % by Contract type
     - Churn % by Internet Service
     - Top-K customers by churn risk
  2. Output to console or file
  
- **Example Output:**
  ```
  === Batch Summary ===
  Total Customers: 1000
  Predicted Churn: 265 (26.5%)
  
  Churn by Contract:
  - Month-to-month: 42.1%
  - One year: 11.3%
  - Two year: 2.8%
  
  Top 10 High-Risk Customers:
  1. customerID-1234: 0.95
  2. customerID-5678: 0.93
  ...
  ```

---

### 4️⃣ Integration & Reliability (20 marks) ⚠️

**Score: 1/4 requirements covered (partial)**

| ID | Requirement | Status | Current State | Gap |
|----|-------------|--------|---------------|-----|
| INTG-01 | End-to-end flow | ❌ Missing | CSV → model → CSV | Need Kafka flow |
| INTG-02 | Configuration management | ⚠️ Partial | `config.py` exists | Need Kafka config |
| INTG-03 | Error handling & DLQ | ❌ Missing | Basic try-catch | Need DLQ pattern |
| INTG-04 | Graceful shutdown | ❌ Missing | No signal handling | Need cleanup |

#### 📝 Action Items

**INTG-02: Kafka Configuration**
- **Priority:** 🔴 Critical
- **Effort:** Low (1 hour)
- **Tasks:**
  1. Add Kafka section to `config.yaml`:
     ```yaml
     kafka:
       bootstrap_servers: "localhost:9092"
       topics:
         raw_customers: "telco.raw.customers"
         predictions: "telco.churn.predictions"
         dead_letter: "telco.deadletter"
       producer:
         batch_size: 1000
         events_per_sec: 10
       consumer:
         group_id: "churn-prediction-group"
         auto_offset_reset: "earliest"
         max_poll_records: 500
     ```
  2. Update `config.py` to load Kafka config

**INTG-03: Dead Letter Queue**
- **Priority:** 🟡 Medium
- **Effort:** Medium (2 hours)
- **Implementation:**
  ```python
  def send_to_dlq(producer, message, error):
      dlq_message = {
          "original_message": message,
          "error_type": type(error).__name__,
          "error_message": str(error),
          "timestamp": datetime.utcnow().isoformat(),
          "retry_count": message.get('retry_count', 0) + 1
      }
      producer.send('telco.deadletter', value=dlq_message)
  ```

**INTG-04: Graceful Shutdown**
- **Priority:** 🟡 Medium
- **Effort:** Medium (2 hours)
- **Implementation:**
  ```python
  import signal
  
  running = True
  
  def signal_handler(sig, frame):
      global running
      print("Shutting down gracefully...")
      running = False
  
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)
  
  while running:
      # Process messages
      pass
  
  consumer.close()
  producer.flush()
  producer.close()
  ```

---

### 5️⃣ Testing & Observability (10 marks) ⚠️

**Score: 1/5 requirements covered (partial)**

| ID | Requirement | Status | Current State | Gap |
|----|-------------|--------|---------------|-----|
| TEST-01 | Producer unit tests | ❌ Missing | 98 tests exist, no Kafka | Need mock tests |
| TEST-02 | Consumer unit tests | ❌ Missing | Inference tests exist | Need mock tests |
| TEST-03 | Integration tests | ⚠️ Partial | `test_integration.py` | Need Kafka tests |
| TEST-04 | Logs & screenshots | ❌ Missing | Has logging | Need Kafka logs |
| TEST-05 | Monitoring & metrics | ❌ Missing | MLflow only | Need Kafka metrics |

#### 📝 Action Items

**TEST-01, 02: Unit Tests with Mocking**
- **Priority:** 🔴 Critical
- **Effort:** Medium (4 hours)
- **Tasks:**
  1. Create `tests/test_kafka_producer.py`
  2. Create `tests/test_kafka_consumer.py`
  3. Use `unittest.mock` to mock `KafkaProducer` and `KafkaConsumer`
  
- **Example:**
  ```python
  # tests/test_kafka_producer.py
  from unittest.mock import Mock, patch
  import pytest
  
  @patch('src.streaming.producer.KafkaProducer')
  def test_streaming_producer(mock_producer):
      # Test producer initialization
      # Test message sending
      # Test error handling
      pass
  ```

**TEST-03: Integration Tests**
- **Priority:** 🟡 Medium
- **Effort:** High (4 hours)
- **Options:**
  1. **Testcontainers:** Spin up real Kafka in Docker for tests
  2. **Embedded Kafka:** Use `kafka-python-ng` testing utilities
- **Files:** `tests/test_kafka_integration.py`

**TEST-04: Logs & Screenshots**
- **Priority:** 🔴 Critical (for submission)
- **Effort:** Low (1 hour)
- **Tasks:**
  1. Run producer → capture logs
  2. Run consumer → capture logs
  3. Open Kafka UI → screenshot topics, messages
  4. Save to `docs/kafka_screenshots/`
  
- **Required Screenshots:**
  - Kafka UI showing topics
  - Messages in `telco.raw.customers`
  - Messages in `telco.churn.predictions`
  - Producer terminal logs
  - Consumer terminal logs

**TEST-05: Monitoring**
- **Priority:** 🟢 Low
- **Effort:** Medium (3 hours)
- **Metrics to Track:**
  - Producer: messages/sec, errors, lag
  - Consumer: messages/sec, processing time, lag
  - Kafka: topic size, partition distribution
- **Tools:** Prometheus + Grafana (optional), Kafka UI built-in metrics

---

### 6️⃣ Documentation (10 marks) ✅

**Score: 1/3 requirements covered**

| ID | Requirement | Status | Current State | Gap |
|----|-------------|--------|---------------|-----|
| DOC-01 | README updates | ✅ Covered | 1,336-line README | Add Kafka section |
| DOC-02 | Config documentation | ❌ Missing | Good docs exist | Add Kafka guide |
| DOC-03 | Usage examples | ❌ Missing | API examples exist | Add CLI examples |

#### 📝 Action Items

**DOC-01: README Update**
- **Priority:** 🔴 Critical
- **Effort:** Low (2 hours)
- **Sections to Add:**
  ```markdown
  ## 🌊 Kafka Streaming Integration (NEW)
  
  ### Quick Start
  1. Start Kafka: `docker-compose up -d`
  2. Run producer: `python src/streaming/producer.py --mode streaming`
  3. Run consumer: `python src/streaming/consumer.py --mode streaming`
  
  ### Producer Modes
  - Streaming: continuous event generation
  - Batch: process CSV in chunks
  
  ### Consumer Modes
  - Streaming: real-time inference
  - Batch: windowed processing with analytics
  
  ### Topics
  - telco.raw.customers
  - telco.churn.predictions
  - telco.deadletter
  ```

**DOC-02, 03: Configuration & Examples**
- **Priority:** 🟡 Medium
- **Effort:** Low (1 hour)
- **Create:** `docs/kafka_guide.md` with detailed setup and usage

---

### 7️⃣ BONUS: Airflow DAGs (+10 marks) ⚠️

**Score: 1/3 requirements covered (partial)**

| ID | Requirement | Status | Current State | Gap |
|----|-------------|--------|---------------|-----|
| BONUS-01 | Streaming DAG | ⚠️ Partial | Airflow infrastructure exists | Need new DAG |
| BONUS-02 | Batch Kafka DAG | ⚠️ Partial | Batch DAG exists | Adapt for Kafka |
| BONUS-03 | Airflow screenshots | ✅ Covered | 4 screenshots exist | Need new ones |

#### 📝 Action Items

**BONUS-01: Streaming DAG**
- **Priority:** 🟢 Low (bonus)
- **Effort:** High (4 hours)
- **Tasks:**
  1. Create `dags/kafka_streaming_dag.py`
  2. Tasks:
     - Health check (ping Kafka)
     - Start long-running consumer
     - Monitor consumer process
     - Alert on failures
  
- **Template:**
  ```python
  from airflow import DAG
  from airflow.operators.bash import BashOperator
  from airflow.operators.python import PythonOperator
  
  dag = DAG(
      'kafka_streaming_health',
      schedule='@hourly',
      catchup=False
  )
  
  health_check = BashOperator(
      task_id='check_kafka',
      bash_command='kafka-topics.sh --list --bootstrap-server localhost:9092'
  )
  
  start_consumer = BashOperator(
      task_id='start_consumer',
      bash_command='python src/streaming/consumer.py --mode streaming &'
  )
  ```

**BONUS-02: Batch Kafka DAG**
- **Priority:** 🟢 Low (bonus)
- **Effort:** Medium (3 hours)
- **Tasks:**
  1. Create `dags/kafka_batch_dag.py`
  2. Chain: producer (batch) → wait → consumer (batch) → summary
  3. Schedule: daily or hourly

---

## 🔄 Reusable Components from MP1

These components are **ready to use** with minimal or no modification:

| Component | File/Location | Usage in MP2 | Status |
|-----------|---------------|--------------|--------|
| **ML Model** | `artifacts/models/sklearn_pipeline_mlflow.joblib` | Load in consumer for predictions | ✅ Ready |
| **Preprocessing** | Embedded in model pipeline | Auto-applied during prediction | ✅ Ready |
| **Prediction Logic** | `src/inference/predict.py` | Adapt `predict_from_dict()` for Kafka | ✅ Ready |
| **Batch Inference** | `src/inference/batch_predict.py` | Template for batch consumer | ✅ Ready |
| **Configuration** | `config.py`, `config.yaml` | Extend with Kafka config | ✅ Ready |
| **Testing Framework** | `tests/`, `pytest.ini` | Add Kafka tests | ✅ Ready |
| **Docker** | `Dockerfile` | Use in docker-compose | ✅ Ready |
| **Airflow** | `dags/telco_churn_dag.py` | Template for Kafka DAGs | ✅ Ready |

---

## 📦 New Dependencies Required

### Python Packages

Add to `requirements.txt`:
```txt
# Kafka streaming
kafka-python>=2.0.0
confluent-kafka>=2.0.0  # Alternative, more performant
```

### Infrastructure

Add to `docker-compose.yml`:
```yaml
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
  
  kafka:
    image: confluentinc/cp-kafka:7.5.0
  
  kafka-ui:
    image: provectuslabs/kafka-ui:latest  # Optional
```

---

## 🗓️ Implementation Roadmap

### Phase 1: Infrastructure (Critical) - 4 hours
- [ ] Create `docker-compose.yml` with Kafka, Zookeeper, Kafka UI
- [ ] Create topic initialization scripts
- [ ] Update `requirements.txt` with `kafka-python`
- [ ] Test Kafka connectivity

### Phase 2: Producer (Critical) - 8 hours
- [ ] Create `src/streaming/producer.py`
- [ ] Implement streaming mode (continuous sampling)
- [ ] Implement batch mode (CSV chunks)
- [ ] Add argparse configuration
- [ ] Implement checkpoint mechanism
- [ ] Add logging and error handling

### Phase 3: Consumer (Critical) - 10 hours
- [ ] Create `src/streaming/consumer.py`
- [ ] Implement streaming mode (continuous inference)
- [ ] Implement batch mode (windowed processing)
- [ ] Integrate model loading and prediction
- [ ] Add batch summary analytics
- [ ] Publish to predictions topic
- [ ] Add error handling and DLQ

### Phase 4: Testing (High Priority) - 6 hours
- [ ] Create `tests/test_kafka_producer.py`
- [ ] Create `tests/test_kafka_consumer.py`
- [ ] Add integration tests
- [ ] Generate logs and screenshots
- [ ] Test end-to-end flow

### Phase 5: Documentation (High Priority) - 3 hours
- [ ] Update `README.md` with Kafka section
- [ ] Add configuration documentation
- [ ] Add usage examples
- [ ] Update architecture diagrams

### Phase 6: Bonus Airflow (Low Priority) - 6 hours
- [ ] Create `dags/kafka_streaming_dag.py`
- [ ] Create `dags/kafka_batch_dag.py`
- [ ] Add health checks and monitoring
- [ ] Capture Airflow screenshots

**Total Estimated Effort:** **37 hours**

---

## ⚠️ Risk Assessment

### 🟢 Low Risk
- Infrastructure setup (well-documented by Confluent)
- Model integration (already working perfectly)
- Configuration management (patterns exist from MP1)

### 🟡 Medium Risk
- Checkpoint mechanism (needs design decision: file vs Redis)
- Error handling and DLQ (needs thorough testing)
- Batch analytics (aggregation logic needs validation)

### 🔴 High Risk
- **Streaming mode correctness** (continuous operation, no message loss)
- **Performance at scale** (needs load testing with high throughput)
- **Integration testing** (testcontainers setup can be complex)

**Mitigation Strategies:**
1. Start with batch mode (simpler) before streaming
2. Use Kafka UI for visual debugging
3. Implement comprehensive logging from day 1
4. Test with small datasets first

---

## 📊 Evaluation Alignment

| Category | Marks | Coverage Status | Notes |
|----------|-------|-----------------|-------|
| **Producers** | 20 | ❌ 20% (partial schema) | Need full implementation |
| **Consumers** | 40 | ⚠️ 50% (model + preprocessing ready) | Need Kafka integration |
| **Integration & Reliability** | 20 | ⚠️ 25% (config partial) | Need end-to-end flow |
| **Testing & Observability** | 10 | ⚠️ 20% (framework exists) | Need Kafka tests |
| **Documentation** | 10 | ✅ 33% (README ready) | Need Kafka sections |
| **Bonus: Airflow** | +10 | ⚠️ 33% (infra exists) | Need Kafka DAGs |

**Current Readiness:** ~30% (foundation from MP1)  
**Work Remaining:** ~70% (Kafka-specific components)

---

## ✅ Next Steps

### Immediate Actions (This Week)

1. **Setup Infrastructure** (Day 1)
   - Create `docker-compose.yml`
   - Start Kafka and verify connectivity
   - Create topics manually via Kafka UI

2. **Build Producer** (Days 2-3)
   - Implement `producer.py` with both modes
   - Test with small batches
   - Verify messages in Kafka UI

3. **Build Consumer** (Days 4-5)
   - Implement `consumer.py` with streaming mode
   - Integrate model inference
   - Test end-to-end flow

4. **Testing & Documentation** (Days 6-7)
   - Write unit tests
   - Capture screenshots
   - Update README

### Optional (Bonus Points)

5. **Airflow Integration** (Days 8-9)
   - Create Kafka DAGs
   - Test orchestration
   - Capture Airflow screenshots

---

## 📞 Support Resources

- **Kafka Documentation:** https://kafka.apache.org/documentation/
- **kafka-python Docs:** https://kafka-python.readthedocs.io/
- **Confluent Docker Images:** https://hub.docker.com/u/confluentinc
- **MP1 Reference:** Your existing codebase (excellent foundation!)

---

**Report Generated:** October 10, 2025  
**Status:** Ready for implementation  
**Estimated Timeline:** 1-2 weeks (full-time) or 3-4 weeks (part-time)
