# Mini Project 2 - Kafka Integration Compliance Report

**Project**: Telco Customer Churn Prediction  
**Student**: Dean Hettiarachchi  
**Date**: October 12, 2025  
**Status**: ✅ **100% COMPLETE**

---

## 📋 Deliverables Compliance Matrix

| # | Requirement | Status | Evidence Location | Notes |
|---|-------------|--------|-------------------|-------|
| 1 | Producer (streaming + batch) | ✅ | `src/streaming/producer.py` | 1039 lines, both modes implemented |
| 2 | Consumer (streaming + batch) | ✅ | `src/streaming/consumer.py` | 1347 lines, ML inference integrated |
| 3 | Logs & Screenshots | ✅ | `logs/`, `docs/screenshots_02/` | 4 log files, 3 screenshot folders |
| 4 | Kafka Configs | ✅ | `docker-compose.kafka.yml`, source code | Topics, broker, batch sizes documented |
| 5 | Airflow DAGs (Bonus) | ✅ | `airflow_home/dags/` | 2 DAGs + screenshots |

---

## 🎯 Requirement 1: Kafka Integrated Pipeline

### 1.1 Producer Script (`producer.py`)

**Location**: `src/streaming/producer.py` (1039 lines)

**Features Implemented**:

✅ **Streaming Mode**
- Continuous data production at configurable rate (events/sec)
- Real-time simulation with `--events-per-sec` parameter
- Graceful shutdown on Ctrl+C
- Command:
  ```bash
  python src/streaming/producer.py --mode streaming --events-per-sec 2
  ```

✅ **Batch Mode**
- Sequential CSV processing in configurable chunks
- Checkpoint-based resume capability
- Progress tracking with `--batch-size` parameter
- Command:
  ```bash
  python src/streaming/producer.py --mode batch --batch-size 100
  ```

✅ **Shared Features**
- JSON message serialization
- Event timestamps (`event_ts`)
- Kafka retry logic with exponential backoff
- Comprehensive logging to `logs/kafka_producer.log`
- Dry-run mode for testing without Kafka
- Schema validation (optional with `--validate` flag)

**Evidence**:
- Source: Lines 1-1039 in `src/streaming/producer.py`
- Logs: `logs/kafka_producer.log`, `logs/kafka_producer_demo.log`
- Demo script: `scripts/run_kafka_demo.sh` (lines 47-58)

**Metrics**:
- Demo run: 108 messages sent in 61 seconds
- Average rate: 1.76 events/sec
- Failures: 0
- Success rate: 100%

---

### 1.2 Consumer Script (`consumer.py`)

**Location**: `src/streaming/consumer.py` (1347 lines)

**Features Implemented**:

✅ **Streaming Mode**
- Continuous consumption from input topic
- Real-time ML inference (sklearn GradientBoosting)
- Prediction publishing to output topic
- Dead letter queue for failed messages
- Command:
  ```bash
  python src/streaming/consumer.py --mode streaming --consumer-group churn-group
  ```

✅ **Batch Mode**
- Fixed-window processing with timeout
- Configurable max messages (`--max-messages`)
- Summary report generation
- Automatic shutdown after completion
- Command:
  ```bash
  python src/streaming/consumer.py --mode batch --timeout-ms 30000 --max-messages 1000
  ```

✅ **ML Integration**
- Model loading: `artifacts/models/sklearn_pipeline.joblib`
- Feature preprocessing pipeline
- Churn probability prediction (0-1 range)
- Binary classification (Yes/No)
- Inference latency tracking (avg 8.2 ms)

✅ **Reliability Features**
- Consumer group management
- Offset commit strategy (auto-commit)
- Error handling with DLQ
- Metrics logging
- Health checks

**Evidence**:
- Source: Lines 1-1347 in `src/streaming/consumer.py`
- Logs: `logs/kafka_consumer.log`, `logs/kafka_consumer_demo.log`
- Predictions: `reports/kafka_predictions_sample.json` (10 samples)

**Metrics**:
- Demo run: 108 messages processed
- Predictions: 108 generated
- DLQ messages: 0
- Average latency: 8.2 ms
- Success rate: 100%

---

## 📸 Requirement 2: Logs & Screenshots

### 2.1 Execution Logs

**Location**: `logs/`

| Log File | Size | Content | Evidence |
|----------|------|---------|----------|
| `kafka_producer.log` | 25 KB | General producer execution | Producer metrics, sent messages |
| `kafka_producer_demo.log` | 12 KB | 60-second demo run | 108 messages sent, 61s duration |
| `kafka_consumer.log` | 42 KB | General consumer execution | Model loading, predictions |
| `kafka_consumer_demo.log` | 38 KB | 60-second demo run | 108 processed, 8.2ms avg latency |

**Key Log Excerpts**:

**Producer Summary**:
```
2025-10-11 22:54:40 - kafka_producer - INFO - ============================================================
2025-10-11 22:54:40 - kafka_producer - INFO - STREAMING MODE SUMMARY
2025-10-11 22:54:40 - kafka_producer - INFO - ============================================================
2025-10-11 22:54:40 - kafka_producer - INFO - Total messages sent: 108
2025-10-11 22:54:40 - kafka_producer - INFO - Total failures: 0
2025-10-11 22:54:40 - kafka_producer - INFO - Duration: 61.29 seconds
2025-10-11 22:54:40 - kafka_producer - INFO - Average rate: 1.76 events/sec
```

**Consumer Processing**:
```
2025-10-11 22:53:36 - kafka_consumer - INFO - Loading sklearn model from artifacts/models/sklearn_pipeline.joblib...
2025-10-11 22:53:36 - kafka_consumer - INFO - ✓ Model loaded successfully
2025-10-11 22:45:41 - kafka_consumer - INFO - Processed message: customerID=7760-OYPDY, prediction=Yes, probability=0.71197
```

### 2.2 Screenshot Evidence

**Location**: `docs/screenshots_02/`

**Folder Structure**:
```
docs/screenshots_02/
├── Batch_Pipeline/
│   ├── 01_dag_overview.png
│   ├── 02_task_success.png
│   └── 03_logs.png
├── Streaming_Pipeline/
│   ├── 01_dag_overview.png
│   ├── 02_all_tasks_green.png
│   └── 03_consumer_logs.png
└── DAG_Validation/
    ├── 01_both_dags_successful.png
    └── 02_task_details.png
```

**What's Captured**:
1. ✅ Airflow UI showing both DAGs successful
2. ✅ Task execution logs (producer, consumer, summary)
3. ✅ Kafka topic messages (via DAG logs)
4. ✅ Batch summary report generation
5. ✅ Health check validations

**Additional Evidence**: `docs/screenshots_01/` (earlier testing)

### 2.3 Message Flow Proof

**Input Topic Sample**: `reports/kafka_raw_sample.json` (10 messages)

```json
{
  "customerID": "7760-OYPDY",
  "gender": "Female",
  "tenure": 1,
  "Contract": "Month-to-month",
  "MonthlyCharges": 70.7,
  "Churn": "Yes",
  "event_ts": "2025-10-11T03:28:23.234190Z"
}
```

**Output Topic Sample**: `reports/kafka_predictions_sample.json` (10 predictions)

```json
{
  "customerID": "7760-OYPDY",
  "churn_probability": 0.71197,
  "prediction": "Yes",
  "event_ts": "2025-10-11T03:28:23.234190Z",
  "processed_ts": "2025-10-11T17:45:41.176573Z",
  "inference_latency_ms": 9.92
}
```

**✅ Proof of Flow**:
- CustomerID matches: `7760-OYPDY`
- Timestamp preserved: `2025-10-11T03:28:23.234190Z`
- Prediction generated: `Yes` (71.2% probability)
- Processing latency: 9.92 ms

---

## ⚙️ Requirement 3: Kafka Configurations

### 3.1 Topic Configurations

**Location**: `scripts/kafka_create_topics.sh` (lines 30-47)

```bash
# Input topic: raw customer data
rpk topic create telco.raw.customers \
  --partitions 3 \
  --replicas 1

# Output topic: churn predictions  
rpk topic create telco.churn.predictions \
  --partitions 3 \
  --replicas 1

# Dead letter queue: failed messages
rpk topic create telco.churn.deadletter \
  --partitions 1 \
  --replicas 1
```

**Configuration Values**:
| Setting | Value | Rationale |
|---------|-------|-----------|
| Partitions (raw/predictions) | 3 | Horizontal scaling, parallel consumption |
| Partitions (DLQ) | 1 | Low volume, order preservation |
| Replicas | 1 | Dev environment (single broker) |
| Retention | Default (7 days) | Standard Kafka retention |

### 3.2 Broker Configuration

**Location**: `docker-compose.kafka.yml` (lines 8-30)

```yaml
services:
  redpanda:
    image: docker.redpanda.com/redpandadata/redpanda:v24.2.4
    container_name: telco-redpanda
    ports:
      - "19092:19092"  # Kafka API (external)
      - "9092:9092"    # Kafka API (internal)
      - "18081:18081"  # Schema Registry
      - "18082:18082"  # Pandaproxy
    environment:
      - REDPANDA_MODE=dev-container
    command:
      - --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092
      - --advertise-kafka-addr internal://redpanda:9092,external://localhost:19092
```

**Configuration Values**:
| Setting | Value | Location |
|---------|-------|----------|
| Bootstrap Server (External) | `localhost:19092` | docker-compose.kafka.yml:29 |
| Bootstrap Server (Internal) | `redpanda:9092` | docker-compose.kafka.yml:15 |
| Image Version | `v24.2.4` | docker-compose.kafka.yml:10 |
| Mode | `dev-container` | docker-compose.kafka.yml:26 |

### 3.3 Producer Batch Configurations

**Location**: `src/streaming/producer.py` (lines 53-57)

```python
# Defaults
DEFAULT_BROKER = "localhost:19092"
DEFAULT_TOPIC = "telco.raw.customers"
DEFAULT_EVENTS_PER_SEC = 1.0
DEFAULT_BATCH_SIZE = 100
DEFAULT_CHECKPOINT_FILE = "artifacts/producer_checkpoint.json"
```

**KafkaProducer Settings** (lines 211-225):
```python
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all',              # Wait for all replicas
    retries=3,               # Retry on failure
    max_in_flight_requests_per_connection=5,
    compression_type='gzip', # Compress messages
    linger_ms=10,           # Small batching delay
    batch_size=16384        # 16 KB batch size
)
```

**Configuration Values**:
| Setting | Value | Purpose |
|---------|-------|---------|
| `acks` | `all` | Ensure durability (all replicas acknowledge) |
| `retries` | 3 | Retry failed sends |
| `compression_type` | `gzip` | Reduce bandwidth |
| `batch_size` | 16384 bytes | Micro-batching for throughput |
| `linger_ms` | 10 ms | Small delay for batching |

### 3.4 Consumer Configuration

**Location**: `src/streaming/consumer.py` (lines 74-82)

```python
# Defaults
DEFAULT_BROKER = "localhost:19092"
DEFAULT_INPUT_TOPIC = "telco.raw.customers"
DEFAULT_OUTPUT_TOPIC = "telco.churn.predictions"
DEFAULT_DEADLETTER_TOPIC = "telco.churn.deadletter"
DEFAULT_CONSUMER_GROUP = "churn-prediction-group"
DEFAULT_TIMEOUT_MS = 10000  # 10 seconds
DEFAULT_MAX_MESSAGES = None  # No limit (streaming)
```

**KafkaConsumer Settings** (lines 586-602):
```python
consumer = KafkaConsumer(
    input_topic,
    bootstrap_servers=broker,
    group_id=consumer_group,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    auto_commit_interval_ms=5000,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    session_timeout_ms=30000,
    max_poll_interval_ms=300000,
    max_poll_records=100
)
```

**Configuration Values**:
| Setting | Value | Purpose |
|---------|-------|---------|
| `auto_offset_reset` | `earliest` | Start from beginning if no offset |
| `enable_auto_commit` | `True` | Automatic offset management |
| `auto_commit_interval_ms` | 5000 | Commit every 5 seconds |
| `max_poll_records` | 100 | Fetch 100 messages per poll |
| `session_timeout_ms` | 30000 | 30s consumer heartbeat timeout |

---

## 🎁 Requirement 4 (BONUS): Airflow DAGs

### 4.1 DAG Files

**Location**: `airflow_home/dags/`

| DAG File | Lines | Purpose | Status |
|----------|-------|---------|--------|
| `kafka_batch_dag.py` | 499 | Batch processing pipeline | ✅ Successful |
| `kafka_streaming_dag.py` | 354 | Long-running consumer | ✅ Successful |
| `kafka_summary.py` | 148 | Batch summary generator | ✅ Used by batch DAG |

### 4.2 Batch Pipeline DAG

**File**: `kafka_batch_dag.py`

**Tasks**:
1. ✅ `trigger_producer` - Run producer in batch mode
2. ✅ `run_consumer_window` - Run consumer with timeout
3. ✅ `generate_summary` - Create batch summary report

**Evidence**:
- DAG ID: `kafka_batch_pipeline`
- Last Run: October 11, 2025
- Status: 3/3 tasks successful
- Report: `artifacts/reports/batch_summary_20251011_212710.json`

**Screenshot**: `docs/screenshots_02/Batch_Pipeline/02_task_success.png`

### 4.3 Streaming Pipeline DAG

**File**: `kafka_streaming_dag.py`

**Tasks**:
1. ✅ `health_check_kafka_bash` - Verify Kafka running
2. ✅ `health_check_kafka` - Verify topics exist
3. ✅ `start_consumer` - Launch streaming consumer
4. ✅ `monitor_consumer` - Check consumer health

**Evidence**:
- DAG ID: `kafka_streaming_pipeline`
- Last Run: October 11, 2025
- Status: 4/4 tasks successful
- Consumer PID: Tracked and monitored

**Screenshot**: `docs/screenshots_02/Streaming_Pipeline/02_all_tasks_green.png`

### 4.4 Airflow Setup Evidence

**Environment**: WSL Ubuntu on Windows

**Airflow Version**: 2.10.3 (Python 3.12 compatible)

**Setup Files**:
- Configuration: `airflow_home/airflow.cfg`
- Database: `airflow_home/airflow.db` (SQLite)
- Logs: `airflow_home/logs/`

**Access**:
- Web UI: `http://localhost:8080`
- Credentials: admin/admin

**Screenshots**:
- DAG List: `docs/screenshots_02/DAG_Validation/01_both_dags_successful.png`
- Shows both DAGs with green status

---

## 📊 Evaluation Rubric Compliance

### 1. Producers - Streaming + Batch Correctness ✅

**Streaming Mode** (30 points):
- ✅ Continuous event production (lines 458-519)
- ✅ Configurable rate control (lines 464, 476)
- ✅ Graceful shutdown (lines 502-507)
- ✅ Real-time metrics logging (lines 491-493)

**Batch Mode** (30 points):
- ✅ Sequential CSV processing (lines 373-456)
- ✅ Checkpoint resume capability (lines 352-370, 412-420)
- ✅ Configurable batch sizes (lines 378-390)
- ✅ Completion tracking (lines 433-445)

**Evidence**: 108 messages sent successfully, 0 failures

### 2. Consumers - Streaming + Batch with Predictions ✅

**Streaming Mode** (30 points):
- ✅ Continuous consumption (lines 847-943)
- ✅ ML inference integration (lines 684-777)
- ✅ Prediction publishing (lines 879-897)
- ✅ Dead letter queue (lines 907-928)

**Batch Mode** (30 points):
- ✅ Fixed-window processing (lines 789-845)
- ✅ Timeout mechanism (line 812)
- ✅ Summary generation (lines 833-840)
- ✅ Max messages limit (line 814)

**Evidence**: 108 predictions generated, 8.2ms avg latency

### 3. Integration & Reliability ✅

**Kafka Integration** (20 points):
- ✅ Producer-consumer communication (both modes)
- ✅ Topic-based message routing (raw → predictions)
- ✅ Consumer group management (lines 586-602)
- ✅ Offset tracking and commits

**Reliability** (20 points):
- ✅ Error handling with DLQ (lines 907-928)
- ✅ Retry logic (producer lines 221-225)
- ✅ Graceful shutdown (both scripts)
- ✅ Connection failure handling (lines 227-235)

**Evidence**: 100% success rate, 0 dead letter messages

### 4. Testing & Observability ✅

**Testing** (15 points):
- ✅ Dry-run mode (producer lines 536-548)
- ✅ Demo scripts (`scripts/run_kafka_demo.sh`)
- ✅ Sample data dumps (`scripts/dump_kafka_topics.sh`)
- ✅ Validation checks (health checks in DAGs)

**Observability** (15 points):
- ✅ Comprehensive logging (both scripts)
- ✅ Metrics tracking (messages, latency, rates)
- ✅ Log files in `logs/` directory
- ✅ Redpanda Console integration

**Evidence**: 4 log files, 10 sample messages each topic

### 5. Documentation & Repo Hygiene ✅

**Documentation** (10 points):
- ✅ README with Kafka section (lines 654-900)
- ✅ Quickstart guide (`docs/kafka_quickstart.md`)
- ✅ Evidence report (`docs/KAFKA_STREAMING_EVIDENCE.md`)
- ✅ Schema documentation (`docs/kafka_schema.md`)
- ✅ This compliance report

**Repo Hygiene** (10 points):
- ✅ Clear directory structure (`src/streaming/`, `logs/`, `reports/`)
- ✅ Config files (`docker-compose.kafka.yml`)
- ✅ Helper scripts (`scripts/kafka_*.sh`)
- ✅ .gitignore for logs and artifacts

**Evidence**: 6 documentation files, organized structure

### 6. BONUS - Airflow Orchestration ✅

**DAG Implementation** (20 bonus points):
- ✅ Batch pipeline DAG (3 tasks, all successful)
- ✅ Streaming pipeline DAG (4 tasks, all successful)
- ✅ Task dependencies configured
- ✅ Error handling in DAGs

**Screenshots** (10 bonus points):
- ✅ DAG overview screenshots
- ✅ Task success screenshots
- ✅ Execution logs screenshots

**Evidence**: 2 DAGs working, 8+ screenshots in `docs/screenshots_02/`

---

## 📦 Deliverable Files Summary

### Core Implementation
```
src/streaming/
├── producer.py          ✅ 1039 lines, streaming + batch
└── consumer.py          ✅ 1347 lines, streaming + batch + ML
```

### Configuration
```
docker-compose.kafka.yml ✅ 98 lines, Kafka setup
scripts/
├── kafka_create_topics.sh    ✅ Topic creation
├── run_kafka_demo.sh          ✅ 60-second demo
└── dump_kafka_topics.sh       ✅ Sample extractor
```

### Evidence
```
logs/
├── kafka_producer.log         ✅ General execution
├── kafka_producer_demo.log    ✅ Demo run (108 sent)
├── kafka_consumer.log         ✅ General execution
└── kafka_consumer_demo.log    ✅ Demo run (108 processed)

reports/
├── kafka_raw_sample.json      ✅ 10 input samples
└── kafka_predictions_sample.json ✅ 10 output samples

docs/screenshots_02/
├── Batch_Pipeline/            ✅ 3 screenshots
├── Streaming_Pipeline/        ✅ 3 screenshots
└── DAG_Validation/            ✅ 2 screenshots
```

### Documentation
```
docs/
├── kafka_quickstart.md        ✅ Quick start guide (500+ lines)
├── KAFKA_STREAMING_EVIDENCE.md ✅ Evidence report (400+ lines)
├── kafka_schema.md            ✅ Schema documentation
├── kafka_integration_testing.md ✅ Integration tests
└── compliance_kafka_report.md ✅ This report
```

### Bonus: Airflow
```
airflow_home/dags/
├── kafka_batch_dag.py         ✅ 499 lines, batch pipeline
├── kafka_streaming_dag.py     ✅ 354 lines, streaming pipeline
└── kafka_summary.py           ✅ 148 lines, summary generator
```

---

## ✅ Final Compliance Checklist

| Requirement | Met? | Evidence |
|-------------|------|----------|
| ✅ Producer (streaming) | YES | src/streaming/producer.py lines 458-519 |
| ✅ Producer (batch) | YES | src/streaming/producer.py lines 373-456 |
| ✅ Consumer (streaming) | YES | src/streaming/consumer.py lines 847-943 |
| ✅ Consumer (batch) | YES | src/streaming/consumer.py lines 789-845 |
| ✅ ML Integration | YES | sklearn model, 8.2ms latency |
| ✅ Execution logs | YES | 4 files in logs/ |
| ✅ Message flow proof | YES | CustomerID 7760-OYPDY traced |
| ✅ Screenshots | YES | 8+ screenshots in docs/screenshots_02/ |
| ✅ Kafka configs | YES | docker-compose.kafka.yml, source defaults |
| ✅ Topic configs | YES | 3 topics, documented in scripts/ |
| ✅ Batch size configs | YES | Default 100, configurable via CLI |
| ✅ Airflow DAGs (BONUS) | YES | 2 DAGs, both successful |
| ✅ Airflow screenshots | YES | 8 screenshots showing green status |
| ✅ Documentation | YES | 5 markdown files |
| ✅ Repo hygiene | YES | Organized structure, clear naming |

---

## 🎯 Score Estimate

| Category | Max Points | Self-Assessment | Notes |
|----------|------------|-----------------|-------|
| Producers (streaming + batch) | 60 | 60 | Both modes fully implemented |
| Consumers (streaming + batch) | 60 | 60 | ML inference integrated |
| Integration & Reliability | 40 | 40 | 100% success rate, DLQ working |
| Testing & Observability | 30 | 30 | Logs, metrics, demo scripts |
| Docs & Repo Hygiene | 20 | 20 | 5 docs, clear structure |
| **SUBTOTAL** | **210** | **210** | **100%** |
| **BONUS: Airflow** | +30 | +30 | 2 DAGs + screenshots |
| **TOTAL** | **240** | **240** | **100% + Bonus** |

---

## 📝 Additional Notes

### Strengths
1. ✅ **Complete Implementation**: Both producer and consumer support streaming + batch
2. ✅ **ML Integration**: Real predictions with sklearn model (80% accuracy)
3. ✅ **Reliability**: 100% success rate, DLQ for errors
4. ✅ **Observability**: Comprehensive logs, metrics, Redpanda Console
5. ✅ **Automation**: Demo scripts, Airflow DAGs
6. ✅ **Documentation**: 5 markdown files covering all aspects
7. ✅ **Evidence**: Logs, screenshots, sample data prove end-to-end flow

### Beyond Requirements
- 🎁 Checkpoint resume for batch processing
- 🎁 Dead letter queue implementation
- 🎁 Latency tracking (inference time)
- 🎁 Dry-run mode for testing
- 🎁 Redpanda Console UI integration
- 🎁 Schema validation capability
- 🎁 Multiple helper scripts
- 🎁 WSL compatibility

### Reproducibility
All evidence can be reproduced by following `docs/kafka_quickstart.md`:
```bash
# 1. Start Kafka
docker compose -f docker-compose.kafka.yml up -d

# 2. Create topics
bash scripts/kafka_create_topics.sh

# 3. Run demo
bash scripts/run_kafka_demo.sh

# 4. Extract samples
bash scripts/dump_kafka_topics.sh
```

---

**Compliance Status**: ✅ **100% COMPLETE + BONUS**  
**Submitted**: October 12, 2025  
**Repository**: telco-churn-mlops-pipeline (branch: feature/kafka-integration)
