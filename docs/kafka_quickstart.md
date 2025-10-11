# Kafka Integration - Quick Start Guide

**MP2: Kafka Streaming for Telco Churn Prediction**

This guide shows you how to run the complete Kafka streaming pipeline locally in under 10 minutes.

---

## 🎯 What You'll Build

```
Dataset (CSV) → Producer → Kafka Topic → Consumer → ML Model → Predictions Topic
```

**Components**:
- **Producer**: Sends customer data to Kafka (`telco.raw.customers`)
- **Consumer**: Reads data, runs ML model, writes predictions (`telco.churn.predictions`)
- **Kafka**: Message broker (Redpanda - Kafka-compatible)

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Start Kafka

```bash
# Start Kafka and Redpanda Console
docker compose -f docker-compose.kafka.yml up -d

# Verify services are running
docker ps
```

**Expected Output**:
```
CONTAINER ID   IMAGE                          STATUS
abc123         redpanda:v24.2.4               Up 10 seconds
def456         redpandadata/console:latest    Up 10 seconds
```

**Ports**:
- Kafka: `localhost:19092`
- Redpanda Console: `http://localhost:8080`

### Step 2: Create Topics

```bash
# Linux/WSL
bash scripts/kafka_create_topics.sh

# Windows PowerShell
.\scripts\kafka_create_topics.ps1
```

**Topics Created**:
- `telco.raw.customers` (3 partitions) - Input
- `telco.churn.predictions` (3 partitions) - Output
- `telco.churn.deadletter` (1 partition) - Failed messages

### Step 3: Run Producer (Streaming Mode)

```bash
python src/streaming/producer.py \
  --mode streaming \
  --dataset-path data/raw/Telco-Customer-Churn.csv \
  --events-per-sec 2 \
  --broker localhost:19092 \
  --topic telco.raw.customers
```

**What Happens**:
- Sends 2 customer records/second to Kafka
- Continues until Ctrl+C
- Logs to `logs/kafka_producer.log`

### Step 4: Run Consumer (Streaming Mode)

```bash
# In a new terminal
python src/streaming/consumer.py \
  --mode streaming \
  --broker localhost:19092 \
  --input-topic telco.raw.customers \
  --output-topic telco.churn.predictions \
  --deadletter-topic telco.churn.deadletter \
  --consumer-group churn-prediction-group
```

**What Happens**:
- Reads customer records from Kafka
- Runs ML model (sklearn GradientBoosting)
- Writes predictions to output topic
- Logs to `logs/kafka_consumer.log`

### Step 5: View Results

**Option 1: Check Logs**
```bash
# Producer stats
tail -30 logs/kafka_producer.log | grep "SUMMARY"

# Consumer stats  
tail -30 logs/kafka_consumer.log | grep "Processed"
```

**Option 2: Redpanda Console**
- Open: `http://localhost:8080`
- Click **Topics** → `telco.churn.predictions`
- View real-time predictions

**Option 3: Console Consumer**
```bash
docker exec telco-redpanda rpk topic consume telco.churn.predictions --num 5
```

**Sample Output**:
```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.73,
  "prediction": "Yes",
  "processed_ts": "2025-10-12T10:15:30Z",
  "inference_latency_ms": 8.5
}
```

---

## 🔄 Batch Mode (Alternative)

**Use Case**: Process entire dataset in batches with checkpointing

### Producer (Batch)

```bash
python src/streaming/producer.py \
  --mode batch \
  --dataset-path data/raw/Telco-Customer-Churn.csv \
  --batch-size 100 \
  --broker localhost:19092 \
  --topic telco.raw.customers
```

**Features**:
- Processes CSV in chunks of 100 rows
- Saves checkpoint to `artifacts/producer_checkpoint.json`
- Resumes from last checkpoint if interrupted
- Exits when dataset is complete

### Consumer (Batch)

```bash
python src/streaming/consumer.py \
  --mode batch \
  --broker localhost:19092 \
  --input-topic telco.raw.customers \
  --output-topic telco.churn.predictions \
  --deadletter-topic telco.churn.deadletter \
  --consumer-group churn-batch-group \
  --timeout-ms 30000 \
  --max-messages 1000
```

**Features**:
- Processes up to 1000 messages
- Exits after 30-second timeout
- Generates summary report in `artifacts/reports/`

---

## 🚀 Full Demo (60 Seconds)

**Automated script to run producer + consumer together:**

```bash
# Linux/WSL
bash scripts/run_kafka_demo.sh

# What it does:
# 1. Starts consumer in background
# 2. Runs producer for 60 seconds (2 events/sec)
# 3. Waits for consumer to finish
# 4. Saves logs to logs/kafka_*_demo.log
```

**Then extract topic samples:**
```bash
bash scripts/dump_kafka_topics.sh

# Creates:
# - reports/kafka_raw_sample.json (10 inputs)
# - reports/kafka_predictions_sample.json (10 outputs)
```

---

## 📊 Kafka Topics Schema

### Input Topic: `telco.raw.customers`

```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "tenure": 1,
  "MonthlyCharges": 29.85,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "Churn": "No",
  "event_ts": "2025-10-12T10:15:25.123Z"
}
```

**Fields**: 21 total (19 features + customerID + event_ts)

### Output Topic: `telco.churn.predictions`

```json
{
  "customerID": "7590-VHVEG",
  "churn_probability": 0.73,
  "prediction": "Yes",
  "event_ts": "2025-10-12T10:15:25.123Z",
  "processed_ts": "2025-10-12T10:15:30.456Z",
  "inference_latency_ms": 8.5,
  "model_version": "sklearn_pipeline_v1"
}
```

**Fields**: 7 total (prediction results + metadata)

---

## ⚙️ Configuration

### Kafka Broker

**File**: `docker-compose.kafka.yml`

```yaml
services:
  redpanda:
    ports:
      - "19092:19092"  # Kafka API
      - "18081:18081"  # Schema Registry
      - "18082:18082"  # Pandaproxy
    environment:
      - REDPANDA_MODE=dev-container
```

### Producer Defaults

**File**: `src/streaming/producer.py`

```python
DEFAULT_BROKER = "localhost:19092"
DEFAULT_TOPIC = "telco.raw.customers"
DEFAULT_EVENTS_PER_SEC = 1.0
DEFAULT_BATCH_SIZE = 100
```

### Consumer Defaults

**File**: `src/streaming/consumer.py`

```python
DEFAULT_BROKER = "localhost:19092"
DEFAULT_INPUT_TOPIC = "telco.raw.customers"
DEFAULT_OUTPUT_TOPIC = "telco.churn.predictions"
DEFAULT_DEADLETTER_TOPIC = "telco.churn.deadletter"
DEFAULT_CONSUMER_GROUP = "churn-prediction-group"
DEFAULT_TIMEOUT_MS = 10000  # 10 seconds
```

---

## 🔍 Monitoring & Observability

### 1. Redpanda Console (Recommended)

Open `http://localhost:8080`

**Features**:
- Real-time topic messages
- Consumer group lag
- Broker health
- Topic configuration

### 2. Command-Line Tools

```bash
# List topics
docker exec telco-redpanda rpk topic list

# Topic details
docker exec telco-redpanda rpk topic describe telco.raw.customers

# Consumer group status
docker exec telco-redpanda rpk group describe churn-prediction-group

# Broker health
docker exec telco-redpanda rpk cluster health
```

### 3. Application Logs

```bash
# Producer metrics
tail -f logs/kafka_producer.log

# Consumer metrics
tail -f logs/kafka_consumer.log

# Check for errors
grep ERROR logs/kafka_*.log
```

---

## 🐛 Troubleshooting

### Issue: "No brokers available"

**Cause**: Kafka not running

**Solution**:
```bash
docker compose -f docker-compose.kafka.yml up -d
docker ps  # Verify running
```

### Issue: "Topic does not exist"

**Cause**: Topics not created

**Solution**:
```bash
bash scripts/kafka_create_topics.sh
docker exec telco-redpanda rpk topic list  # Verify
```

### Issue: Consumer not processing messages

**Cause**: Consumer group offset at end of topic

**Solution**:
```bash
# Reset consumer group
docker exec telco-redpanda rpk group seek churn-prediction-group --to start

# Or use new consumer group name
python src/streaming/consumer.py --consumer-group my-new-group ...
```

### Issue: Model loading failed

**Cause**: sklearn version mismatch

**Solution**:
```bash
# Retrain model with current sklearn
python src/models/train.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --output artifacts/models/sklearn_pipeline.joblib \
  --save-preprocessor
```

### Issue: Connection timeout

**Cause**: Firewall or wrong broker address

**Solution**:
```bash
# Test connectivity
telnet localhost 19092

# Try alternative broker address
python src/streaming/producer.py --broker localhost:9092 ...
```

---

## 📁 Key Files & Directories

```
├── src/streaming/
│   ├── producer.py           # Kafka producer (streaming + batch)
│   └── consumer.py           # Kafka consumer with ML inference
├── scripts/
│   ├── kafka_create_topics.sh      # Topic creation
│   ├── run_kafka_demo.sh           # 60-second demo
│   └── dump_kafka_topics.sh        # Extract samples
├── docker-compose.kafka.yml  # Kafka infrastructure
├── logs/
│   ├── kafka_producer.log          # Producer execution
│   └── kafka_consumer.log          # Consumer execution
├── reports/
│   ├── kafka_raw_sample.json       # Input samples
│   └── kafka_predictions_sample.json  # Output samples
└── docs/
    ├── KAFKA_STREAMING_EVIDENCE.md # Detailed evidence
    └── kafka_quickstart.md          # This file
```

---

## 🎓 Learning Path

### Beginner (Start Here)
1. Run **Quick Start (5 Minutes)** above
2. Observe messages in Redpanda Console
3. Check logs for producer/consumer stats

### Intermediate
1. Run **Batch Mode** with different batch sizes
2. Modify topic partitions (3 → 6)
3. Test with multiple consumers (horizontal scaling)

### Advanced
1. Integrate with **Airflow** (see `airflow_home/dags/kafka_*.py`)
2. Add schema validation with Avro
3. Implement exactly-once semantics

---

## 📖 Related Documentation

- **Evidence Report**: `docs/KAFKA_STREAMING_EVIDENCE.md`
- **Schema Details**: `docs/kafka_schema.md`
- **Integration Testing**: `docs/kafka_integration_testing.md`
- **Airflow DAGs**: `airflow_home/dags/kafka_batch_dag.py`, `kafka_streaming_dag.py`

---

## ✅ Success Criteria

After completing this guide, you should have:

- [x] Kafka running locally (docker-compose)
- [x] Topics created (raw.customers, churn.predictions, deadletter)
- [x] Producer sending customer data
- [x] Consumer generating predictions
- [x] Logs showing successful processing
- [x] Sample predictions in Redpanda Console or JSON files

**Next Steps**: Run Airflow DAGs for automated orchestration (see Step 10 documentation)

---

**Questions?** Check `docs/KAFKA_STREAMING_EVIDENCE.md` for detailed troubleshooting and evidence.
