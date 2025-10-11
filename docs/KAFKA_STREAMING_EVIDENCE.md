# Kafka Streaming Evidence - Step 10

**Date**: October 11, 2025  
**Environment**: WSL Ubuntu on Windows  
**Kafka Version**: Confluent Platform 7.5.0  
**Model**: sklearn GradientBoostingClassifier (v1.7.2)

---

## 📋 Executive Summary

Successfully demonstrated end-to-end Kafka streaming pipeline for telco churn predictions:

- **Producer**: Sent 108 customer records to `telco.raw.customers` topic over 60 seconds
- **Consumer**: Processed messages in real-time, generated predictions, sent to `telco.churn.predictions` topic
- **Average Latency**: ~8.2 ms per prediction
- **Success Rate**: 100% (no dead letter queue messages)

---

## 🏗️ Infrastructure Status

### Kafka Cluster
```bash
docker ps --filter "name=kafka"
```

**Status**: ✅ HEALTHY
- Kafka Broker: Running for 2+ hours
  - Port 9092 (internal)
  - Port 9093 (external)
- Zookeeper: Running for 2+ hours
  - Port 2181

### Topics Created
| Topic Name | Partitions | Purpose |
|------------|------------|---------|
| `telco.raw.customers` | 3 | Input customer data |
| `telco.churn.predictions` | 3 | Output predictions |
| `telco.churn.deadletter` | 1 | Failed messages |

---

## 📊 Demo Execution Results

### Producer Performance
**Script**: `scripts/run_kafka_demo.sh`  
**Command**:
```bash
python3 src/streaming/producer.py \
  --mode streaming \
  --broker localhost:9093 \
  --topic telco.raw.customers \
  --dataset-path data/raw/Telco-Customer-Churn.csv \
  --events-per-sec 2
```

**Results**:
- Total messages sent: **108**
- Total failures: **0**
- Duration: **61.29 seconds**
- Average rate: **1.76 events/sec** (target: 2.0)

**Log**: `logs/kafka_producer_demo.log`

### Consumer Performance
**Command**:
```bash
python3 src/streaming/consumer.py \
  --mode streaming \
  --broker localhost:9093 \
  --input-topic telco.raw.customers \
  --output-topic telco.churn.predictions \
  --deadletter-topic telco.churn.deadletter \
  --consumer-group demo-consumer-group \
  --model-path artifacts/models/sklearn_pipeline.joblib
```

**Results**:
- Messages processed: **108**
- Predictions generated: **108**
- Failed messages: **0**
- Average inference latency: **8.2 ms**

**Log**: `logs/kafka_consumer_demo.log`

---

## 📦 Data Flow Evidence

### Sample Input Message
**Topic**: `telco.raw.customers`

```json
{
  "customerID": "7760-OYPDY",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.7,
  "TotalCharges": "151.65",
  "Churn": "Yes",
  "event_ts": "2025-10-11T03:28:23.234190Z"
}
```

### Corresponding Output Message
**Topic**: `telco.churn.predictions`

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

**Analysis**:
- ✅ Customer correctly predicted as **High Churn Risk** (71.2% probability)
- ✅ Inference latency: **9.92 ms** (well within SLA)
- ✅ Event timestamp preserved from input
- ✅ Processing timestamp added

---

## 📈 Prediction Distribution

**Sample Size**: 10 predictions (from `reports/kafka_predictions_sample.json`)

| Customer ID | Prediction | Probability | Latency (ms) |
|-------------|------------|-------------|--------------|
| 9867-JCZSP | No | 4.4% | 7.3 |
| 4671-VJLCL | No | 3.2% | 7.9 |
| **7760-OYPDY** | **Yes** | **71.2%** | 9.9 |
| 4667-QONEA | No | 7.1% | 8.2 |
| 8769-KKTPH | No | 15.0% | 6.3 |
| 3957-SQXML | No | 2.7% | 10.5 |
| 0434-CSFON | No | 49.3% | 13.1 |
| 0526-SXDJP | No | 3.6% | 7.2 |
| 5122-CYFXA | No | 44.0% | 5.5 |
| 8627-ZYGSZ | No | 23.0% | 8.5 |

**Insights**:
- Churn predictions: 1 Yes (10%), 9 No (90%)
- Average latency: **8.42 ms**
- Latency range: **5.5 - 13.1 ms**
- All predictions within acceptable range

---

## 🔧 Technical Details

### Model Information
- **Framework**: scikit-learn 1.7.2
- **Algorithm**: GradientBoostingClassifier
- **Training Accuracy**: 81.58%
- **Test ROC AUC**: 0.8466
- **Model File**: `artifacts/models/sklearn_pipeline.joblib`

### Kafka Configuration
- **Bootstrap Servers**: localhost:9092, localhost:9093
- **Consumer Group**: demo-consumer-group
- **Auto Offset Reset**: earliest
- **Max Poll Interval**: 300000 ms

---

## ✅ Validation Checklist

- [x] Kafka infrastructure running and healthy
- [x] Producer successfully sent 108 messages
- [x] Consumer processed all messages without errors
- [x] Predictions written to output topic
- [x] No messages in dead letter queue
- [x] Average latency < 15 ms (achieved 8.2 ms)
- [x] Sample data dumps created:
  - `reports/kafka_raw_sample.json` (10 messages)
  - `reports/kafka_predictions_sample.json` (10 messages)
- [x] Logs captured:
  - `logs/kafka_producer_demo.log`
  - `logs/kafka_consumer_demo.log`

---

## 📂 Artifacts Location

```
├── logs/
│   ├── kafka_producer_demo.log      # Producer execution log
│   └── kafka_consumer_demo.log      # Consumer execution log
├── reports/
│   ├── kafka_raw_sample.json        # 10 input customer records
│   └── kafka_predictions_sample.json # 10 output predictions
├── scripts/
│   ├── run_kafka_demo.sh            # Demo automation script
│   └── dump_kafka_topics.sh         # Topic dumping script
└── artifacts/
    ├── models/
    │   └── sklearn_pipeline.joblib  # Trained model (v1.7.2)
    └── metrics/
        └── sklearn_metrics.json     # Model performance metrics
```

---

## 🚀 Reproducing the Demo

### Prerequisites
1. Kafka running in Docker (ports 9092, 9093)
2. Zookeeper running (port 2181)
3. Python 3.12 virtual environment activated
4. Dataset: `data/raw/Telco-Customer-Churn.csv`

### Step 1: Run the Demo
```bash
cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1
./scripts/run_kafka_demo.sh
```

**Expected Output**:
- Consumer starts in background
- Producer sends ~108 messages over 60 seconds
- Consumer processes messages in real-time
- Logs saved to `logs/` directory

### Step 2: Dump Topics
```bash
./scripts/dump_kafka_topics.sh
```

**Expected Output**:
- 10 raw customer messages → `reports/kafka_raw_sample.json`
- 10 prediction messages → `reports/kafka_predictions_sample.json`
- Validation: ✓ PASS (>= 5 predictions found)

### Step 3: Verify Logs
```bash
# Check producer log
tail -30 logs/kafka_producer_demo.log

# Check consumer log
tail -50 logs/kafka_consumer_demo.log
```

---

## 🐛 Troubleshooting

### Issue 1: "Model loading failed - sklearn version mismatch"
**Cause**: Model trained with sklearn 1.6.1, environment has 1.7.2  
**Solution**: Retrain model with current version
```bash
source airflow_venv/bin/activate
python3 src/models/train.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --output artifacts/models/sklearn_pipeline.joblib \
  --save-preprocessor
```

### Issue 2: "No predictions in output topic"
**Cause**: Consumer failed to load model or process messages  
**Solution**: Check consumer log for errors
```bash
tail -100 logs/kafka_consumer_demo.log
```

### Issue 3: "Kafka not running"
**Cause**: Docker containers stopped  
**Solution**: Start Kafka and Zookeeper
```bash
cd /path/to/kafka-docker
docker-compose up -d
```

---

## 📸 Screenshots

See `docs/screenshots_02/` for:
- Kafka UI showing topic messages
- Producer/consumer metrics
- Airflow DAG execution history
- Topic partition details

---

## 🎯 Key Achievements

1. ✅ **End-to-End Pipeline**: Dataset → Kafka → ML Model → Predictions
2. ✅ **Real-Time Processing**: 8.2 ms average latency
3. ✅ **High Reliability**: 100% success rate (0 dead letter messages)
4. ✅ **Scalability**: 3 partitions for horizontal scaling
5. ✅ **Observability**: Comprehensive logging and metrics
6. ✅ **Reproducibility**: Automated scripts for easy replication

---

## 📝 Notes

- Demo runs for 60 seconds to balance between thoroughness and efficiency
- Producer rate: 2 events/sec (achieves ~1.76 events/sec actual)
- Consumer processes messages in streaming mode (no timeout)
- Model retrained to match current sklearn version (1.7.2)
- All 9 issues from Step 9 resolved and incorporated into scripts

---

**Generated**: October 11, 2025  
**Status**: ✅ COMPLETE  
**Evidence Quality**: HIGH (logs + topic dumps + documentation)
