# Kafka Streaming Observability Guide

Comprehensive guide for monitoring and observability of the Telco Churn Kafka streaming pipeline.

## Table of Contents

- [Overview](#overview)
- [Structured Logging](#structured-logging)
- [Prometheus Metrics](#prometheus-metrics)
- [Health Checks](#health-checks)
- [Prometheus Setup](#prometheus-setup)
- [Grafana Dashboards](#grafana-dashboards)
- [Log Analysis](#log-analysis)
- [Alert Rules](#alert-rules)
- [Troubleshooting](#troubleshooting)

---

## Overview

The streaming pipeline provides comprehensive observability through:

1. **Structured JSON Logging**: Machine-readable logs with rich context
2. **Prometheus Metrics**: Real-time performance and health metrics
3. **Health Endpoints**: Service health status checks
4. **Dual Output**: Human-readable console + JSON file logs

### Architecture

```
┌─────────────────┐
│  Kafka Producer │
│  (JSON Logs)    │──► logs/kafka_producer.log
└─────────────────┘

┌─────────────────┐
│  Kafka Consumer │
│  (JSON Logs +   │──► logs/kafka_consumer.log
│   Prometheus)   │──► http://localhost:8000/metrics
└─────────────────┘──► http://localhost:8000/health
```

---

## Structured Logging

### Log Format

Both producer and consumer output structured JSON logs with the following standard fields:

```json
{
  "timestamp": "2025-01-11T02:15:30.123Z",
  "level": "INFO",
  "logger": "kafka_consumer",
  "message": "Message processed successfully | Prediction: 1 | Probability: 0.8542",
  "module": "consumer",
  "function": "process_message",
  "line": 865
}
```

### Extra Fields

Additional context fields are included based on the event type:

#### Producer Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `message_id` | string | Customer ID or message key | `"5678-ABCDE"` |
| `topic` | string | Kafka topic name | `"telco.raw.customers"` |
| `partition` | integer | Partition number | `2` |
| `offset` | integer | Message offset | `1234` |
| `latency_ms` | float | Publishing latency in milliseconds | `15.42` |
| `event_type` | string | Event classification | `"message_published"` |
| `error_type` | string | Error classification (if failed) | `"schema_validation"` |
| `batch_size` | integer | Number of messages in batch | `100` |
| `success_count` | integer | Successful messages in batch | `98` |
| `failure_count` | integer | Failed messages in batch | `2` |

#### Consumer Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `message_id` | string | Customer ID or message key | `"5678-ABCDE"` |
| `topic` | string | Kafka topic name | `"telco.raw.customers"` |
| `partition` | integer | Partition number | `2` |
| `offset` | integer | Message offset | `1234` |
| `latency_ms` | float | Processing latency in milliseconds | `42.18` |
| `event_type` | string | Event classification | `"message_processed"` |
| `error_type` | string | Error classification (if failed) | `"inference_error"` |
| `prediction` | integer | Churn prediction (0 or 1) | `1` |
| `churn_probability` | float | Churn probability | `0.8542` |

### Event Types

#### Producer Events

- `validation_failure`: Message failed schema validation before publishing
- `message_published`: Message successfully published to Kafka
- `publish_failure`: Kafka publish operation failed

#### Consumer Events

- `validation_failure`: Input message failed schema validation
- `transformation_failure`: Feature transformation failed
- `inference_failure`: Model inference failed
- `output_validation_failure`: Output message failed schema validation (bug)
- `message_processed`: Message successfully processed
- `processing_exception`: Unexpected error during processing

### Error Types

- `schema_validation`: Schema validation error
- `feature_transformation`: Feature engineering error
- `model_inference`: Inference error
- `kafka_error`: Kafka connection/operation error
- `unknown_error`: Unexpected/uncategorized error

### Log Files

| Component | File Path | Format | Rotation |
|-----------|-----------|--------|----------|
| Producer | `logs/kafka_producer.log` | JSON | 10MB, 5 backups |
| Consumer | `logs/kafka_consumer.log` | JSON | 10MB, 5 backups |
| Consumer (timestamped) | `artifacts/logs/consumer_YYYYMMDD_HHMMSS.log` | Human-readable | None |

---

## Prometheus Metrics

The consumer exposes Prometheus metrics on **http://localhost:8000/metrics**.

### Available Metrics

#### Counters

**`kafka_messages_processed_total{status, topic}`**
- Total number of messages processed
- Labels:
  - `status`: `success` or `failed`
  - `topic`: Source topic name
- Example:
  ```promql
  kafka_messages_processed_total{status="success",topic="telco.raw.customers"}
  ```

**`kafka_messages_failed_total{error_type, topic}`**
- Total number of failed messages by error type
- Labels:
  - `error_type`: `validation_error`, `processing_error`, `inference_error`, `unknown_error`
  - `topic`: Source topic name
- Example:
  ```promql
  kafka_messages_failed_total{error_type="validation_error",topic="telco.raw.customers"}
  ```

#### Histograms

**`kafka_processing_latency_seconds{operation}`**
- Processing latency distribution in seconds
- Labels:
  - `operation`: `validation`, `transformation`, `inference`, `total`
- Buckets: Default Prometheus buckets (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
- Provides `_sum`, `_count`, and `_bucket` metrics
- Example:
  ```promql
  # Average latency
  rate(kafka_processing_latency_seconds_sum[5m]) / rate(kafka_processing_latency_seconds_count[5m])
  
  # 95th percentile
  histogram_quantile(0.95, rate(kafka_processing_latency_seconds_bucket[5m]))
  ```

#### Gauges

**`kafka_consumer_lag{topic, partition}`**
- Current consumer lag (messages behind)
- Labels:
  - `topic`: Topic name
  - `partition`: Partition number
- Example:
  ```promql
  kafka_consumer_lag{topic="telco.raw.customers",partition="0"}
  ```

**`kafka_consumer_model_loaded`**
- Model loaded status (1=loaded, 0=not loaded)
- Example:
  ```promql
  kafka_consumer_model_loaded == 0  # Alert if model not loaded
  ```

**`kafka_consumer_broker_connected`**
- Broker connection status (1=connected, 0=disconnected)
- Example:
  ```promql
  kafka_consumer_broker_connected == 0  # Alert if disconnected
  ```

#### Info

**`kafka_consumer_info`**
- Consumer metadata and configuration
- Labels:
  - `version`: Consumer version
  - `mode`: `streaming` or `batch`
  - `broker`: Kafka broker address
  - `consumer_group`: Consumer group ID
  - `model_backend`: `sklearn` or `spark`
- Example:
  ```promql
  kafka_consumer_info{mode="streaming",model_backend="sklearn"}
  ```

---

## Health Checks

### Health Endpoint

**URL**: `http://localhost:8000/health` (planned - not yet implemented)

**Response Format**:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-11T02:15:30.123Z",
  "checks": {
    "broker_connection": "pass",
    "model_loaded": "pass"
  }
}
```

**Status Values**:
- `status`: `healthy` or `unhealthy`
- `checks.<check_name>`: `pass`, `fail`, `not_initialized`, `unknown`

**Health Check Logic**:
- `broker_connection`: Tests `consumer.topics()` - fails if broker unreachable
- `model_loaded`: Checks if model loaded successfully during startup
- Overall `status` is `unhealthy` if any check is `fail`

---

## Prometheus Setup

### 1. Install Prometheus

**Linux/WSL**:
```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar -xzf prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64
```

**macOS**:
```bash
brew install prometheus
```

**Windows**:
Download from https://prometheus.io/download/ and extract.

### 2. Configure Prometheus

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s  # Scrape targets every 15 seconds
  evaluation_interval: 15s  # Evaluate rules every 15 seconds

scrape_configs:
  - job_name: 'kafka_consumer'
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'telco-churn-consumer'
          environment: 'dev'
```

### 3. Start Prometheus

```bash
./prometheus --config.file=prometheus.yml
```

Access Prometheus UI at **http://localhost:9090**.

### 4. Verify Metrics

1. Navigate to **http://localhost:9090/targets**
2. Check that `kafka_consumer` target is **UP**
3. Go to **http://localhost:9090/graph**
4. Query: `kafka_messages_processed_total`

---

## Grafana Dashboards

### 1. Install Grafana

**Linux/WSL**:
```bash
wget https://dl.grafana.com/oss/release/grafana-10.0.0.linux-amd64.tar.gz
tar -xzf grafana-10.0.0.linux-amd64.tar.gz
cd grafana-10.0.0/bin
./grafana-server
```

**macOS**:
```bash
brew install grafana
brew services start grafana
```

**Windows**:
Download from https://grafana.com/grafana/download.

Access Grafana at **http://localhost:3000** (default credentials: admin/admin).

### 2. Add Prometheus Data Source

1. Go to **Configuration → Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. Set URL: `http://localhost:9090`
5. Click **Save & Test**

### 3. Sample Dashboard Panels

#### Panel 1: Message Processing Rate

**Query**:
```promql
rate(kafka_messages_processed_total[5m])
```

**Type**: Graph
**Title**: Message Processing Rate (msg/sec)
**Legend**: `{{status}} - {{topic}}`

#### Panel 2: Success Rate

**Query**:
```promql
100 * sum(rate(kafka_messages_processed_total{status="success"}[5m])) / sum(rate(kafka_messages_processed_total[5m]))
```

**Type**: Stat
**Title**: Success Rate (%)
**Unit**: Percent (0-100)
**Thresholds**: <95 (red), 95-99 (yellow), >99 (green)

#### Panel 3: Average Processing Latency

**Query**:
```promql
rate(kafka_processing_latency_seconds_sum{operation="total"}[5m]) / rate(kafka_processing_latency_seconds_count{operation="total"}[5m])
```

**Type**: Graph
**Title**: Average Processing Latency (seconds)
**Unit**: seconds (s)

#### Panel 4: Error Breakdown

**Query**:
```promql
sum by (error_type) (rate(kafka_messages_failed_total[5m]))
```

**Type**: Pie Chart
**Title**: Error Types Distribution
**Legend**: `{{error_type}}`

#### Panel 5: P95 Latency by Operation

**Query**:
```promql
histogram_quantile(0.95, rate(kafka_processing_latency_seconds_bucket[5m]))
```

**Type**: Graph
**Title**: 95th Percentile Latency by Operation
**Legend**: `{{operation}}`

#### Panel 6: Consumer Health

**Queries**:
```promql
kafka_consumer_model_loaded
kafka_consumer_broker_connected
```

**Type**: Stat
**Title**: Consumer Health
**Value Mappings**: 0=Unhealthy (red), 1=Healthy (green)

### 4. Import Dashboard JSON

Create `grafana_dashboard.json`:

```json
{
  "dashboard": {
    "title": "Kafka Consumer Observability",
    "panels": [
      {
        "id": 1,
        "title": "Message Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(kafka_messages_processed_total[5m])",
            "legendFormat": "{{status}} - {{topic}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Success Rate (%)",
        "type": "stat",
        "targets": [
          {
            "expr": "100 * sum(rate(kafka_messages_processed_total{status=\"success\"}[5m])) / sum(rate(kafka_messages_processed_total[5m]))"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 95, "color": "yellow"},
                {"value": 99, "color": "green"}
              ]
            }
          }
        }
      }
    ]
  }
}
```

Import via **Dashboards → Import → Upload JSON file**.

---

## Log Analysis

### Using `jq` for JSON Logs

#### View All Logs

```bash
cat logs/kafka_consumer.log | jq .
```

#### Filter by Log Level

```bash
# Errors only
cat logs/kafka_consumer.log | jq 'select(.level == "ERROR")'

# Warnings and errors
cat logs/kafka_consumer.log | jq 'select(.level == "WARNING" or .level == "ERROR")'
```

#### Filter by Event Type

```bash
# Failed messages
cat logs/kafka_consumer.log | jq 'select(.event_type == "validation_failure")'

# Successful processing
cat logs/kafka_consumer.log | jq 'select(.event_type == "message_processed")'
```

#### Filter by Customer ID

```bash
cat logs/kafka_consumer.log | jq 'select(.message_id == "5678-ABCDE")'
```

#### Extract Specific Fields

```bash
# Show only timestamp, level, and message
cat logs/kafka_consumer.log | jq '{timestamp, level, message}'

# Show processing latency
cat logs/kafka_consumer.log | jq 'select(.latency_ms != null) | {timestamp, message_id, latency_ms}'
```

#### Calculate Average Latency

```bash
cat logs/kafka_consumer.log | \
  jq 'select(.latency_ms != null) | .latency_ms' | \
  awk '{sum+=$1; count++} END {print "Average:", sum/count, "ms"}'
```

#### Count Events by Type

```bash
cat logs/kafka_consumer.log | \
  jq -r '.event_type' | \
  sort | uniq -c | sort -nr
```

#### Find High Latency Messages (>100ms)

```bash
cat logs/kafka_consumer.log | jq 'select(.latency_ms > 100)'
```

### Using `grep` for Pattern Matching

```bash
# Find all errors
grep '"level": "ERROR"' logs/kafka_consumer.log

# Find specific customer
grep '"message_id": "5678-ABCDE"' logs/kafka_consumer.log

# Find schema validation failures
grep '"error_type": "schema_validation"' logs/kafka_consumer.log
```

### Log Rotation Analysis

```bash
# View all rotated logs
ls -lh logs/kafka_consumer.log*

# Search across all rotated logs
zgrep '"event_type": "inference_failure"' logs/kafka_consumer.log*
```

### Real-Time Log Monitoring

```bash
# Tail JSON logs with pretty printing
tail -f logs/kafka_consumer.log | jq .

# Monitor errors only
tail -f logs/kafka_consumer.log | jq 'select(.level == "ERROR")'
```

---

## Alert Rules

### Prometheus Alert Rules

Create `alerts.yml`:

```yaml
groups:
  - name: kafka_consumer_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(kafka_messages_failed_total[5m])) /
            sum(rate(kafka_messages_processed_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # Consumer disconnected
      - alert: ConsumerDisconnected
        expr: kafka_consumer_broker_connected == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Consumer disconnected from Kafka broker"
          description: "Consumer has been disconnected for more than 1 minute"

      # Model not loaded
      - alert: ModelNotLoaded
        expr: kafka_consumer_model_loaded == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "ML model not loaded"
          description: "Consumer cannot process messages without a loaded model"

      # High processing latency
      - alert: HighProcessingLatency
        expr: |
          histogram_quantile(0.95,
            rate(kafka_processing_latency_seconds_bucket{operation="total"}[5m])
          ) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High processing latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 1s)"

      # Consumer lag increasing
      - alert: ConsumerLagIncreasing
        expr: |
          kafka_consumer_lag > 1000 and
          deriv(kafka_consumer_lag[5m]) > 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Consumer lag is increasing"
          description: "Lag is {{ $value }} messages and growing"

      # Low throughput
      - alert: LowThroughput
        expr: |
          sum(rate(kafka_messages_processed_total[5m])) < 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low message throughput"
          description: "Processing {{ $value }} messages/sec (expected: >1)"
```

Add to `prometheus.yml`:

```yaml
rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']  # Alertmanager address
```

### Alertmanager Configuration

Create `alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'email-notifications'

receivers:
  - name: 'email-notifications'
    email_configs:
      - to: 'team@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alertmanager@example.com'
        auth_password: 'password'
```

---

## Troubleshooting

### Issue: Metrics endpoint not accessible

**Symptoms**:
- `curl http://localhost:8000/metrics` returns connection refused
- Prometheus target shows as DOWN

**Diagnosis**:
```bash
# Check if consumer is running
ps aux | grep consumer

# Check port 8000 is listening
netstat -an | grep 8000  # Linux/WSL
Get-NetTCPConnection -LocalPort 8000  # PowerShell
```

**Solutions**:
1. Ensure consumer is started: `python -m src.streaming.consumer --mode streaming`
2. Check if port 8000 is already in use
3. Verify prometheus_client installed: `pip show prometheus-client`
4. Check consumer logs for metrics server errors

### Issue: JSON logs not created

**Symptoms**:
- `logs/kafka_consumer.log` doesn't exist
- Only timestamped logs in `artifacts/logs/`

**Diagnosis**:
```bash
# Check if logs directory exists
ls -la logs/

# Check file permissions
ls -la logs/kafka_consumer.log
```

**Solutions**:
1. Ensure `logs/` directory exists (created automatically by `setup_logging()`)
2. Check disk space: `df -h`
3. Verify write permissions
4. Check for errors in console output

### Issue: Structured log fields missing

**Symptoms**:
- JSON logs missing `message_id`, `latency_ms`, etc.
- Only standard fields present

**Diagnosis**:
```bash
# Check if extra fields are present
cat logs/kafka_consumer.log | jq 'select(.message_id != null)' | head -1
```

**Solutions**:
1. Ensure using latest consumer code with structured logging
2. Verify JSONFormatter is configured in setup_logging()
3. Check log statements use `extra={}` parameter
4. Restart consumer to reload code changes

### Issue: Prometheus metrics not updating

**Symptoms**:
- Metrics endpoint returns old values
- Counter/gauge values frozen

**Diagnosis**:
```promql
# Check last scrape time in Prometheus
up{job="kafka_consumer"}

# Verify metric labels
kafka_messages_processed_total
```

**Solutions**:
1. Check Prometheus scrape_interval (default: 15s)
2. Verify consumer is processing messages
3. Check metric labels match Prometheus queries
4. Restart consumer if metrics server hung

### Issue: High memory usage from log files

**Symptoms**:
- `logs/kafka_consumer.log` growing very large
- Disk space running low

**Diagnosis**:
```bash
# Check log file sizes
du -h logs/*.log

# Check rotation status
ls -lh logs/kafka_consumer.log*
```

**Solutions**:
1. Log rotation configured (10MB, 5 backups by default)
2. Reduce log level: `--log-level WARNING` (instead of DEBUG)
3. Manually archive old logs:
   ```bash
   gzip logs/kafka_consumer.log.1
   ```
4. Increase rotation maxBytes if needed (edit consumer.py)

### Issue: Permission denied writing to logs/

**Symptoms**:
- Consumer crashes with `PermissionError`
- Cannot create log files

**Diagnosis**:
```bash
# Check directory permissions
ls -ld logs/

# Check user/group ownership
ls -l logs/
```

**Solutions**:
```bash
# Fix permissions
chmod 755 logs/
chown $USER:$USER logs/

# Or run consumer with sudo (not recommended)
```

### Diagnostic Commands Reference

```bash
# Check consumer process
ps aux | grep "src.streaming.consumer"

# Check Kafka connectivity
nc -zv localhost 9092  # Kafka broker

# Check metrics endpoint
curl -s http://localhost:8000/metrics | head -20

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.job == "kafka_consumer")'

# Monitor log file growth
watch -n 5 'ls -lh logs/kafka_consumer.log'

# Check recent errors
tail -100 logs/kafka_consumer.log | jq 'select(.level == "ERROR")'
```

---

## Best Practices

### 1. Log Level Management

- **Development**: `--log-level DEBUG` for detailed troubleshooting
- **Production**: `--log-level INFO` for balance
- **High-volume**: `--log-level WARNING` to reduce I/O

### 2. Metric Cardinality

- Avoid high-cardinality labels (customer IDs, timestamps)
- Use aggregation for detailed analysis (logs)
- Keep label values bounded (error_types, topics)

### 3. Alert Tuning

- Set appropriate thresholds for your workload
- Use `for:` clause to avoid flapping alerts
- Adjust based on historical baselines

### 4. Log Retention

- Rotate logs to prevent disk exhaustion
- Archive old logs to S3/object storage
- Set retention policies (30 days default)

### 5. Dashboard Organization

- Separate operational (real-time) and analytical dashboards
- Use consistent time ranges (Last 1h, Last 24h)
- Add annotations for deployments/incidents

### 6. Performance Impact

- Structured logging adds ~5-10% overhead
- Prometheus scraping is lightweight (<1% CPU)
- Monitor metrics server memory usage

---

## Summary

The observability stack provides:

✅ **Structured JSON logs** with rich context for debugging  
✅ **Prometheus metrics** for real-time monitoring  
✅ **Health checks** for service status  
✅ **Grafana dashboards** for visualization  
✅ **Alert rules** for proactive incident response  
✅ **Log analysis tools** (jq, grep) for troubleshooting  

**Next Steps**:
1. Configure Prometheus and Grafana
2. Import sample dashboard
3. Set up alert rules
4. Monitor metrics during load testing
5. Tune thresholds based on baselines

For additional questions, refer to:
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [kafka-python Documentation](https://kafka-python.readthedocs.io/)
