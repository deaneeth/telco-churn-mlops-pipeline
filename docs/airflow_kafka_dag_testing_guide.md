# Airflow Kafka DAG Testing Guide
## Step-by-Step Instructions with Screenshot Evidence

This guide provides detailed instructions for testing the Kafka DAGs in Airflow and capturing screenshot evidence.

---

## Prerequisites

### 1. Environment Setup
Ensure you have the following running:
- ✅ Airflow environment (`airflow_env` virtual environment activated)
- ✅ Kafka broker (Docker Compose or standalone)
- ✅ Project directory: `E:\ZuuCrew\telco-churn-prediction-mini-project-1`

### 2. Verify Airflow Installation
```powershell
# Activate Airflow environment
.\airflow_env\Scripts\Activate.ps1

# Verify Airflow version
airflow version

# Check Airflow home directory
echo $env:AIRFLOW_HOME
# Should show: E:\ZuuCrew\telco-churn-prediction-mini-project-1\airflow_home
```

### 3. Start Kafka Broker
```powershell
# Start production Kafka (if not already running)
docker compose up -d

# OR start test Kafka
docker compose -f docker-compose.test.yml up -d

# Verify Kafka is running
docker ps | Select-String "redpanda"
```

---

## Part 1: DAG Validation

### Step 1.1: List All DAGs
```powershell
# Navigate to project directory
cd E:\ZuuCrew\telco-churn-prediction-mini-project-1

# List all DAGs
airflow dags list
```

**Expected Output:**
```
dag_id                           | filepath                        | owner
=================================|=================================|====================
kafka_batch_pipeline             | kafka_batch_dag.py              | data-engineering-team
kafka_streaming_pipeline         | kafka_streaming_dag.py          | data-engineering-team
telco_churn_prediction_pipeline  | telco_churn_dag.py              | data-science-team
```

📸 **Screenshot 1**: Take screenshot of `airflow dags list` output
- **Filename**: `01_airflow_dags_list.png`
- **Highlight**: Both Kafka DAGs present

### Step 1.2: Show DAG Details
```powershell
# Show streaming DAG details
airflow dags show kafka_streaming_pipeline

# Show batch DAG details
airflow dags show kafka_batch_pipeline
```

📸 **Screenshot 2**: Take screenshot of streaming DAG details
- **Filename**: `02_streaming_dag_details.png`

📸 **Screenshot 3**: Take screenshot of batch DAG details
- **Filename**: `03_batch_dag_details.png`

### Step 1.3: List DAG Tasks
```powershell
# List streaming DAG tasks
airflow tasks list kafka_streaming_pipeline

# List batch DAG tasks
airflow tasks list kafka_batch_pipeline
```

**Expected Streaming Tasks:**
```
health_check_kafka
health_check_kafka_bash
monitor_consumer
start_consumer
```

**Expected Batch Tasks:**
```
generate_summary
run_consumer_window
trigger_producer
trigger_producer_bash
```

📸 **Screenshot 4**: Take screenshot of task lists
- **Filename**: `04_dag_tasks_list.png`
- **Show both**: Streaming and batch tasks

---

## Part 2: Streaming DAG Testing

### Step 2.1: Test Kafka Health Check
```powershell
# Test health check task
airflow tasks test kafka_streaming_pipeline health_check_kafka 2025-10-11
```

**Expected Output:**
```
✓ Kafka health check passed: {
  "broker": "localhost:9093",
  "timestamp": "2025-10-11T...",
  "status": "HEALTHY",
  "topics_found": ["telco.raw.customers", "telco.churn.predictions", ...],
  "input_topic_exists": true,
  "output_topic_exists": true
}
```

📸 **Screenshot 5**: Take screenshot of health check output
- **Filename**: `05_streaming_health_check.png`
- **Highlight**: "HEALTHY" status

### Step 2.2: Test Consumer Start
```powershell
# Test start consumer task
airflow tasks test kafka_streaming_pipeline start_consumer 2025-10-11
```

**Expected Output:**
```
Starting consumer process...
Command: python src\streaming\consumer.py --broker localhost:9093 ...
✓ Consumer started successfully: PID=12345
✓ Log file: artifacts\logs\kafka_consumer_streaming_20251011_...log
```

📸 **Screenshot 6**: Take screenshot of consumer start
- **Filename**: `06_streaming_start_consumer.png`
- **Highlight**: Consumer PID and log file path

### Step 2.3: Test Consumer Monitoring
```powershell
# Wait a few seconds for consumer to process messages
Start-Sleep -Seconds 10

# Test monitor task
airflow tasks test kafka_streaming_pipeline monitor_consumer 2025-10-11
```

**Expected Output:**
```
✓ Consumer is running (PID=12345)

Consumer health: {
  "pid": 12345,
  "process_running": true,
  "log_file": "artifacts\\logs\\kafka_consumer_streaming_...log",
  "log_size_bytes": 15420,
  "timestamp": "2025-10-11T...",
  "status": "HEALTHY"
}
```

📸 **Screenshot 7**: Take screenshot of monitor output
- **Filename**: `07_streaming_monitor_consumer.png`
- **Highlight**: "process_running": true

### Step 2.4: View Consumer Logs
```powershell
# Find the latest consumer log
Get-ChildItem artifacts\logs\kafka_consumer_streaming_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# View last 30 lines of log
Get-Content (Get-ChildItem artifacts\logs\kafka_consumer_streaming_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 30
```

📸 **Screenshot 8**: Take screenshot of consumer log content
- **Filename**: `08_streaming_consumer_logs.png`
- **Highlight**: Messages being processed

---

## Part 3: Batch DAG Testing

### Step 3.1: Test Producer Trigger
```powershell
# Test producer task
airflow tasks test kafka_batch_pipeline trigger_producer 2025-10-11
```

**Expected Output:**
```
Triggering producer batch...
Command: python src\streaming\producer.py --broker localhost:9093 ...
Input file: data\processed\test.csv
Batch size: 1000

✓ Producer batch completed:
  Messages sent: 1000
  Messages failed: 0
  Success rate: 100.0%
```

📸 **Screenshot 9**: Take screenshot of producer output
- **Filename**: `09_batch_trigger_producer.png`
- **Highlight**: Messages sent count and success rate

### Step 3.2: Test Consumer Window
```powershell
# Test consumer window task
airflow tasks test kafka_batch_pipeline run_consumer_window 2025-10-11
```

**Expected Output:**
```
Running consumer for time window...
Timeout: 60 seconds
Max messages: 1000

✓ Consumer window completed:
  Messages processed: 1000
  Predictions made: 1000
  Duration: 25.34 seconds
  Throughput: 39.47 msg/sec
```

📸 **Screenshot 10**: Take screenshot of consumer window output
- **Filename**: `10_batch_consumer_window.png`
- **Highlight**: Messages processed and throughput

### Step 3.3: Test Summary Generation
```powershell
# Test summary task
airflow tasks test kafka_batch_pipeline generate_summary 2025-10-11
```

**Expected Output:**
```
Generating summary report...
Retrieved 1000 predictions from output topic

================================================================
BATCH PROCESSING SUMMARY
================================================================
Total Predictions: 1000
Churn Rate: 26.5%
  - Predicted Churn (Yes): 265
  - Predicted No Churn (No): 735
Average Churn Probability: 0.3421
High Risk Customers (>70%): 87

Pipeline Performance:
  - Messages Sent: 1000
  - Messages Processed: 1000
  - Success Rate: 100.0%
  - Throughput: 39.47 msg/sec
  - Duration: 25.34 sec
================================================================

✓ Summary JSON saved: artifacts\reports\batch_summary_20251011_...json
✓ Predictions CSV saved: artifacts\reports\batch_summary_20251011_...csv
```

📸 **Screenshot 11**: Take screenshot of summary output
- **Filename**: `11_batch_generate_summary.png`
- **Highlight**: Churn statistics and high-risk customer count

### Step 3.4: View Summary Files
```powershell
# List summary reports
Get-ChildItem artifacts\reports\batch_summary_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 3

# View latest summary JSON
Get-Content (Get-ChildItem artifacts\reports\batch_summary_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

📸 **Screenshot 12**: Take screenshot of summary JSON content
- **Filename**: `12_batch_summary_json.png`
- **Highlight**: Statistics and high-risk customers

---

## Part 4: Airflow Web UI (Optional)

### Step 4.1: Start Airflow Web Server
```powershell
# In first terminal: Start web server
airflow webserver --port 8080

# In second terminal: Start scheduler
airflow scheduler
```

### Step 4.2: Access Web UI
1. Open browser to: `http://localhost:8080`
2. Login with your credentials (default: admin/admin)
3. Navigate to DAGs page

📸 **Screenshot 13**: Take screenshot of Airflow home page
- **Filename**: `13_airflow_web_ui_home.png`
- **Highlight**: Kafka DAGs visible

### Step 4.3: View Streaming DAG
1. Click on `kafka_streaming_pipeline`
2. View Graph view
3. View Task details

📸 **Screenshot 14**: Take screenshot of streaming DAG graph
- **Filename**: `14_streaming_dag_graph.png`

### Step 4.4: View Batch DAG
1. Click on `kafka_batch_pipeline`
2. View Graph view
3. View Task details

📸 **Screenshot 15**: Take screenshot of batch DAG graph
- **Filename**: `15_batch_dag_graph.png`

### Step 4.5: Manual DAG Execution
1. Enable `kafka_batch_pipeline` DAG
2. Click "Trigger DAG" button
3. Wait for execution to complete
4. View run logs

📸 **Screenshot 16**: Take screenshot of DAG run results
- **Filename**: `16_batch_dag_execution.png`
- **Highlight**: All tasks green (success)

---

## Part 5: Validation Checklist

### ✅ DAG Validation
- [ ] Both Kafka DAGs appear in `airflow dags list`
- [ ] Streaming DAG has 4 tasks (health_check, start_consumer, monitor_consumer, health_check_bash)
- [ ] Batch DAG has 4 tasks (trigger_producer, run_consumer_window, generate_summary, trigger_producer_bash)
- [ ] DAG owners set correctly (data-engineering-team)

### ✅ Streaming DAG Tests
- [ ] Health check task runs successfully
- [ ] Kafka broker connectivity confirmed
- [ ] Consumer starts and gets PID
- [ ] Consumer process stays running
- [ ] Monitor task detects running consumer
- [ ] Consumer logs show message processing

### ✅ Batch DAG Tests
- [ ] Producer task sends messages successfully
- [ ] Consumer window processes messages
- [ ] Summary task generates reports
- [ ] JSON summary contains statistics
- [ ] CSV files created in artifacts/reports/
- [ ] High-risk customers identified

### ✅ File Outputs
- [ ] Consumer logs in `artifacts/logs/`
- [ ] Producer logs in `artifacts/logs/`
- [ ] Summary JSON in `artifacts/reports/`
- [ ] Predictions CSV in `artifacts/reports/`
- [ ] High-risk CSV in `artifacts/reports/`

---

## Screenshot Summary

| # | Filename | Description | Evidence |
|---|----------|-------------|----------|
| 1 | `01_airflow_dags_list.png` | DAG list showing both Kafka DAGs | DAGs present |
| 2 | `02_streaming_dag_details.png` | Streaming DAG details | DAG structure |
| 3 | `03_batch_dag_details.png` | Batch DAG details | DAG structure |
| 4 | `04_dag_tasks_list.png` | Task lists for both DAGs | All tasks present |
| 5 | `05_streaming_health_check.png` | Health check output | Kafka connectivity |
| 6 | `06_streaming_start_consumer.png` | Consumer start output | Consumer running |
| 7 | `07_streaming_monitor_consumer.png` | Monitor output | Consumer health |
| 8 | `08_streaming_consumer_logs.png` | Consumer log content | Messages processed |
| 9 | `09_batch_trigger_producer.png` | Producer output | Messages sent |
| 10 | `10_batch_consumer_window.png` | Consumer window output | Messages processed |
| 11 | `11_batch_generate_summary.png` | Summary statistics | Churn analysis |
| 12 | `12_batch_summary_json.png` | Summary JSON content | Report generated |
| 13 | `13_airflow_web_ui_home.png` | Airflow UI home | DAGs in UI |
| 14 | `14_streaming_dag_graph.png` | Streaming DAG graph | Task dependencies |
| 15 | `15_batch_dag_graph.png` | Batch DAG graph | Task dependencies |
| 16 | `16_batch_dag_execution.png` | DAG execution results | Successful run |

---

## Troubleshooting

### Issue: Kafka broker not accessible
**Solution:**
```powershell
# Check Docker containers
docker ps

# Restart Kafka
docker compose down
docker compose up -d

# Wait for broker to be ready
Start-Sleep -Seconds 10
```

### Issue: Consumer doesn't start
**Solution:**
```powershell
# Check if consumer is already running
Get-Process | Where-Object {$_.ProcessName -like "*python*"}

# Kill existing consumer processes
Stop-Process -Name python -Force

# Retry test
airflow tasks test kafka_streaming_pipeline start_consumer 2025-10-11
```

### Issue: No predictions in output topic
**Solution:**
```powershell
# Publish some test messages first
python src/streaming/producer.py --broker localhost:9093 --topic telco.raw.customers --input-file data/processed/test.csv --batch-size 10

# Wait for processing
Start-Sleep -Seconds 5

# Retry summary
airflow tasks test kafka_batch_pipeline generate_summary 2025-10-11
```

### Issue: Airflow DAGs not found
**Solution:**
```powershell
# Verify AIRFLOW_HOME is set
echo $env:AIRFLOW_HOME

# Check DAG files exist
Get-ChildItem airflow_home\dags\kafka_*.py

# Refresh DAGs
airflow dags list-import-errors
```

---

## Additional Commands

### View All Logs
```powershell
# List all Kafka-related logs
Get-ChildItem artifacts\logs\kafka_*.log

# View specific log
Get-Content artifacts\logs\kafka_consumer_streaming_20251011_142530.log
```

### Clean Up
```powershell
# Stop consumer processes
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Where-Object {$_.CommandLine -like "*consumer.py*"} | Stop-Process -Force

# Clean old logs (older than 7 days)
Get-ChildItem artifacts\logs\*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item

# Clean old reports
Get-ChildItem artifacts\reports\batch_summary_*.* | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item
```

### Export Evidence Package
```powershell
# Create evidence directory
New-Item -ItemType Directory -Force -Path evidence\step9_airflow_kafka

# Copy screenshot files
Copy-Item *.png evidence\step9_airflow_kafka\

# Copy latest logs
Copy-Item artifacts\logs\kafka_*.log evidence\step9_airflow_kafka\

# Copy latest summary
Copy-Item (Get-ChildItem artifacts\reports\batch_summary_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName evidence\step9_airflow_kafka\

# Create archive
Compress-Archive -Path evidence\step9_airflow_kafka\* -DestinationPath evidence\step9_airflow_kafka_evidence.zip
```

---

## Success Criteria

✅ **All tasks must pass:**
1. DAGs visible in `airflow dags list`
2. All tasks run successfully with `airflow tasks test`
3. Consumer processes messages (visible in logs)
4. Summary reports generated with statistics
5. Screenshot evidence collected for all steps

✅ **Evidence package contains:**
- 16 screenshots showing all test steps
- Sample consumer logs
- Sample summary JSON
- README with screenshot descriptions

---

## Notes

- Replace `2025-10-11` with current date when testing
- Adjust paths if your project is in a different location
- Some tasks may require Kafka to have existing data - run producer first if needed
- Consumer logs continue to grow - check regularly for disk space
- Use `Ctrl+C` to stop long-running processes

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-11  
**Author**: AI Assistant  
**Purpose**: Step 9 validation evidence collection
