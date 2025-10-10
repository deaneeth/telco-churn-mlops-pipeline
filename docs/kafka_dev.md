# 🌊 Kafka Development Environment Guide

**Project:** Telco Customer Churn Prediction - Kafka Integration  
**Last Updated:** October 10, 2025

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Topic Management](#topic-management)
- [Testing & Debugging](#testing--debugging)
- [Troubleshooting](#troubleshooting)
- [Port Reference](#port-reference)

---

## 🎯 Overview

This project uses **Redpanda** as the Kafka-compatible streaming platform for local development. Redpanda was chosen over Confluent Kafka for its simplicity (no Zookeeper required) while maintaining full Kafka API compatibility.

### Key Components

- **Redpanda**: Kafka-compatible message broker (single-node, dev mode)
- **Redpanda Console**: Web UI for monitoring topics, messages, and cluster health
- **Three Topics**:
  - `telco.raw.customers` - Input stream of customer data
  - `telco.churn.predictions` - Output predictions with churn probabilities
  - `telco.deadletter` - Failed/invalid messages (error handling)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kafka Streaming Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Producer                Redpanda              Consumer       │
│  ┌──────────┐           ┌──────────┐          ┌──────────┐  │
│  │          │           │          │          │          │  │
│  │ Dataset  │──────────>│  Topic:  │─────────>│  Model   │  │
│  │ Sampler  │   Publish │   raw    │ Consume  │ Inference│  │
│  │          │           │customers │          │          │  │
│  └──────────┘           └──────────┘          └──────────┘  │
│                               │                     │         │
│                               │                     │         │
│                               v                     v         │
│                         ┌──────────┐          ┌──────────┐  │
│                         │  Topic:  │<─────────│  Publish │  │
│                         │deadletter│          │ Results  │  │
│                         └──────────┘          └──────────┘  │
│                                                     │         │
│                                                     v         │
│                                               ┌──────────┐  │
│                                               │  Topic:  │  │
│                                               │predictions│ │
│                                               └──────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Start Kafka Infrastructure

**Option A: Using Docker Compose (Recommended)**
```bash
# Start Redpanda + Console
docker compose -f docker-compose.kafka.yml up -d

# Verify services are running
docker compose -f docker-compose.kafka.yml ps
```

**Option B: Using Makefile**
```bash
make kafka-up      # Start Kafka
make kafka-status  # Check status
```

### 2. Create Topics

**Bash (Linux/Mac/WSL):**
```bash
bash scripts/kafka_create_topics.sh
```

**PowerShell (Windows):**
```powershell
.\scripts\kafka_create_topics.ps1
```

**Makefile:**
```bash
make kafka-topics
```

### 3. Access Web UI

Open browser: **http://localhost:8080** (Redpanda Console)

### 4. Stop Kafka

```bash
docker compose -f docker-compose.kafka.yml down

# Or with Makefile
make kafka-down
```

---

## 🔧 Detailed Setup

### Prerequisites

- Docker Desktop or Docker Engine (20.10+)
- Docker Compose (v2.0+)
- Git Bash (Windows) or Terminal (Mac/Linux)

### Step-by-Step Installation

#### 1. Verify Docker is Running

```bash
docker --version
docker compose version
```

Expected output:
```
Docker version 24.0.x
Docker Compose version v2.x.x
```

#### 2. Start Redpanda

```bash
# Navigate to project root
cd /path/to/telco-churn-prediction-mini-project-1

# Start services
docker compose -f docker-compose.kafka.yml up -d
```

Expected output:
```
[+] Running 3/3
 ✔ Network telco-kafka-network     Created
 ✔ Container telco-redpanda         Started
 ✔ Container telco-kafka-console    Started
```

#### 3. Wait for Services to be Healthy

```bash
# Check health status
docker compose -f docker-compose.kafka.yml ps
```

Wait until both services show `healthy` status (may take 30-60 seconds).

#### 4. Create Topics

**Using Bash:**
```bash
bash scripts/kafka_create_topics.sh
```

**Using PowerShell:**
```powershell
.\scripts\kafka_create_topics.ps1
```

Expected output:
```
========================================
Telco Churn - Kafka Topic Setup
========================================

✓ Redpanda container is running
✓ Redpanda is healthy

Creating Kafka topics...

Creating topic: telco.raw.customers
  Partitions: 3
  Replication Factor: 1
  ✓ Topic created successfully

Creating topic: telco.churn.predictions
  Partitions: 3
  Replication Factor: 1
  ✓ Topic created successfully

Creating topic: telco.deadletter
  Partitions: 1
  Replication Factor: 1
  ✓ Topic created successfully
```

#### 5. Verify Setup

**Access Redpanda Console:**
- URL: http://localhost:8080
- Navigate to "Topics" to see all 3 topics

**Using CLI:**
```bash
# List all topics
docker exec telco-redpanda rpk topic list

# Describe a topic
docker exec telco-redpanda rpk topic describe telco.raw.customers
```

---

## 📊 Topic Management

### List Topics

```bash
# Using rpk (Redpanda CLI)
docker exec telco-redpanda rpk topic list

# Using kafka-console tools (Kafka-compatible)
docker exec telco-redpanda kafka-topics --list --bootstrap-server localhost:9092
```

### Describe Topic Details

```bash
# Show partitions, replicas, configuration
docker exec telco-redpanda rpk topic describe telco.raw.customers
```

### Delete a Topic (Careful!)

```bash
docker exec telco-redpanda rpk topic delete telco.raw.customers
```

### Create Additional Topics

```bash
docker exec telco-redpanda rpk topic create my-new-topic \
  --partitions 3 \
  --replicas 1
```

---

## 🧪 Testing & Debugging

### Console Producer (Send Test Messages)

**Send messages to `telco.raw.customers`:**

```bash
# Interactive mode - type JSON messages
docker exec -it telco-redpanda rpk topic produce telco.raw.customers

# Then paste JSON messages (one per line):
{"customerID":"TEST-001","gender":"Male","SeniorCitizen":0,"tenure":12,"MonthlyCharges":50.0,"event_ts":"2025-10-10T10:00:00Z"}
{"customerID":"TEST-002","gender":"Female","SeniorCitizen":1,"tenure":24,"MonthlyCharges":80.0,"event_ts":"2025-10-10T10:01:00Z"}

# Press Ctrl+C to exit
```

**Send from file:**
```bash
cat test_messages.json | docker exec -i telco-redpanda rpk topic produce telco.raw.customers
```

### Console Consumer (Read Messages)

**Read all messages from beginning:**

```bash
docker exec -it telco-redpanda rpk topic consume telco.raw.customers --from-beginning
```

**Read only new messages:**

```bash
docker exec -it telco-redpanda rpk topic consume telco.raw.customers
```

**Read last N messages:**

```bash
docker exec -it telco-redpanda rpk topic consume telco.raw.customers --num 10
```

**Filter by partition:**

```bash
docker exec -it telco-redpanda rpk topic consume telco.raw.customers --partition 0
```

### Kafka-Compatible Console Consumer

```bash
# Using standard Kafka tools (also available in Redpanda)
docker exec -it telco-redpanda kafka-console-consumer \
  --topic telco.raw.customers \
  --from-beginning \
  --bootstrap-server localhost:9092
```

### Check Consumer Groups

```bash
# List all consumer groups
docker exec telco-redpanda rpk group list

# Describe a consumer group
docker exec telco-redpanda rpk group describe churn-prediction-group

# Check consumer lag
docker exec telco-redpanda rpk group describe churn-prediction-group -v
```

### View Cluster Health

```bash
# Cluster health status
docker exec telco-redpanda rpk cluster health

# Cluster info
docker exec telco-redpanda rpk cluster info
```

---

## 🐛 Troubleshooting

### Issue: Container Not Starting

**Check logs:**
```bash
docker compose -f docker-compose.kafka.yml logs redpanda
docker compose -f docker-compose.kafka.yml logs console
```

**Common causes:**
- Port conflict (19092, 8080, 18081 already in use)
- Insufficient Docker resources (increase memory to 4GB+)
- Corrupted volume data

**Solution:**
```bash
# Stop and remove everything
docker compose -f docker-compose.kafka.yml down -v

# Restart
docker compose -f docker-compose.kafka.yml up -d
```

### Issue: Topics Not Creating

**Check Redpanda health:**
```bash
docker exec telco-redpanda rpk cluster health
```

**Wait for startup:**
```bash
# Redpanda may take 30-60 seconds to become fully ready
# Re-run topic creation script after waiting
bash scripts/kafka_create_topics.sh
```

### Issue: Cannot Connect from Producer/Consumer

**Verify bootstrap server address:**
- Inside Docker network: `redpanda:9092`
- From host machine: `localhost:19092`

**Test connectivity:**
```bash
# From host
telnet localhost 19092

# Should connect without error
```

### Issue: Console UI Not Loading

**Check console container:**
```bash
docker compose -f docker-compose.kafka.yml logs console
```

**Restart console:**
```bash
docker compose -f docker-compose.kafka.yml restart console
```

### Issue: Permission Denied (Scripts)

**Make scripts executable:**
```bash
chmod +x scripts/kafka_create_topics.sh
```

### Clean Slate Reset

```bash
# Stop everything
docker compose -f docker-compose.kafka.yml down -v

# Remove volumes
docker volume rm telco-redpanda-data

# Remove network
docker network rm telco-kafka-network

# Restart fresh
docker compose -f docker-compose.kafka.yml up -d
bash scripts/kafka_create_topics.sh
```

---

## 🔌 Port Reference

| Service | Port | Purpose | Access |
|---------|------|---------|--------|
| **Redpanda Kafka API** | 19092 | Kafka protocol (external) | `localhost:19092` |
| **Redpanda Kafka API (internal)** | 9092 | Kafka protocol (Docker network) | `redpanda:9092` |
| **Redpanda Schema Registry** | 18081 | Schema management | `localhost:18081` |
| **Redpanda Pandaproxy** | 18082 | HTTP Proxy for Kafka | `localhost:18082` |
| **Redpanda Admin API** | 9644 | Cluster administration | `localhost:9644` |
| **Redpanda Console** | 8080 | Web UI | `http://localhost:8080` |

### Connection Strings

**From Python/Producer/Consumer (host machine):**
```python
bootstrap_servers = "localhost:19092"
```

**From Docker containers in same network:**
```python
bootstrap_servers = "redpanda:9092"
```

---

## 📚 Additional Resources

### Redpanda Documentation
- [Redpanda Quickstart](https://docs.redpanda.com/current/get-started/quick-start/)
- [rpk CLI Reference](https://docs.redpanda.com/current/reference/rpk/)
- [Redpanda Console](https://docs.redpanda.com/current/manage/console/)

### Kafka Compatibility
- [Kafka Protocol](https://kafka.apache.org/protocol)
- [kafka-python Library](https://kafka-python.readthedocs.io/)

### Project-Specific Guides
- [Producer Implementation Guide](../src/streaming/README.md) (TODO)
- [Consumer Implementation Guide](../src/streaming/README.md) (TODO)
- [Testing Guide](../tests/README_KAFKA.md) (TODO)

---

## 🔄 Daily Development Workflow

### Morning Startup
```bash
# 1. Start Kafka
docker compose -f docker-compose.kafka.yml up -d

# 2. Verify health
docker compose -f docker-compose.kafka.yml ps

# 3. Check topics (should already exist)
docker exec telco-redpanda rpk topic list
```

### Development
```bash
# 1. Open Redpanda Console in browser
open http://localhost:8080

# 2. Run producer in one terminal
python src/streaming/producer.py --mode streaming

# 3. Run consumer in another terminal
python src/streaming/consumer.py --mode streaming

# 4. Monitor messages in Console UI
```

### Evening Shutdown
```bash
# Stop Kafka (keeps volumes)
docker compose -f docker-compose.kafka.yml down

# Or stop but keep running (uses resources)
docker compose -f docker-compose.kafka.yml stop
```

---

## ✅ Validation Checklist

Before proceeding to producer/consumer development, verify:

- [ ] Docker Compose starts without errors
- [ ] Both containers show `healthy` status
- [ ] Redpanda Console accessible at http://localhost:8080
- [ ] All 3 topics created successfully:
  - [ ] `telco.raw.customers` (3 partitions)
  - [ ] `telco.churn.predictions` (3 partitions)
  - [ ] `telco.deadletter` (1 partition)
- [ ] Console producer can send test messages
- [ ] Console consumer can read test messages
- [ ] No port conflicts or errors in logs

**Test Command:**
```bash
# Send test message
echo '{"customerID":"TEST","tenure":10}' | docker exec -i telco-redpanda rpk topic produce telco.raw.customers

# Read it back
docker exec telco-redpanda rpk topic consume telco.raw.customers --num 1
```

If all checks pass, you're ready for **Step 3: Producer Implementation**! 🚀

---

**Last Updated:** October 10, 2025  
**Maintained By:** Telco Churn MLOps Team
