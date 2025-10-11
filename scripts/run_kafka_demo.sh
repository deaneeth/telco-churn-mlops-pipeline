#!/bin/bash
################################################################################
# Kafka Streaming Demo Script
# Purpose: Demonstrate end-to-end Kafka streaming with producer and consumer
# Duration: 60 seconds
################################################################################

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Directories
LOGS_DIR="$PROJECT_ROOT/logs"
REPORTS_DIR="$PROJECT_ROOT/reports"
SRC_DIR="$PROJECT_ROOT/src/streaming"

# Kafka configuration
KAFKA_BROKER="localhost:9093"
INPUT_TOPIC="telco.raw.customers"
OUTPUT_TOPIC="telco.churn.predictions"
DLQ_TOPIC="telco.churn.deadletter"

# Demo parameters
DEMO_DURATION=60  # seconds
EVENTS_PER_SEC=2  # streaming rate

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Kafka Streaming Demo - Step 10${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""

# Activate virtual environment if exists
if [ -d "airflow_venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source airflow_venv/bin/activate
fi

# Check if Kafka is running
echo -e "${YELLOW}Checking Kafka availability...${NC}"
if ! docker ps | grep -q kafka; then
    echo -e "${RED}ERROR: Kafka is not running!${NC}"
    echo "Please start Kafka with: docker-compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ Kafka is running${NC}"

# Create necessary directories
mkdir -p "$LOGS_DIR"
mkdir -p "$REPORTS_DIR"

# Log files
PRODUCER_LOG="$LOGS_DIR/kafka_producer_demo.log"
CONSUMER_LOG="$LOGS_DIR/kafka_consumer_demo.log"

echo ""
echo -e "${YELLOW}Starting Kafka Consumer (background)...${NC}"
echo "Log file: $CONSUMER_LOG"

# Start consumer in background
python3 "$SRC_DIR/consumer.py" \
    --mode streaming \
    --broker "$KAFKA_BROKER" \
    --input-topic "$INPUT_TOPIC" \
    --output-topic "$OUTPUT_TOPIC" \
    --deadletter-topic "$DLQ_TOPIC" \
    --consumer-group "demo-consumer-group" \
    > "$CONSUMER_LOG" 2>&1 &

CONSUMER_PID=$!
echo -e "${GREEN}✓ Consumer started (PID: $CONSUMER_PID)${NC}"

# Wait for consumer to initialize
sleep 5

echo ""
echo -e "${YELLOW}Starting Kafka Producer (streaming mode)...${NC}"
echo "Log file: $PRODUCER_LOG"
echo "Duration: ${DEMO_DURATION} seconds"
echo "Rate: ${EVENTS_PER_SEC} events/second"

# Start producer in foreground with timeout
timeout ${DEMO_DURATION}s python3 "$SRC_DIR/producer.py" \
    --mode streaming \
    --broker "$KAFKA_BROKER" \
    --topic "$INPUT_TOPIC" \
    --dataset-path "data/raw/Telco-Customer-Churn.csv" \
    --events-per-sec "$EVENTS_PER_SEC" \
    > "$PRODUCER_LOG" 2>&1 || true

echo -e "${GREEN}✓ Producer completed${NC}"

# Give consumer a few more seconds to process remaining messages
echo ""
echo -e "${YELLOW}Allowing consumer to process remaining messages...${NC}"
sleep 5

# Stop consumer gracefully
echo -e "${YELLOW}Stopping consumer...${NC}"
kill -TERM $CONSUMER_PID 2>/dev/null || true
sleep 2

# Force kill if still running
if kill -0 $CONSUMER_PID 2>/dev/null; then
    kill -9 $CONSUMER_PID 2>/dev/null || true
fi

echo -e "${GREEN}✓ Consumer stopped${NC}"

echo ""
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Demo Complete!${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""
echo "Logs saved to:"
echo "  - $PRODUCER_LOG"
echo "  - $CONSUMER_LOG"
echo ""
echo "Next steps:"
echo "  1. Run: bash scripts/dump_kafka_topics.sh"
echo "  2. Capture screenshots of Kafka UI"
echo "  3. Review generated artifacts"
