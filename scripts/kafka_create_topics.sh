#!/bin/bash

# Kafka Topic Creation Script for Telco Churn Prediction
# Creates all required topics for streaming and batch processing
# Compatible with Redpanda (Kafka-compatible)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Telco Churn - Kafka Topic Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
BOOTSTRAP_SERVER="${KAFKA_BOOTSTRAP_SERVER:-localhost:19092}"
CONTAINER_NAME="${KAFKA_CONTAINER:-telco-redpanda}"

# Topic configurations
declare -A TOPICS=(
    ["telco.raw.customers"]="3:1"           # 3 partitions, replication factor 1
    ["telco.churn.predictions"]="3:1"       # 3 partitions, replication factor 1
    ["telco.deadletter"]="1:1"              # 1 partition, replication factor 1
)

# Check if Redpanda container is running
echo -e "${YELLOW}Checking Redpanda container...${NC}"
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo -e "${RED}Error: Redpanda container '$CONTAINER_NAME' is not running!${NC}"
    echo -e "${YELLOW}Please start it with: docker compose -f docker-compose.kafka.yml up -d${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Redpanda container is running${NC}"
echo ""

# Wait for Kafka to be ready
echo -e "${YELLOW}Waiting for Redpanda to be ready...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec $CONTAINER_NAME rpk cluster health &> /dev/null; then
        echo -e "${GREEN}✓ Redpanda is healthy${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -e "  Attempt $RETRY_COUNT/$MAX_RETRIES..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}Error: Redpanda did not become healthy in time${NC}"
    exit 1
fi
echo ""

# Create topics
echo -e "${BLUE}Creating Kafka topics...${NC}"
echo ""

for TOPIC in "${!TOPICS[@]}"; do
    IFS=':' read -r PARTITIONS REPLICATION <<< "${TOPICS[$TOPIC]}"
    
    echo -e "${YELLOW}Creating topic: $TOPIC${NC}"
    echo -e "  Partitions: $PARTITIONS"
    echo -e "  Replication Factor: $REPLICATION"
    
    # Check if topic already exists
    if docker exec $CONTAINER_NAME rpk topic list | grep -q "^$TOPIC"; then
        echo -e "${YELLOW}  ⚠ Topic already exists, skipping...${NC}"
    else
        # Create topic using rpk (Redpanda CLI)
        docker exec $CONTAINER_NAME rpk topic create "$TOPIC" \
            --partitions "$PARTITIONS" \
            --replicas "$REPLICATION" \
            --topic-config retention.ms=604800000 \
            --topic-config cleanup.policy=delete
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✓ Topic created successfully${NC}"
        else
            echo -e "${RED}  ✗ Failed to create topic${NC}"
        fi
    fi
    echo ""
done

# List all topics
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Current Topics:${NC}"
echo -e "${BLUE}========================================${NC}"
docker exec $CONTAINER_NAME rpk topic list
echo ""

# Show topic details
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Topic Details:${NC}"
echo -e "${BLUE}========================================${NC}"
for TOPIC in "${!TOPICS[@]}"; do
    echo -e "${YELLOW}Topic: $TOPIC${NC}"
    docker exec $CONTAINER_NAME rpk topic describe "$TOPIC" | head -20
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Kafka topics setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Access Redpanda Console: ${YELLOW}http://localhost:8080${NC}"
echo -e "  2. Test producer: ${YELLOW}bash scripts/kafka_test_producer.sh${NC}"
echo -e "  3. Test consumer: ${YELLOW}bash scripts/kafka_test_consumer.sh${NC}"
echo ""
