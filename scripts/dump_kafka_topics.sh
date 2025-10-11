#!/bin/bash
################################################################################
# Kafka Topic Dump Script
# Purpose: Extract sample messages from Kafka topics for evidence
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

REPORTS_DIR="$PROJECT_ROOT/reports"
mkdir -p "$REPORTS_DIR"

# Output files
RAW_SAMPLE="$REPORTS_DIR/kafka_raw_sample.json"
PREDICTIONS_SAMPLE="$REPORTS_DIR/kafka_predictions_sample.json"

# Kafka topics
INPUT_TOPIC="telco.raw.customers"
OUTPUT_TOPIC="telco.churn.predictions"

# Number of messages to dump
SAMPLE_SIZE=10

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Kafka Topic Dump - Step 10${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""

# Check if Kafka is running
if ! docker ps | grep -q kafka; then
    echo -e "\033[0;31mERROR: Kafka is not running!\033[0m"
    exit 1
fi

echo -e "${YELLOW}Dumping messages from ${INPUT_TOPIC}...${NC}"

# Dump raw customer messages
docker exec kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic "$INPUT_TOPIC" \
    --from-beginning \
    --max-messages "$SAMPLE_SIZE" \
    --timeout-ms 5000 2>/dev/null > "$RAW_SAMPLE.tmp" || echo "" > "$RAW_SAMPLE.tmp"

# Convert to JSON array
python3 -c "
import json
import sys

messages = []
with open('$RAW_SAMPLE.tmp', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                messages.append(json.loads(line))
            except:
                pass

with open('$RAW_SAMPLE', 'w') as f:
    json.dump(messages, f, indent=2)
" 2>/dev/null || echo "[]" > "$RAW_SAMPLE"

RAW_COUNT=$(python3 -c "import json; print(len(json.load(open('$RAW_SAMPLE'))))" 2>/dev/null || echo "0")
echo -e "${GREEN}✓ Dumped $RAW_COUNT messages to: $RAW_SAMPLE${NC}"

echo ""
echo -e "${YELLOW}Dumping messages from ${OUTPUT_TOPIC}...${NC}"

# Dump prediction messages
docker exec kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic "$OUTPUT_TOPIC" \
    --from-beginning \
    --max-messages "$SAMPLE_SIZE" \
    --timeout-ms 5000 2>/dev/null > "$PREDICTIONS_SAMPLE.tmp" || echo "" > "$PREDICTIONS_SAMPLE.tmp"

# Convert to JSON array
python3 -c "
import json
import sys

messages = []
with open('$PREDICTIONS_SAMPLE.tmp', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                messages.append(json.loads(line))
            except:
                pass

with open('$PREDICTIONS_SAMPLE', 'w') as f:
    json.dump(messages, f, indent=2)
" 2>/dev/null || echo "[]" > "$PREDICTIONS_SAMPLE"

PRED_COUNT=$(python3 -c "import json; print(len(json.load(open('$PREDICTIONS_SAMPLE'))))" 2>/dev/null || echo "0")
echo -e "${GREEN}✓ Dumped $PRED_COUNT messages to: $PREDICTIONS_SAMPLE${NC}"

echo ""
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Topic Dump Complete!${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""

# Validation
echo "Validation:"
if [ "$PRED_COUNT" -ge 5 ]; then
    echo -e "${GREEN}✓ PASS: Found $PRED_COUNT predictions (>= 5 required)${NC}"
    
    # Show sample prediction
    echo ""
    echo "Sample prediction:"
    python3 -c "
import json
data = json.load(open('$PREDICTIONS_SAMPLE'))
if data:
    p = data[0]
    print(json.dumps({k: p.get(k) for k in ['customerID', 'churn_probability', 'prediction', 'processed_ts'] if k in p}, indent=2))
" 2>/dev/null || echo "Unable to parse prediction"
else
    echo -e "\033[0;31m✗ FAIL: Only found $PRED_COUNT predictions (5 required)${NC}"
    echo "Try running the demo again or increase duration."
fi

# Cleanup temp files
rm -f "$RAW_SAMPLE.tmp" "$PREDICTIONS_SAMPLE.tmp"

echo ""
echo "Files created:"
echo "  - $RAW_SAMPLE"
echo "  - $PREDICTIONS_SAMPLE"
