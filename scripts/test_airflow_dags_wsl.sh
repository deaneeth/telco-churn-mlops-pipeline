#!/bin/bash
# Airflow DAG Testing Script for WSL
# This script executes all tests and generates screenshot instructions

set -e

PROJECT_ROOT="/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1"
AIRFLOW_HOME="$PROJECT_ROOT/airflow_home"
SCREENSHOTS_DIR="$PROJECT_ROOT/artifacts/screenshots"
TEST_DATE=$(date +%Y-%m-%d)

# Set environment
export AIRFLOW_HOME="$AIRFLOW_HOME"

# Create screenshots directory
mkdir -p "$SCREENSHOTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

function print_header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

function print_section() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo ""
}

function print_screenshot() {
    local number=$1
    local title=$2
    local description=$3
    local filename=$4
    
    echo ""
    echo -e "${MAGENTA}📸 SCREENSHOT $number - $title${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$description${NC}"
    echo -e "Save as: ${GREEN}$SCREENSHOTS_DIR/$filename${NC}"
    echo ""
    read -p "Press Enter when screenshot is captured..."
}

function execute_command() {
    local description=$1
    local command=$2
    
    echo ""
    echo -e "${GREEN}⚡ Executing: $description${NC}"
    echo -e "${NC}Command: $command${NC}"
    echo -e "${NC}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    eval "$command"
    
    echo -e "${NC}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_header "Airflow DAG Testing & Screenshot Guide"

echo "This script will guide you through testing both Airflow DAGs"
echo "and collecting screenshot evidence for documentation."
echo ""
echo "Prerequisites:"
echo "  ✓ Airflow is configured in WSL"
echo "  ✓ Kafka is running (docker-compose up -d from Windows)"
echo "  ✓ AIRFLOW_HOME is set"
echo ""
read -p "Press Enter to start testing..."

# ============================================================================
# PART 1: DAG VALIDATION
# ============================================================================

print_section "PART 1: DAG VALIDATION (3 steps)"

# Screenshot 01
execute_command "List all DAGs" "airflow dags list"
print_screenshot "01" "DAG List" \
    "Capture terminal showing both Kafka DAGs in the list (kafka_streaming_pipeline, kafka_batch_pipeline)" \
    "01_dag_list.png"

# Screenshot 02
execute_command "Show streaming DAG structure" "airflow dags show kafka_streaming_pipeline"
print_screenshot "02" "Streaming DAG Structure" \
    "Capture the DAG structure output showing task dependencies" \
    "02_streaming_dag_structure.png"

# Screenshot 03
execute_command "List batch DAG tasks" "airflow tasks list kafka_batch_pipeline"
print_screenshot "03" "Batch DAG Tasks" \
    "Capture terminal showing all 4 tasks for batch DAG" \
    "03_batch_dag_tasks.png"

# ============================================================================
# PART 2: STREAMING DAG TESTING
# ============================================================================

print_section "PART 2: STREAMING DAG (4 steps)"

echo -e "${YELLOW}Testing Streaming DAG tasks...${NC}"
echo ""

# Screenshot 04
execute_command "Test health check task" \
    "airflow tasks test kafka_streaming_pipeline health_check_kafka $TEST_DATE"
print_screenshot "04" "Health Check Execution" \
    "Capture terminal showing successful Kafka health check" \
    "04_health_check_test.png"

# Screenshot 05
echo ""
echo -e "${YELLOW}⚠️  WARNING: The next task starts a long-running consumer!${NC}"
echo "    It will run in the background. You can stop it later."
read -p "Press Enter to continue..."

execute_command "Start consumer process" \
    "airflow tasks test kafka_streaming_pipeline start_consumer $TEST_DATE"
print_screenshot "05" "Consumer Start" \
    "Capture terminal showing consumer started successfully with PID" \
    "05_consumer_start.png"

# Screenshot 06
execute_command "Monitor consumer health" \
    "airflow tasks test kafka_streaming_pipeline monitor_consumer $TEST_DATE"
print_screenshot "06" "Consumer Monitoring" \
    "Capture terminal showing consumer health status" \
    "06_consumer_monitor.png"

# Screenshot 07
LOG_FILE=$(ls -t $PROJECT_ROOT/artifacts/logs/kafka_consumer_streaming_*.log 2>/dev/null | head -1)
if [ -f "$LOG_FILE" ]; then
    execute_command "View consumer logs" "tail -30 $LOG_FILE"
else
    echo -e "${YELLOW}No consumer log file found${NC}"
fi
print_screenshot "07" "Consumer Logs" \
    "Capture log file content showing consumer activity" \
    "07_consumer_logs.png"

# ============================================================================
# PART 3: BATCH DAG TESTING
# ============================================================================

print_section "PART 3: BATCH DAG (4 steps)"

echo -e "${YELLOW}Testing Batch DAG tasks...${NC}"
echo ""

# Screenshot 08
execute_command "Trigger producer to send batch" \
    "airflow tasks test kafka_batch_pipeline trigger_producer $TEST_DATE"
print_screenshot "08" "Producer Execution" \
    "Capture terminal showing batch messages sent to Kafka" \
    "08_producer_execution.png"

# Screenshot 09
execute_command "Run consumer for time window" \
    "airflow tasks test kafka_batch_pipeline run_consumer_window $TEST_DATE"
print_screenshot "09" "Consumer Window" \
    "Capture terminal showing consumer processing messages for 60 seconds" \
    "09_consumer_window.png"

# Screenshot 10
execute_command "Generate summary report" \
    "airflow tasks test kafka_batch_pipeline generate_summary $TEST_DATE"
print_screenshot "10" "Summary Generation" \
    "Capture terminal showing summary statistics and report files created" \
    "10_summary_generation.png"

# Screenshot 11
SUMMARY_FILE=$(ls -t $PROJECT_ROOT/artifacts/reports/batch_summary_*.json 2>/dev/null | head -1)
if [ -f "$SUMMARY_FILE" ]; then
    execute_command "View summary report" "cat $SUMMARY_FILE | python -m json.tool"
else
    echo -e "${YELLOW}No summary report found${NC}"
fi
print_screenshot "11" "Summary Report Content" \
    "Capture JSON summary showing churn statistics and high-risk customers" \
    "11_summary_report.png"

# ============================================================================
# PART 4: WEB UI SCREENSHOTS
# ============================================================================

print_section "PART 4: WEB UI (5 steps)"

echo ""
echo -e "${YELLOW}Web UI Screenshots - Manual Instructions:${NC}"
echo ""
echo "Please open http://localhost:8080 in your browser"
echo "Login with: admin / admin"
echo ""
read -p "Press Enter when ready to continue with Web UI screenshots..."

print_screenshot "12" "DAGs Homepage" \
    "Capture Airflow homepage showing both Kafka DAGs in the list" \
    "12_web_ui_dags.png"

print_screenshot "13" "Streaming DAG Graph View" \
    "Click on kafka_streaming_pipeline, then Graph tab. Capture the task graph." \
    "13_streaming_dag_graph.png"

print_screenshot "14" "Batch DAG Graph View" \
    "Click on kafka_batch_pipeline, then Graph tab. Capture the task graph." \
    "14_batch_dag_graph.png"

print_screenshot "15" "DAG Trigger" \
    "Click 'Trigger DAG' button for batch DAG. Capture the run starting." \
    "15_dag_trigger.png"

print_screenshot "16" "DAG Run Details" \
    "Click on a running/completed DAG run. Capture the task status view." \
    "16_dag_run_details.png"

# ============================================================================
# COMPLETION
# ============================================================================

print_header "TESTING COMPLETE!"

echo ""
echo -e "${YELLOW}Screenshots saved to:${NC}"
echo -e "${GREEN}  $SCREENSHOTS_DIR${NC}"
echo ""
echo -e "${CYAN}Screenshot Summary:${NC}"
echo "  01-03: DAG Validation (3 screenshots)"
echo "  04-07: Streaming DAG Testing (4 screenshots)"
echo "  08-11: Batch DAG Testing (4 screenshots)"
echo "  12-16: Web UI Screenshots (5 screenshots)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review all screenshots in $SCREENSHOTS_DIR"
echo "  2. Create screenshot archive:"
echo "     cd $PROJECT_ROOT"
echo "     zip -r airflow_dag_screenshots.zip artifacts/screenshots/"
echo "  3. Update Step 9 completion report with evidence"
echo ""

# List captured screenshots
echo -e "${CYAN}Captured Screenshots:${NC}"
ls -1 "$SCREENSHOTS_DIR"/*.png 2>/dev/null | while read file; do
    echo -e "  ${GREEN}✓ $(basename $file)${NC}"
done || echo "  (Screenshots will appear as you capture them)"

echo ""
echo -e "${GREEN}Testing session complete! 🎉${NC}"
echo ""
echo "To create evidence package:"
echo "  cd $PROJECT_ROOT"
echo "  zip -r step9_evidence.zip artifacts/screenshots/ artifacts/reports/ artifacts/logs/kafka_*"
