#!/bin/bash
#
# Kafka Integration Test Runner
# 
# This script orchestrates the complete integration test workflow:
# 1. Start Kafka test environment (Docker Compose)
# 2. Wait for broker to be ready
# 3. Run integration tests
# 4. Collect logs and generate reports
# 5. Tear down test environment
#
# Usage:
#   bash scripts/run_kafka_integration_tests.sh [--keep-containers] [--verbose]
#
# Options:
#   --keep-containers   Don't tear down containers after tests (for debugging)
#   --verbose          Show detailed output from Docker and tests
#
# Author: AI Assistant
# Created: 2025-01-11

set -e  # Exit on error

# ==================== CONFIGURATION ====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.test.yml"
REPORT_DIR="$PROJECT_ROOT/reports"
LOG_DIR="$PROJECT_ROOT/logs"
TEST_LOG="$REPORT_DIR/kafka_integration.log"

BROKER_HOST="localhost:19093"
BROKER_READY_TIMEOUT=60
TEST_TIMEOUT=300

KEEP_CONTAINERS=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== HELPER FUNCTIONS ====================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        log_error "Python is not installed. Please install Python 3.8+."
        exit 1
    fi
    
    # Determine Python command
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    # Check pytest
    if ! $PYTHON_CMD -m pytest --version &> /dev/null; then
        log_error "pytest is not installed. Run: pip install pytest"
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

cleanup_containers() {
    if [ "$KEEP_CONTAINERS" = true ]; then
        log_warning "Keeping containers running (--keep-containers flag set)"
        log_info "To stop manually: docker-compose -f $COMPOSE_FILE down"
        return
    fi
    
    log_info "Tearing down test environment..."
    
    if [ "$VERBOSE" = true ]; then
        docker-compose -f "$COMPOSE_FILE" down -v
    else
        docker-compose -f "$COMPOSE_FILE" down -v > /dev/null 2>&1
    fi
    
    log_success "Test environment cleaned up"
}

wait_for_broker() {
    log_info "Waiting for Kafka broker to be ready (timeout: ${BROKER_READY_TIMEOUT}s)..."
    
    local elapsed=0
    local retry_interval=2
    
    while [ $elapsed -lt $BROKER_READY_TIMEOUT ]; do
        # Try to list topics using rpk (Redpanda CLI) inside container
        if docker exec telco-redpanda-test rpk cluster health > /dev/null 2>&1; then
            log_success "Kafka broker is ready"
            return 0
        fi
        
        sleep $retry_interval
        elapsed=$((elapsed + retry_interval))
        
        if [ $((elapsed % 10)) -eq 0 ]; then
            log_info "Still waiting... ($elapsed/${BROKER_READY_TIMEOUT}s)"
        fi
    done
    
    log_error "Kafka broker failed to become ready within ${BROKER_READY_TIMEOUT}s"
    return 1
}

# ==================== MAIN WORKFLOW ====================

main() {
    log_info "========================================="
    log_info "Kafka Integration Test Runner"
    log_info "========================================="
    
    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --keep-containers)
                KEEP_CONTAINERS=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [--keep-containers] [--verbose]"
                exit 1
                ;;
        esac
    done
    
    # Setup trap for cleanup
    trap cleanup_containers EXIT
    
    # Create directories
    mkdir -p "$REPORT_DIR"
    mkdir -p "$LOG_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Navigate to project root
    cd "$PROJECT_ROOT"
    
    # STEP 1: Start Docker Compose
    log_info "STEP 1: Starting Kafka test environment..."
    
    if [ "$VERBOSE" = true ]; then
        docker-compose -f "$COMPOSE_FILE" up -d
    else
        docker-compose -f "$COMPOSE_FILE" up -d > /dev/null 2>&1
    fi
    
    log_success "Docker Compose started"
    
    # STEP 2: Wait for broker
    log_info "STEP 2: Waiting for broker to be ready..."
    
    if ! wait_for_broker; then
        log_error "Failed to start Kafka broker"
        docker-compose -f "$COMPOSE_FILE" logs redpanda-test
        exit 1
    fi
    
    # STEP 3: Run integration tests
    log_info "STEP 3: Running integration tests..."
    
    # Clear previous test log
    > "$TEST_LOG"
    
    # Run pytest with integration markers
    log_info "Executing: pytest tests/test_kafka_integration.py"
    
    if [ "$VERBOSE" = true ]; then
        $PYTHON_CMD -m pytest \
            tests/test_kafka_integration.py \
            -v \
            -m "integration and kafka" \
            --tb=short \
            --maxfail=1 \
            --timeout=$TEST_TIMEOUT \
            --log-cli-level=INFO \
            2>&1 | tee "$TEST_LOG"
    else
        $PYTHON_CMD -m pytest \
            tests/test_kafka_integration.py \
            -v \
            -m "integration and kafka" \
            --tb=short \
            --maxfail=1 \
            --timeout=$TEST_TIMEOUT \
            > "$TEST_LOG" 2>&1
    fi
    
    TEST_EXIT_CODE=$?
    
    # STEP 4: Collect logs
    log_info "STEP 4: Collecting container logs..."
    
    docker-compose -f "$COMPOSE_FILE" logs redpanda-test > "$REPORT_DIR/redpanda.log" 2>&1
    
    # STEP 5: Generate summary report
    log_info "STEP 5: Generating summary report..."
    
    cat > "$REPORT_DIR/kafka_integration_summary.txt" <<EOF
Kafka Integration Test Summary
Generated: $(date)
=====================================

Test Environment:
- Broker: $BROKER_HOST
- Compose File: $COMPOSE_FILE
- Python: $($PYTHON_CMD --version)
- pytest: $($PYTHON_CMD -m pytest --version)

Test Results:
- Exit Code: $TEST_EXIT_CODE
- Log File: $TEST_LOG
- Container Logs: $REPORT_DIR/redpanda.log

Status: $([ $TEST_EXIT_CODE -eq 0 ] && echo "PASSED ✓" || echo "FAILED ✗")
=====================================
EOF
    
    cat "$REPORT_DIR/kafka_integration_summary.txt"
    
    # Final status
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        log_success "========================================="
        log_success "All integration tests PASSED ✓"
        log_success "========================================="
        log_info "Reports saved to: $REPORT_DIR/"
        log_info "  - kafka_integration.log"
        log_info "  - kafka_integration_summary.txt"
        log_info "  - redpanda.log"
        
        return 0
    else
        log_error "========================================="
        log_error "Integration tests FAILED ✗"
        log_error "========================================="
        log_error "Check logs at: $TEST_LOG"
        log_info "To debug, re-run with --verbose and --keep-containers"
        
        return 1
    fi
}

# Run main function
main "$@"
exit $?
