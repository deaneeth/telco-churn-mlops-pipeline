#!/bin/bash
# Setup Airflow in WSL for Testing
# This script configures Airflow in WSL and prepares for DAG testing

set -e

echo "========================================"
echo "Airflow WSL Setup Script"
echo "========================================"
echo ""

PROJECT_ROOT="/mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1"
AIRFLOW_HOME="$PROJECT_ROOT/airflow_home"

# Set environment variable
echo "[1/7] Setting AIRFLOW_HOME environment variable..."
export AIRFLOW_HOME="$AIRFLOW_HOME"
echo "✓ AIRFLOW_HOME = $AIRFLOW_HOME"
echo ""

# Check if Airflow is installed
echo "[2/7] Checking Airflow installation..."
if ! command -v airflow &> /dev/null; then
    echo "⚠ Airflow not found in WSL. Installing..."
    pip install apache-airflow==2.7.3 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-3.8.txt"
    echo "✓ Airflow installed"
else
    echo "✓ Airflow already installed: $(airflow version)"
fi
echo ""

# Verify airflow.cfg exists (already configured for WSL paths)
echo "[3/7] Verifying airflow.cfg..."
if [ -f "$AIRFLOW_HOME/airflow.cfg" ]; then
    echo "✓ Config file found"
    
    # Update load_examples to False
    sed -i 's/load_examples = True/load_examples = False/g' "$AIRFLOW_HOME/airflow.cfg"
    echo "✓ Disabled example DAGs"
else
    echo "✗ Config file not found: $AIRFLOW_HOME/airflow.cfg"
    echo "  Initializing Airflow first..."
    airflow db init
fi
echo ""

# Initialize Airflow database
echo "[4/7] Initializing Airflow database..."
airflow db init 2>&1 | tail -5
echo "✓ Database initialized"
echo ""

# Create admin user (non-interactive)
echo "[5/7] Creating Airflow admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>&1 | grep -E "created|already exists" || echo "✓ Admin user ready"
echo ""

# Create artifacts directories
echo "[6/7] Creating artifact directories..."
mkdir -p "$PROJECT_ROOT/artifacts/reports"
mkdir -p "$PROJECT_ROOT/artifacts/logs"
mkdir -p "$PROJECT_ROOT/artifacts/screenshots"
echo "✓ Created artifact directories"
echo ""

# Verify DAG files exist
echo "[7/7] Verifying DAG files..."
DAG_FILES=(
    "$AIRFLOW_HOME/dags/kafka_streaming_dag.py"
    "$AIRFLOW_HOME/dags/kafka_batch_dag.py"
    "$AIRFLOW_HOME/dags/kafka_summary.py"
)

all_exist=true
for dag in "${DAG_FILES[@]}"; do
    if [ -f "$dag" ]; then
        echo "✓ Found: $(basename $dag)"
    else
        echo "✗ Missing: $(basename $dag)"
        all_exist=false
    fi
done
echo ""

# Summary
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Verify Kafka is running (from Windows):"
echo "   docker-compose up -d"
echo ""
echo "2. List DAGs:"
echo "   airflow dags list"
echo ""
echo "3. Run the automated test script:"
echo "   bash scripts/test_airflow_dags_wsl.sh"
echo ""
echo "4. Start Airflow webserver (in new WSL terminal):"
echo "   export AIRFLOW_HOME='$AIRFLOW_HOME'"
echo "   airflow webserver --port 8080"
echo ""
echo "5. Start Airflow scheduler (in another WSL terminal):"
echo "   export AIRFLOW_HOME='$AIRFLOW_HOME'"
echo "   airflow scheduler"
echo ""
echo "6. Access Web UI from Windows browser:"
echo "   http://localhost:8080 (admin/admin)"
echo ""

if [ "$all_exist" = true ]; then
    echo "✓ All DAG files present - Ready to test!"
else
    echo "✗ Some DAG files missing - Please verify"
fi
echo ""

# Add AIRFLOW_HOME to bashrc for persistence
if ! grep -q "AIRFLOW_HOME" ~/.bashrc; then
    echo "# Airflow Configuration" >> ~/.bashrc
    echo "export AIRFLOW_HOME='$AIRFLOW_HOME'" >> ~/.bashrc
    echo "✓ Added AIRFLOW_HOME to ~/.bashrc for future sessions"
fi
