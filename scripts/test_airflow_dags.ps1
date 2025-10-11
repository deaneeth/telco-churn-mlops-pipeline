# Airflow DAG Testing Script with Screenshot Collection
# This script executes all tests from the testing guide and creates screenshot prompts

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Airflow DAG Testing & Screenshot Guide" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$PROJECT_ROOT = "E:\ZuuCrew\telco-churn-prediction-mini-project-1"
$AIRFLOW_HOME = "$PROJECT_ROOT\airflow_home"
$SCREENSHOTS_DIR = "$PROJECT_ROOT\artifacts\screenshots"

# Set environment
$env:AIRFLOW_HOME = $AIRFLOW_HOME

# Create screenshots directory
if (-not (Test-Path $SCREENSHOTS_DIR)) {
    New-Item -ItemType Directory -Path $SCREENSHOTS_DIR -Force | Out-Null
}

function Prompt-Screenshot {
    param(
        [string]$Number,
        [string]$Title,
        [string]$Description,
        [string]$Filename
    )
    
    Write-Host ""
    Write-Host "📸 SCREENSHOT $Number - $Title" -ForegroundColor Magenta
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
    Write-Host $Description -ForegroundColor White
    Write-Host ""
    Write-Host "Save as: $SCREENSHOTS_DIR\$Filename" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key when screenshot is captured..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Execute-Command {
    param(
        [string]$Command,
        [string]$Description
    )
    
    Write-Host ""
    Write-Host "⚡ Executing: $Description" -ForegroundColor Green
    Write-Host "Command: $Command" -ForegroundColor Gray
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    
    # Execute and capture output
    Invoke-Expression $Command
    
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
}

Write-Host "This script will guide you through testing both Airflow DAGs" -ForegroundColor Cyan
Write-Host "and collecting screenshot evidence for documentation." -ForegroundColor Cyan
Write-Host ""
Write-Host "Prerequisites:" -ForegroundColor Yellow
Write-Host "  ✓ Airflow is configured (run scripts/setup_airflow_windows.ps1)" -ForegroundColor White
Write-Host "  ✓ Kafka is running (docker-compose up -d)" -ForegroundColor White
Write-Host "  ✓ Webserver running in another terminal" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to start testing"

# ============================================================================
# PART 1: DAG VALIDATION
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  PART 1: DAG VALIDATION (3 steps)     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan

# Screenshot 01
Execute-Command -Command "airflow dags list" -Description "List all DAGs"
Prompt-Screenshot `
    -Number "01" `
    -Title "DAG List" `
    -Description "Capture terminal showing both Kafka DAGs in the list (kafka_streaming_pipeline, kafka_batch_pipeline)" `
    -Filename "01_dag_list.png"

# Screenshot 02
Execute-Command -Command "airflow dags show kafka_streaming_pipeline" -Description "Show streaming DAG structure"
Prompt-Screenshot `
    -Number "02" `
    -Title "Streaming DAG Structure" `
    -Description "Capture the DAG structure output showing task dependencies" `
    -Filename "02_streaming_dag_structure.png"

# Screenshot 03
Execute-Command -Command "airflow tasks list kafka_batch_pipeline" -Description "List batch DAG tasks"
Prompt-Screenshot `
    -Number "03" `
    -Title "Batch DAG Tasks" `
    -Description "Capture terminal showing all 4 tasks for batch DAG" `
    -Filename "03_batch_dag_tasks.png"

# ============================================================================
# PART 2: STREAMING DAG TESTING
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  PART 2: STREAMING DAG (4 steps)      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan

$test_date = Get-Date -Format "yyyy-MM-dd"

# Screenshot 04
Execute-Command `
    -Command "airflow tasks test kafka_streaming_pipeline health_check_kafka $test_date" `
    -Description "Test health check task"
Prompt-Screenshot `
    -Number "04" `
    -Title "Health Check Execution" `
    -Description "Capture terminal showing successful Kafka health check" `
    -Filename "04_health_check_test.png"

# Screenshot 05
Write-Host ""
Write-Host "⚠️  WARNING: The next task starts a long-running consumer!" -ForegroundColor Yellow
Write-Host "    It will run in the background. You can stop it later." -ForegroundColor Yellow
Read-Host "Press Enter to continue"

Execute-Command `
    -Command "airflow tasks test kafka_streaming_pipeline start_consumer $test_date" `
    -Description "Start consumer process"
Prompt-Screenshot `
    -Number "05" `
    -Title "Consumer Start" `
    -Description "Capture terminal showing consumer started successfully with PID" `
    -Filename "05_consumer_start.png"

# Screenshot 06
Execute-Command `
    -Command "airflow tasks test kafka_streaming_pipeline monitor_consumer $test_date" `
    -Description "Monitor consumer health"
Prompt-Screenshot `
    -Number "06" `
    -Title "Consumer Monitoring" `
    -Description "Capture terminal showing consumer health status" `
    -Filename "06_consumer_monitor.png"

# Screenshot 07
$log_file = Get-ChildItem "$PROJECT_ROOT\artifacts\logs\kafka_consumer_streaming_*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($log_file) {
    Execute-Command -Command "Get-Content '$($log_file.FullName)' -Tail 30" -Description "View consumer logs"
} else {
    Write-Host "No consumer log file found" -ForegroundColor Yellow
}
Prompt-Screenshot `
    -Number "07" `
    -Title "Consumer Logs" `
    -Description "Capture log file content showing consumer activity" `
    -Filename "07_consumer_logs.png"

# ============================================================================
# PART 3: BATCH DAG TESTING
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  PART 3: BATCH DAG (4 steps)          ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan

# Screenshot 08
Execute-Command `
    -Command "airflow tasks test kafka_batch_pipeline trigger_producer $test_date" `
    -Description "Trigger producer to send batch"
Prompt-Screenshot `
    -Number "08" `
    -Title "Producer Execution" `
    -Description "Capture terminal showing batch messages sent to Kafka" `
    -Filename "08_producer_execution.png"

# Screenshot 09
Execute-Command `
    -Command "airflow tasks test kafka_batch_pipeline run_consumer_window $test_date" `
    -Description "Run consumer for time window"
Prompt-Screenshot `
    -Number "09" `
    -Title "Consumer Window" `
    -Description "Capture terminal showing consumer processing messages for 60 seconds" `
    -Filename "09_consumer_window.png"

# Screenshot 10
Execute-Command `
    -Command "airflow tasks test kafka_batch_pipeline generate_summary $test_date" `
    -Description "Generate summary report"
Prompt-Screenshot `
    -Number "10" `
    -Title "Summary Generation" `
    -Description "Capture terminal showing summary statistics and report files created" `
    -Filename "10_summary_generation.png"

# Screenshot 11
$summary_file = Get-ChildItem "$PROJECT_ROOT\artifacts\reports\batch_summary_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($summary_file) {
    Execute-Command -Command "Get-Content '$($summary_file.FullName)' | ConvertFrom-Json | ConvertTo-Json -Depth 5" -Description "View summary report"
} else {
    Write-Host "No summary report found" -ForegroundColor Yellow
}
Prompt-Screenshot `
    -Number "11" `
    -Title "Summary Report Content" `
    -Description "Capture JSON summary showing churn statistics and high-risk customers" `
    -Filename "11_summary_report.png"

# ============================================================================
# PART 4: AIRFLOW WEB UI
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  PART 4: WEB UI (5 steps)             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host ""
Write-Host "Opening Airflow Web UI..." -ForegroundColor Yellow
Start-Process "http://localhost:8080"

Write-Host ""
Write-Host "Web UI Screenshots - Manual Instructions:" -ForegroundColor Cyan
Write-Host ""

Prompt-Screenshot `
    -Number "12" `
    -Title "DAGs Homepage" `
    -Description "Capture Airflow homepage showing both Kafka DAGs in the list" `
    -Filename "12_web_ui_dags.png"

Prompt-Screenshot `
    -Number "13" `
    -Title "Streaming DAG Graph View" `
    -Description "Click on kafka_streaming_pipeline, then Graph tab. Capture the task graph." `
    -Filename "13_streaming_dag_graph.png"

Prompt-Screenshot `
    -Number "14" `
    -Title "Batch DAG Graph View" `
    -Description "Click on kafka_batch_pipeline, then Graph tab. Capture the task graph." `
    -Filename "14_batch_dag_graph.png"

Prompt-Screenshot `
    -Number "15" `
    -Title "DAG Trigger" `
    -Description "Click 'Trigger DAG' button for batch DAG. Capture the run starting." `
    -Filename "15_dag_trigger.png"

Prompt-Screenshot `
    -Number "16" `
    -Title "DAG Run Details" `
    -Description "Click on a running/completed DAG run. Capture the task status view." `
    -Filename "16_dag_run_details.png"

# ============================================================================
# COMPLETION
# ============================================================================

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  TESTING COMPLETE!                    ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Screenshots saved to:" -ForegroundColor Yellow
Write-Host "  $SCREENSHOTS_DIR" -ForegroundColor White
Write-Host ""
Write-Host "Screenshot Summary:" -ForegroundColor Cyan
Write-Host "  01-03: DAG Validation (3 screenshots)" -ForegroundColor White
Write-Host "  04-07: Streaming DAG Testing (4 screenshots)" -ForegroundColor White
Write-Host "  08-11: Batch DAG Testing (4 screenshots)" -ForegroundColor White
Write-Host "  12-16: Web UI Screenshots (5 screenshots)" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review all screenshots in $SCREENSHOTS_DIR" -ForegroundColor White
Write-Host "  2. Create screenshot archive: Compress-Archive -Path '$SCREENSHOTS_DIR\*' -DestinationPath '$PROJECT_ROOT\airflow_dag_screenshots.zip'" -ForegroundColor White
Write-Host "  3. Update Step 9 completion report with evidence" -ForegroundColor White
Write-Host ""

# List captured screenshots
Write-Host "Captured Screenshots:" -ForegroundColor Cyan
Get-ChildItem $SCREENSHOTS_DIR -Filter "*.png" | Sort-Object Name | ForEach-Object {
    Write-Host "  ✓ $($_.Name)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Testing session complete! 🎉" -ForegroundColor Green
