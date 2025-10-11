# Setup Airflow for Windows Testing
# This script configures Airflow to work on Windows and prepares for DAG testing

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Airflow Windows Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$PROJECT_ROOT = "E:\ZuuCrew\telco-churn-prediction-mini-project-1"
$AIRFLOW_HOME = "$PROJECT_ROOT\airflow_home"

# Set environment variable
Write-Host "[1/6] Setting AIRFLOW_HOME environment variable..." -ForegroundColor Yellow
$env:AIRFLOW_HOME = $AIRFLOW_HOME
Write-Host "Done: AIRFLOW_HOME = $AIRFLOW_HOME" -ForegroundColor Green
Write-Host ""

# Fix airflow.cfg paths from WSL to Windows
Write-Host "[2/6] Updating airflow.cfg paths from WSL to Windows..." -ForegroundColor Yellow

$config_file = "$AIRFLOW_HOME\airflow.cfg"

if (Test-Path $config_file) 
{
    # Read the config file
    $content = Get-Content $config_file -Raw
    
    # Replace WSL paths with Windows paths
    $content = $content -replace '/mnt/e/ZuuCrew', 'E:/ZuuCrew'
    $content = $content -replace 'sqlite:////', 'sqlite:///'
    $content = $content -replace 'load_examples = True', 'load_examples = False'
    
    # Write back
    Set-Content -Path $config_file -Value $content
    
    Write-Host "Done: Updated paths" -ForegroundColor Green
    Write-Host "  - /mnt/e/ZuuCrew -> E:/ZuuCrew" -ForegroundColor Gray
    Write-Host "  - Fixed SQLite URI format" -ForegroundColor Gray
    Write-Host "  - Disabled example DAGs" -ForegroundColor Gray
}
else 
{
    Write-Host "Error: Config file not found: $config_file" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Initialize Airflow database
Write-Host "[3/6] Initializing Airflow database..." -ForegroundColor Yellow
try 
{
    airflow db init 2>&1 | Out-Null
    Write-Host "Done: Database initialized successfully" -ForegroundColor Green
}
catch 
{
    Write-Host "Warning: Database initialization failed: $_" -ForegroundColor Red
    Write-Host "Continuing anyway..." -ForegroundColor Yellow
}
Write-Host ""

# Create admin user (non-interactive)
Write-Host "[4/6] Creating Airflow admin user..." -ForegroundColor Yellow
try 
{
    $output = airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin 2>&1
    Write-Host "Done: Admin user created (username: admin, password: admin)" -ForegroundColor Green
}
catch 
{
    Write-Host "Note: User may already exist (this is OK)" -ForegroundColor Yellow
}
Write-Host ""

# Create artifacts directories
Write-Host "[5/6] Creating artifact directories..." -ForegroundColor Yellow
$dirs = @(
    "$PROJECT_ROOT\artifacts\reports",
    "$PROJECT_ROOT\artifacts\logs",
    "$PROJECT_ROOT\artifacts\screenshots"
)

foreach ($dir in $dirs) 
{
    if (-not (Test-Path $dir)) 
    {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Done: Created $dir" -ForegroundColor Green
    }
    else 
    {
        Write-Host "Done: Exists $dir" -ForegroundColor Gray
    }
}
Write-Host ""

# Verify DAG files exist
Write-Host "[6/6] Verifying DAG files..." -ForegroundColor Yellow
$dag_files = @(
    "$AIRFLOW_HOME\dags\kafka_streaming_dag.py",
    "$AIRFLOW_HOME\dags\kafka_batch_dag.py",
    "$AIRFLOW_HOME\dags\kafka_summary.py"
)

$all_exist = $true
foreach ($dag in $dag_files) 
{
    if (Test-Path $dag) 
    {
        Write-Host "Done: Found $(Split-Path $dag -Leaf)" -ForegroundColor Green
    }
    else 
    {
        Write-Host "Error: Missing $(Split-Path $dag -Leaf)" -ForegroundColor Red
        $all_exist = $false
    }
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Verify Kafka is running:" -ForegroundColor White
Write-Host "   docker-compose up -d" -ForegroundColor Gray
Write-Host ""
Write-Host "2. List DAGs:" -ForegroundColor White
Write-Host "   airflow dags list" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Start Airflow webserver (in new terminal):" -ForegroundColor White
Write-Host "   `$env:AIRFLOW_HOME='$AIRFLOW_HOME'; airflow webserver --port 8080" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Start Airflow scheduler (in another terminal):" -ForegroundColor White
Write-Host "   `$env:AIRFLOW_HOME='$AIRFLOW_HOME'; airflow scheduler" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Access Web UI:" -ForegroundColor White
Write-Host "   http://localhost:8080 (admin/admin)" -ForegroundColor Gray
Write-Host ""
Write-Host "6. Test DAG tasks:" -ForegroundColor White
Write-Host "   See docs/airflow_kafka_dag_testing_guide.md for detailed steps" -ForegroundColor Gray
Write-Host ""

if ($all_exist) 
{
    Write-Host "Ready: All DAG files present - Ready to test!" -ForegroundColor Green
}
else 
{
    Write-Host "Error: Some DAG files missing - Please verify" -ForegroundColor Red
}

