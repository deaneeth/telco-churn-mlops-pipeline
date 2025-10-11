# PowerShell Helper: Launch WSL for Airflow Testing
# Run this from Windows to open WSL and start the testing process

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Airflow DAG Testing - WSL Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "[1/3] Checking Docker status..." -ForegroundColor Yellow
$dockerRunning = docker ps 2>&1
if ($LASTEXITCODE -eq 0) 
{
    Write-Host "Done: Docker is running" -ForegroundColor Green
}
else 
{
    Write-Host "Warning: Docker may not be running" -ForegroundColor Yellow
    Write-Host "Please start Docker Desktop if needed" -ForegroundColor Yellow
}
Write-Host ""

# Start Kafka if not already running
Write-Host "[2/3] Starting Kafka infrastructure..." -ForegroundColor Yellow
$composeCheck = docker-compose ps kafka 2>&1 | Select-String "Up"
if ($composeCheck) 
{
    Write-Host "Done: Kafka is already running" -ForegroundColor Green
}
else 
{
    Write-Host "Starting Kafka with docker-compose..." -ForegroundColor Yellow
    docker-compose up -d
    Start-Sleep -Seconds 5
    Write-Host "Done: Kafka started" -ForegroundColor Green
}
Write-Host ""

# Verify Kafka is accessible
Write-Host "[3/3] Verifying Kafka health..." -ForegroundColor Yellow
$kafkaStatus = docker-compose ps kafka | Select-String "Up"
if ($kafkaStatus) 
{
    Write-Host "Done: Kafka is healthy" -ForegroundColor Green
}
else 
{
    Write-Host "Warning: Kafka may not be ready" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ready to Test!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Opening WSL terminal..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Commands to run in WSL:" -ForegroundColor Cyan
Write-Host "  1. cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1" -ForegroundColor White
Write-Host "  2. bash scripts/setup_airflow_wsl.sh" -ForegroundColor White
Write-Host "  3. bash scripts/test_airflow_dags_wsl.sh" -ForegroundColor White
Write-Host ""
Write-Host "Full command guide available in: WSL_COMMANDS.sh" -ForegroundColor Gray
Write-Host ""

# Create a WSL command script
$wslScript = @"
#!/bin/bash
cd /mnt/e/ZuuCrew/telco-churn-prediction-mini-project-1
echo ""
echo "========================================"
echo "Airflow DAG Testing in WSL"
echo "========================================"
echo ""
echo "Step 1: Run setup script"
echo "  bash scripts/setup_airflow_wsl.sh"
echo ""
echo "Step 2: Run testing script"  
echo "  bash scripts/test_airflow_dags_wsl.sh"
echo ""
echo "Or see: WSL_COMMANDS.sh for all commands"
echo ""
exec bash
"@

$wslScript | Out-File -FilePath "launch_wsl.sh" -Encoding UTF8 -NoNewline

Write-Host "Launching WSL..." -ForegroundColor Green
Write-Host ""

# Launch WSL with the project directory
wsl -d Ubuntu bash launch_wsl.sh
