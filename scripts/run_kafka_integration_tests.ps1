# Kafka Integration Test Runner (PowerShell)
#
# This script orchestrates the complete integration test workflow on Windows:
# 1. Start Kafka test environment (Docker Compose)
# 2. Wait for broker to be ready
# 3. Run integration tests
# 4. Collect logs and generate reports
# 5. Tear down test environment
#
# Usage:
#   .\scripts\run_kafka_integration_tests.ps1 [-KeepContainers] [-Verbose]
#
# Parameters:
#   -KeepContainers   Don't tear down containers after tests (for debugging)
#   -Verbose          Show detailed output from Docker and tests
#
# Author: AI Assistant
# Created: 2025-01-11

[CmdletBinding()]
param(
    [switch]$KeepContainers = $false,
    [switch]$VerboseOutput = $false
)

# Stop on errors
$ErrorActionPreference = "Stop"

# ==================== CONFIGURATION ====================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ComposeFile = Join-Path $ProjectRoot "docker-compose.test.yml"
$ReportDir = Join-Path $ProjectRoot "reports"
$LogDir = Join-Path $ProjectRoot "logs"
$TestLog = Join-Path $ReportDir "kafka_integration.log"

$BrokerHost = "localhost:19093"
$BrokerReadyTimeout = 60
$TestTimeout = 300

# ==================== HELPER FUNCTIONS ====================

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] " -NoNewline
    Write-Host $Message -ForegroundColor $Color
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput "[INFO] $Message" "Cyan"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "[SUCCESS] $Message" "Green"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "[WARNING] $Message" "Yellow"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "[ERROR] $Message" "Red"
}

function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check Docker
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed. Please install Docker Desktop."
        exit 1
    }
    
    # Check Docker Compose
    $composeV2 = docker compose version 2>&1
    $composeV1 = docker-compose version 2>&1
    
    if (-not ($composeV2 -or $composeV1)) {
        Write-Error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    }
    
    # Determine Docker Compose command
    if ($composeV2) {
        $script:DockerComposeCmd = "docker compose"
    } else {
        $script:DockerComposeCmd = "docker-compose"
    }
    
    # Check Python
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "Python is not installed. Please install Python 3.8+."
        exit 1
    }
    
    # Check pytest
    $pytestCheck = python -m pytest --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pytest is not installed. Run: pip install pytest"
        exit 1
    }
    
    Write-Success "All prerequisites satisfied"
}

function Stop-TestEnvironment {
    if ($KeepContainers) {
        Write-Warning "Keeping containers running (-KeepContainers flag set)"
        Write-Info "To stop manually: $DockerComposeCmd -f $ComposeFile down"
        return
    }
    
    Write-Info "Tearing down test environment..."
    
    if ($VerboseOutput) {
        & $DockerComposeCmd -f $ComposeFile down -v
    } else {
        & $DockerComposeCmd -f $ComposeFile down -v 2>&1 | Out-Null
    }
    
    Write-Success "Test environment cleaned up"
}

function Wait-ForBroker {
    Write-Info "Waiting for Kafka broker to be ready (timeout: ${BrokerReadyTimeout}s)..."
    
    $elapsed = 0
    $retryInterval = 2
    
    while ($elapsed -lt $BrokerReadyTimeout) {
        # Try to check broker health
        $healthCheck = docker exec telco-redpanda-test rpk cluster health 2>&1
        
        if ($LASTEXITCODE -eq 0 -and $healthCheck -match "Healthy") {
            Write-Success "Kafka broker is ready"
            return $true
        }
        
        Start-Sleep -Seconds $retryInterval
        $elapsed += $retryInterval
        
        if ($elapsed % 10 -eq 0) {
            Write-Info "Still waiting... ($elapsed/${BrokerReadyTimeout}s)"
        }
    }
    
    Write-Error "Kafka broker failed to become ready within ${BrokerReadyTimeout}s"
    return $false
}

# ==================== MAIN WORKFLOW ====================

try {
    Write-Info "========================================="
    Write-Info "Kafka Integration Test Runner"
    Write-Info "========================================="
    
    # Create directories
    New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    
    # Check prerequisites
    Test-Prerequisites
    
    # Navigate to project root
    Set-Location $ProjectRoot
    
    # STEP 1: Start Docker Compose
    Write-Info "STEP 1: Starting Kafka test environment..."
    
    if ($VerboseOutput) {
        & $DockerComposeCmd -f $ComposeFile up -d
    } else {
        & $DockerComposeCmd -f $ComposeFile up -d 2>&1 | Out-Null
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Docker Compose"
    }
    
    Write-Success "Docker Compose started"
    
    # STEP 2: Wait for broker
    Write-Info "STEP 2: Waiting for broker to be ready..."
    
    if (-not (Wait-ForBroker)) {
        Write-Error "Failed to start Kafka broker"
        & $DockerComposeCmd -f $ComposeFile logs redpanda-test
        throw "Broker startup failed"
    }
    
    # STEP 3: Run integration tests
    Write-Info "STEP 3: Running integration tests..."
    
    # Clear previous test log
    if (Test-Path $TestLog) {
        Remove-Item $TestLog -Force
    }
    
    Write-Info "Executing: pytest tests/test_kafka_integration.py"
    
    if ($VerboseOutput) {
        python -m pytest `
            tests/test_kafka_integration.py `
            -v `
            -m "integration and kafka" `
            --tb=short `
            --maxfail=1 `
            --timeout=$TestTimeout `
            --log-cli-level=INFO `
            2>&1 | Tee-Object -FilePath $TestLog
    } else {
        python -m pytest `
            tests/test_kafka_integration.py `
            -v `
            -m "integration and kafka" `
            --tb=short `
            --maxfail=1 `
            --timeout=$TestTimeout `
            *> $TestLog
    }
    
    $testExitCode = $LASTEXITCODE
    
    # STEP 4: Collect logs
    Write-Info "STEP 4: Collecting container logs..."
    
    $redpandaLogPath = Join-Path $ReportDir "redpanda.log"
    & $DockerComposeCmd -f $ComposeFile logs redpanda-test > $redpandaLogPath 2>&1
    
    # STEP 5: Generate summary report
    Write-Info "STEP 5: Generating summary report..."
    
    $pythonVersion = python --version 2>&1
    $pytestVersion = python -m pytest --version 2>&1
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    $summaryPath = Join-Path $ReportDir "kafka_integration_summary.txt"
    
    $summaryContent = @"
Kafka Integration Test Summary
Generated: $timestamp
=====================================

Test Environment:
- Broker: $BrokerHost
- Compose File: $ComposeFile
- Python: $pythonVersion
- pytest: $pytestVersion

Test Results:
- Exit Code: $testExitCode
- Log File: $TestLog
- Container Logs: $redpandaLogPath

Status: $(if ($testExitCode -eq 0) { "PASSED ✓" } else { "FAILED ✗" })
=====================================
"@
    
    $summaryContent | Out-File -FilePath $summaryPath -Encoding UTF8
    Write-Host $summaryContent
    
    # Final status
    if ($testExitCode -eq 0) {
        Write-Success "========================================="
        Write-Success "All integration tests PASSED ✓"
        Write-Success "========================================="
        Write-Info "Reports saved to: $ReportDir\"
        Write-Info "  - kafka_integration.log"
        Write-Info "  - kafka_integration_summary.txt"
        Write-Info "  - redpanda.log"
        
        exit 0
    } else {
        Write-Error "========================================="
        Write-Error "Integration tests FAILED ✗"
        Write-Error "========================================="
        Write-Error "Check logs at: $TestLog"
        Write-Info "To debug, re-run with -Verbose and -KeepContainers"
        
        exit 1
    }
    
} catch {
    Write-Error "Fatal error: $_"
    Write-Error $_.ScriptStackTrace
    exit 1
    
} finally {
    # Cleanup
    Stop-TestEnvironment
}
