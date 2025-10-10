# Kafka Topic Creation Script for Telco Churn Prediction (PowerShell)
# Creates all required topics for streaming and batch processing
# Compatible with Redpanda (Kafka-compatible)

$ErrorActionPreference = "Stop"

# Configuration
$BootstrapServer = if ($env:KAFKA_BOOTSTRAP_SERVER) { $env:KAFKA_BOOTSTRAP_SERVER } else { "localhost:19092" }
$ContainerName = if ($env:KAFKA_CONTAINER) { $env:KAFKA_CONTAINER } else { "telco-redpanda" }

# Topic configurations: Name => @{Partitions, ReplicationFactor}
$Topics = @{
    "telco.raw.customers" = @{Partitions=3; Replication=1}
    "telco.churn.predictions" = @{Partitions=3; Replication=1}
    "telco.deadletter" = @{Partitions=1; Replication=1}
}

Write-Host "========================================" -ForegroundColor Blue
Write-Host "Telco Churn - Kafka Topic Setup" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Check if Redpanda container is running
Write-Host "Checking Redpanda container..." -ForegroundColor Yellow
$containerRunning = docker ps --filter "name=$ContainerName" --format "{{.Names}}" | Select-String -Pattern $ContainerName -Quiet

if (-not $containerRunning) {
    Write-Host "Error: Redpanda container '$ContainerName' is not running!" -ForegroundColor Red
    Write-Host "Please start it with: docker compose -f docker-compose.kafka.yml up -d" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Redpanda container is running" -ForegroundColor Green
Write-Host ""

# Wait for Kafka to be ready
Write-Host "Waiting for Redpanda to be ready..." -ForegroundColor Yellow
$maxRetries = 30
$retryCount = 0

while ($retryCount -lt $maxRetries) {
    $healthCheck = docker exec $ContainerName rpk cluster health 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Redpanda is healthy" -ForegroundColor Green
        break
    }
    $retryCount++
    Write-Host "  Attempt $retryCount/$maxRetries..."
    Start-Sleep -Seconds 2
}

if ($retryCount -eq $maxRetries) {
    Write-Host "Error: Redpanda did not become healthy in time" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Create topics
Write-Host "Creating Kafka topics..." -ForegroundColor Blue
Write-Host ""

foreach ($topicName in $Topics.Keys) {
    $config = $Topics[$topicName]
    $partitions = $config.Partitions
    $replication = $config.Replication
    
    Write-Host "Creating topic: $topicName" -ForegroundColor Yellow
    Write-Host "  Partitions: $partitions"
    Write-Host "  Replication Factor: $replication"
    
    # Check if topic already exists
    $existingTopics = docker exec $ContainerName rpk topic list 2>&1 | Out-String
    if ($existingTopics -match $topicName) {
        Write-Host "  ⚠ Topic already exists, skipping..." -ForegroundColor Yellow
    } else {
        # Create topic using rpk (Redpanda CLI)
        docker exec $ContainerName rpk topic create $topicName --partitions $partitions --replicas $replication --topic-config retention.ms=604800000 --topic-config cleanup.policy=delete 2>&1 | Out-Null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Topic created successfully" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Failed to create topic" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# List all topics
Write-Host "========================================" -ForegroundColor Blue
Write-Host "Current Topics:" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
docker exec $ContainerName rpk topic list
Write-Host ""

# Show topic details
Write-Host "========================================" -ForegroundColor Blue
Write-Host "Topic Details:" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
foreach ($topicName in $Topics.Keys) {
    Write-Host "Topic: $topicName" -ForegroundColor Yellow
    docker exec $ContainerName rpk topic describe $topicName | Select-Object -First 20
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Kafka topics setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Blue
Write-Host "  1. Access Redpanda Console: " -NoNewline; Write-Host "http://localhost:8080" -ForegroundColor Yellow
Write-Host "  2. Test producer: " -NoNewline; Write-Host "python src/streaming/producer.py --mode streaming" -ForegroundColor Yellow
Write-Host "  3. Test consumer: " -NoNewline; Write-Host "python src/streaming/consumer.py --mode streaming" -ForegroundColor Yellow
Write-Host ""
