<#
start-monitoring.ps1

Usage: Run this from PowerShell (recommended: run in an elevated prompt if you hit permission issues).
Docker Desktop should be running and configured to use Linux containers (WSL2 recommended).
#>

$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Write-Host "Using compose file: $scriptDir/docker-compose.yml"

Write-Host "Starting monitoring stack (Prometheus + Grafana + exporter)..."
docker compose -f "$scriptDir/docker-compose.yml" up -d

Write-Host "Containers started. Listing running containers..."
docker ps --format "table {{.Names}}	{{.Image}}	{{.Status}}	{{.Ports}}"

Write-Host "Prometheus: http://localhost:9090  Grafana: http://localhost:3000 (admin/admin)"
