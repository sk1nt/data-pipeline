<#
stop-monitoring.ps1

Usage: Stops and removes containers started by the compose file.
#>

$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Write-Host "Bringing down monitoring stack using: $scriptDir/docker-compose.yml"
docker compose -f "$scriptDir/docker-compose.yml" down

Write-Host "Done."
