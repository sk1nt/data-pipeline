# Running the monitoring stack on Windows (Docker Desktop)

Prerequisites
- Docker Desktop installed and running.
- WSL2 integration enabled (recommended) and using Linux containers.
- Clone of this repo available on the host file system accessible to Docker.

Quick start (PowerShell)
1. Open PowerShell (or WSL shell). If using Docker Desktop with WSL2, running from WSL is preferable.
2. From any location run the helper script in this directory:

```powershell
# from PowerShell
./monitoring/start-monitoring.ps1

# or (to stop)
./monitoring/stop-monitoring.ps1
```

Alternative (explicit compose)
```powershell
docker compose -f monitoring/docker-compose.yml up -d
```

Skip the metrics exporter (if it fails to build)
-----------------------------------------------
If the `metrics_exporter` service fails during build (for example because the local `metrics_exporter` folder is missing or network pulls are restricted), you can skip building it by using the provided override file. The override replaces the exporter with a tiny placeholder container so Prometheus and Grafana can start.

```powershell
# Start using the override (no-build exporter placeholder)
docker compose -f monitoring/docker-compose.yml -f monitoring/docker-compose.override.yml up -d
```

Verify
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

Troubleshooting
- If volume mounts fail with "file not found" make sure Docker Desktop has access to the filesystem (Settings → Resources → File Sharing) or run the compose from WSL where files are already accessible.
- If using Windows paths in compose, convert to WSL style or run from WSL.
- If ports are already in use, stop conflicting services or update `monitoring/docker-compose.yml` ports.
