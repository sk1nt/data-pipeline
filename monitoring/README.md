# Monitoring (Prometheus + Grafana)

This folder provides a Docker Compose setup to run a Prometheus + Grafana stack
and a lightweight Prometheus exporter for the data-pipeline.

Services:
- `metrics_exporter`: Exposes metrics on port `8000` (Prometheus client)
- `prometheus`: Prometheus server (9090)
- `grafana`: Grafana UI (3000)

Quick start (requires Docker & Docker Compose):

1. Change to the monitoring directory:

```bash
cd monitoring
docker-compose up -d
```

2. Open Grafana: http://localhost:3000 (admin/admin)
3. Open Prometheus: http://localhost:9090

The Grafana container is pre-provisioned with a simple dashboard located at
`grafana/dashboards/gex_dashboard.json` and a Prometheus datasource.

Exporter
--------
The exporter is a tiny Python process that exposes metrics for imports. It is
built as part of the `metrics_exporter` service. You can run it locally with:

```bash
python src/metrics_exporter.py
```

Metrics exposed (examples):
- `gex_import_records_total{ticker=...}`: counter of imported records per ticker
- `gex_import_failures_total`: counter of import failures
- `gex_import_duration_seconds`: histogram of import duration
- `gex_last_import_timestamp`: gauge with last successful import unix timestamp
