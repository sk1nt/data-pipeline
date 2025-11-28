"""Prometheus metrics exporter for the data-pipeline.

Exposes metrics that track imports, durations, failures, and last-run timestamps.
Run this alongside the application (or as a separate container) and configure
Prometheus to scrape `/metrics` on port 8000.
"""

from __future__ import annotations

import time
import logging
from prometheus_client import start_http_server, Counter, Gauge, Histogram

LOG = logging.getLogger("metrics_exporter")

# Metrics
IMPORT_RECORDS = Counter(
    "gex_import_records_total", "Total number of records imported", ["ticker"]
)
IMPORT_FAILURES = Counter(
    "gex_import_failures_total", "Total number of import failures"
)
LAST_IMPORT_DURATION = Histogram(
    "gex_import_duration_seconds", "Duration of last import in seconds"
)
LAST_IMPORT_TIME = Gauge(
    "gex_last_import_timestamp", "Unix timestamp of last successful import"
)
IMPORT_IN_PROGRESS = Gauge(
    "gex_import_in_progress", "1 if an import is running, 0 otherwise"
)


def record_import(ticker: str, records: int, duration: float) -> None:
    IMPORT_RECORDS.labels(ticker=ticker).inc(records)
    LAST_IMPORT_DURATION.observe(duration)
    LAST_IMPORT_TIME.set_to_current_time()


def record_failure() -> None:
    IMPORT_FAILURES.inc()


def start_server(port: int = 8000) -> None:
    start_http_server(port)
    LOG.info("Prometheus metrics exporter started on :%d", port)


def run_forever(port: int = 8000) -> None:
    start_server(port)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        LOG.info("metrics exporter stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever(8000)
