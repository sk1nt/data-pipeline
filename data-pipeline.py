#!/usr/bin/env python3
"""Minimal GEX history bridge server.

This server only implements the /gex_history_url endpoint that accepts a JSON
payload with a signed download URL, ticker, and endpoint label. Requests are
persisted into the DuckDB-backed queue exposed by src.lib.gex_history_queue so
that background workers (e.g. scripts/import_gex_history.py) can process them.

The goal is to keep the implementation self-contained without depending on the
old torch-market repository or any legacy-private packages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.lib.gex_history_queue import gex_history_queue  # noqa: E402
from src.import_gex_history import process_historical_imports  # noqa: E402

LOG = logging.getLogger("gex_history_bridge")
_QUEUE_WORKER_ACTIVE = threading.Event()


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class HistoryBridgeHandler(BaseHTTPRequestHandler):
    """HTTP handler that only supports /gex_history_url and /health."""

    server_version = "GEXHistoryBridge/1.0"

    def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path in {"", "/"}:
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "healthy"})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:  # noqa: N802
        """Only /gex_history_url is supported."""
        path = self.path.rstrip("/")
        if path != "/gex_history_url":
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return
        self._handle_history_request()

    # ------------------------------------------------------------------ internals

    def _handle_history_request(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_error(HTTPStatus.BAD_REQUEST, "Empty request body")
            return

        raw_body = self.rfile.read(length)
        LOG.debug("History request body: %s", raw_body)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON payload")
            return

        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "Payload must be a JSON object")
            return

        url = self._normalize_string(payload.get("url"))
        ticker = self._normalize_string(payload.get("ticker"))
        endpoint = self._normalize_string(
            payload.get("endpoint") or payload.get("feed") or payload.get("kind")
        )
        if not endpoint:
            endpoint = self._infer_endpoint(url)
        metadata = payload.get("metadata")
        if metadata and not isinstance(metadata, dict):
            metadata = None

        # Extract ticker from URL if not provided or incorrect
        # URL format: https://hist.gex.bot/gb-nqndx/2025-10-21_NQ_NDX_classic_gex_zero.json
        if not ticker or ticker == "NDX":
            import re
            # Try to extract ticker from filename in URL
            url_match = re.search(r'/(\d{4}-\d{2}-\d{2})_([^_]+_[^_]+)_classic', url)
            if url_match:
                ticker = url_match.group(2)  # Extract NQ_NDX from filename
                LOG.info(f"Extracted ticker '{ticker}' from URL")

        if not url or not ticker:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing url or ticker")
            return

        if not endpoint:
            endpoint = "gex_zero"

        LOG.info("Queueing history request url=%s ticker=%s endpoint=%s", url, ticker, endpoint)

        try:
            queue_id = gex_history_queue.enqueue_request(
                url=url,
                ticker=ticker,
                endpoint=endpoint,
                payload=metadata or {},
            )
            trigger_queue_processing()
        except Exception as exc:  # pragma: no cover
            LOG.exception("Failed to enqueue history request")
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "status": "queued",
                "id": queue_id,
                "url": url,
                "ticker": ticker,
                "endpoint": endpoint,
            },
        )

    @staticmethod
    def _normalize_string(value: Optional[Any]) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    @staticmethod
    def _infer_endpoint(url: str) -> str:
        import re
        match = re.search(r'_((?:gex_zero|gex_one|gex_full))\.json', url)
        if match:
            inferred = match.group(1)
            LOG.info("Inferred endpoint '%s' from URL", inferred)
            return inferred
        return "gex_zero"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        """Route BaseHTTPRequestHandler logging through logging module."""
        LOG.info("%s - %s", self.address_string(), fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEX history bridge server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=8877, help="Port for incoming POSTs (default: 8877)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def trigger_queue_processing() -> None:
    """Kick off background processing if no worker is currently active."""
    if _QUEUE_WORKER_ACTIVE.is_set():
        LOG.debug("Queue processor already running; skipping trigger")
        return

    def _worker():
        if _QUEUE_WORKER_ACTIVE.is_set():
            return
        _QUEUE_WORKER_ACTIVE.set()
        LOG.info("Starting background queue processor")
        try:
            process_historical_imports()
        except Exception:  # pragma: no cover - logged by importer
            LOG.exception("Background queue processor failed")
        finally:
            LOG.info("Queue processor finished")
            _QUEUE_WORKER_ACTIVE.clear()

    thread = threading.Thread(target=_worker, name="gex-history-processor", daemon=True)
    thread.start()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    server = ThreadingHTTPServer((args.host, args.port), HistoryBridgeHandler)
    LOG.info("Serving /gex_history_url on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
