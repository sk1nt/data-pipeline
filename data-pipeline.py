#!/usr/bin/env python3
"""Unified data pipeline server keeping legacy GEX endpoints and new services."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import settings
from src.import_gex_history import process_historical_imports
from src.lib.gex_history_queue import gex_history_queue
from src.lib.redis_client import RedisClient
from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings
from src.services.tastytrade_streamer import StreamerSettings, TastyTradeStreamer
from src.services.redis_timeseries import RedisTimeSeriesClient
from src.services.redis_flush_worker import FlushWorkerSettings, RedisFlushWorker

LOGGER = logging.getLogger("data_pipeline")


class HistoryPayload(BaseModel):
    url: str
    ticker: Optional[str] = None
    endpoint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ServiceManager:
    def __init__(self) -> None:
        self.tastytrade: Optional[TastyTradeStreamer] = None
        self.gex_poller: Optional[GEXBotPoller] = None
        self.redis_client: Optional[RedisClient] = None
        self.rts: Optional[RedisTimeSeriesClient] = None
        self.flush_worker: Optional[RedisFlushWorker] = None
        self.trade_count = 0
        self.depth_count = 0
        self.last_trade_ts: Optional[str] = None
        self.last_depth_ts: Optional[str] = None

    def start(self) -> None:
        self.redis_client = RedisClient(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
        )
        self.rts = RedisTimeSeriesClient(self.redis_client.client)

        if settings.tastytrade_stream_enabled:
            self.tastytrade = TastyTradeStreamer(
                StreamerSettings(
                    client_id=settings.tastytrade_client_id or "",
                    client_secret=settings.tastytrade_client_secret or "",
                    refresh_token=settings.tastytrade_refresh_token or "",
                    symbols=settings.tastytrade_symbol_list,
                    depth_levels=settings.tastytrade_depth_levels,
                ),
                on_trade=self._handle_trade_event,
                on_depth=self._handle_depth_event,
            )
            self.tastytrade.start()
            LOGGER.info("TastyTrade streamer started")

        if settings.gex_polling_enabled and settings.gexbot_api_key:
            self.gex_poller = GEXBotPoller(
                GEXBotPollerSettings(
                    api_key=settings.gexbot_api_key,
                    symbols=settings.gex_symbol_list,
                    interval_seconds=settings.gex_poll_interval_seconds,
                ),
                redis_client=self.redis_client,
                ts_client=self.rts,
            )
            self.gex_poller.start()
            LOGGER.info("GEXBot poller started")

        if self.rts and self.redis_client:
            flush_settings = FlushWorkerSettings()
            self.flush_worker = RedisFlushWorker(self.redis_client, self.rts, flush_settings)
            self.flush_worker.start()
            LOGGER.info("Redis flush worker started")

    async def stop(self) -> None:
        if self.tastytrade:
            await self.tastytrade.stop()
            LOGGER.info("TastyTrade streamer stopped")
        if self.gex_poller:
            await self.gex_poller.stop()
            LOGGER.info("GEXBot poller stopped")
        if self.flush_worker:
            await self.flush_worker.stop()
            LOGGER.info("Redis flush worker stopped")

    def status(self) -> Dict[str, Any]:
        return {
            "tastytrade_streamer": {
                "running": self.tastytrade is not None,
                "trade_samples": self.trade_count,
                "last_trade_ts": self.last_trade_ts,
                "depth_samples": self.depth_count,
                "last_depth_ts": self.last_depth_ts,
            },
            "gex_poller": getattr(self.gex_poller, "status", lambda: {})(),
            "redis_flush_worker": getattr(self.flush_worker, "status", lambda: {})(),
        }

    async def _handle_trade_event(self, payload: Dict[str, Any]) -> None:
        if not self.rts:
            return
        await asyncio.to_thread(self._write_trade_timeseries, payload)

    async def _handle_depth_event(self, payload: Dict[str, Any]) -> None:
        if not self.rts:
            return
        await asyncio.to_thread(self._write_depth_timeseries, payload)

    def _write_trade_timeseries(self, payload: Dict[str, Any]) -> None:
        symbol = payload.get("symbol", "").upper() or "UNKNOWN"
        timestamp_ms = _timestamp_ms(payload.get("timestamp"))
        price = float(payload.get("price", 0.0))
        size = float(payload.get("size", 0.0))
        samples = [
            (f"ts:trade:price:{symbol}", timestamp_ms, price, {"symbol": symbol, "type": "trade", "field": "price"}),
            (f"ts:trade:size:{symbol}", timestamp_ms, size, {"symbol": symbol, "type": "trade", "field": "size"}),
        ]
        self.rts.multi_add(samples)
        self.trade_count += 1
        self.last_trade_ts = payload.get("timestamp") or datetime.utcnow().isoformat()

    def _write_depth_timeseries(self, payload: Dict[str, Any]) -> None:
        symbol = payload.get("symbol", "").upper() or "UNKNOWN"
        timestamp_ms = _timestamp_ms(payload.get("timestamp"))
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []
        samples = []
        depth_levels = settings.tastytrade_depth_levels
        for idx, level in enumerate(bids[:depth_levels], start=1):
            price = float(level.get("price", 0.0))
            size = float(level.get("size", 0.0))
            samples.append(
                (
                    f"ts:depth:{symbol}:bid:{idx}:price",
                    timestamp_ms,
                    price,
                    {"symbol": symbol, "type": "depth", "side": "bid", "level": str(idx), "field": "price"},
                )
            )
            samples.append(
                (
                    f"ts:depth:{symbol}:bid:{idx}:size",
                    timestamp_ms,
                    size,
                    {"symbol": symbol, "type": "depth", "side": "bid", "level": str(idx), "field": "size"},
                )
            )
        for idx, level in enumerate(asks[:depth_levels], start=1):
            price = float(level.get("price", 0.0))
            size = float(level.get("size", 0.0))
            samples.append(
                (
                    f"ts:depth:{symbol}:ask:{idx}:price",
                    timestamp_ms,
                    price,
                    {"symbol": symbol, "type": "depth", "side": "ask", "level": str(idx), "field": "price"},
                )
            )
            samples.append(
                (
                    f"ts:depth:{symbol}:ask:{idx}:size",
                    timestamp_ms,
                    size,
                    {"symbol": symbol, "type": "depth", "side": "ask", "level": str(idx), "field": "size"},
                )
            )
        if samples and self.rts:
            self.rts.multi_add(samples)
            self.depth_count += 1
            self.last_depth_ts = payload.get("timestamp") or datetime.utcnow().isoformat()


service_manager = ServiceManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("Starting services during FastAPI lifespan")
    service_manager.start()
    try:
        yield
    finally:
        LOGGER.info("Stopping services during FastAPI lifespan")
        await service_manager.stop()


app = FastAPI(title="Data Pipeline", lifespan=lifespan)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/status")
async def status() -> Dict[str, Any]:
    return service_manager.status()


STATUS_PAGE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Data Pipeline Status</title>
  <style>
    body { font-family: Arial, sans-serif; background: #111; color: #f1f1f1; }
    pre { background: #222; padding: 1rem; border-radius: 8px; }
    .warning { color: #ffcc00; }
  </style>
</head>
<body>
  <h1>Data Pipeline Status</h1>
  <p class=\"warning\">This page refreshes every 3 seconds.</p>
  <pre id=\"status\">Loading...</pre>
  <script>
    async function refresh() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById('status').textContent = 'Error: ' + err;
      }
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


@app.get("/status.html", response_class=HTMLResponse)
async def status_page() -> str:
    return STATUS_PAGE


@app.post("/gex_history_url")
async def gex_history_endpoint(payload: HistoryPayload, background_tasks: BackgroundTasks):
    url = payload.url.strip()
    ticker = _normalize_string(payload.ticker)
    endpoint = _normalize_string(payload.endpoint)
    metadata = payload.metadata if isinstance(payload.metadata, dict) else None

    if not endpoint:
        endpoint = _infer_endpoint(url)
    if not ticker:
        ticker = _extract_ticker_from_url(url)

    if not url or not ticker:
        raise HTTPException(status_code=400, detail="Missing url or ticker")

    try:
        queue_id = gex_history_queue.enqueue_request(
            url=url,
            ticker=ticker,
            endpoint=endpoint or "gex_zero",
            payload=metadata or {},
        )
    except Exception as exc:
        LOGGER.exception("Failed to enqueue history request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(_trigger_queue_processing)

    return {
        "status": "queued",
        "id": queue_id,
        "url": url,
        "ticker": ticker,
        "endpoint": endpoint or "gex_zero",
    }


async def _trigger_queue_processing() -> None:
    try:
        await asyncio.to_thread(process_historical_imports)
    except Exception:
        LOGGER.exception("Background import processing failed")


def _normalize_string(value: Optional[Any]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _infer_endpoint(url: str) -> str:
    import re

    match = re.search(r"_((?:gex_zero|gex_one|gex_full))\\.json", url)
    if match:
        return match.group(1)
    return "gex_zero"


def _extract_ticker_from_url(url: str) -> str:
    import re

    match = re.search(r"/(\\d{4}-\\d{2}-\\d{2})_([^_]+_[^_]+)_classic", url)
    if match:
        return match.group(2)
    return ""


def _timestamp_ms(value: Optional[str]) -> int:
    if isinstance(value, (int, float)):
        return int(float(value))
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
    return int(datetime.utcnow().timestamp() * 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data pipeline server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
