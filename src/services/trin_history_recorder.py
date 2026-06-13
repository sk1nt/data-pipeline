"""Persist live TRIN updates to DuckDB and Parquet for historical backfill."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from ..config import settings

LOGGER = logging.getLogger(__name__)


@dataclass
class TrinHistoryRecorderSettings:
    """Runtime settings for TRIN history persistence."""

    db_path: Path = Path(settings.data_path / "trin_history.db")
    parquet_dir: Path = Path(settings.data_path / "parquet" / "trin")
    flush_interval_seconds: int = 60
    table_name: str = "trin_trade_history"


class TrinHistoryRecorder:
    """Buffer live TRIN updates and persist them in a queryable form."""

    def __init__(
        self,
        settings: Optional[TrinHistoryRecorderSettings] = None,
    ) -> None:
        self.settings = settings or TrinHistoryRecorderSettings()
        self.settings.flush_interval_seconds = max(5, int(self.settings.flush_interval_seconds))
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("TRIN history recorder already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="trin-history-recorder")

    async def stop(self) -> None:
        self._stop_event.set()
        await self.flush()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def record_trade(self, payload: Dict[str, Any]) -> None:
        symbol = str(payload.get("symbol", "")).upper()
        if not symbol.startswith("$TRIN"):
            return

        ts_ms = self._timestamp_ms(payload.get("timestamp"))
        timestamp_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(
            tzinfo=None
        )
        normalized = {
            "event_id": self._event_id(payload, ts_ms),
            "timestamp_ms": ts_ms,
            "timestamp_utc": timestamp_utc,
            "day": timestamp_utc.date(),
            "symbol": symbol,
            "price": float(payload.get("price", 0.0) or 0.0),
            "size": float(payload.get("size", 0.0) or 0.0),
            "source": str(payload.get("source", "tastytrade")).lower(),
            "payload_json": json.dumps(payload, sort_keys=True, default=str),
        }
        with self._buffer_lock:
            self._buffer.append(normalized)

    async def flush(self) -> int:
        records = self._drain_buffer()
        if not records:
            return 0
        return await asyncio.to_thread(self._persist_records, records)

    async def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.settings.flush_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                await self.flush()
        finally:
            await self.flush()

    def _drain_buffer(self) -> List[Dict[str, Any]]:
        with self._buffer_lock:
            if not self._buffer:
                return []
            records = self._buffer
            self._buffer = []
            return records

    def _persist_records(self, records: List[Dict[str, Any]]) -> int:
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.parquet_dir.mkdir(parents=True, exist_ok=True)
        records = list({record["event_id"]: record for record in records}.values())
        if not records:
            return 0

        conn = duckdb.connect(str(self.settings.db_path))
        try:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.settings.table_name} (
                    event_id VARCHAR,
                    timestamp_ms BIGINT,
                    timestamp_utc TIMESTAMP,
                    day DATE,
                    symbol VARCHAR,
                    price DOUBLE,
                    size DOUBLE,
                    source VARCHAR,
                    payload_json VARCHAR
                )
                """
            )

            event_ids = [r["event_id"] for r in records]
            if event_ids:
                placeholders = ", ".join(["?" for _ in event_ids])
                conn.execute(
                    f"DELETE FROM {self.settings.table_name} WHERE event_id IN ({placeholders})",
                    event_ids,
                )

            insert_rows = [
                (
                    r["event_id"],
                    r["timestamp_ms"],
                    r["timestamp_utc"],
                    r["day"],
                    r["symbol"],
                    r["price"],
                    r["size"],
                    r["source"],
                    r["payload_json"],
                )
                for r in records
            ]
            conn.executemany(
                f"""
                INSERT INTO {self.settings.table_name}
                (event_id, timestamp_ms, timestamp_utc, day, symbol, price, size, source, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )

            df = pd.DataFrame(records)
            if df.empty:
                return 0

            df["day"] = pd.to_datetime(df["day"])
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
            for (symbol, day), group in df.groupby(["symbol", "day"], sort=True):
                safe_symbol = str(symbol).replace("/", "_")
                out_dir = self.settings.parquet_dir / safe_symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{day.strftime('%Y%m%d')}.parquet"
                self._write_day_parquet(out_path, group)
            return len(records)
        finally:
            conn.close()

    def _write_day_parquet(self, out_path: Path, group: pd.DataFrame) -> None:
        columns = [
            "event_id",
            "timestamp_ms",
            "timestamp_utc",
            "day",
            "symbol",
            "price",
            "size",
            "source",
            "payload_json",
        ]
        group = group[columns].sort_values(["timestamp_ms", "event_id"])
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            combined = pd.concat([existing, group], ignore_index=True)
            combined = combined.drop_duplicates(subset=["event_id"], keep="last")
            combined = combined.sort_values(["timestamp_ms", "event_id"])
        else:
            combined = group
        combined.to_parquet(out_path, index=False, compression="zstd")

    @staticmethod
    def _timestamp_ms(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                return int(datetime.now(timezone.utc).timestamp() * 1000)
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    @staticmethod
    def _event_id(payload: Dict[str, Any], ts_ms: int) -> str:
        normalized = dict(payload)
        normalized["timestamp_ms"] = ts_ms
        encoded = json.dumps(normalized, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()
