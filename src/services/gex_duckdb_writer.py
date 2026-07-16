"""Shared writer queue for live GEX snapshot DuckDB persistence."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

from ..config import settings as config_settings
from .gex_wall_utils import build_compact_wall_fields

LOGGER = logging.getLogger(__name__)


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(float(value))
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


@dataclass
class GEXDuckDBWriterSettings:
    db_path: Path = Path(config_settings.data_path / "gex_data.db")
    batch_size: int = 500
    flush_interval_seconds: float = 5.0
    queue_warning_threshold: int = 250


class GEXDuckDBWriter:
    """Serialize live GEX snapshot writes into a single DuckDB writer thread."""

    def __init__(self, settings: GEXDuckDBWriterSettings = GEXDuckDBWriterSettings()) -> None:
        self.settings = settings
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._schema_ready = False
        self._last_snapshot_ms_by_symbol: Dict[str, int] = {}
        self._enqueued_count = 0
        self._written_snapshots = 0
        self._last_write_ts: Optional[str] = None
        self._last_batch_size = 0
        self._last_error: Optional[str] = None
        self._queue_depth = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            LOGGER.warning("GEX DuckDB writer already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="gex-duckdb-writer",
            daemon=True,
        )
        self._thread.start()

    async def stop(self) -> None:
        self._stop_event.set()
        self._queue.put_nowait(None)
        thread = self._thread
        if thread:
            # Join off-thread so shutdown doesn't block the event loop.
            import asyncio

            await asyncio.to_thread(thread.join, 5.0)
            self._thread = None

    def enqueue_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._queue.put_nowait(snapshot)
        self._enqueued_count += 1
        self._queue_depth = self._queue.qsize()
        if self._queue_depth >= self.settings.queue_warning_threshold:
            LOGGER.warning("GEX DuckDB writer queue depth high: %s", self._queue_depth)

    def _run(self) -> None:
        LOGGER.info("GEX DuckDB writer started (db=%s)", self.settings.db_path)
        db_path = self.settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            while not self._stop_event.is_set():
                batch = self._collect_batch()
                if not batch:
                    continue
                try:
                    self._flush_batch(db_path, batch)
                    self._last_batch_size = len(batch)
                    self._last_write_ts = datetime.now(timezone.utc).isoformat()
                except Exception:
                    self._last_error = "flush failed"
                    LOGGER.exception("GEX DuckDB writer failed to flush batch")
            # Drain any final items after stop has been requested.
            remaining = self._drain_remaining()
            if remaining:
                try:
                    self._flush_batch(db_path, remaining)
                    self._last_batch_size = len(remaining)
                    self._last_write_ts = datetime.now(timezone.utc).isoformat()
                except Exception:
                    self._last_error = "flush failed"
                    LOGGER.exception("GEX DuckDB writer failed during shutdown flush")
        except Exception:
            self._last_error = "connection failed"
            LOGGER.exception("GEX DuckDB writer could not open DuckDB file")
        finally:
            LOGGER.info("GEX DuckDB writer stopped")

    def _collect_batch(self) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        try:
            first = self._queue.get(timeout=self.settings.flush_interval_seconds)
        except queue.Empty:
            return batch
        if first is None:
            return batch
        batch.append(first)
        deadline = time.monotonic() + self.settings.flush_interval_seconds
        for _ in range(self.settings.batch_size - 1):
            if time.monotonic() >= deadline:
                break
            try:
                timeout = max(0.0, deadline - time.monotonic())
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if item is None:
                self._stop_event.set()
                break
            batch.append(item)
        self._queue_depth = self._queue.qsize()
        return batch

    def _drain_remaining(self) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            batch.append(item)
            if len(batch) >= self.settings.batch_size:
                break
        self._queue_depth = self._queue.qsize()
        return batch

    def _flush_batch(self, db_path: Path, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        snapshot_rows: List[tuple[Any, ...]] = []
        for snapshot in batch:
            row = self._build_snapshot_row(snapshot)
            if row is None:
                continue
            snapshot_rows.append(row)
        if not snapshot_rows:
            return
        with duckdb.connect(str(db_path)) as conn:
            self._ensure_schema(conn)
            conn.executemany(
                """
                INSERT INTO gex_snapshots (
                    timestamp,
                    ticker,
                    spot_price,
                    zero_gamma,
                    net_gex,
                    min_dte,
                    sec_min_dte,
                    major_pos_vol,
                    major_pos_oi,
                    major_pos_vol_gamma,
                    major_neg_vol,
                    major_neg_oi,
                    major_neg_vol_gamma,
                    sum_gex_vol,
                    sum_gex_oi,
                    delta_risk_reversal,
                    max_priors,
                    pos_can1_strike,
                    pos_can1_value,
                    pos_can1_pct,
                    pos_can2_strike,
                    pos_can2_value,
                    pos_can2_pct,
                    neg_can1_strike,
                    neg_can1_value,
                    neg_can1_pct,
                    neg_can2_strike,
                    neg_can2_value,
                    neg_can2_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                snapshot_rows,
            )
        self._written_snapshots += len(snapshot_rows)

    def _build_snapshot_row(self, snapshot: Dict[str, Any]) -> Optional[tuple[Any, ...]]:
        ticker = (snapshot.get("symbol") or snapshot.get("ticker") or "").upper()
        epoch_ms = _timestamp_ms(snapshot.get("timestamp"))
        if not ticker:
            return None
        last_ms = self._last_snapshot_ms_by_symbol.get(ticker)
        if last_ms is not None and epoch_ms <= last_ms:
            epoch_ms = last_ms + 1
        self._last_snapshot_ms_by_symbol[ticker] = epoch_ms

        max_priors = snapshot.get("max_priors")
        if max_priors is not None:
            try:
                max_priors = json.dumps(max_priors)
            except (TypeError, ValueError):
                max_priors = None
        compact_fields = build_compact_wall_fields(snapshot)
        return (
            epoch_ms,
            ticker,
            _to_float(snapshot.get("spot")),
            _to_float(snapshot.get("zero_gamma")),
            _to_float(snapshot.get("net_gex")),
            snapshot.get("min_dte"),
            snapshot.get("sec_min_dte"),
            _to_float(snapshot.get("major_pos_vol")),
            _to_float(snapshot.get("major_pos_oi")),
            _to_float(snapshot.get("major_pos_vol_gamma")),
            _to_float(snapshot.get("major_neg_vol")),
            _to_float(snapshot.get("major_neg_oi")),
            _to_float(snapshot.get("major_neg_vol_gamma")),
            _to_float(snapshot.get("sum_gex_vol")),
            _to_float(snapshot.get("sum_gex_oi")),
            _to_float(snapshot.get("delta_risk_reversal")),
            max_priors,
            _to_float(compact_fields.get("pos_can1_strike")),
            _to_float(compact_fields.get("pos_can1_value")),
            _to_float(compact_fields.get("pos_can1_pct")),
            _to_float(compact_fields.get("pos_can2_strike")),
            _to_float(compact_fields.get("pos_can2_value")),
            _to_float(compact_fields.get("pos_can2_pct")),
            _to_float(compact_fields.get("neg_can1_strike")),
            _to_float(compact_fields.get("neg_can1_value")),
            _to_float(compact_fields.get("neg_can1_pct")),
            _to_float(compact_fields.get("neg_can2_strike")),
            _to_float(compact_fields.get("neg_can2_value")),
            _to_float(compact_fields.get("neg_can2_pct")),
        )

    @staticmethod
    def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gex_snapshots (
                timestamp BIGINT,
                ticker VARCHAR,
                spot_price DOUBLE,
                zero_gamma DOUBLE,
                net_gex DOUBLE,
                min_dte INTEGER,
                sec_min_dte INTEGER,
                major_pos_vol DOUBLE,
                major_pos_oi DOUBLE,
                major_pos_vol_gamma DOUBLE,
                major_neg_vol DOUBLE,
                major_neg_oi DOUBLE,
                major_neg_vol_gamma DOUBLE,
                sum_gex_vol DOUBLE,
                sum_gex_oi DOUBLE,
                delta_risk_reversal DOUBLE,
                max_priors VARCHAR,
                strikes VARCHAR,
                pos_can1_strike DOUBLE,
                pos_can1_value DOUBLE,
                pos_can1_pct DOUBLE,
                pos_can2_strike DOUBLE,
                pos_can2_value DOUBLE,
                pos_can2_pct DOUBLE,
                neg_can1_strike DOUBLE,
                neg_can1_value DOUBLE,
                neg_can1_pct DOUBLE,
                neg_can2_strike DOUBLE,
                neg_can2_value DOUBLE,
                neg_can2_pct DOUBLE
            )
            """
        )
        for column in (
            "strikes",
            "major_pos_vol_gamma",
            "major_neg_vol_gamma",
            "pos_can1_strike",
            "pos_can1_value",
            "pos_can1_pct",
            "pos_can2_strike",
            "pos_can2_value",
            "pos_can2_pct",
            "neg_can1_strike",
            "neg_can1_value",
            "neg_can1_pct",
            "neg_can2_strike",
            "neg_can2_value",
            "neg_can2_pct",
        ):
            conn.execute(
                f"ALTER TABLE gex_snapshots ADD COLUMN IF NOT EXISTS {column} "
                f"{'VARCHAR' if column == 'strikes' else 'DOUBLE'}"
            )

    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "db_path": str(self.settings.db_path),
            "flush_interval_seconds": self.settings.flush_interval_seconds,
            "queued_items": self._queue.qsize(),
            "queue_depth": self._queue_depth,
            "enqueued_count": self._enqueued_count,
            "written_snapshots": self._written_snapshots,
            "last_write_ts": self._last_write_ts,
            "last_batch_size": self._last_batch_size,
            "last_error": self._last_error,
        }
