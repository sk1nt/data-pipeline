"""Background worker that flushes RedisTimeSeries data to DuckDB/Parquet."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import duckdb
import pandas as pd
import redis

from ..config import settings
from ..lib.redis_client import RedisClient
from .redis_timeseries import RedisTimeSeriesClient

LOGGER = logging.getLogger(__name__)


@dataclass
class FlushWorkerSettings:
    interval_seconds: int = settings.flush_interval_seconds
    key_pattern: str = "ts:*"
    last_hash: str = "ts_meta:last_flushed"
    db_path: Path = Path(settings.timeseries_db_path)
    parquet_dir: Path = Path(settings.timeseries_parquet_dir)
    gex_snapshot_db: Path = settings.data_path / "gex_data.db"
    gex_snapshot_prefix: str = "gex:snapshot:"
    gex_dynamic_key: str = "gexbot:symbols:dynamic"


class RedisFlushWorker:
    def __init__(
        self,
        redis_client: RedisClient,
        ts_client: RedisTimeSeriesClient,
        settings: FlushWorkerSettings = FlushWorkerSettings(),
    ) -> None:
        self.redis_client = redis_client
        self.ts_client = ts_client
        self.settings = settings
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._last_summary: Dict[str, Any] = {}
        self._migrate_legacy_metadata()

    def _migrate_legacy_metadata(self) -> None:
        """Migrate legacy metadata key to new location."""
        legacy_key = "ts:last_flushed"
        new_key = self.settings.last_hash
        
        if self.redis_client.client.exists(legacy_key) and not self.redis_client.client.exists(new_key):
            LOGGER.info("Migrating legacy metadata from %s to %s", legacy_key, new_key)
            legacy_data = self.redis_client.client.hgetall(legacy_key)
            if legacy_data:
                self.redis_client.client.hset(new_key, mapping=legacy_data)
            # Delete the legacy key to prevent conflicts
            self.redis_client.client.delete(legacy_key)
            LOGGER.info("Deleted legacy metadata key %s", legacy_key)

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("Redis flush worker already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="redis-flush-worker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        LOGGER.info("Redis flush worker started (interval=%ss)", self.settings.interval_seconds)
        while not self._stop_event.is_set():
            await self._flush_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.settings.interval_seconds)
            except asyncio.TimeoutError:
                continue
        LOGGER.info("Redis flush worker stopped")

    async def _flush_once(self) -> None:
        try:
            await asyncio.to_thread(self._flush_sync)
        except Exception:
            LOGGER.exception("Redis flush worker encountered an error")

    def _flush_sync(self) -> None:
        start_time = time.perf_counter()
        keys = list(self.redis_client.client.scan_iter(match=self.settings.key_pattern))
        if not keys:
            gex_summary = self._flush_gex_snapshots()
            self._last_summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "samples": 0,
                "keys": 0,
                "duration": 0.0,
            }
            self._last_summary.update(gex_summary)
            return
        last_hash = self.settings.last_hash
        new_records: List[Tuple[str, int, float]] = []
        last_updates = {}
        legacy_key = "ts:last_flushed"
        for key in keys:
            # Normalize key to string for comparisons and logging
            k = key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
            # Skip metadata keys that share the `ts:` prefix to avoid WRONGTYPE errors
            if k == last_hash or k == legacy_key:
                LOGGER.debug("Skipping metadata key %s during flush", k)
                continue
            last_ts = self.redis_client.client.hget(last_hash, key)
            start = int(last_ts) + 1 if last_ts is not None else 0
            try:
                samples = self.ts_client.range(key, start, "+")
            except redis.ResponseError as exc:
                # If the key exists but is not a RedisTimeSeries key (wrong type), skip it.
                # This can happen if non-TS keys share the `ts:*` prefix.
                msg = str(exc)
                if "WRONGTYPE" in msg or "WRONGTYPE" in msg.upper():
                    # If it's a known metadata key we already filtered, we shouldn't reach here.
                    # For other non-TS keys, warn once and continue.
                    LOGGER.warning("Skipping non-timeseries key %s: %s", k, msg)
                    continue
                # Re-raise unexpected ResponseError
                raise
            if not samples:
                continue
            new_records.extend((key, ts, value) for ts, value in samples)
            last_updates[key] = samples[-1][0]
        if not new_records:
            gex_summary = self._flush_gex_snapshots()
            duration = time.perf_counter() - start_time
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "samples": 0,
                "keys": 0,
                "duration": duration,
            }
            summary.update(gex_summary)
            self._last_summary = summary
            return
        df = pd.DataFrame(new_records, columns=["key", "ts", "value"])
        df["day"] = pd.to_datetime(df["ts"], unit="ms").dt.date
        self._write_to_duckdb(df)
        self._write_parquet(df)
        gex_summary = self._flush_gex_snapshots()
        if last_updates:
            self.redis_client.client.hset(last_hash, mapping=last_updates)
        duration = time.perf_counter() - start_time
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "samples": int(len(df)),
            "keys": int(len(last_updates)),
            "duration": duration,
        }
        summary.update(gex_summary)
        self._last_summary = summary
        LOGGER.info(
            "Redis flush worker: %s samples from %s keys in %.2fs",
            summary["samples"],
            summary["keys"],
            summary["duration"],
        )

    def _write_to_duckdb(self, df: pd.DataFrame) -> None:
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(self.settings.db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS redis_timeseries (
                key VARCHAR,
                ts BIGINT,
                value DOUBLE,
                day DATE
            )
            """
        )
        conn.register("flush_df", df)
        conn.execute("INSERT INTO redis_timeseries SELECT key, ts, value, day FROM flush_df")
        conn.close()

    def _write_parquet(self, df: pd.DataFrame) -> None:
        base_dir = self.settings.parquet_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        flush_ts = int(datetime.utcnow().timestamp())
        for day, group in df.groupby("day"):
            day_dir = base_dir / str(day)
            day_dir.mkdir(parents=True, exist_ok=True)
            filename = day_dir / f"flush_{flush_ts}.parquet"
            group.to_parquet(filename, index=False)

    def _flush_gex_snapshots(self) -> Dict[str, int]:
        try:
            symbols = self._collect_gex_symbols()
            if not symbols:
                return {"gex_snapshots": 0, "gex_strikes": 0}
            snapshot_rows: List[Dict[str, Any]] = []
            strike_rows: List[Dict[str, Any]] = []
            for symbol in symbols:
                snapshot = self._load_snapshot(symbol)
                if not snapshot:
                    continue
                snapshot_row = self._build_snapshot_row(snapshot)
                if not snapshot_row:
                    continue
                snapshot_rows.append(snapshot_row)
                strike_rows.extend(self._build_strike_rows(snapshot, snapshot_row["epoch_ms"]))
            if not snapshot_rows:
                return {"gex_snapshots": 0, "gex_strikes": 0}
            self._write_gex_tables(snapshot_rows, strike_rows)
            return {
                "gex_snapshots": len(snapshot_rows),
                "gex_strikes": len(strike_rows),
            }
        except Exception:
            LOGGER.exception("Failed to flush GEX snapshots")
            return {"gex_snapshots": 0, "gex_strikes": 0}

    def _collect_gex_symbols(self) -> Set[str]:
        symbols = {s.strip().upper() for s in settings.gex_symbol_list if s.strip()}
        dynamic_raw = self.redis_client.client.get(self.settings.gex_dynamic_key)
        if dynamic_raw:
            try:
                decoded = dynamic_raw.decode() if isinstance(dynamic_raw, (bytes, bytearray)) else dynamic_raw
                dynamic_symbols = json.loads(decoded)
                if isinstance(dynamic_symbols, list):
                    symbols.update(str(item).upper() for item in dynamic_symbols if isinstance(item, str))
            except json.JSONDecodeError:
                LOGGER.warning("Invalid dynamic symbol cache; ignoring")
        return symbols

    def _load_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = f"{self.settings.gex_snapshot_prefix}{symbol.upper()}"
        payload = self.redis_client.client.get(key)
        if not payload:
            return None
        try:
            decoded = payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
            data = json.loads(decoded)
            return data
        except json.JSONDecodeError:
            LOGGER.warning("Failed to decode snapshot for %s", symbol)
            return None

    def _parse_snapshot_epoch_ms(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        ts: Optional[datetime] = None
        if isinstance(value, datetime):
            ts = value.astimezone(timezone.utc)
        elif isinstance(value, (int, float)):
            ts = datetime.fromtimestamp(float(value), tz=timezone.utc)
        elif isinstance(value, str):
            try:
                ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            ts = ts.astimezone(timezone.utc)
        if ts is None:
            return None
        return int(ts.timestamp() * 1000)

    def _build_snapshot_row(self, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ticker = (snapshot.get("symbol") or snapshot.get("ticker") or "").upper()
        epoch_ms = self._parse_snapshot_epoch_ms(snapshot.get("timestamp"))
        if not ticker or epoch_ms is None:
            return None
        max_priors = snapshot.get("max_priors")
        if max_priors is not None:
            try:
                max_priors_str = json.dumps(max_priors)
            except (TypeError, ValueError):
                max_priors_str = None
        else:
            max_priors_str = None
        return {
            "epoch_ms": epoch_ms,
            "ticker": ticker,
            "spot_price": snapshot.get("spot"),
            "zero_gamma": snapshot.get("zero_gamma"),
            "net_gex": snapshot.get("net_gex"),
            "min_dte": snapshot.get("min_dte"),
            "sec_min_dte": snapshot.get("sec_min_dte"),
            "major_pos_vol": snapshot.get("major_pos_vol"),
            "major_pos_oi": snapshot.get("major_pos_oi"),
            "major_neg_vol": snapshot.get("major_neg_vol"),
            "major_neg_oi": snapshot.get("major_neg_oi"),
            "sum_gex_vol": snapshot.get("sum_gex_vol"),
            "sum_gex_oi": snapshot.get("sum_gex_oi"),
            "delta_risk_reversal": snapshot.get("delta_risk_reversal"),
            "max_priors": max_priors_str,
        }

    def _build_strike_rows(self, snapshot: Dict[str, Any], epoch_ms: int) -> List[Dict[str, Any]]:
        ticker = (snapshot.get("symbol") or snapshot.get("ticker") or "").upper()
        strikes = snapshot.get("strikes") or []
        rows: List[Dict[str, Any]] = []
        if not ticker:
            return rows
        for entry in strikes:
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            try:
                strike = float(entry[0])
            except (TypeError, ValueError):
                continue
            gamma = self._safe_float(entry, 1)
            oi_gamma = self._safe_float(entry, 2)
            priors = entry[3] if len(entry) > 3 else None
            priors_str = None
            if priors is not None:
                try:
                    priors_str = json.dumps(priors)
                except (TypeError, ValueError):
                    priors_str = None
            rows.append(
                {
                    "epoch_ms": epoch_ms,
                    "ticker": ticker,
                    "strike": strike,
                    "gamma": gamma,
                    "oi_gamma": oi_gamma,
                    "priors": priors_str,
                }
            )
        return rows

    @staticmethod
    def _safe_float(entry: Iterable[Any], idx: int) -> Optional[float]:
        try:
            value = entry[idx]
        except (IndexError, TypeError):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write_gex_tables(self, snapshots: List[Dict[str, Any]], strikes: List[Dict[str, Any]]) -> None:
        df_snapshots = pd.DataFrame(snapshots)
        df_strikes = pd.DataFrame(strikes) if strikes else None
        conn = duckdb.connect(str(self.settings.gex_snapshot_db))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gex_snapshots (
                epoch_ms BIGINT,
                ticker VARCHAR,
                spot_price DOUBLE,
                zero_gamma DOUBLE,
                net_gex DOUBLE,
                min_dte INTEGER,
                sec_min_dte INTEGER,
                major_pos_vol DOUBLE,
                major_pos_oi DOUBLE,
                major_neg_vol DOUBLE,
                major_neg_oi DOUBLE,
                sum_gex_vol DOUBLE,
                sum_gex_oi DOUBLE,
                delta_risk_reversal DOUBLE,
                max_priors VARCHAR
            )
            """
        )
        conn.register("gex_snapshots_flush", df_snapshots)
        conn.execute(
            """
            DELETE FROM gex_snapshots
            USING gex_snapshots_flush
            WHERE gex_snapshots.ticker = gex_snapshots_flush.ticker
              AND gex_snapshots.epoch_ms = gex_snapshots_flush.epoch_ms
            """
        )
        conn.execute(
            """
            INSERT INTO gex_snapshots
            SELECT
                epoch_ms,
                ticker,
                spot_price,
                zero_gamma,
                net_gex,
                min_dte,
                sec_min_dte,
                major_pos_vol,
                major_pos_oi,
                major_neg_vol,
                major_neg_oi,
                sum_gex_vol,
                sum_gex_oi,
                delta_risk_reversal,
                max_priors
            FROM gex_snapshots_flush
            """
        )
        if df_strikes is not None and not df_strikes.empty:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gex_strikes (
                    epoch_ms BIGINT,
                    ticker VARCHAR,
                    strike DOUBLE,
                    gamma DOUBLE,
                    oi_gamma DOUBLE,
                    priors VARCHAR
                )
                """
            )
            conn.register("gex_strikes_flush", df_strikes)
            conn.execute(
                """
                DELETE FROM gex_strikes
                USING gex_strikes_flush
                WHERE gex_strikes.ticker = gex_strikes_flush.ticker
                  AND gex_strikes.epoch_ms = gex_strikes_flush.epoch_ms
                  AND gex_strikes.strike = gex_strikes_flush.strike
                """
            )
            conn.execute(
                """
                INSERT INTO gex_strikes
                SELECT epoch_ms, ticker, strike, gamma, oi_gamma, priors
                FROM gex_strikes_flush
                """
            )
        conn.close()

    def status(self) -> Dict[str, Any]:
        running = self._task is not None and not self._task.done()
        summary = dict(self._last_summary)
        summary.setdefault("running", running)
        summary.setdefault("samples", 0)
        summary.setdefault("keys", 0)
        summary.setdefault("duration", 0.0)
        summary.setdefault("timestamp", None)
        return summary
