"""Background worker that flushes RedisTimeSeries data to DuckDB/Parquet."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import duckdb
import polars as pl
import redis

from ..config import settings as config_settings
from lib.redis_client import RedisClient
from .gex_wall_utils import build_compact_wall_fields, parse_gex_strikes, summarize_wall_candidates
from .redis_timeseries import RedisTimeSeriesClient

LOGGER = logging.getLogger(__name__)


@dataclass
class FlushWorkerSettings:
    # Use reasonable compile-time defaults; runtime overrides use config_settings
    interval_seconds: int = 600
    schedule_mode: str = "interval"
    daily_time: str = "00:30"
    key_pattern: str = "ts:*"
    last_hash: str = "ts_meta:last_flushed"
    db_path: Path = Path(config_settings.timeseries_db_path)
    parquet_dir: Path = Path(config_settings.timeseries_parquet_dir)
    gex_snapshot_db: Path = config_settings.data_path / "gex_data.db"
    gex_snapshot_prefix: str = "gex:snapshot:"
    tick_db_path: Path = Path(config_settings.tick_db_path)
    depth_db_path: Path = Path(config_settings.depth_db_path)
    tick_parquet_dir: Path = Path(config_settings.tick_parquet_dir)
    depth_parquet_dir: Path = Path(config_settings.depth_parquet_dir)


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
        # Ensure runtime settings are sourced from config where appropriate
        try:
            self.settings.interval_seconds = int(
                getattr(
                    config_settings,
                    "flush_interval_seconds",
                    self.settings.interval_seconds,
                )
            )
        except Exception:
            pass
        self.settings.schedule_mode = getattr(
            config_settings, "flush_schedule_mode", self.settings.schedule_mode
        )
        self.settings.daily_time = getattr(
            config_settings, "flush_daily_time", self.settings.daily_time
        )
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._last_summary: Dict[str, Any] = {}
        self._migrate_legacy_metadata()

    def _migrate_legacy_metadata(self) -> None:
        """Migrate legacy metadata key to new location."""
        legacy_key = "ts:last_flushed"
        new_key = self.settings.last_hash

        if self.redis_client.client.exists(
            legacy_key
        ) and not self.redis_client.client.exists(new_key):
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
        mode = (self.settings.schedule_mode or "interval").lower()
        if mode == "daily":
            LOGGER.info(
                "Redis flush worker started (daily schedule at %sZ)",
                self.settings.daily_time,
            )
            await self._run_daily()
        else:
            LOGGER.info(
                "Redis flush worker started (interval=%ss)",
                self.settings.interval_seconds,
            )
            await self._run_interval()
        LOGGER.info("Redis flush worker stopped")

    async def _run_interval(self) -> None:
        while not self._stop_event.is_set():
            await self._flush_once()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.settings.interval_seconds
                )
            except asyncio.TimeoutError:
                continue

    async def _run_daily(self) -> None:
        while not self._stop_event.is_set():
            wait_seconds = self._seconds_until_daily_run()
            LOGGER.info("Next Redis flush scheduled in %.0fs", wait_seconds)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_seconds)
                if self._stop_event.is_set():
                    break
            except asyncio.TimeoutError:
                pass
            if self._stop_event.is_set():
                break
            await self._flush_once()

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
            uw_summary = self._flush_uw_messages()
            self._last_summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "samples": 0,
                "keys": 0,
                "duration": 0.0,
            }
            self._last_summary.update(gex_summary)
            self._last_summary.update(uw_summary)
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
        gex_ts_records = self._collect_gex_candidate_timeseries_records()
        if gex_ts_records:
            self.ts_client.multi_add(
                (
                    (
                        key,
                        ts,
                        value,
                        {
                            "type": "gex",
                            "field": key.split(":")[2],
                            "symbol": key.rsplit(":", 1)[-1],
                        },
                    )
                    for key, ts, value in gex_ts_records
                )
            )
            new_records.extend(gex_ts_records)

        if not new_records:
            gex_summary = self._flush_gex_snapshots()
            uw_summary = self._flush_uw_messages()
            duration = time.perf_counter() - start_time
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "samples": 0,
                "keys": 0,
                "duration": duration,
            }
            summary.update(gex_summary)
            summary.update(uw_summary)
            self._last_summary = summary
            return
        df = pl.DataFrame(
            {
                "key": [self._normalize_key(r[0]) for r in new_records],
                "ts": [r[1] for r in new_records],
                "value": [float(r[2]) for r in new_records],
            }
        )
        df = df.with_columns(
            pl.col("ts").cast(pl.Datetime("ms")).dt.date().alias("day")
        )
        self._write_to_duckdb(df)
        self._write_tick_outputs(df)
        self._write_depth_outputs(df)
        self._write_parquet(df)
        gex_summary = self._flush_gex_snapshots()
        uw_summary = self._flush_uw_messages()
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
        summary.update(uw_summary)
        self._last_summary = summary
        LOGGER.info(
            "Redis flush worker: %s samples from %s keys in %.2fs",
            summary["samples"],
            summary["keys"],
            summary["duration"],
        )

    def _collect_gex_candidate_timeseries_records(self) -> List[Tuple[str, int, float]]:
        records: List[Tuple[str, int, float]] = []
        for symbol in self._collect_gex_symbols():
            snapshot = self._load_snapshot(symbol)
            if not snapshot:
                continue
            epoch_ms = self._parse_snapshot_epoch_ms(snapshot.get("timestamp"))
            if epoch_ms is None:
                continue
            for field, value in self._gex_candidate_fields(snapshot).items():
                numeric = self._coerce_float(value)
                if numeric is None:
                    continue
                records.append((f"ts:gex:{field}:{symbol}", epoch_ms, numeric))
        return records

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _gex_candidate_fields(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        compact = build_compact_wall_fields(snapshot)
        for prefix in ("pos", "neg"):
            for idx in (1, 2):
                for name in ("strike", "value", "pct"):
                    field = f"{prefix}_can{idx}_{name}"
                    fields[field] = compact.get(field, snapshot.get(field))
        if any(value is not None for value in fields.values()):
            return fields

        strikes = parse_gex_strikes(snapshot.get("strikes"))
        fields.update(
            {
                f"pos_can{idx}_{name}": value
                for idx, entry in enumerate(
                    summarize_wall_candidates(
                        snapshot.get("major_pos_vol"), strikes, prefer_positive=True
                    ),
                    start=1,
                )
                for name, value in entry.items()
            }
        )
        fields.update(
            {
                f"neg_can{idx}_{name}": value
                for idx, entry in enumerate(
                    summarize_wall_candidates(
                        snapshot.get("major_neg_vol"), strikes, prefer_positive=False
                    ),
                    start=1,
                )
                for name, value in entry.items()
            }
        )
        return fields

    def _seconds_until_daily_run(self) -> float:
        try:
            hour, minute = [int(part) for part in self.settings.daily_time.split(":")]
        except (ValueError, AttributeError):
            hour, minute = 0, 30
        target = dt_time(hour=hour % 24, minute=minute % 60)
        now = datetime.now(timezone.utc)
        today_target = now.replace(
            hour=target.hour,
            minute=target.minute,
            second=0,
            microsecond=0,
        )
        if today_target <= now:
            today_target += timedelta(days=1)
        return max(0.0, (today_target - now).total_seconds())

    def _write_to_duckdb(self, df: pl.DataFrame) -> None:
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
        conn.execute(
            "INSERT INTO redis_timeseries SELECT key, ts, value, day FROM flush_df"
        )
        conn.close()

    def _write_parquet(self, df: pl.DataFrame) -> None:
        base_dir = self.settings.parquet_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        flush_ts = int(datetime.utcnow().timestamp())
        for group in df.partition_by(["day"]):
            day = group["day"][0]
            day_dir = base_dir / str(day)
            day_dir.mkdir(parents=True, exist_ok=True)
            filename = day_dir / f"flush_{flush_ts}.parquet"
            group.write_parquet(filename)

    def _write_tick_outputs(self, df: pl.DataFrame) -> None:
        if df.is_empty() or "key" not in df.columns:
            return
        tick_df = df.filter(pl.col("key").str.starts_with("ts:trade:"))
        if tick_df.is_empty():
            return
        tick_df = tick_df.with_columns(
            pl.col("key").str.split(":").alias("_parts")
        ).filter(
            pl.col("_parts").list.len() >= 5
        ).with_columns(
            pl.col("_parts").list.get(2).alias("metric"),
            pl.col("_parts").list.get(3).str.to_uppercase().alias("symbol"),
            pl.col("_parts").list.get(4).str.to_uppercase().alias("source"),
        ).drop("_parts")
        pivot = tick_df.pivot(
            on="metric",
            index=["symbol", "source", "ts"],
            values="value",
            aggregate_function="last",
        )
        if pivot.is_empty():
            return
        if "price" not in pivot.columns:
            return
        before_drop = len(pivot)
        pivot = pivot.drop_nulls(subset=["price"])
        if pivot.is_empty():
            LOGGER.debug(
                "Skip tick flush slice due to missing price column values (before=%s)",
                before_drop,
            )
            return
        if "size" not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0.0).alias("size"))
        else:
            pivot = pivot.with_columns(pl.col("size").fill_null(0.0))
        pivot = pivot.with_columns(
            pl.col("ts").cast(pl.Datetime("ms", "UTC")).alias("timestamp"),
        ).with_columns(
            pl.col("timestamp").dt.strftime("%Y%m%d").alias("day"),
        )
        parquet_df = pivot.select(
            ["symbol", "source", "timestamp", "ts", "price", "size", "day"]
        ).rename({"ts": "timestamp_ms"})
        self._append_tick_parquet(parquet_df)
        self._insert_tick_rows(parquet_df)

    def _write_depth_outputs(self, df: pl.DataFrame) -> None:
        if df.is_empty() or "key" not in df.columns:
            return
        depth_df = df.filter(pl.col("key").str.starts_with("ts:depth:"))
        if depth_df.is_empty():
            return
        depth_df = depth_df.with_columns(
            pl.col("key").str.split(":").alias("_parts")
        ).filter(
            pl.col("_parts").list.len() >= 7
        ).with_columns(
            pl.col("_parts").list.get(2).str.to_uppercase().alias("symbol"),
            pl.col("_parts").list.get(3).str.to_uppercase().alias("source"),
            pl.col("_parts").list.get(4).alias("side"),
            pl.col("_parts").list.get(5).cast(pl.Int64, strict=False).alias("level"),
            pl.col("_parts").list.get(6).alias("field"),
        ).drop("_parts").drop_nulls(subset=["level"])
        if depth_df.is_empty():
            return
        pivot = depth_df.pivot(
            on="field",
            index=["symbol", "source", "ts", "side", "level"],
            values="value",
            aggregate_function="last",
        )
        if pivot.is_empty():
            return
        if "price" not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(None).cast(pl.Float64).alias("price"))
        if "size" not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(None).cast(pl.Float64).alias("size"))
        pivot = pivot.with_columns(
            pl.col("ts").cast(pl.Datetime("ms", "UTC")).alias("timestamp"),
        ).with_columns(
            pl.col("timestamp").dt.strftime("%Y%m%d").alias("day"),
        )
        parquet_df = pivot.select(
            ["symbol", "source", "side", "level", "timestamp", "ts", "price", "size", "day"]
        ).rename({"ts": "timestamp_ms"})
        self._append_depth_parquet(parquet_df)

    def _append_tick_parquet(self, parquet_df: pl.DataFrame) -> None:
        if parquet_df.is_empty():
            return
        manifests: List[Tuple[str, str, str, str, int, datetime, datetime]] = []
        parquet_dir = self.settings.tick_parquet_dir
        parquet_dir.mkdir(parents=True, exist_ok=True)
        for group in parquet_df.partition_by(["symbol", "day"]):
            symbol = group["symbol"][0]
            day = group["day"][0]
            safe_symbol = self._sanitize_symbol_dir(symbol)
            dest = parquet_dir / safe_symbol / f"{day}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            subset = group.drop(["day"]).sort("timestamp")
            subset.write_parquet(dest)
            manifests.append(
                (
                    symbol,
                    ",".join(sorted(subset["source"].unique().to_list())),
                    day,
                    dest.as_posix(),
                    len(subset),
                    subset["timestamp"][0],
                    subset["timestamp"][-1],
                )
            )
        self._update_tick_manifest(manifests)

    def _append_depth_parquet(self, parquet_df: pl.DataFrame) -> None:
        if parquet_df.is_empty():
            return
        manifests: List[Tuple[str, str, str, str, int, datetime, datetime]] = []
        parquet_dir = self.settings.depth_parquet_dir
        parquet_dir.mkdir(parents=True, exist_ok=True)
        for group in parquet_df.partition_by(["symbol", "day"]):
            symbol = group["symbol"][0]
            day = group["day"][0]
            safe_symbol = self._sanitize_symbol_dir(symbol)
            dest = parquet_dir / safe_symbol / f"{day}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            subset = group.drop(["day"]).sort(["timestamp", "side", "level"])
            subset.write_parquet(dest)
            manifests.append(
                (
                    symbol,
                    ",".join(sorted(subset["source"].unique().to_list())),
                    day,
                    dest.as_posix(),
                    len(subset),
                    subset["timestamp"][0],
                    subset["timestamp"][-1],
                )
            )
        self._update_depth_manifest(manifests)

    @staticmethod
    def _sanitize_symbol_dir(symbol: str) -> str:
        """Ensure symbol-derived directories remain relative and filesystem-safe."""
        cleaned = (symbol or "UNKNOWN").strip().replace("/", "_")
        if not cleaned:
            return "UNKNOWN"
        return cleaned

    def _insert_tick_rows(self, parquet_df: pl.DataFrame) -> None:
        if parquet_df.is_empty():
            return
        rows = parquet_df.with_columns(
            pl.col("size").fill_null(0.0).round(0).cast(pl.Int64).alias("volume"),
            pl.lit("trade").alias("tick_type"),
        )
        conn = duckdb.connect(str(self.settings.tick_db_path))
        try:
            conn.execute("DESCRIBE tick_data")
        except duckdb.CatalogException:
            LOGGER.warning(
                "tick_data table missing in %s; skipping tick inserts",
                self.settings.tick_db_path,
            )
            conn.close()
            return
        conn.register(
            "tick_flush_df",
            rows.select(["symbol", "timestamp", "price", "volume", "tick_type", "source"]),
        )
        next_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM tick_data").fetchone()[
            0
        ]
        conn.execute(
            """
            INSERT INTO tick_data
            SELECT
                row_number() OVER () + ? AS id,
                symbol,
                timestamp,
                price,
                volume,
                tick_type,
                source
            FROM tick_flush_df
            """,
            [next_id],
        )
        conn.close()

    def _update_tick_manifest(
        self,
        rows: List[Tuple[str, str, str, str, int, datetime, datetime]],
    ) -> None:
        if not rows:
            return
        conn = duckdb.connect(str(self.settings.tick_db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tick_parquet_manifest (
                symbol VARCHAR,
                sources VARCHAR,
                day VARCHAR,
                file_path VARCHAR,
                rows BIGINT,
                first_ts TIMESTAMP,
                last_ts TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, day)
            )
            """
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO tick_parquet_manifest
            (symbol, sources, day, file_path, rows, first_ts, last_ts, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            rows,
        )
        pattern = (self.settings.tick_parquet_dir / "*" / "*.parquet").as_posix()
        escaped = pattern.replace("'", "''")
        conn.execute(
            f"""
            CREATE OR REPLACE VIEW tick_parquet AS
            SELECT *
            FROM read_parquet('{escaped}', union_by_name=true)
            """
        )
        conn.close()

    def _update_depth_manifest(
        self,
        rows: List[Tuple[str, str, str, str, int, datetime, datetime]],
    ) -> None:
        if not rows:
            return
        conn = duckdb.connect(str(self.settings.depth_db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS depth_parquet_manifest (
                symbol VARCHAR,
                sources VARCHAR,
                day VARCHAR,
                file_path VARCHAR,
                rows BIGINT,
                first_ts TIMESTAMP,
                last_ts TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, day)
            )
            """
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO depth_parquet_manifest
            (symbol, sources, day, file_path, rows, first_ts, last_ts, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            rows,
        )
        pattern = (self.settings.depth_parquet_dir / "*" / "*.parquet").as_posix()
        escaped = pattern.replace("'", "''")
        conn.execute(
            f"""
            CREATE OR REPLACE VIEW depth_parquet AS
            SELECT *
            FROM read_parquet('{escaped}', union_by_name=true)
            """
        )
        conn.close()

    @staticmethod
    def _normalize_key(key: Any) -> str:
        if isinstance(key, (bytes, bytearray)):
            return key.decode()
        return str(key)

    def _flush_gex_snapshots(self) -> Dict[str, int]:
        # Live GEX snapshots now persist directly from the poller into DuckDB.
        # Keep this hook as a no-op so the flush worker does not collapse the
        # event stream back down to the latest blob per symbol.
        return {"gex_snapshots": 0, "gex_strikes": 0}

    def _collect_gex_symbols(self) -> Set[str]:
        symbols = {
            s.strip().upper()
            for s in getattr(config_settings, "gex_symbol_list", [])
            if s.strip()
        }
        prefix = self.settings.gex_snapshot_prefix
        try:
            for key in self.redis_client.client.scan_iter(match=f"{prefix}*"):
                normalized = self._normalize_key(key)
                symbol = normalized.removeprefix(prefix).strip().upper()
                if symbol:
                    symbols.add(symbol)
        except Exception:
            LOGGER.debug("Failed to scan GEX snapshot keys", exc_info=True)
        return symbols

    @staticmethod
    def _parse_dynamic_expiry(raw: Any) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        if isinstance(raw, str):
            try:
                dt_value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
            if dt_value.tzinfo is None:
                dt_value = dt_value.replace(tzinfo=timezone.utc)
            return dt_value.astimezone(timezone.utc)
        return None

    def _load_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = f"{self.settings.gex_snapshot_prefix}{symbol.upper()}"
        payload = self.redis_client.client.get(key)
        if not payload:
            return None
        try:
            decoded = (
                payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
            )
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
        candidate_fields = self._gex_candidate_fields(snapshot)
        max_priors = snapshot.get("max_priors")
        if max_priors is not None:
            try:
                max_priors_str = json.dumps(max_priors)
            except (TypeError, ValueError):
                max_priors_str = None
        else:
            max_priors_str = None
        return {
            "timestamp": epoch_ms,
            "ticker": ticker,
            "spot_price": snapshot.get("spot"),
            "zero_gamma": snapshot.get("zero_gamma"),
            "net_gex": snapshot.get("net_gex"),
            "min_dte": snapshot.get("min_dte"),
            "sec_min_dte": snapshot.get("sec_min_dte"),
            "major_pos_vol": snapshot.get("major_pos_vol"),
            "major_pos_oi": snapshot.get("major_pos_oi"),
            "major_pos_vol_gamma": snapshot.get("major_pos_vol_gamma"),
            "major_neg_vol": snapshot.get("major_neg_vol"),
            "major_neg_oi": snapshot.get("major_neg_oi"),
            "major_neg_vol_gamma": snapshot.get("major_neg_vol_gamma"),
            "sum_gex_vol": snapshot.get("sum_gex_vol"),
            "sum_gex_oi": snapshot.get("sum_gex_oi"),
            "gex_delta_15s": snapshot.get("gex_delta_15s"),
            "delta_risk_reversal": snapshot.get("delta_risk_reversal"),
            "max_priors": max_priors_str,
            "pos_can1_strike": snapshot.get("pos_can1_strike")
            if snapshot.get("pos_can1_strike") is not None
            else candidate_fields.get("pos_can1_strike"),
            "pos_can1_value": snapshot.get("pos_can1_value")
            if snapshot.get("pos_can1_value") is not None
            else candidate_fields.get("pos_can1_value"),
            "pos_can1_pct": snapshot.get("pos_can1_pct")
            if snapshot.get("pos_can1_pct") is not None
            else candidate_fields.get("pos_can1_pct"),
            "pos_can2_strike": snapshot.get("pos_can2_strike")
            if snapshot.get("pos_can2_strike") is not None
            else candidate_fields.get("pos_can2_strike"),
            "pos_can2_value": snapshot.get("pos_can2_value")
            if snapshot.get("pos_can2_value") is not None
            else candidate_fields.get("pos_can2_value"),
            "pos_can2_pct": snapshot.get("pos_can2_pct")
            if snapshot.get("pos_can2_pct") is not None
            else candidate_fields.get("pos_can2_pct"),
            "neg_can1_strike": snapshot.get("neg_can1_strike")
            if snapshot.get("neg_can1_strike") is not None
            else candidate_fields.get("neg_can1_strike"),
            "neg_can1_value": snapshot.get("neg_can1_value")
            if snapshot.get("neg_can1_value") is not None
            else candidate_fields.get("neg_can1_value"),
            "neg_can1_pct": snapshot.get("neg_can1_pct")
            if snapshot.get("neg_can1_pct") is not None
            else candidate_fields.get("neg_can1_pct"),
            "neg_can2_strike": snapshot.get("neg_can2_strike")
            if snapshot.get("neg_can2_strike") is not None
            else candidate_fields.get("neg_can2_strike"),
            "neg_can2_value": snapshot.get("neg_can2_value")
            if snapshot.get("neg_can2_value") is not None
            else candidate_fields.get("neg_can2_value"),
            "neg_can2_pct": snapshot.get("neg_can2_pct")
            if snapshot.get("neg_can2_pct") is not None
            else candidate_fields.get("neg_can2_pct"),
        }

    def _build_strike_rows(
        self, snapshot: Dict[str, Any], epoch_ms: int
    ) -> List[Dict[str, Any]]:
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
                    "timestamp": epoch_ms,
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

    def _write_gex_tables(
        self, snapshots: List[Dict[str, Any]], strikes: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        snapshot_rows = self._dedupe_rows(snapshots, ("ticker", "timestamp"))
        strike_rows = self._dedupe_rows(
            strikes, ("ticker", "timestamp", "strike")
        ) if strikes else []
        df_snapshots = pl.from_dicts(snapshot_rows)
        df_strikes = pl.from_dicts(strike_rows) if strike_rows else None
        conn = duckdb.connect(str(self.settings.gex_snapshot_db))
        self._ensure_gex_snapshot_columns(conn)
        conn.register("gex_snapshots_flush", df_snapshots)
        conn.execute(
            """
            DELETE FROM gex_snapshots
            USING gex_snapshots_flush
            WHERE gex_snapshots.ticker = gex_snapshots_flush.ticker
              AND gex_snapshots.timestamp = gex_snapshots_flush.timestamp
            """
        )
        conn.execute(
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
                gex_delta_15s,
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
                neg_can2_pct,
                strikes
            )
            SELECT
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
                gex_delta_15s,
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
                neg_can2_pct,
                NULL AS strikes
            FROM gex_snapshots_flush
            """
        )
        if df_strikes is not None and not df_strikes.is_empty():
            conn.register("gex_strikes_flush", df_strikes)
            conn.execute(
                """
                DELETE FROM gex_strikes
                USING gex_strikes_flush
                WHERE gex_strikes.ticker = gex_strikes_flush.ticker
                  AND gex_strikes.timestamp = gex_strikes_flush.timestamp
                  AND gex_strikes.strike = gex_strikes_flush.strike
                """
            )
            conn.execute(
                """
                INSERT INTO gex_strikes
                SELECT timestamp, ticker, strike, gamma, oi_gamma, priors
                FROM gex_strikes_flush
                """
            )
        conn.close()
        return len(snapshot_rows), len(strike_rows)

    def _ensure_gex_snapshot_columns(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Add compact candidate columns to legacy gex_snapshots tables."""
        for column in (
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
            "major_pos_vol_gamma",
            "major_neg_vol_gamma",
            "gex_delta_15s",
            "strikes",
        ):
            conn.execute(
                f"ALTER TABLE gex_snapshots ADD COLUMN IF NOT EXISTS {column} "
                f"{'VARCHAR' if column == 'strikes' else 'DOUBLE'}"
            )

    @staticmethod
    def _dedupe_rows(
        rows: List[Dict[str, Any]], key_fields: Tuple[str, ...]
    ) -> List[Dict[str, Any]]:
        """Collapse duplicate rows by key, keeping the last row for each key."""
        deduped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for row in rows:
            key = tuple(row.get(field) for field in key_fields)
            deduped[key] = row
        return list(deduped.values())

    def status(self) -> Dict[str, Any]:
        running = self._task is not None and not self._task.done()
        summary = dict(self._last_summary)
        summary.setdefault("running", running)
        summary.setdefault("samples", 0)
        summary.setdefault("keys", 0)
        summary.setdefault("duration", 0.0)
        summary.setdefault("timestamp", None)
        return summary

    def _flush_uw_messages(self) -> Dict[str, int]:
        """Flush UW messages from Redis to DuckDB."""
        try:
            uw_db_path = config_settings.data_path / "uw_messages.db"
            uw_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Collect market_agg history
            market_agg_count = 0
            market_agg_key = "uw:market_agg:history"
            market_agg_raw = self.redis_client.client.lrange(market_agg_key, 0, -1)
            if market_agg_raw:
                market_agg_records = []
                for raw in market_agg_raw:
                    try:
                        data = json.loads(raw)
                        record = {
                            "received_at": data.get("received_at"),
                            "date": data.get("data", {}).get("date"),
                            "call_premium": data.get("data", {}).get("call_premium"),
                            "put_premium": data.get("data", {}).get("put_premium"),
                            "call_premium_otm_only": data.get("data", {}).get("call_premium_otm_only"),
                            "put_premium_otm_only": data.get("data", {}).get("put_premium_otm_only"),
                            "delta": data.get("data", {}).get("delta"),
                            "gamma": data.get("data", {}).get("gamma"),
                            "theta": data.get("data", {}).get("theta"),
                            "vega": data.get("data", {}).get("vega"),
                        }
                        market_agg_records.append(record)
                    except Exception as e:
                        LOGGER.warning(f"Failed to parse market_agg record: {e}")
                
                if market_agg_records:
                    df = pl.from_dicts(market_agg_records)
                    conn = duckdb.connect(str(uw_db_path))
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS market_agg_state (
                            received_at TIMESTAMP,
                            date VARCHAR,
                            call_premium DOUBLE,
                            put_premium DOUBLE,
                            call_premium_otm_only DOUBLE,
                            put_premium_otm_only DOUBLE,
                            delta DOUBLE,
                            gamma DOUBLE,
                            theta DOUBLE,
                            vega DOUBLE
                        )
                        """
                    )
                    conn.register("market_agg_flush", df)
                    conn.execute("INSERT INTO market_agg_state SELECT * FROM market_agg_flush")
                    conn.close()
                    market_agg_count = len(df)
                    # Clear history after flush
                    self.redis_client.client.delete(market_agg_key)
            
            # Collect option_trade history
            option_trade_count = 0
            option_trade_key = "uw:option_trade:history"
            option_trade_raw = self.redis_client.client.lrange(option_trade_key, 0, -1)
            if option_trade_raw:
                option_trade_records = []
                for raw in option_trade_raw:
                    try:
                        data = json.loads(raw)
                        record = {
                            "received_at": data.get("received_at"),
                            "topic": data.get("topic"),
                            "topic_symbol": data.get("topic_symbol"),
                            "is_index_option": data.get("data", {}).get("is_index_option"),
                            "ticker": data.get("data", {}).get("ticker"),
                            "option_chain_id": data.get("data", {}).get("option_chain_id"),
                            "type": data.get("data", {}).get("type"),
                            "strike": data.get("data", {}).get("strike"),
                            "expiry": data.get("data", {}).get("expiry"),
                            "dte": data.get("data", {}).get("dte"),
                            "cost_basis": data.get("data", {}).get("cost_basis"),
                            "volume": data.get("data", {}).get("volume"),
                            "price": data.get("data", {}).get("price"),
                            "tags": json.dumps(data.get("data", {}).get("tags", [])),
                            "implied_volatility": data.get("data", {}).get("implied_volatility"),
                            "delta": data.get("data", {}).get("delta"),
                            "gamma": data.get("data", {}).get("gamma"),
                            "theta": data.get("data", {}).get("theta"),
                            "vega": data.get("data", {}).get("vega"),
                            "rho": data.get("data", {}).get("rho"),
                            "premium": data.get("data", {}).get("premium"),
                            "size": data.get("data", {}).get("size"),
                            "open_interest": data.get("data", {}).get("open_interest"),
                            "underlying_price": data.get("data", {}).get("underlying_price"),
                        }
                        option_trade_records.append(record)
                    except Exception as e:
                        LOGGER.warning(f"Failed to parse option_trade record: {e}")
                
                if option_trade_records:
                    df = pl.from_dicts(option_trade_records)
                    conn = duckdb.connect(str(uw_db_path))
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS option_trades (
                            received_at TIMESTAMP,
                            topic VARCHAR,
                            topic_symbol VARCHAR,
                            is_index_option BOOLEAN,
                            ticker VARCHAR,
                            option_chain_id VARCHAR,
                            type VARCHAR,
                            strike DOUBLE,
                            expiry TIMESTAMP,
                            dte INTEGER,
                            cost_basis DOUBLE,
                            volume BIGINT,
                            price DOUBLE,
                            tags VARCHAR,
                            implied_volatility DOUBLE,
                            delta DOUBLE,
                            gamma DOUBLE,
                            theta DOUBLE,
                            vega DOUBLE,
                            rho DOUBLE,
                            premium DOUBLE,
                            size BIGINT,
                            open_interest BIGINT,
                            underlying_price DOUBLE
                        )
                        """
                    )
                    # Migrate tables created before IV/greeks columns existed.
                    # CREATE TABLE IF NOT EXISTS is a no-op when the table exists,
                    # so old schemas never pick up new columns without this.
                    for _col, _typ in [
                        ("implied_volatility", "DOUBLE"),
                        ("delta", "DOUBLE"),
                        ("gamma", "DOUBLE"),
                        ("theta", "DOUBLE"),
                        ("vega", "DOUBLE"),
                        ("rho", "DOUBLE"),
                        ("premium", "DOUBLE"),
                        ("size", "BIGINT"),
                        ("open_interest", "BIGINT"),
                        ("underlying_price", "DOUBLE"),
                    ]:
                        conn.execute(
                            f"ALTER TABLE option_trades ADD COLUMN IF NOT EXISTS {_col} {_typ}"
                        )
                    conn.register("option_trade_flush", df)
                    conn.execute("INSERT INTO option_trades SELECT * FROM option_trade_flush")
                    conn.close()
                    option_trade_count = len(df)
                    # Clear history after flush
                    self.redis_client.client.delete(option_trade_key)
            
            LOGGER.info(
                f"Flushed {market_agg_count} market_agg and {option_trade_count} option_trade messages to DuckDB"
            )
            return {
                "uw_market_agg": market_agg_count,
                "uw_option_trades": option_trade_count,
            }
        except Exception:
            LOGGER.exception("Failed to flush UW messages")
            return {"uw_market_agg": 0, "uw_option_trades": 0}
