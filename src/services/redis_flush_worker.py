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
import pandas as pd
import redis

from ..config import settings as config_settings
from lib.redis_client import RedisClient
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
        df = pd.DataFrame(new_records, columns=["key", "ts", "value"])
        df["key"] = df["key"].apply(self._normalize_key)
        df["day"] = pd.to_datetime(df["ts"], unit="ms").dt.date
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
        conn.execute(
            "INSERT INTO redis_timeseries SELECT key, ts, value, day FROM flush_df"
        )
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

    def _write_tick_outputs(self, df: pd.DataFrame) -> None:
        if df.empty or "key" not in df:
            return
        tick_df = df[df["key"].str.startswith("ts:trade:")].copy()
        if tick_df.empty:
            return
        parts = tick_df["key"].str.split(":", expand=True)
        if parts.shape[1] < 5:
            return
        tick_df["metric"] = parts[2]
        tick_df["symbol"] = parts[3].str.upper()
        tick_df["source"] = parts[4].str.upper()
        pivot = tick_df.pivot_table(
            index=["symbol", "source", "ts"],
            columns="metric",
            values="value",
            aggfunc="last",
        ).reset_index()
        if pivot.empty:
            return
        pivot.columns = [
            col if isinstance(col, str) else col[0] for col in pivot.columns
        ]
        if "price" not in pivot.columns:
            return
        before_drop = len(pivot)
        pivot = pivot.dropna(subset=["price"])
        if pivot.empty:
            LOGGER.debug(
                "Skip tick flush slice due to missing price column values (before=%s)",
                before_drop,
            )
            return
        pivot["size"] = pivot.get("size", 0.0).fillna(0.0)
        pivot["timestamp"] = pd.to_datetime(pivot["ts"], unit="ms", utc=True)
        pivot["day"] = pivot["timestamp"].dt.strftime("%Y%m%d")
        parquet_df = pivot[
            ["symbol", "source", "timestamp", "ts", "price", "size", "day"]
        ].rename(columns={"ts": "timestamp_ms"})
        self._append_tick_parquet(parquet_df)
        self._insert_tick_rows(parquet_df)

    def _write_depth_outputs(self, df: pd.DataFrame) -> None:
        if df.empty or "key" not in df:
            return
        depth_df = df[df["key"].str.startswith("ts:depth:")].copy()
        if depth_df.empty:
            return
        parts = depth_df["key"].str.split(":", expand=True)
        if parts.shape[1] < 7:
            return
        depth_df["symbol"] = parts[2].str.upper()
        depth_df["source"] = parts[3].str.upper()
        depth_df["side"] = parts[4]
        depth_df["level"] = pd.to_numeric(parts[5], errors="coerce").astype("Int64")
        depth_df["field"] = parts[6]
        depth_df = depth_df.dropna(subset=["level"])
        if depth_df.empty:
            return
        pivot = depth_df.pivot_table(
            index=["symbol", "source", "ts", "side", "level"],
            columns="field",
            values="value",
            aggfunc="last",
        ).reset_index()
        if pivot.empty:
            return
        pivot.columns = [
            col if isinstance(col, str) else col[0] for col in pivot.columns
        ]
        pivot["price"] = pivot.get("price", pd.NA)
        pivot["size"] = pivot.get("size", pd.NA)
        pivot["timestamp"] = pd.to_datetime(pivot["ts"], unit="ms", utc=True)
        pivot["day"] = pivot["timestamp"].dt.strftime("%Y%m%d")
        parquet_df = pivot[
            [
                "symbol",
                "source",
                "side",
                "level",
                "timestamp",
                "ts",
                "price",
                "size",
                "day",
            ]
        ].rename(columns={"ts": "timestamp_ms"})
        self._append_depth_parquet(parquet_df)

    def _append_tick_parquet(self, parquet_df: pd.DataFrame) -> None:
        if parquet_df.empty:
            return
        manifests: List[Tuple[str, str, str, str, int, datetime, datetime]] = []
        parquet_dir = self.settings.tick_parquet_dir
        parquet_dir.mkdir(parents=True, exist_ok=True)
        for (symbol, day), group in parquet_df.groupby(["symbol", "day"]):
            safe_symbol = self._sanitize_symbol_dir(symbol)
            dest = parquet_dir / safe_symbol / f"{day}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            subset = group.drop(columns=["day"]).sort_values("timestamp")
            subset.to_parquet(dest, index=False)
            manifests.append(
                (
                    symbol,
                    ",".join(sorted(set(subset["source"]))),
                    day,
                    dest.as_posix(),
                    len(subset),
                    subset["timestamp"].iloc[0].to_pydatetime(),
                    subset["timestamp"].iloc[-1].to_pydatetime(),
                )
            )
        self._update_tick_manifest(manifests)

    def _append_depth_parquet(self, parquet_df: pd.DataFrame) -> None:
        if parquet_df.empty:
            return
        manifests: List[Tuple[str, str, str, str, int, datetime, datetime]] = []
        parquet_dir = self.settings.depth_parquet_dir
        parquet_dir.mkdir(parents=True, exist_ok=True)
        for (symbol, day), group in parquet_df.groupby(["symbol", "day"]):
            safe_symbol = self._sanitize_symbol_dir(symbol)
            dest = parquet_dir / safe_symbol / f"{day}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            subset = group.drop(columns=["day"]).sort_values(
                ["timestamp", "side", "level"]
            )
            subset.to_parquet(dest, index=False)
            manifests.append(
                (
                    symbol,
                    ",".join(sorted(set(subset["source"]))),
                    day,
                    dest.as_posix(),
                    len(subset),
                    subset["timestamp"].iloc[0].to_pydatetime(),
                    subset["timestamp"].iloc[-1].to_pydatetime(),
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

    def _insert_tick_rows(self, parquet_df: pd.DataFrame) -> None:
        if parquet_df.empty:
            return
        rows = parquet_df.copy()
        rows["volume"] = rows["size"].fillna(0).round().astype(int)
        rows["tick_type"] = "trade"
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
            rows[["symbol", "timestamp", "price", "volume", "tick_type", "source"]],
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
                strike_rows.extend(
                    self._build_strike_rows(snapshot, snapshot_row["timestamp"])
                )
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
        symbols = {
            s.strip().upper()
            for s in getattr(config_settings, "gex_symbol_list", [])
            if s.strip()
        }
        # Dynamic symbol enrollment no longer used — only use configured symbol list
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
            "major_neg_vol": snapshot.get("major_neg_vol"),
            "major_neg_oi": snapshot.get("major_neg_oi"),
            "sum_gex_vol": snapshot.get("sum_gex_vol"),
            "sum_gex_oi": snapshot.get("sum_gex_oi"),
            "delta_risk_reversal": snapshot.get("delta_risk_reversal"),
            "max_priors": max_priors_str,
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
    ) -> None:
        df_snapshots = pd.DataFrame(snapshots)
        df_strikes = pd.DataFrame(strikes) if strikes else None
        conn = duckdb.connect(str(self.settings.gex_snapshot_db))
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
            INSERT INTO gex_snapshots
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
            from pathlib import Path
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
                    df = pd.DataFrame(market_agg_records)
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
                        }
                        option_trade_records.append(record)
                    except Exception as e:
                        LOGGER.warning(f"Failed to parse option_trade record: {e}")
                
                if option_trade_records:
                    df = pd.DataFrame(option_trade_records)
                    conn = duckdb.connect(str(uw_db_path))
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS option_trades (
                            received_at TIMESTAMP,
                            topic VARCHAR,
                            topic_symbol VARCHAR,
                            is_index_option BOOLEAN,
                            ticker VARCHAR,
                            option_chain_id BIGINT,
                            type VARCHAR,
                            strike DOUBLE,
                            expiry TIMESTAMP,
                            dte INTEGER,
                            cost_basis DOUBLE,
                            volume BIGINT,
                            price DOUBLE,
                            tags VARCHAR
                        )
                        """
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
