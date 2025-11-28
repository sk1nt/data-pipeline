"""Lookup helpers for historical trades and depth comparison metadata."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb
import redis

from ..config import settings
from ..lib.redis_client import RedisClient
from .redis_timeseries import RedisTimeSeriesClient


class LookupService:
    """Expose historical lookup helpers backed by Redis and DuckDB."""

    def __init__(
        self,
        redis_client: RedisClient,
        ts_client: RedisTimeSeriesClient,
    ) -> None:
        self.redis_client = redis_client
        self.ts_client = ts_client

    def trade_history(
        self, symbol: str, source: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return the most-recent trade prices for a symbol/source."""
        key = self._trade_price_key(symbol, source)
        try:
            rows = self.ts_client.revrange(key, "+", "-", count=limit)
        except redis.ResponseError:
            return []
        history: List[Dict[str, Any]] = []
        for ts, value in reversed(rows):
            history.append(
                {
                    "symbol": symbol,
                    "source": source,
                    "timestamp": self._format_timestamp(ts),
                    "price": value,
                }
            )
        return history

    def lookup_history(
        self,
        symbol: str,
        limit: int = 100,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query DuckDB tick history for a symbol."""
        db_path = settings.data_path / "tick_data.db"
        if not db_path.exists():
            return []

        params: List[Any] = [symbol]
        filters = ["symbol = ?"]
        if start_time:
            filters.append("timestamp >= ?")
            params.append(self._normalize_iso(start_time))
        if end_time:
            filters.append("timestamp <= ?")
            params.append(self._normalize_iso(end_time))
        query = (
            "SELECT symbol, timestamp, price, volume, source FROM tick_data WHERE "
            + " AND ".join(filters)
        )
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            conn = duckdb.connect(str(db_path))
            rows = conn.execute(query, params).fetchall()
        except duckdb.Error:
            return []
        finally:
            if "conn" in locals():
                conn.close()

        history = []
        for symbol_val, timestamp_val, price_val, volume_val, source_val in rows:
            history.append(
                {
                    "symbol": symbol_val,
                    "timestamp": self._format_timestamp_from_value(timestamp_val),
                    "price": float(price_val) if price_val is not None else None,
                    "volume": int(volume_val) if volume_val is not None else None,
                    "source": source_val,
                }
            )
        return history

    def store_depth_comparison(self, symbol: str, payload: Dict[str, Any]) -> None:
        """Persist latest depth comparison summary for quick lookup."""
        key = self._depth_comparison_key(symbol)
        self.redis_client.client.set(key, json.dumps(payload))

    def get_depth_comparison(self, symbol: str) -> Dict[str, Any]:
        """Read cached depth comparison data."""
        key = self._depth_comparison_key(symbol)
        raw = self.redis_client.client.get(key)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _trade_price_key(symbol: str, source: str) -> str:
        return f"ts:trade:price:{symbol}:{source}"

    @staticmethod
    def _depth_comparison_key(symbol: str) -> str:
        return f"depth:comparison:{symbol}"

    @staticmethod
    def _format_timestamp(value: int) -> str:
        return datetime.utcfromtimestamp(value / 1000).isoformat() + "Z"

    @staticmethod
    def _format_timestamp_from_value(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        try:
            return str(value)
        except Exception:
            return ""

    @staticmethod
    def _normalize_iso(value: str) -> str:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.isoformat()
        except ValueError:
            return value
