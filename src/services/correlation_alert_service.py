"""Service for formatting correlation alerts and persisting them to DuckDB."""

from __future__ import annotations

import json
import logging
import os
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import duckdb

LOGGER = logging.getLogger(__name__)

CORRELATION_DB_PATH = os.getenv(
    "CORRELATION_DB_PATH", "data/correlation_events.db"
)
TABLE_NAME = "correlation_events"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    timestamp              TIMESTAMP NOT NULL,
    social_event_id        VARCHAR NOT NULL,
    social_source          VARCHAR NOT NULL,
    social_author          VARCHAR NOT NULL,
    social_text            VARCHAR,
    social_score           INTEGER NOT NULL,
    social_url             VARCHAR,
    alert_type             VARCHAR,
    alert_fired            BOOLEAN NOT NULL DEFAULT FALSE,
    signals_triggered      VARCHAR,
    volume_ratio           DOUBLE,
    gex_change_pct         DOUBLE,
    price_change_pct       DOUBLE,
    config_snapshot        VARCHAR,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Realized market impact (back-filled by MarketMoverAnalyzer)
    realized_impact_score  DOUBLE,
    price_t0               DOUBLE,
    price_t15              DOUBLE,
    price_ticker           VARCHAR,
    is_noise               BOOLEAN
);
"""

# Columns added after the initial schema; safely ignored if already present.
_MIGRATE_COLUMNS = [
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS realized_impact_score DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_t0              DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_t15             DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_ticker          VARCHAR",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS is_noise              BOOLEAN",
]


class CorrelationAlertService:
    """Format correlation alerts for Discord and persist to DuckDB."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or CORRELATION_DB_PATH
        self._ensure_table()

    def _ensure_dir(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _ensure_table(self) -> None:
        try:
            self._ensure_dir()
            with closing(duckdb.connect(self.db_path)) as conn:
                conn.execute(_CREATE_TABLE_SQL)
                for stmt in _MIGRATE_COLUMNS:
                    try:
                        conn.execute(stmt)
                    except Exception:
                        pass  # column already exists or not supported — safe to ignore
        except Exception:
            LOGGER.exception("Failed to create correlation_events table")

    def format_alert_message(self, alert: Dict[str, Any]) -> str:
        """Format a correlation alert dict into a Discord-safe message string."""
        alert_type = alert.get("alert_type", "unknown")
        message = alert.get("message", "")
        severity = alert.get("severity", "medium")
        signals = alert.get("signals_triggered", [])
        social = alert.get("social_event", {})
        timestamp = alert.get("timestamp", "")

        # Severity prefix
        if severity == "high":
            header = "🔴 **HIGH PRIORITY CORRELATION ALERT**"
        else:
            header = "🟡 **CORRELATION ALERT**"

        # Sanitize social text for Discord (strip markdown injection, limit length)
        social_text = self._sanitize_text(social.get("text", ""))
        author = self._sanitize_text(social.get("author", "unknown"))
        source = social.get("source", "unknown")

        parts = [
            header,
            f"**Source**: {source} — {author}",
            f"**Type**: `{alert_type}` | **Signals**: {', '.join(f'`{s}`' for s in signals)}",
            "",
            message,
            "",
            f"_Event time: {timestamp}_",
        ]

        return "\n".join(parts)

    def log_correlation_event(
        self,
        alert: Dict[str, Any],
        alert_fired: bool = True,
        *,
        realized_impact_score: Optional[float] = None,
        price_t0: Optional[float] = None,
        price_t15: Optional[float] = None,
        price_ticker: Optional[str] = None,
        is_noise: Optional[bool] = None,
    ) -> None:
        """Persist a correlation event (alert or no-alert) to DuckDB.

        The ``realized_impact_*`` keyword arguments are populated asynchronously
        by :class:`~src.services.market_mover_analyzer.MarketMoverAnalyzer` once
        enough post-event market data exists.  They default to NULL on initial
        insert and can be back-filled via :meth:`backfill_realized_impact`.
        """
        try:
            social = alert.get("social_event", {})
            signals = alert.get("market_signals", {})

            row = {
                "timestamp": alert.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "social_event_id": social.get("event_id", ""),
                "social_source": social.get("source", ""),
                "social_author": social.get("author", ""),
                "social_text": (social.get("text", "") or "")[:500],
                "social_score": social.get("relevance_score", 0),
                "social_url": social.get("url"),
                "alert_type": alert.get("alert_type"),
                "alert_fired": alert_fired,
                "signals_triggered": json.dumps(alert.get("signals_triggered", [])),
                "volume_ratio": signals.get("volume_ratio"),
                "gex_change_pct": signals.get("gex_change_pct"),
                "price_change_pct": signals.get("price_change_pct"),
                "config_snapshot": None,
                "realized_impact_score": realized_impact_score,
                "price_t0": price_t0,
                "price_t15": price_t15,
                "price_ticker": price_ticker,
                "is_noise": is_noise,
            }

            self._insert_row(row)
        except Exception:
            LOGGER.exception("Failed to log correlation event")

    def backfill_realized_impact(
        self,
        social_event_id: str,
        realized_impact_score: float,
        price_t0: Optional[float],
        price_t15: Optional[float],
        price_ticker: str,
        is_noise: bool,
    ) -> None:
        """Update realized impact columns for an event already in the DB."""
        try:
            self._ensure_dir()
            with closing(duckdb.connect(self.db_path)) as conn:
                conn.execute(
                    f"""
                    UPDATE {TABLE_NAME}
                    SET realized_impact_score = ?,
                        price_t0             = ?,
                        price_t15            = ?,
                        price_ticker         = ?,
                        is_noise             = ?
                    WHERE social_event_id = ?
                    """,
                    [
                        realized_impact_score, price_t0, price_t15,
                        price_ticker, is_noise, social_event_id,
                    ],
                )
        except Exception:
            LOGGER.exception(
                "Failed to backfill realized impact for event %s", social_event_id
            )

    def query_events(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        source: Optional[str] = None,
        alert_fired_only: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query historical correlation events."""
        conditions = []
        params = []

        if start:
            conditions.append("timestamp >= ?")
            params.append(start.isoformat())
        if end:
            conditions.append("timestamp <= ?")
            params.append(end.isoformat())
        if source:
            conditions.append("social_source = ?")
            params.append(source)
        if alert_fired_only:
            conditions.append("alert_fired = true")

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT * FROM {TABLE_NAME}{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        try:
            self._ensure_dir()
            with closing(duckdb.connect(self.db_path)) as conn:
                result = conn.execute(query, params)
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception:
            LOGGER.exception("Failed to query correlation events")
            return []

    def _insert_row(self, row: Dict[str, Any]) -> None:
        columns = list(row.keys())
        placeholders = ", ".join(["?" for _ in columns])
        col_names = ", ".join(columns)
        query = f"INSERT INTO {TABLE_NAME} ({col_names}) VALUES ({placeholders})"
        values = tuple(row[c] for c in columns)

        self._ensure_dir()
        with closing(duckdb.connect(self.db_path)) as conn:
            conn.execute(query, values)

    @staticmethod
    def _sanitize_text(text: str, max_length: int = 200) -> str:
        """Strip potentially dangerous markdown/mentions and truncate."""
        # Remove @everyone/@here to prevent Discord pings
        sanitized = text.replace("@everyone", "@ everyone").replace("@here", "@ here")
        # Escape backticks to prevent code block injection
        sanitized = sanitized.replace("`", "'")
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "…"
        return sanitized
