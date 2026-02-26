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
    timestamp       TIMESTAMP NOT NULL,
    social_event_id VARCHAR NOT NULL,
    social_source   VARCHAR NOT NULL,
    social_author   VARCHAR NOT NULL,
    social_text     VARCHAR,
    social_score    INTEGER NOT NULL,
    social_url      VARCHAR,
    alert_type      VARCHAR,
    alert_fired     BOOLEAN NOT NULL DEFAULT FALSE,
    signals_triggered VARCHAR,
    volume_ratio    DOUBLE,
    gex_change_pct  DOUBLE,
    price_change_pct DOUBLE,
    uw_ratio_change DOUBLE,
    config_snapshot VARCHAR,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


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
    ) -> None:
        """Persist a correlation event (alert or no-alert) to DuckDB."""
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
                "uw_ratio_change": None,
                "config_snapshot": None,
            }

            # Compute UW ratio change if available
            uw_cur = signals.get("uw_put_call_ratio")
            uw_prev = signals.get("uw_prev_ratio")
            if uw_cur is not None and uw_prev is not None and uw_prev != 0:
                row["uw_ratio_change"] = (float(uw_cur) - float(uw_prev)) / float(uw_prev)

            self._insert_row(row)
        except Exception:
            LOGGER.exception("Failed to log correlation event")

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
