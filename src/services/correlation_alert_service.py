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
    -- Classification (populated at insert by EventCategorizer)
    event_category         VARCHAR,
    category_severity      VARCHAR,
    source_type            VARCHAR,
    text_fingerprint       VARCHAR,
    is_first_mention       BOOLEAN,
    first_mention_id       VARCHAR,
    credibility_rank       DOUBLE,
    -- GEX regime at alert time (populated at insert)
    gex_regime             VARCHAR,
    net_gex_at_alert       DOUBLE,
    -- Realized market impact (back-filled by MarketMoverAnalyzer)
    realized_impact_score  DOUBLE,
    price_ticker           VARCHAR,
    is_noise               BOOLEAN,
    -- GEX spot prices from timeseries parquet (back-filled)
    price_t0               DOUBLE,
    price_t5               DOUBLE,
    price_t15              DOUBLE,
    price_t30              DOUBLE,
    price_t60              DOUBLE,
    -- High-resolution tick prices from MNQ tick parquet (back-filled)
    tick_price_t0          DOUBLE,
    tick_price_t5          DOUBLE,
    tick_price_t15         DOUBLE,
    tick_price_t30         DOUBLE,
    tick_price_t60         DOUBLE,
    -- Move summaries (back-filled)
    tick_move_t5_pct       DOUBLE,
    tick_move_t15_pct      DOUBLE,
    tick_move_t30_pct      DOUBLE,
    tick_move_t60_pct      DOUBLE,
    tick_move_direction    VARCHAR
);
"""

# Columns added after the initial schema; safely ignored if already present.
_MIGRATE_COLUMNS = [
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS realized_impact_score DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_t0              DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_t15             DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_ticker          VARCHAR",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS is_noise              BOOLEAN",
    # Phase 2: classification + multi-horizon tick data
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS event_category        VARCHAR",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS category_severity     VARCHAR",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS source_type           VARCHAR",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS text_fingerprint      VARCHAR",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS is_first_mention      BOOLEAN",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS first_mention_id      VARCHAR",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS credibility_rank      DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS gex_regime            VARCHAR",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS net_gex_at_alert      DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_t5              DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_t30             DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS price_t60             DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_price_t0         DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_price_t5         DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_price_t15        DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_price_t30        DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_price_t60        DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_move_t5_pct      DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_move_t15_pct     DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_move_t30_pct     DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_move_t60_pct     DOUBLE",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS tick_move_direction   VARCHAR",
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
        # Phase 2: classification fields (from EventCategorizer)
        event_category: Optional[str] = None,
        category_severity: Optional[str] = None,
        source_type: Optional[str] = None,
        text_fingerprint: Optional[str] = None,
        is_first_mention: Optional[bool] = None,
        first_mention_id: Optional[str] = None,
        credibility_rank: Optional[float] = None,
        # Phase 2: GEX regime at alert time
        gex_regime: Optional[str] = None,
        net_gex_at_alert: Optional[float] = None,
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
                # Classification
                "event_category": event_category or alert.get("event_category"),
                "category_severity": category_severity or alert.get("category_severity"),
                "source_type": source_type or alert.get("source_type"),
                "text_fingerprint": text_fingerprint or alert.get("text_fingerprint"),
                "is_first_mention": is_first_mention,
                "first_mention_id": first_mention_id,
                "credibility_rank": credibility_rank,
                "gex_regime": gex_regime or alert.get("gex_regime"),
                "net_gex_at_alert": net_gex_at_alert,
                # Realized impact (back-filled later)
                "realized_impact_score": realized_impact_score,
                "price_ticker": price_ticker,
                "is_noise": is_noise,
                "price_t0": price_t0,
                "price_t5": None,
                "price_t15": price_t15,
                "price_t30": None,
                "price_t60": None,
                "tick_price_t0": None,
                "tick_price_t5": None,
                "tick_price_t15": None,
                "tick_price_t30": None,
                "tick_price_t60": None,
                "tick_move_t5_pct": None,
                "tick_move_t15_pct": None,
                "tick_move_t30_pct": None,
                "tick_move_t60_pct": None,
                "tick_move_direction": None,
            }

            self._insert_row(row)
        except Exception:
            LOGGER.exception("Failed to log correlation event")

    def backfill_realized_impact(
        self,
        social_event_id: str,
        realized_impact_score: float,
        price_ticker: str,
        is_noise: bool,
        price_t0: Optional[float] = None,
        price_t5: Optional[float] = None,
        price_t15: Optional[float] = None,
        price_t30: Optional[float] = None,
        price_t60: Optional[float] = None,
        tick_price_t0: Optional[float] = None,
        tick_price_t5: Optional[float] = None,
        tick_price_t15: Optional[float] = None,
        tick_price_t30: Optional[float] = None,
        tick_price_t60: Optional[float] = None,
        tick_move_t5_pct: Optional[float] = None,
        tick_move_t15_pct: Optional[float] = None,
        tick_move_t30_pct: Optional[float] = None,
        tick_move_t60_pct: Optional[float] = None,
        tick_move_direction: Optional[str] = None,
    ) -> None:
        """Update realized impact columns for an event already in the DB."""
        try:
            self._ensure_dir()
            with closing(duckdb.connect(self.db_path)) as conn:
                conn.execute(
                    f"""
                    UPDATE {TABLE_NAME}
                    SET realized_impact_score = ?,
                        price_ticker         = ?,
                        is_noise             = ?,
                        price_t0             = ?,
                        price_t5             = ?,
                        price_t15            = ?,
                        price_t30            = ?,
                        price_t60            = ?,
                        tick_price_t0        = ?,
                        tick_price_t5        = ?,
                        tick_price_t15       = ?,
                        tick_price_t30       = ?,
                        tick_price_t60       = ?,
                        tick_move_t5_pct     = ?,
                        tick_move_t15_pct    = ?,
                        tick_move_t30_pct    = ?,
                        tick_move_t60_pct    = ?,
                        tick_move_direction  = ?
                    WHERE social_event_id = ?
                    """,
                    [
                        realized_impact_score, price_ticker, is_noise,
                        price_t0, price_t5, price_t15, price_t30, price_t60,
                        tick_price_t0, tick_price_t5, tick_price_t15,
                        tick_price_t30, tick_price_t60,
                        tick_move_t5_pct, tick_move_t15_pct,
                        tick_move_t30_pct, tick_move_t60_pct,
                        tick_move_direction,
                        social_event_id,
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
