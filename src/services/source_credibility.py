"""Source credibility service — tracks per-author/outlet historical move impact
and assigns a credibility rank used to weight alert priority.

How it works
------------
Every time a correlation event is back-filled with a realized_impact_score the
score is attributed to the source (social_author) that produced it.  Over time
each source accumulates:

  - total_events         : number of events attributed
  - avg_impact           : rolling average realized_impact_score
  - high_impact_count    : events with score >= HIGH_IMPACT_THRESHOLD
  - noise_count          : events flagged is_noise = True
  - signal_ratio         : high_impact_count / total_events
  - credibility_rank     : 0–100 composite (higher = more trustworthy moves)
  - source_type          : institution / executive / media / government / social / unknown
  - category_breakdown   : {category: count} — what topics this source covers

Storage
-------
Persisted to the same DuckDB file as correlation_events (separate table:
source_credibility).  Also cached in Redis as source_credibility:{author_key}
(JSON, 1-hour TTL) for fast lookup during real-time alert enrichment.

Credibility rank formula
------------------------
  rank = (signal_ratio * 60) + (avg_impact / 100 * 30) + (log(total_events+1) / log(50) * 10)

  - 60 pts for signal_ratio (fraction of events that moved the market)
  - 30 pts for average realized impact
  - 10 pts for volume (more events = more data = higher confidence)
  All capped at 100.
"""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import duckdb

from src.lib.redis_client import RedisClient

LOGGER = logging.getLogger(__name__)

CORRELATION_DB_PATH = os.getenv("CORRELATION_DB_PATH", "data/correlation_events.db")
SOURCE_TABLE = "source_credibility"
REDIS_KEY_PREFIX = "source_credibility:"
REDIS_TTL = 3600  # 1 hour

HIGH_IMPACT_THRESHOLD = 15.0   # realized_impact_score >= this counts as a "move"
MIN_EVENTS_FOR_RANK   = 3      # need at least this many events before rank is meaningful

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {SOURCE_TABLE} (
    author_key           VARCHAR PRIMARY KEY,
    source_type          VARCHAR NOT NULL DEFAULT 'unknown',
    display_name         VARCHAR NOT NULL,
    total_events         INTEGER NOT NULL DEFAULT 0,
    avg_impact           DOUBLE  NOT NULL DEFAULT 0.0,
    high_impact_count    INTEGER NOT NULL DEFAULT 0,
    noise_count          INTEGER NOT NULL DEFAULT 0,
    signal_ratio         DOUBLE  NOT NULL DEFAULT 0.0,
    credibility_rank     DOUBLE  NOT NULL DEFAULT 0.0,
    category_breakdown   VARCHAR NOT NULL DEFAULT '{{}}',
    first_seen           TIMESTAMP,
    last_seen            TIMESTAMP,
    last_updated         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);
"""


def _author_key(author: str) -> str:
    """Normalise author to a stable lookup key."""
    return author.lower().strip().replace(" ", "_")[:80]


def _compute_rank(
    total_events: int,
    avg_impact: float,
    high_impact_count: int,
    noise_count: int,
) -> float:
    non_noise = max(total_events - noise_count, 0)
    signal_ratio = high_impact_count / non_noise if non_noise > 0 else 0.0
    volume_score = math.log(total_events + 1) / math.log(50) if total_events > 0 else 0.0
    rank = (signal_ratio * 60) + (min(avg_impact, 100) / 100 * 30) + (min(volume_score, 1.0) * 10)
    return round(min(rank, 100.0), 2)


class SourceCredibilityService:
    """Track and query per-author move-credibility scores."""

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.redis = redis_client
        self.db_path = db_path or CORRELATION_DB_PATH
        self._ensure_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rank(self, author: str, source_type: str = "unknown") -> Dict[str, Any]:
        """Return credibility record for an author (fast Redis → DB fallback).

        Always returns a dict even if the author has no history (defaults to
        credibility_rank=0, signal_ratio=0).
        """
        key = _author_key(author)

        # 1) Redis cache
        if self.redis:
            try:
                cached = self.redis.client.get(f"{REDIS_KEY_PREFIX}{key}")
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        # 2) DuckDB
        record = self._fetch_from_db(key)
        if record:
            self._cache(key, record)
            return record

        # 3) Brand-new author — return defaults
        return {
            "author_key": key,
            "display_name": author,
            "source_type": source_type,
            "total_events": 0,
            "avg_impact": 0.0,
            "high_impact_count": 0,
            "noise_count": 0,
            "signal_ratio": 0.0,
            "credibility_rank": 0.0,
            "category_breakdown": {},
            "sufficient_data": False,
        }

    def record_event_impact(
        self,
        author: str,
        source_type: str,
        realized_impact: float,
        is_noise: bool,
        category: str = "uncategorized",
        event_time: Optional[datetime] = None,
    ) -> None:
        """Upsert the author's credibility record after a back-fill."""
        key = _author_key(author)
        now = event_time or datetime.now(timezone.utc)

        try:
            self._ensure_dir()
            with closing(duckdb.connect(self.db_path)) as conn:
                existing = conn.execute(
                    f"SELECT * FROM {SOURCE_TABLE} WHERE author_key = ?", [key]
                ).fetchone()

                if existing:
                    cols = [d[0] for d in conn.execute(f"DESCRIBE {SOURCE_TABLE}").fetchall()]
                    rec = dict(zip(cols, existing))
                    total = rec["total_events"] + 1
                    avg = (rec["avg_impact"] * rec["total_events"] + realized_impact) / total
                    hi  = rec["high_impact_count"] + (1 if realized_impact >= HIGH_IMPACT_THRESHOLD else 0)
                    nc  = rec["noise_count"] + (1 if is_noise else 0)
                    try:
                        cat_bd = json.loads(rec.get("category_breakdown") or "{}")
                    except Exception:
                        cat_bd = {}
                    cat_bd[category] = cat_bd.get(category, 0) + 1
                    rank = _compute_rank(total, avg, hi, nc)
                    conn.execute(
                        f"""UPDATE {SOURCE_TABLE} SET
                            total_events      = ?,
                            avg_impact        = ?,
                            high_impact_count = ?,
                            noise_count       = ?,
                            signal_ratio      = ?,
                            credibility_rank  = ?,
                            category_breakdown = ?,
                            last_seen         = ?,
                            last_updated      = CURRENT_TIMESTAMP
                        WHERE author_key = ?""",
                        [
                            total, round(avg, 3), hi, nc,
                            round(hi / max(total - nc, 1), 4),
                            rank,
                            json.dumps(cat_bd),
                            now.isoformat(),
                            key,
                        ],
                    )
                else:
                    hi   = 1 if realized_impact >= HIGH_IMPACT_THRESHOLD else 0
                    nc   = 1 if is_noise else 0
                    rank = _compute_rank(1, realized_impact, hi, nc)
                    conn.execute(
                        f"""INSERT INTO {SOURCE_TABLE}
                            (author_key, source_type, display_name, total_events,
                             avg_impact, high_impact_count, noise_count, signal_ratio,
                             credibility_rank, category_breakdown, first_seen, last_seen)
                        VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            key, source_type, author, round(realized_impact, 3),
                            hi, nc,
                            round(hi / max(1 - nc, 1), 4),
                            rank,
                            json.dumps({category: 1}),
                            now.isoformat(), now.isoformat(),
                        ],
                    )

        except Exception:
            LOGGER.exception("Failed to record event impact for author %s", author)
            return

        # Invalidate Redis cache
        if self.redis:
            try:
                self.redis.client.delete(f"{REDIS_KEY_PREFIX}{key}")
            except Exception:
                pass

    def top_sources(
        self,
        limit: int = 50,
        source_type: Optional[str] = None,
        min_events: int = MIN_EVENTS_FOR_RANK,
    ) -> List[Dict[str, Any]]:
        """Return the most credible sources ranked by credibility_rank desc."""
        try:
            self._ensure_dir()
            conditions = [f"total_events >= {min_events}"]
            params: List[Any] = []
            if source_type:
                conditions.append("source_type = ?")
                params.append(source_type)
            where = " WHERE " + " AND ".join(conditions)
            with closing(duckdb.connect(self.db_path, read_only=True)) as conn:
                rows = conn.execute(
                    f"SELECT * FROM {SOURCE_TABLE}{where} ORDER BY credibility_rank DESC LIMIT {limit}",
                    params,
                ).fetchall()
                cols = [d[0] for d in conn.execute(f"DESCRIBE {SOURCE_TABLE}").fetchall()]
                return [dict(zip(cols, row)) for row in rows]
        except Exception:
            LOGGER.exception("Failed to query top sources")
            return []

    def rebuild_from_correlation_events(self) -> int:
        """Recompute all source credibility records from the full correlation_events table.

        Useful on first run or after schema migration.  Returns number of authors updated.
        """
        try:
            self._ensure_dir()
            with closing(duckdb.connect(self.db_path)) as conn:
                rows = conn.execute("""
                    SELECT
                        social_author,
                        social_source,
                        COUNT(*)                                           AS total,
                        AVG(COALESCE(realized_impact_score, 0))            AS avg_imp,
                        SUM(CASE WHEN realized_impact_score >= 15 THEN 1 ELSE 0 END) AS hi,
                        SUM(CASE WHEN is_noise = true THEN 1 ELSE 0 END)  AS nc,
                        MIN(timestamp)                                     AS first_seen,
                        MAX(timestamp)                                     AS last_seen
                    FROM correlation_events
                    WHERE social_author IS NOT NULL AND social_author != ''
                    GROUP BY social_author, social_source
                """).fetchall()

                updated = 0
                for row in rows:
                    author, stype, total, avg_imp, hi, nc, first_seen, last_seen = row
                    avg_imp = avg_imp or 0.0
                    hi = hi or 0
                    nc = nc or 0
                    key = _author_key(author)
                    rank = _compute_rank(total, avg_imp, hi, nc)
                    conn.execute(
                        f"""INSERT OR REPLACE INTO {SOURCE_TABLE}
                            (author_key, source_type, display_name, total_events,
                             avg_impact, high_impact_count, noise_count, signal_ratio,
                             credibility_rank, category_breakdown, first_seen, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{{}}', ?, ?)""",
                        [
                            key, stype or "unknown", author, total,
                            round(avg_imp, 3), hi, nc,
                            round(hi / max(total - nc, 1), 4),
                            rank,
                            str(first_seen), str(last_seen),
                        ],
                    )
                    updated += 1
                return updated
        except Exception:
            LOGGER.exception("Failed to rebuild source credibility")
            return 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _ensure_table(self) -> None:
        try:
            self._ensure_dir()
            with closing(duckdb.connect(self.db_path)) as conn:
                conn.execute(_CREATE_TABLE)
        except Exception:
            LOGGER.exception("Failed to create %s table", SOURCE_TABLE)

    def _fetch_from_db(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            self._ensure_dir()
            with closing(duckdb.connect(self.db_path, read_only=True)) as conn:
                row = conn.execute(
                    f"SELECT * FROM {SOURCE_TABLE} WHERE author_key = ?", [key]
                ).fetchone()
                if row:
                    cols = [d[0] for d in conn.execute(f"DESCRIBE {SOURCE_TABLE}").fetchall()]
                    rec = dict(zip(cols, row))
                    try:
                        rec["category_breakdown"] = json.loads(rec.get("category_breakdown") or "{}")
                    except Exception:
                        rec["category_breakdown"] = {}
                    rec["sufficient_data"] = rec["total_events"] >= MIN_EVENTS_FOR_RANK
                    return rec
        except Exception:
            LOGGER.debug("DB lookup failed for author %s", key, exc_info=True)
        return None

    def _cache(self, key: str, record: Dict[str, Any]) -> None:
        if not self.redis:
            return
        try:
            self.redis.client.setex(
                f"{REDIS_KEY_PREFIX}{key}",
                REDIS_TTL,
                json.dumps(record, default=str),
            )
        except Exception:
            pass
