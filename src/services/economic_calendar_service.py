"""Economic calendar service — fetches upcoming high-impact events and fires pre-event
warnings plus post-release result alerts through the correlation alert channel.

Data source: Forex Factory JSON feed (no API key required).
Alert types published to CORRELATION_ALERT_CHANNEL:
  - "calendar_warning"  — T-30, T-10, T-2 min before a high-impact event
  - "calendar_result"   — actual vs forecast vs previous after release
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, Field

from src.lib.redis_client import RedisClient
from src.services.correlation_engine import CORRELATION_ALERT_CHANNEL

LOGGER = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

CALENDAR_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CALENDAR_REDIS_KEY = "calendar:events:today"
CALENDAR_REDIS_TTL = 3600  # 1 hour cache

# Only alert on these impact levels
HIGH_IMPACT_LEVELS = {"High", "Medium"}
# Warning thresholds in minutes before event
WARNING_MINUTES = [30, 10, 2]
# Poll interval for checking warnings (seconds)
POLL_INTERVAL = 30
# How often to refresh the calendar feed (seconds)
FEED_REFRESH_INTERVAL = 1800  # 30 min

# High-impact event names to always include regardless of impact field
ALWAYS_WATCH = {
    "cpi", "core cpi", "pce", "core pce",
    "fomc", "fed rate", "interest rate decision",
    "nonfarm", "nfp", "jobs", "unemployment",
    "gdp", "ism", "pmi", "retail sales",
    "jobless claims", "initial claims",
    "producer price", "ppi",
    "consumer confidence", "consumer sentiment",
    "ism manufacturing", "ism services",
    "trade balance", "current account",
    "housing starts", "building permits",
    "durable goods",
}


class CalendarEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    country: str
    event_time: datetime  # UTC
    impact: str  # High / Medium / Low
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None


class CalendarAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    alert_type: str  # "calendar_warning" or "calendar_result"
    severity: str = "high"
    event_title: str
    event_time: datetime
    minutes_until: Optional[int] = None  # for warnings
    actual: Optional[str] = None         # for results
    forecast: Optional[str] = None
    previous: Optional[str] = None
    surprise: Optional[str] = None       # "beat" / "miss" / "inline"
    message: str = ""


def _is_high_impact(event: CalendarEvent) -> bool:
    if event.country.upper() != "USD":
        return False
    if event.impact in HIGH_IMPACT_LEVELS:
        return True
    title_lower = event.title.lower()
    return any(kw in title_lower for kw in ALWAYS_WATCH)


def _parse_ff_events(raw: List[Dict[str, Any]]) -> List[CalendarEvent]:
    events: List[CalendarEvent] = []
    for item in raw:
        try:
            # Forex Factory timestamps: "04-30-2026T08:30:00-04:00" (local ET offset)
            ts_str = item.get("date", "")
            if not ts_str:
                continue
            try:
                event_time = datetime.fromisoformat(ts_str).astimezone(timezone.utc)
            except ValueError:
                continue
            events.append(CalendarEvent(
                title=item.get("title", "Unknown"),
                country=item.get("country", ""),
                event_time=event_time,
                impact=item.get("impact", "Low"),
                forecast=item.get("forecast") or None,
                previous=item.get("previous") or None,
                actual=item.get("actual") or None,
            ))
        except Exception:
            continue
    return events


def _compute_surprise(actual_str: str, forecast_str: str, event_title: str) -> Optional[str]:
    """Return 'beat', 'miss', or 'inline' by comparing numeric actual vs forecast."""
    try:
        # Strip common suffixes: %, K, M, B
        def _parse(s: str) -> float:
            s = s.strip().replace("%", "").replace(",", "")
            multiplier = 1.0
            if s.endswith("K"):
                multiplier = 1_000
                s = s[:-1]
            elif s.endswith("M"):
                multiplier = 1_000_000
                s = s[:-1]
            elif s.endswith("B"):
                multiplier = 1_000_000_000
                s = s[:-1]
            return float(s) * multiplier

        actual = _parse(actual_str)
        forecast = _parse(forecast_str)
        diff = actual - forecast

        # For unemployment/claims lower is better
        lower_is_better = any(kw in event_title.lower() for kw in (
            "unemployment", "claims", "jobless", "deficit", "cpi", "pce", "ppi", "inflation"
        ))

        if abs(diff) < 0.01 * abs(forecast) if forecast != 0 else abs(diff) < 0.001:
            return "inline"
        if lower_is_better:
            return "beat" if diff < 0 else "miss"
        else:
            return "beat" if diff > 0 else "miss"
    except Exception:
        return None


class EconomicCalendarService:
    """Poll Forex Factory calendar, fire pre-event warnings and post-release result alerts."""

    def __init__(self, redis_client: RedisClient) -> None:
        self.redis = redis_client
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._events: List[CalendarEvent] = []
        self._last_feed_fetch: Optional[datetime] = None
        # Track which (event_id, warning_minutes) combos have already fired
        self._fired_warnings: set = set()
        # Track which event_ids have had their result published
        self._fired_results: set = set()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="economic-calendar")
        LOGGER.info("EconomicCalendarService started")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._stop_event.set()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        LOGGER.info("EconomicCalendarService stopped")

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._refresh_feed_if_needed()
                self._check_warnings_and_results()
            except Exception:
                LOGGER.exception("EconomicCalendarService loop error")
            await asyncio.sleep(POLL_INTERVAL)

    async def _refresh_feed_if_needed(self) -> None:
        now = datetime.now(timezone.utc)
        if (
            self._last_feed_fetch is not None
            and (now - self._last_feed_fetch).total_seconds() < FEED_REFRESH_INTERVAL
        ):
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(CALENDAR_FEED_URL)
                resp.raise_for_status()
                raw = resp.json()
            self._events = _parse_ff_events(raw)
            self._last_feed_fetch = now
            # Cache today's high-impact events in Redis for the frontend /api/calendar endpoint
            today_str = now.astimezone(_ET).date().isoformat()
            today_events = [
                e.model_dump(mode="json")
                for e in self._events
                if _is_high_impact(e) and e.event_time.astimezone(_ET).date().isoformat() == today_str
            ]
            self.redis.client.setex(
                CALENDAR_REDIS_KEY,
                CALENDAR_REDIS_TTL,
                json.dumps(today_events, default=str),
            )
            LOGGER.info(
                "Calendar feed refreshed: %d total events, %d high-impact today",
                len(self._events), len(today_events),
            )
        except Exception:
            LOGGER.exception("Failed to fetch calendar feed")

    def _check_warnings_and_results(self) -> None:
        now = datetime.now(timezone.utc)
        for event in self._events:
            if not _is_high_impact(event):
                continue

            minutes_until = (event.event_time - now).total_seconds() / 60

            # Pre-event warnings
            for warn_min in WARNING_MINUTES:
                key = (event.event_id, warn_min)
                if key in self._fired_warnings:
                    continue
                # Fire when we're within [warn_min, warn_min + POLL_INTERVAL/60 + 1] min
                tolerance = POLL_INTERVAL / 60 + 0.5
                if warn_min >= minutes_until > warn_min - tolerance:
                    self._fire_warning(event, int(minutes_until))
                    self._fired_warnings.add(key)

            # Post-release result
            if (
                event.event_id not in self._fired_results
                and event.actual is not None
                and event.actual.strip() != ""
                and minutes_until < 5  # event has passed or just released
            ):
                self._fire_result(event)
                self._fired_results.add(event.event_id)

    def _fire_warning(self, event: CalendarEvent, minutes_until: int) -> None:
        urgency = "🔴" if minutes_until <= 2 else "🟡" if minutes_until <= 10 else "⚠️"
        severity = "high" if minutes_until <= 10 else "medium"
        action = "GET FLAT or confirm your edge" if minutes_until <= 5 else "Consider sizing down"
        msg = (
            f"{urgency} **{event.title}** in {minutes_until} min — {action}\n"
            f"Forecast: {event.forecast or 'N/A'} | Previous: {event.previous or 'N/A'}"
        )
        alert = CalendarAlert(
            alert_type="calendar_warning",
            severity=severity,
            event_title=event.title,
            event_time=event.event_time,
            minutes_until=minutes_until,
            forecast=event.forecast,
            previous=event.previous,
            message=msg,
        )
        self._publish(alert)
        LOGGER.info("Calendar warning fired: %s in %d min", event.title, minutes_until)

    def _fire_result(self, event: CalendarEvent) -> None:
        surprise = None
        if event.actual and event.forecast:
            surprise = _compute_surprise(event.actual, event.forecast, event.title)

        surprise_str = ""
        if surprise == "beat":
            surprise_str = " ✅ BEAT"
        elif surprise == "miss":
            surprise_str = " ❌ MISS"
        elif surprise == "inline":
            surprise_str = " ➡️ inline"

        msg = (
            f"📋 **{event.title}** result{surprise_str}\n"
            f"Actual: **{event.actual}** | Forecast: {event.forecast or 'N/A'} | Previous: {event.previous or 'N/A'}"
        )
        severity = "high" if surprise in ("beat", "miss") else "medium"
        alert = CalendarAlert(
            alert_type="calendar_result",
            severity=severity,
            event_title=event.title,
            event_time=event.event_time,
            actual=event.actual,
            forecast=event.forecast,
            previous=event.previous,
            surprise=surprise,
            message=msg,
        )
        self._publish(alert)
        LOGGER.info(
            "Calendar result fired: %s actual=%s forecast=%s surprise=%s",
            event.title, event.actual, event.forecast, surprise,
        )

    def _publish(self, alert: CalendarAlert) -> None:
        try:
            payload = alert.model_dump(mode="json")
            serialized = json.dumps(payload, default=str)
            self.redis.client.publish(CORRELATION_ALERT_CHANNEL, serialized)
        except Exception:
            LOGGER.exception("Failed to publish calendar alert")
