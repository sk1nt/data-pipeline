"""Background service for polling GEXBot API at fixed intervals."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp


LOGGER = logging.getLogger(__name__)


@dataclass
class GEXBotPollerSettings:
    api_key: str
    symbols: List[str] = field(default_factory=lambda: ["NQ_NDX", "ES_SPX", "SPY", "QQQ", "SPX", "NDX"])
    interval_seconds: int = 60


class GEXBotPoller:
    """Poll GEXBot classic endpoints and cache latest snapshots."""

    def __init__(self, settings: GEXBotPollerSettings) -> None:
        self.settings = settings
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self.latest: Dict[str, Dict[str, Any]] = {}

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("GEXBot poller already running")
            return
        if not self.settings.api_key:
            LOGGER.warning("GEXBot API key missing; poller will not start")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="gexbot-poller")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        LOGGER.info(
            "Starting GEXBot poller (symbols=%s interval=%ss)",
            ",".join(self.settings.symbols),
            self.settings.interval_seconds,
        )
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(limit=8, force_close=True)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while not self._stop_event.is_set():
                for symbol in self.settings.symbols:
                    try:
                        snapshot = await self._fetch_symbol(session, symbol)
                        if snapshot:
                            self.latest[symbol.upper()] = snapshot
                    except Exception:  # pragma: no cover - defensive logging
                        LOGGER.exception("Failed to poll GEXBot for %s", symbol)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.settings.interval_seconds
                    )
                except asyncio.TimeoutError:
                    continue
        LOGGER.info("GEXBot poller stopped")

    async def _fetch_symbol(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        base_url = f"https://api.gexbot.com/{symbol}/classic"

        async def _endpoint(name: str) -> Optional[Dict[str, Any]]:
            url = f"{base_url}/{name}?key={self.settings.api_key}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    LOGGER.debug("GEXBot %s %s returned %s", symbol, name, resp.status)
                    return None
                return await resp.json()

        zero = await _endpoint("zero")
        majors = await _endpoint("majors") if zero else None
        maxchange = await _endpoint("maxchange") if zero else None
        if not zero and not majors and not maxchange:
            return None

        result = self._combine_payloads(symbol, zero, majors, maxchange)
        result["raw"] = {"zero": zero, "majors": majors, "maxchange": maxchange}
        return result

    def _combine_payloads(
        self,
        symbol: str,
        zero: Optional[Dict[str, Any]],
        majors: Optional[Dict[str, Any]],
        maxchange: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _first(*values):
            for value in values:
                if value not in (None, "", []):
                    return value
            return None

        timestamp = _first(
            zero.get("timestamp") if zero else None,
            majors.get("timestamp") if majors else None,
            maxchange.get("timestamp") if maxchange else None,
        )
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()

        return {
            "symbol": symbol.upper(),
            "timestamp": timestamp,
            "spot": _first(zero.get("spot") if zero else None, majors.get("spot") if majors else None),
            "zero_gamma": _first(zero.get("zero_gamma") if zero else None, maxchange.get("zero_gamma") if maxchange else None),
            "net_gex": _first(zero.get("net_gex") if zero else None, zero.get("net_gex_vol") if zero else None),
            "major_pos_vol": _first(
                majors.get("major_pos_vol") if majors else None,
                maxchange.get("major_pos_vol") if maxchange else None,
            ),
            "major_neg_vol": _first(
                majors.get("major_neg_vol") if majors else None,
                maxchange.get("major_neg_vol") if maxchange else None,
            ),
            "major_pos_oi": _first(
                majors.get("major_pos_oi") if majors else None,
                maxchange.get("major_pos_oi") if maxchange else None,
            ),
            "major_neg_oi": _first(
                majors.get("major_neg_oi") if majors else None,
                maxchange.get("major_neg_oi") if maxchange else None,
            ),
            "major_pos_strike": _first(
                majors.get("major_pos_strike") if majors else None,
                maxchange.get("major_pos_strike") if maxchange else None,
            ),
            "major_neg_strike": _first(
                majors.get("major_neg_strike") if majors else None,
                maxchange.get("major_neg_strike") if maxchange else None,
            ),
            "delta_risk_reversal": _first(
                zero.get("delta_risk_reversal") if zero else None,
                maxchange.get("delta_risk_reversal") if maxchange else None,
            ),
            "maxchange": maxchange or {},
        }

