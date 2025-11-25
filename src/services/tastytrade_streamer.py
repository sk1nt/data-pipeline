"""Async TastyTrade DXLink streamer service used by data-pipeline."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

try:  # pragma: no cover - optional dependency in some environments
    from tastytrade import DXLinkStreamer
    from tastytrade.session import Session
    from tastytrade.dxfeed import Quote, Trade
except ImportError:  # pragma: no cover
    DXLinkStreamer = Session = Quote = Trade = None  # type: ignore


LOGGER = logging.getLogger(__name__)


TradeHandler = Callable[[Dict[str, float]], Awaitable[None]]
DepthHandler = Callable[[Dict[str, object]], Awaitable[None]]


@dataclass
class StreamerSettings:
    client_id: str
    client_secret: str
    refresh_token: str
    symbols: List[str]
    depth_levels: int = 40


class TastyTradeStreamer:
    """Manage DXLink streaming lifecycle with pluggable callbacks."""

    def __init__(
        self,
        settings: StreamerSettings,
        *,
        on_trade: Optional[TradeHandler] = None,
        on_depth: Optional[DepthHandler] = None,
    ) -> None:
        if DXLinkStreamer is None:
            raise RuntimeError("tastytrade SDK is not installed; cannot start streamer")

        self.settings = settings
        self._on_trade = on_trade
        self._on_depth = on_depth
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("TastyTrade streamer already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="tastytrade-streamer")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        LOGGER.info(
            "Starting TastyTrade DXLink streamer (symbols=%s, depth_levels=%s)",
            self.settings.symbols,
            self.settings.depth_levels,
        )
        session = Session(
            provider_secret=self.settings.client_secret,
            refresh_token=self.settings.refresh_token,
        )
        try:
            async with DXLinkStreamer(session) as streamer:
                formatted_symbols = [self._format_symbol(sym) for sym in self.settings.symbols]
                await streamer.subscribe(Trade, formatted_symbols)
                await streamer.subscribe(Quote, formatted_symbols)
                LOGGER.info("Subscribed to DXLink trades + quotes for %s", formatted_symbols)

                while not self._stop_event.is_set():
                    try:
                        trade = await asyncio.wait_for(streamer.get_event(Trade), timeout=1.0)
                        await self._handle_trade(trade)
                    except asyncio.TimeoutError:
                        pass
                    except Exception:  # pragma: no cover - defensive logging
                        LOGGER.exception("Error processing trade event")

                    try:
                        quote = await asyncio.wait_for(streamer.get_event(Quote), timeout=0.1)
                        await self._handle_quote(quote)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        LOGGER.exception("Error processing quote event")
        finally:
            LOGGER.info("TastyTrade DXLink streamer stopped")

    async def _handle_trade(self, trade) -> None:
        if not trade:
            return
        payload = {
            "symbol": self._normalize_symbol(trade.event_symbol),
            "price": float(getattr(trade, "price", 0.0) or 0.0),
            "size": int(getattr(trade, "size", 0) or 0),
            "timestamp": self._ts_from_ms(getattr(trade, "time", 0)),
        }
        if self._on_trade:
            await self._on_trade(payload)
        else:
            LOGGER.debug("Trade: %s", payload)

    async def _handle_quote(self, quote) -> None:
        if not quote:
            return
        bids = self._extract_levels(getattr(quote, "bid_prices", []) or [], getattr(quote, "bid_sizes", []) or [])
        asks = self._extract_levels(getattr(quote, "ask_prices", []) or [], getattr(quote, "ask_sizes", []) or [])
        depth_payload = {
            "symbol": self._normalize_symbol(getattr(quote, "event_symbol", "")),
            "timestamp": self._ts_from_ms(getattr(quote, "time", 0)),
            "bids": bids,
            "asks": asks,
        }
        if self._on_depth:
            await self._on_depth(depth_payload)

    def _extract_levels(
        self,
        prices: Sequence[float],
        sizes: Sequence[float],
    ) -> List[Dict[str, float]]:
        normalized: List[Dict[str, float]] = []
        for price, size in zip(prices, sizes):
            if price is None:
                continue
            normalized.append({"price": float(price), "size": float(size or 0.0)})
            if len(normalized) >= self.settings.depth_levels:
                break
        # pad to requested depth
        while len(normalized) < self.settings.depth_levels:
            normalized.append({"price": 0.0, "size": 0.0})
        return normalized

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        symbol = symbol.upper().strip()
        futures = {"MNQ", "MES", "NQ", "ES"}
        if symbol in futures:
            return f"/{symbol}:XCME"
        return symbol

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if not symbol:
            return "UNKNOWN"
        return symbol.lstrip("/").split(":", 1)[0].upper()

    @staticmethod
    def _ts_from_ms(value: Optional[int]) -> str:
        """Return an ISO-8601 UTC timestamp for DXLink millisecond values."""
        try:
            if not value:
                raise ValueError("missing timestamp")
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()
