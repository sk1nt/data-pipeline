"""Async TastyTrade DXLink streamer service used by data-pipeline."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Sequence

try:  # pragma: no cover - optional dependency in some environments
    from tastytrade import DXLinkStreamer
    from tastytrade.dxfeed import Greeks, Quote, TimeAndSale, Trade
except ImportError:  # pragma: no cover
    DXLinkStreamer = Quote = TimeAndSale = Trade = Greeks = None  # type: ignore

try:
    from src.services.tastytrade_auth_service import get_tastytrade_auth_service
except Exception:  # pragma: no cover
    from services.tastytrade_auth_service import get_tastytrade_auth_service  # type: ignore


LOGGER = logging.getLogger(__name__)


TradeHandler = Callable[[Dict[str, float]], Awaitable[None]]
DepthHandler = Callable[[Dict[str, object]], Awaitable[None]]
GreeksHandler = Callable[[Dict[str, float]], Awaitable[None]]


@dataclass
class StreamerSettings:
    client_id: str
    client_secret: str
    refresh_token: str
    symbols: List[str]
    depth_levels: int = 40
    enable_depth: bool = False
    # Option OCC symbols for Greeks streaming (e.g. [".SPY251219C600"])
    greeks_symbols: List[str] = field(default_factory=list)
    enable_greeks: bool = False


class TastyTradeStreamer:
    """Manage DXLink streaming lifecycle with pluggable callbacks."""

    def __init__(
        self,
        settings: StreamerSettings,
        *,
        on_trade: Optional[TradeHandler] = None,
        on_depth: Optional[DepthHandler] = None,
        on_greeks: Optional[GreeksHandler] = None,
        auth_service=None,
    ) -> None:
        if DXLinkStreamer is None:
            raise RuntimeError("tastytrade SDK is not installed; cannot start streamer")

        self.settings = settings
        self._on_trade = on_trade
        self._on_depth = on_depth
        self._on_greeks = on_greeks
        self._auth_service = auth_service
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
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        LOGGER.info(
            "Starting TastyTrade DXLink streamer (symbols=%s, depth_levels=%s, greeks=%s)",
            self.settings.symbols,
            self.settings.depth_levels,
            self.settings.enable_greeks,
        )
        auth_service = self._auth_service or get_tastytrade_auth_service()
        session = await asyncio.to_thread(auth_service.get_session)
        try:
            async with DXLinkStreamer(session) as streamer:
                formatted_symbols = [
                    self._format_symbol(sym) for sym in self.settings.symbols
                ]
                time_and_sale_symbols = [
                    sym for sym in formatted_symbols if self._uses_time_and_sale(sym)
                ]
                trade_symbols = [
                    sym for sym in formatted_symbols if not self._uses_time_and_sale(sym)
                ]
                await streamer.subscribe(Trade, trade_symbols)
                if time_and_sale_symbols:
                    await streamer.subscribe(TimeAndSale, time_and_sale_symbols)
                if getattr(self.settings, "enable_depth", False):
                    await streamer.subscribe(Quote, formatted_symbols)
                    LOGGER.info(
                        "Subscribed to DXLink trades + quotes for %s",
                        trade_symbols,
                    )
                else:
                    LOGGER.info(
                        "Subscribed to DXLink trades for %s (quotes disabled)",
                        trade_symbols,
                    )
                if time_and_sale_symbols:
                    LOGGER.info(
                        "Subscribed to DXLink time-and-sale indicators for %s",
                        time_and_sale_symbols,
                    )

                # Greeks subscription for option symbols
                greeks_symbols = []
                if getattr(self.settings, "enable_greeks", False) and self.settings.greeks_symbols:
                    greeks_symbols = list(self.settings.greeks_symbols)
                    await streamer.subscribe(Greeks, greeks_symbols)
                    LOGGER.info(
                        "Subscribed to DXLink greeks for %d option symbols",
                        len(greeks_symbols),
                    )

                while not self._stop_event.is_set():
                    try:
                        trade = await asyncio.wait_for(
                            streamer.get_event(Trade), timeout=1.0
                        )
                        await self._handle_trade(trade)
                    except asyncio.TimeoutError:
                        pass
                    except Exception:  # pragma: no cover - defensive logging
                        LOGGER.exception("Error processing trade event")

                    if time_and_sale_symbols:
                        while True:
                            try:
                                time_and_sale = await asyncio.wait_for(
                                    streamer.get_event(TimeAndSale), timeout=0.1
                                )
                                await self._handle_time_and_sale(time_and_sale)
                            except asyncio.TimeoutError:
                                break
                            except Exception:  # pragma: no cover
                                LOGGER.exception("Error processing time-and-sale event")
                                break

                    if getattr(self.settings, "enable_depth", False):
                        try:
                            quote = await asyncio.wait_for(
                                streamer.get_event(Quote), timeout=0.1
                            )
                            await self._handle_quote(quote)
                        except asyncio.TimeoutError:
                            pass
                        except Exception:
                            LOGGER.exception("Error processing quote event")

                    if greeks_symbols:
                        while True:
                            try:
                                greeks_event = await asyncio.wait_for(
                                    streamer.get_event(Greeks), timeout=0.1
                                )
                                await self._handle_greeks(greeks_event)
                            except asyncio.TimeoutError:
                                break
                            except Exception:  # pragma: no cover
                                LOGGER.exception("Error processing greeks event")
                                break

                    if not getattr(self.settings, "enable_depth", False) and not greeks_symbols:
                        await asyncio.sleep(0.05)
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

    async def _handle_time_and_sale(self, time_and_sale) -> None:
        if not time_and_sale:
            return
        price = getattr(time_and_sale, "price", None)
        if price is None or (isinstance(price, float) and math.isnan(price)):
            return
        payload = {
            "symbol": self._normalize_symbol(time_and_sale.event_symbol),
            "price": float(price),
            "size": int(getattr(time_and_sale, "size", 0) or 0),
            "timestamp": self._ts_from_ms(
                getattr(time_and_sale, "time", 0)
                or getattr(time_and_sale, "event_time", 0)
            ),
        }
        if self._on_trade:
            await self._on_trade(payload)
        else:
            LOGGER.debug("Time-and-sale: %s", payload)

    async def _handle_quote(self, quote) -> None:
        if not quote:
            return
        bids = self._extract_levels(
            getattr(quote, "bid_prices", []) or [],
            getattr(quote, "bid_sizes", []) or [],
        )
        asks = self._extract_levels(
            getattr(quote, "ask_prices", []) or [],
            getattr(quote, "ask_sizes", []) or [],
        )
        depth_payload = {
            "symbol": self._normalize_symbol(getattr(quote, "event_symbol", "")),
            "timestamp": self._ts_from_ms(getattr(quote, "time", 0)),
            "bids": bids,
            "asks": asks,
        }
        if self._on_depth:
            await self._on_depth(depth_payload)

    async def _handle_greeks(self, greeks_event) -> None:
        """Handle real-time Greeks events from DXLink.

        Provides delta, gamma, theta, vega, rho, implied_volatility,
        underlying_price, and option price per subscribed option symbol.
        """
        if not greeks_event:
            return
        payload = {
            "symbol": getattr(greeks_event, "event_symbol", ""),
            "delta": float(getattr(greeks_event, "delta", 0.0) or 0.0),
            "gamma": float(getattr(greeks_event, "gamma", 0.0) or 0.0),
            "theta": float(getattr(greeks_event, "theta", 0.0) or 0.0),
            "vega": float(getattr(greeks_event, "vega", 0.0) or 0.0),
            "rho": float(getattr(greeks_event, "rho", 0.0) or 0.0),
            "implied_volatility": float(getattr(greeks_event, "implied_volatility", 0.0) or 0.0),
            "underlying_price": float(getattr(greeks_event, "underlying_price", 0.0) or 0.0),
            "option_price": float(getattr(greeks_event, "price", 0.0) or 0.0),
            "timestamp": self._ts_from_ms(getattr(greeks_event, "time", 0)),
        }
        if self._on_greeks:
            await self._on_greeks(payload)
        else:
            LOGGER.debug("Greeks: %s", payload)

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
        while len(normalized) < self.settings.depth_levels:
            normalized.append({"price": 0.0, "size": 0.0})
        return normalized

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        symbol = symbol.upper().strip()
        futures = {"MNQ", "MES", "NQ", "ES", "VX"}
        if symbol in futures:
            return f"/{symbol}:XCME"
        return symbol

    @staticmethod
    def _uses_time_and_sale(symbol: str) -> bool:
        """TRIN market indicators publish live price-like updates via TimeAndSale."""
        return symbol.upper().startswith("$TRIN")

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
