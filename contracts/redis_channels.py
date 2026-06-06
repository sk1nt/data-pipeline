"""Canonical Redis pub/sub channel names — keep byte-identical in both repos.

Publisher roles:
  gex:snapshot:stream              — data-pipeline (server)
  market:dom/cvd:{symbol}          — data-trading (Sierra bridge)
  sweep:alert/monitor/danger/ack   — data-trading (sweep_runner)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RedisChannels:
    """Channel templates — format with symbol where noted."""

    gex_snapshot_stream: str = "gex:snapshot:stream"

    market_dom: str = "market:dom:{symbol}"
    market_cvd: str = "market:cvd:{symbol}"

    sweep_alert: str = "sweep:alert:{symbol}"
    sweep_monitor: str = "sweep:monitor:{symbol}"
    sweep_danger: str = "sweep:danger:{symbol}"
    sweep_ack: str = "sweep:ack:{symbol}"

    def dom(self, symbol: str) -> str:
        return self.market_dom.format(symbol=symbol)

    def cvd(self, symbol: str) -> str:
        return self.market_cvd.format(symbol=symbol)

    def alert(self, symbol: str) -> str:
        return self.sweep_alert.format(symbol=symbol)

    def monitor(self, symbol: str) -> str:
        return self.sweep_monitor.format(symbol=symbol)

    def danger(self, symbol: str) -> str:
        return self.sweep_danger.format(symbol=symbol)

    def ack(self, symbol: str) -> str:
        return self.sweep_ack.format(symbol=symbol)