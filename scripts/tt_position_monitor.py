#!/usr/bin/env python3
"""
TastyTrade Position P&L Monitor

Monitors open futures positions and:
- Level 1 (60 ticks): Discord alert
- Level 2 (80 ticks): Discord alert  
- Level 3 (100 ticks): Flatten position via market order

Uses DXLink streaming for real-time prices since position API mark is stale.

Usage:
    python scripts/tt_position_monitor.py [--dry-run] [--poll-interval 2]
"""

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Set

import httpx
from dotenv import load_dotenv

# Add discord-bot for TastyTradeClient
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discord-bot"))

from bot.tastytrade_client import TastyTradeClient

# TastyTrade SDK imports
from tastytrade.session import Session
from tastytrade import DXLinkStreamer
from tastytrade.dxfeed import Trade

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for loss thresholds."""
    level1_ticks: int = 60  # Discord alert
    level2_ticks: int = 80  # Discord alert
    level3_ticks: int = 100  # Flatten


@dataclass
class PositionState:
    """Tracks alert state for a position to avoid duplicate alerts."""
    symbol: str
    account: str
    entry_price: float
    quantity: float
    direction: str  # "Long" or "Short"
    level1_triggered: bool = False
    level2_triggered: bool = False
    level3_triggered: bool = False
    last_ticks: float = 0.0


@dataclass
class MonitorConfig:
    """Main monitor configuration."""
    accounts: list = field(default_factory=lambda: ["5WT31787", "5WT31673"])
    poll_interval: float = 2.0
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    discord_webhook_url: Optional[str] = None
    dry_run: bool = True
    tick_size: float = 0.25  # MNQ/NQ tick size
    tick_value: float = 0.50  # MNQ $ per tick (NQ = $5.00)


class PositionMonitor:
    """Monitors TastyTrade positions for P&L thresholds using streaming prices."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.clients: Dict[str, TastyTradeClient] = {}
        self.position_states: Dict[str, PositionState] = {}  # key: f"{account}:{symbol}"
        self._running = False
        self._live_prices: Dict[str, float] = {}  # symbol -> last price
        self._session: Optional[Session] = None
        
        # Initialize clients for each account
        for account in config.accounts:
            self.clients[account] = TastyTradeClient(
                client_secret=os.getenv("TASTYTRADE_CLIENT_SECRET"),
                refresh_token=os.getenv("TASTYTRADE_REFRESH_TOKEN"),
                default_account=account,
                use_sandbox=False,
                dry_run=config.dry_run,
            )
        
        # Discord webhook
        self.config.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("DISCORD_ALERT_WEBHOOK_URL")
        
        logger.info(f"Initialized monitor for accounts: {config.accounts}")
        logger.info(f"Thresholds: L1={config.thresholds.level1_ticks}, L2={config.thresholds.level2_ticks}, L3={config.thresholds.level3_ticks} ticks")
        logger.info(f"Dry run: {config.dry_run}")

    def _get_position_key(self, account: str, symbol: str) -> str:
        return f"{account}:{symbol}"

    def _get_streamer_symbol(self, position_symbol: str) -> str:
        """Convert position symbol to streamer symbol format."""
        # Position: /MNQH6 -> Streamer: /MNQ:XCME (use root symbol)
        # Extract root: /MNQH6 -> /MNQ
        if position_symbol.startswith("/"):
            # Remove the contract month/year suffix
            root = position_symbol[:4]  # e.g., /MNQ from /MNQH6
            return f"{root}:XCME"
        return position_symbol

    def _calculate_loss_ticks(self, entry_price: float, current_price: float, direction: str) -> float:
        """Calculate ticks of loss from entry price."""
        price_change_ticks = (current_price - entry_price) / self.config.tick_size
        
        # Direction: Long loses when price drops, Short loses when price rises
        dir_mult = 1 if direction == "Long" else -1
        pnl_ticks = price_change_ticks * dir_mult
        
        # Return loss as positive number (0 if in profit)
        return -pnl_ticks if pnl_ticks < 0 else 0.0

    async def send_discord_alert(self, message: str, level: int = 1):
        """Send alert to Discord webhook."""
        if not self.config.discord_webhook_url:
            logger.warning(f"No Discord webhook configured. Alert: {message}")
            return
        
        color = {1: 0xFFA500, 2: 0xFF4500, 3: 0xFF0000}.get(level, 0xFFFFFF)  # Orange, OrangeRed, Red
        
        embed = {
            "title": f"⚠️ Position Alert - Level {level}",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        payload = {"embeds": [embed]}
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.config.discord_webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                if resp.status_code == 204:
                    logger.info(f"Discord alert sent: Level {level}")
                else:
                    logger.error(f"Discord webhook failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    async def flatten_position(self, account: str, state: PositionState):
        """Flatten a position via market order."""
        symbol = state.symbol
        qty = abs(state.quantity)
        direction = state.direction
        
        logger.warning(f"FLATTEN: {account} {symbol} {qty:.0f} {direction}")
        
        if self.config.dry_run:
            logger.info("[DRY RUN] Would flatten position")
            return
        
        try:
            client = self.clients.get(account)
            if not client:
                logger.error(f"No client for account {account}")
                return
            
            # Build closing order - opposite direction
            action = "sell_to_close" if direction == "Long" else "buy_to_close"
            
            # Use the client's market order method
            result = client.place_market_order_with_tp(
                symbol=symbol,
                quantity=int(qty),
                action=action,
                take_profit_offset=None,
            )
            
            logger.info(f"Flatten order result: {result}")
            
        except Exception as e:
            logger.error(f"Failed to flatten position: {e}")

    async def check_position(self, account: str, state: PositionState, current_price: float):
        """Check a single position against thresholds."""
        loss_ticks = self._calculate_loss_ticks(state.entry_price, current_price, state.direction)
        state.last_ticks = loss_ticks
        
        # Calculate P&L for logging
        pnl_dollars = -loss_ticks * self.config.tick_value * abs(state.quantity)
        
        # Check thresholds
        thresholds = self.config.thresholds
        
        # Level 3: Flatten (highest priority)
        if loss_ticks >= thresholds.level3_ticks and not state.level3_triggered:
            state.level3_triggered = True
            state.level2_triggered = True  # Suppress lower levels
            state.level1_triggered = True
            msg = f"**FLATTEN** {account[-4:]} {state.symbol}: {state.quantity:.0f} {state.direction}\nLoss: {loss_ticks:.0f} ticks (${pnl_dollars:,.2f})\nPrice: {current_price:.2f} (Entry: {state.entry_price:.2f})"
            logger.critical(msg)
            await self.send_discord_alert(msg, level=3)
            await self.flatten_position(account, state)
            return
        
        # Level 2: Alert
        if loss_ticks >= thresholds.level2_ticks and not state.level2_triggered:
            state.level2_triggered = True
            state.level1_triggered = True  # Suppress lower level
            msg = f"**WARNING** {account[-4:]} {state.symbol}: {state.quantity:.0f} {state.direction}\nLoss: {loss_ticks:.0f} ticks (${pnl_dollars:,.2f})\nPrice: {current_price:.2f} (Entry: {state.entry_price:.2f})"
            logger.warning(msg)
            await self.send_discord_alert(msg, level=2)
            return
        
        # Level 1: Alert
        if loss_ticks >= thresholds.level1_ticks and not state.level1_triggered:
            state.level1_triggered = True
            msg = f"**ALERT** {account[-4:]} {state.symbol}: {state.quantity:.0f} {state.direction}\nLoss: {loss_ticks:.0f} ticks (${pnl_dollars:,.2f})\nPrice: {current_price:.2f} (Entry: {state.entry_price:.2f})"
            logger.warning(msg)
            await self.send_discord_alert(msg, level=1)
            return

    def refresh_positions(self) -> Set[str]:
        """Refresh position list from all accounts. Returns set of streamer symbols needed."""
        active_keys: Set[str] = set()
        streamer_symbols: Set[str] = set()
        
        for account, client in self.clients.items():
            try:
                positions = client.get_positions()
                
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    # Only monitor futures (start with /)
                    if not symbol.startswith("/"):
                        continue
                    
                    key = self._get_position_key(account, symbol)
                    active_keys.add(key)
                    
                    qty = pos.get("quantity", 0)
                    qty_val = float(qty) if isinstance(qty, Decimal) else float(qty)
                    if qty_val == 0:
                        continue
                    
                    entry = pos.get("average_open_price", 0)
                    entry_val = float(entry) if isinstance(entry, Decimal) else float(entry)
                    direction = pos.get("quantity_direction", "Long")
                    
                    # Create or update state
                    if key not in self.position_states:
                        self.position_states[key] = PositionState(
                            symbol=symbol,
                            account=account,
                            entry_price=entry_val,
                            quantity=qty_val,
                            direction=direction,
                        )
                        logger.info(f"Tracking position: {account[-4:]} {symbol} {qty_val:.0f} {direction} @ {entry_val:.2f}")
                    else:
                        # Update quantity/entry if changed
                        state = self.position_states[key]
                        state.quantity = qty_val
                        state.entry_price = entry_val
                        state.direction = direction
                    
                    # Add streamer symbol
                    streamer_sym = self._get_streamer_symbol(symbol)
                    streamer_symbols.add(streamer_sym)
                    
            except Exception as e:
                logger.error(f"Error polling account {account}: {e}")
        
        # Clean up states for closed positions
        closed_keys = set(self.position_states.keys()) - active_keys
        for key in closed_keys:
            logger.info(f"Position closed: {key}")
            del self.position_states[key]
        
        return streamer_symbols

    async def run(self):
        """Main monitoring loop with streaming prices."""
        self._running = True
        logger.info("Starting position monitor...")
        
        # Create streaming session
        self._session = Session(
            provider_secret=os.getenv("TASTYTRADE_CLIENT_SECRET"),
            refresh_token=os.getenv("TASTYTRADE_REFRESH_TOKEN"),
            is_test=False,
        )
        
        # Initial position refresh
        streamer_symbols = self.refresh_positions()
        
        if not streamer_symbols:
            logger.info("No futures positions to monitor. Waiting...")
        
        last_refresh = asyncio.get_event_loop().time()
        refresh_interval = 30.0  # Refresh positions every 30 seconds
        
        while self._running:
            try:
                # Refresh positions periodically
                now = asyncio.get_event_loop().time()
                if now - last_refresh >= refresh_interval:
                    streamer_symbols = self.refresh_positions()
                    last_refresh = now
                
                if not streamer_symbols:
                    await asyncio.sleep(5.0)
                    continue
                
                # Stream prices and check positions
                async with DXLinkStreamer(self._session) as streamer:
                    logger.info(f"Subscribing to: {list(streamer_symbols)}")
                    await streamer.subscribe(Trade, list(streamer_symbols))
                    
                    while self._running:
                        try:
                            trade = await asyncio.wait_for(
                                streamer.get_event(Trade),
                                timeout=self.config.poll_interval,
                            )
                            
                            current_price = float(trade.price)
                            streamer_sym = trade.event_symbol
                            
                            # Update live price cache
                            self._live_prices[streamer_sym] = current_price
                            
                            # Check all positions with matching root symbol
                            for key, state in list(self.position_states.items()):
                                pos_streamer_sym = self._get_streamer_symbol(state.symbol)
                                if pos_streamer_sym == streamer_sym:
                                    account = state.account
                                    await self.check_position(account, state, current_price)
                            
                        except asyncio.TimeoutError:
                            # No trade events, continue
                            pass
                        
                        # Check if we need to refresh positions
                        now = asyncio.get_event_loop().time()
                        if now - last_refresh >= refresh_interval:
                            new_symbols = self.refresh_positions()
                            if new_symbols != streamer_symbols:
                                streamer_symbols = new_symbols
                                logger.info("Position changes detected, reconnecting streamer...")
                                break  # Exit inner loop to reconnect with new symbols
                            last_refresh = now
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5.0)

    def stop(self):
        """Stop the monitor."""
        self._running = False
        logger.info("Stopping position monitor...")


def main():
    parser = argparse.ArgumentParser(description="TastyTrade Position P&L Monitor")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Don't execute trades (default: True)")
    parser.add_argument("--live", action="store_true", help="Execute actual trades (disables dry-run)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Poll interval in seconds")
    parser.add_argument("--accounts", nargs="+", default=["5WT31787", "5WT31673"], help="Account numbers to monitor")
    parser.add_argument("--l1", type=int, default=60, help="Level 1 alert threshold (ticks)")
    parser.add_argument("--l2", type=int, default=80, help="Level 2 alert threshold (ticks)")
    parser.add_argument("--l3", type=int, default=100, help="Level 3 flatten threshold (ticks)")
    
    args = parser.parse_args()
    
    config = MonitorConfig(
        accounts=args.accounts,
        poll_interval=args.poll_interval,
        thresholds=ThresholdConfig(
            level1_ticks=args.l1,
            level2_ticks=args.l2,
            level3_ticks=args.l3,
        ),
        dry_run=not args.live,
    )
    
    monitor = PositionMonitor(config)
    
    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        monitor.stop()
        logger.info("Monitor stopped by user")


if __name__ == "__main__":
    main()
