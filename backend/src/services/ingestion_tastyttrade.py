import asyncio
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from tastytrade import DXLinkStreamer, OAuthSession
from tastytrade.dxfeed import Trade

from backend.src.services.duckdb_service import store_tick_data
from backend.src.services.redis_service import redis_manager

# Keep writer aligned with reader: single template, default GEX snapshot cache
REDIS_TICK_KEY_TEMPLATE = os.getenv("REDIS_TICK_KEY_TEMPLATE", "gex:snapshot:{symbol}")

# Optional symbol aliases (upper-case) e.g., NQ -> NQ_NDX
DEFAULT_SYMBOL_MAP = {
    "NQ": "NQ_NDX",
}

# Load environment variables
load_dotenv()

TASTYTRADE_CLIENT_ID = os.getenv("TASTYTRADE_CLIENT_ID")
TASTYTRADE_CLIENT_SECRET = os.getenv("TASTYTRADE_CLIENT_SECRET")
TASTYTRADE_REFRESH_TOKEN = os.getenv("TASTYTRADE_REFRESH_TOKEN")


class TastyTradeIngestion:
    def __init__(self):
        self.client = None
        self.symbols = ["MES", "MNQ", "NQ", "SPY", "QQQ", "VIX"]
        self.last_save = time.time()
        self.save_interval = 60  # Save every 60 seconds
        self.tick_buffer = []  # Buffer for batch saving

    def process_quote(self, trade):
        """Handle incoming trade data"""
        try:
            # Convert trade to tick format
            # Clean symbol: remove leading slash and exchange suffix
            symbol = trade.event_symbol.lstrip("/").split(":")[0]
            symbol = DEFAULT_SYMBOL_MAP.get(symbol.upper(), symbol.upper())

            tick = {
                "symbol": symbol,
                "timestamp": datetime.fromtimestamp(trade.time / 1000).isoformat(),
                "price": float(trade.price),
                "volume": int(trade.size),
                "tick_type": "trade",
                "source": "tastyttrade",
            }

            print(f"Received tick: {tick}")

            # Add to buffer for batch saving
            self.tick_buffer.append(tick)

            # Cache in Redis for real-time access using unified key template
            sym = tick["symbol"].upper()
            sym = DEFAULT_SYMBOL_MAP.get(sym, sym)
            redis_key = REDIS_TICK_KEY_TEMPLATE.format(symbol=sym)
            tick["symbol"] = sym
            redis_manager.set_tick_data(redis_key, tick)

        except Exception as e:
            print(f"Error processing trade: {e}")

    def save_buffers(self):
        """Save buffered ticks to DuckDB"""
        if self.tick_buffer:
            try:
                store_tick_data(self.tick_buffer)
                print(f"Saved {len(self.tick_buffer)} ticks to DuckDB")
                self.tick_buffer.clear()
                self.last_save = time.time()
            except Exception as e:
                print(f"Error saving to DuckDB: {e}")

    async def run(self):
        """Main ingestion loop"""
        try:
            # Create OAuth2 session - it will automatically manage access tokens
            session = OAuthSession(
                provider_secret=TASTYTRADE_CLIENT_SECRET,
                refresh_token=TASTYTRADE_REFRESH_TOKEN,
            )

            print("✅ Authenticated with TastyTrade OAuth2")

            # Format symbols - futures get /symbol:XCME, equities stay as-is
            futures_symbols = ["MES", "MNQ", "NQ"]  # Common futures
            formatted_symbols = []

            for sym in self.symbols:
                if sym in futures_symbols:
                    # Futures: add slash and exchange
                    formatted_symbols.append(f"/{sym}:XCME")
                else:
                    # Equities/ETFs: use as-is with leading slash
                    formatted_symbols.append(f"/{sym}")

            print(f"Formatted symbols: {formatted_symbols}")

            # Start streaming with async context manager
            async with DXLinkStreamer(session) as streamer:
                print("Subscribing to trades...")
                await streamer.subscribe(Trade, formatted_symbols)
                print("✅ Streaming started successfully")

                # Event loop with periodic saves
                try:
                    while True:
                        # Get next trade event (with timeout)
                        try:
                            trade = await asyncio.wait_for(
                                streamer.get_event(Trade), timeout=1.0
                            )
                            self.process_quote(trade)
                        except asyncio.TimeoutError:
                            # No trade in timeout period, check if should save
                            pass

                        # Check if we should save
                        if time.time() - self.last_save >= self.save_interval:
                            self.save_buffers()

                except KeyboardInterrupt:
                    print("Received shutdown signal")

        except Exception as e:
            print(f"Streaming error: {e}", exc_info=True)
        finally:
            # Save any remaining data
            print("Saving final buffers...")
            self.save_buffers()
            print("Stream service stopped")


if __name__ == "__main__":
    ingestion = TastyTradeIngestion()
    asyncio.run(ingestion.run())
