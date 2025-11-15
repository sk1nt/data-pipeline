#!/usr/bin/env python3
"""
TastyTrade Live Market Data Streaming Service

Streams real-time quotes for futures contracts using TastyTrade's DXLink streaming API.
Saves data to CSV files and forwards to webhook for centralized processing.

Uses refresh token authentication from .env file:
    TASTYTRADE_REFRESH_TOKEN

Usage:
    python scripts/tastytrade_stream_service.py --symbols NQ MNQ ES MES
    
Environment Variables:
    TASTYTRADE_REFRESH_TOKEN - TastyTrade refresh token (client_secret)
    WEBHOOK_URL - Optional webhook endpoint (default: http://127.0.0.1:8877/uw)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

# Import OAuth2 manager
from scripts.tastytrade_oauth2 import TastyTradeOAuth2

# Import TastyTrade SDK (but use OAuth2 for auth)
from tastytrade import OAuthSession, DXLinkStreamer
from tastytrade.dxfeed import Quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory - standard location for raw tick data
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "data" / "raw" / "tastytrade_stream"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class TastyTradeStreamService:
    """Service for streaming live market data from TastyTrade."""
    
    def __init__(
        self,
        symbols: List[str],
        output_dir: Path = OUTPUTS_DIR,
        webhook_url: Optional[str] = None,
        save_interval: int = 60,
    ):
        """
        Initialize streaming service.
        
        Args:
            symbols: List of symbols to stream (e.g., ['NQ', 'MNQ', 'ES', 'MES'])
            output_dir: Directory to save CSV files
            webhook_url: Optional webhook URL for forwarding data
            save_interval: Interval in seconds to flush data to disk
        """
        self.symbols = symbols
        self.output_dir = Path(output_dir)
        self.webhook_url = webhook_url or os.getenv(
            "WEBHOOK_URL", 
            "http://127.0.0.1:8877/uw"
        )
        self.save_interval = save_interval
        
        # Initialize OAuth2 manager
        use_sandbox = os.getenv('TASTYTRADE_USE_SANDBOX', 'false').lower() == 'true'
        self.oauth = TastyTradeOAuth2(use_sandbox=use_sandbox)
        
        # Data buffers for each symbol
        self.buffers: Dict[str, List[Dict]] = {symbol: [] for symbol in symbols}
        
        # Track last save time
        self.last_save = time.time()
        
    def process_quote(self, quote: Quote):
        """Process incoming quote data."""
        try:
            # Extract symbol - remove leading slash and exchange suffix
            # Examples: /NQ:XCME -> NQ, /SPY -> SPY
            symbol = quote.event_symbol.lstrip('/').split(':')[0]
            
            # Structure quote data
            data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'bid': float(quote.bid_price) if quote.bid_price else None,
                'ask': float(quote.ask_price) if quote.ask_price else None,
                'bid_size': float(quote.bid_size) if quote.bid_size else None,
                'ask_size': float(quote.ask_size) if quote.ask_size else None,
                'last': float(quote.ask_price) if quote.ask_price else None,  # Use ask as proxy for last
            }
            
            # Add to buffer
            if symbol in self.buffers:
                self.buffers[symbol].append(data)
            
                # Log periodically
                if len(self.buffers[symbol]) % 100 == 0:
                    if data['bid'] and data['ask']:
                        logger.info(
                            f"{symbol}: {len(self.buffers[symbol])} quotes buffered | "
                            f"Bid: {data['bid']:.2f} Ask: {data['ask']:.2f}"
                        )
                    else:
                        logger.info(f"{symbol}: {len(self.buffers[symbol])} quotes buffered")
            
                # Forward to webhook if configured
                self._forward_to_webhook(symbol, data)
                
        except Exception as e:
            logger.error(f"Error processing quote: {e}")
    
    
    def _forward_to_webhook(self, symbol: str, data: Dict):
        """Forward data to webhook (fire and forget)."""
        try:
            # Format for universal webhook
            payload = {
                'topic': f'ticker:{symbol}',
                'data': json.dumps({
                    'timestamp': data['timestamp'],
                    'bid': data['bid'],
                    'ask': data['ask'],
                    'bid_size': data['bid_size'],
                    'ask_size': data['ask_size'],
                    'last': data['last'],
                })
            }
            requests.post(
                self.webhook_url,
                json=payload,
                timeout=1.0
            )
        except Exception as e:
            # Don't log every failure - would spam logs
            if len(self.buffers.get(symbol, [])) % 1000 == 0:
                logger.warning(f"Webhook forward failed: {e}")
    
    def save_buffers(self):
        """Save buffered data to CSV files."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        for symbol, buffer in self.buffers.items():
            if not buffer:
                continue
                
            try:
                # Create filename with date
                filename = self.output_dir / f"{symbol}_live_{current_date}.csv"
                
                # Convert to DataFrame
                df = pd.DataFrame(buffer)
                
                # Append to existing file or create new
                if filename.exists():
                    df.to_csv(filename, mode='a', header=False, index=False)
                else:
                    df.to_csv(filename, mode='w', header=True, index=False)
                
                logger.info(
                    f"Saved {len(buffer)} records for {symbol} to {filename.name}"
                )
                
                # Clear buffer
                self.buffers[symbol] = []
                
            except Exception as e:
                logger.error(f"Error saving buffer for {symbol}: {e}")
        
        self.last_save = time.time()
    
    async def stream_market_data(self):
        """Main streaming loop using modern async API."""
        logger.info(f"Starting market data stream for: {', '.join(self.symbols)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Webhook URL: {self.webhook_url}")
        logger.info(f"Save interval: {self.save_interval}s")
        
        try:
            # Get OAuth2 credentials
            use_sandbox = os.getenv('TASTYTRADE_USE_SANDBOX', 'false').lower() == 'true'
            client_secret = os.getenv('TASTYTRADE_CLIENT_SECRET')
            refresh_token = os.getenv('TASTYTRADE_REFRESH_TOKEN')
            
            if not client_secret or not refresh_token:
                raise RuntimeError(
                    "Missing TASTYTRADE_CLIENT_SECRET or TASTYTRADE_REFRESH_TOKEN in environment"
                )
            
            logger.info("Creating OAuth2 session...")
            # Create OAuth2 session - it will automatically manage access tokens
            session = OAuthSession(
                provider_secret=client_secret,
                refresh_token=refresh_token,
                is_test=use_sandbox
            )
            
            logger.info("✅ Authenticated with TastyTrade OAuth2")
            
            # Format symbols - futures get /symbol:XCME, equities stay as-is
            futures_symbols = ['NQ', 'MNQ', 'ES', 'MES', 'YM', 'RTY']  # Common futures
            formatted_symbols = []
            
            for sym in self.symbols:
                # Remove leading slash if present
                clean_sym = sym.lstrip('/')
                
                if clean_sym in futures_symbols:
                    # Futures: add slash and exchange
                    formatted_symbols.append(f"/{clean_sym}:XCME")
                else:
                    # Equities/ETFs: use as-is with leading slash
                    formatted_symbols.append(f"/{clean_sym}")
            
            logger.info(f"Formatted symbols: {formatted_symbols}")
            
            # Start streaming with async context manager
            async with DXLinkStreamer(session) as streamer:
                logger.info("Subscribing to quotes...")
                await streamer.subscribe(Quote, formatted_symbols)
                logger.info("✅ Streaming started successfully")
                
                # Event loop with periodic saves
                save_task = None
                try:
                    while True:
                        # Get next quote event (with timeout)
                        try:
                            quote = await asyncio.wait_for(
                                streamer.get_event(Quote),
                                timeout=1.0
                            )
                            self.process_quote(quote)
                        except asyncio.TimeoutError:
                            # No quote in timeout period, check if should save
                            pass
                        
                        # Check if we should save
                        if time.time() - self.last_save >= self.save_interval:
                            self.save_buffers()
                            
                except KeyboardInterrupt:
                    logger.info("Received shutdown signal")
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
        finally:
            # Save any remaining data
            logger.info("Saving final buffers...")
            self.save_buffers()
            logger.info("Stream service stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='TastyTrade live market data streaming service'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['NQ', 'MNQ', 'ES', 'SPY', 'QQQ', 'VIX', 'TLT'],
        help='Symbols to stream (default: NQ MNQ ES SPY QQQ VIX TLT)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUTS_DIR,
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--webhook-url',
        type=str,
        help='Webhook URL for forwarding data (default: http://127.0.0.1:8877/uw)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=60,
        help='Interval in seconds to save data to disk (default: 60)'
    )
    parser.add_argument(
        '--no-webhook',
        action='store_true',
        help='Disable webhook forwarding'
    )
    
    args = parser.parse_args()
    
    # Initialize service
    webhook_url = None if args.no_webhook else args.webhook_url
    
    try:
        service = TastyTradeStreamService(
            symbols=args.symbols,
            output_dir=args.output_dir,
            webhook_url=webhook_url,
            save_interval=args.save_interval,
        )
        
        # Run streaming loop
        asyncio.run(service.stream_market_data())
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
