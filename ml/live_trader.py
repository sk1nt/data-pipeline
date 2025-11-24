#!/usr/bin/env python3
"""Live trading system using enhanced GEX LSTM model.

Consumes real-time tick data from Redis and GEX updates to make live trading decisions.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.lib.redis_client import RedisClient
from ml.backtest_model import load_model_and_scaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveTrader:
    """Live trading system using enhanced GEX LSTM model."""

    def __init__(
        self,
        model_path: str = 'ml/models/enhanced_gex_model.pt',
        symbol: str = 'MNQ',
        sequence_length: int = 60,
        prediction_threshold: float = 0.3,
        max_position_size: int = 1,
        commission_cost: float = 0.42,
        api_endpoint: str = 'http://localhost:8877/ml-trade'
    ):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.prediction_threshold = prediction_threshold
        self.max_position_size = max_position_size
        self.commission_cost = commission_cost
        self.api_endpoint = api_endpoint

        # Redis connections
        self.redis_client = RedisClient()
        # Note: Using synchronous Redis client with asyncio.to_thread for pubsub operations

        # Load model and scaler
        logger.info(f"Loading model from {model_path}")
        self.model, self.scaler, self.checkpoint = load_model_and_scaler(model_path)
        self.model.eval()
        logger.info(f"Model loaded with {self.checkpoint.get('input_dim', 'unknown')} features")

        # Feature columns (must match training)
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 'zero_gamma', 'spot_price',
            'net_gex', 'major_pos_vol', 'major_neg_vol', 'sum_gex_vol', 'delta_risk_reversal',
            'max_priors_current', 'max_priors_1m', 'max_priors_5m',
            'adx', 'di_plus', 'di_minus', 'rsi', 'stoch_k', 'stoch_d'
        ]

        # Rolling data buffer
        self.tick_buffer: List[Dict] = []
        self.gex_data: Optional[Dict] = None
        self.last_tick_timestamp: Optional[float] = None  # Track last processed timestamp

        # Trading state
        self.position = 0  # -1 (short), 0 (flat), 1 (long)
        self.total_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0

        # GEX update tracking
        self.last_gex_update = 0.0

        # Redis channels
        self.tick_channel = "market_data:tastytrade:trades"
        # Map futures symbols to their underlying index GEX data
        gex_symbol_map = {
            'MNQ': 'NQ_NDX',  # Micro E-mini Nasdaq -> Nasdaq-100
            'MES': 'ES_SPX',  # Micro E-mini S&P -> S&P 500
            'NQ': 'NQ_NDX',   # E-mini Nasdaq -> Nasdaq-100
            'ES': 'ES_SPX',   # E-mini S&P -> S&P 500
        }
        gex_key = gex_symbol_map.get(symbol, symbol)
        self.gex_snapshot_key = f"gex:snapshot:{gex_key}"

        logger.info(f"LiveTrader initialized for {symbol}")

    async def get_latest_gex_data(self) -> Optional[Dict]:
        """Get latest GEX data from Redis."""
        try:
            data = await asyncio.to_thread(self.redis_client.client.get, self.gex_snapshot_key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting GEX data: {e}")
        return None

    def preprocess_tick_data(self, ticks: List[Dict]) -> Optional[np.ndarray]:
        """Preprocess tick data into sequence for model."""
        if len(ticks) < self.sequence_length:
            return None

        try:
            # Convert to DataFrame
            df = pd.DataFrame(ticks[-self.sequence_length:])

            # Resample to 1-second OHLCV
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = df.resample('1s').agg({
                'price': ['first', 'max', 'min', 'last'],
                'size': 'sum'
            })
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.dropna()

            if len(df) < self.sequence_length:
                return None

            # Use latest data
            df = df.tail(self.sequence_length).copy()

            # Add GEX features (constant within sequence)
            if self.gex_data:
                df['zero_gamma'] = self.gex_data.get('zero_gamma', 0)
                df['spot_price'] = self.gex_data.get('spot_price', 0)
                df['net_gex'] = self.gex_data.get('net_gex', 0)
                df['major_pos_vol'] = self.gex_data.get('major_pos_vol', 0)
                df['major_neg_vol'] = self.gex_data.get('major_neg_vol', 0)
                df['sum_gex_vol'] = self.gex_data.get('sum_gex_vol', 0)
                df['delta_risk_reversal'] = self.gex_data.get('delta_risk_reversal', 0)

                # Extract max_priors
                priors = self.gex_data.get('max_priors', [])
                df['max_priors_current'] = priors[0][1] if len(priors) > 0 else 0
                df['max_priors_1m'] = priors[1][1] if len(priors) > 1 else 0
                df['max_priors_5m'] = priors[4][1] if len(priors) > 4 else 0
            else:
                # Default values if no GEX data
                for col in ['zero_gamma', 'spot_price', 'net_gex', 'major_pos_vol', 'major_neg_vol',
                           'sum_gex_vol', 'delta_risk_reversal', 'max_priors_current', 'max_priors_1m', 'max_priors_5m']:
                    df[col] = 0.0

            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)

            # Select features
            feature_data = df[self.feature_cols].values

            # Scale features
            if self.scaler:
                feature_data = self.scaler.transform(feature_data)

            return feature_data

        except Exception as e:
            logger.error(f"Error preprocessing tick data: {e}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the sequence."""
        # RSI
        df['rsi'] = 100 - (100 / (1 + df['close'].diff(1).clip(lower=0).rolling(14).mean() /
                                  df['close'].diff(1).clip(upper=0).abs().rolling(14).mean()))

        # ADX and DIs
        df['tr'] = np.maximum(df['high'] - df['low'],
                             np.maximum(abs(df['high'] - df['close'].shift(1)),
                                       abs(df['low'] - df['close'].shift(1))))

        df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                 np.maximum(df['low'].shift(1) - df['low'], 0), 0)

        df['atr'] = df['tr'].rolling(14).mean()
        df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()

        df['di_plus'] = df['plus_di']
        df['di_minus'] = df['minus_di']

        # Stochastic Oscillator
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)

        return df

    def make_prediction(self, sequence: np.ndarray) -> Tuple[float, int]:
        """Make prediction on sequence data."""
        try:
            with torch.no_grad():
                input_tensor = torch.from_numpy(sequence).float().unsqueeze(0)  # Add batch dimension
                logits = self.model(input_tensor)
                prob_up = torch.sigmoid(logits).item()

                # Convert to trade decision
                if prob_up >= self.prediction_threshold:
                    return prob_up, 1  # Long signal
                elif (1 - prob_up) >= self.prediction_threshold:
                    return prob_up, -1  # Short signal
                else:
                    return prob_up, 0  # No signal

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0, 0

    def execute_trade(self, signal: int, current_price: float, confidence: float) -> None:
        """Send trade signal to API endpoint instead of executing real trade."""
        if signal == 0:
            return  # No trade

        timestamp = datetime.now(timezone.utc).isoformat()

        # Determine trade action
        if signal == 1:
            action = "entry"
            direction = "long"
        elif signal == -1:
            action = "entry"
            direction = "short"
        else:
            return

        # Calculate simulated P&L for position changes
        pnl = 0.0
        if (signal == 1 and self.position < 0) or (signal == -1 and self.position > 0):
            # Closing position
            pnl = (self.position * current_price * (-1 if signal == 1 else 1)) - self.commission_cost
            self.total_pnl += pnl

        # Update position
        old_position = self.position
        if signal == 1:
            self.position = min(self.position + 1, self.max_position_size)
        else:
            self.position = max(self.position - 1, -self.max_position_size)

        self.total_trades += 1

        # Prepare trade data
        trade_data = {
            "symbol": self.symbol,
            "action": action,  # "entry" or "exit"
            "direction": direction,  # "long" or "short"
            "price": current_price,
            "confidence": confidence,
            "position_before": old_position,
            "position_after": self.position,
            "pnl": pnl,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "timestamp": timestamp,
            "simulated": True
        }

        # Send to API endpoint
        asyncio.create_task(self.send_trade_update(trade_data))

        # Log locally
        logger.info(f"Trade signal sent: {action} {direction} {self.symbol} at ${current_price:.2f} (confidence: {confidence:.1%})")

    async def update_gex_data(self) -> None:
        """Update GEX data from Redis."""
        new_gex = await self.get_latest_gex_data()
        if new_gex:
            self.gex_data = new_gex
            logger.debug("Updated GEX data")

    async def process_tick(self, tick: Dict) -> None:
        """Process incoming tick data."""
        try:
            # Add to buffer
            self.tick_buffer.append(tick)

            # Keep only recent ticks (last 5 minutes worth)
            cutoff_time = datetime.now(timezone.utc).timestamp() * 1000 - 300000  # 5 minutes ago
            self.tick_buffer = [t for t in self.tick_buffer if t['timestamp'] > cutoff_time]

            # Process if we have enough data
            if len(self.tick_buffer) >= self.sequence_length:
                logger.info(f"Processing sequence of {len(self.tick_buffer)} ticks")
                sequence = self.preprocess_tick_data(self.tick_buffer)
                if sequence is not None:
                    prob, signal = self.make_prediction(sequence)
                    current_price = self.tick_buffer[-1]['price']

                    logger.debug(".3f")

                    # Execute trade
                    self.execute_trade(signal, current_price, prob)
                else:
                    logger.warning("Failed to preprocess tick data")
            else:
                logger.debug(f"Buffer has {len(self.tick_buffer)} ticks, need {self.sequence_length}")

        except Exception as e:
            logger.error(f"Error processing tick: {e}")

    async def run(self) -> None:
        """Run the live trading system using Redis pub/sub."""
        print("Entering run() method")
        logger.info("Starting live trading system...")

        # Update GEX data initially
        await self.update_gex_data()
        self.last_gex_update = datetime.now(timezone.utc).timestamp()
        print("GEX data updated")

        try:
            # Create a queue for messages
            from queue import Queue
            message_queue = Queue()
            
            # Subscribe to Redis pub/sub channel in a separate thread
            def pubsub_listener():
                pubsub = self.redis_client.client.pubsub()
                pubsub.subscribe(self.tick_channel)
                logger.info(f"Subscribed to Redis channel: {self.tick_channel}")
                
                while True:
                    try:
                        message = pubsub.get_message(timeout=1.0)
                        if message and message['type'] == 'message':
                            message_queue.put(message)
                    except Exception as e:
                        logger.error(f"PubSub listener error: {e}")
                        break
            
            import threading
            listener_thread = threading.Thread(target=pubsub_listener, daemon=True)
            listener_thread.start()
            
            logger.info("PubSub listener thread started")
            
            while True:
                try:
                    # Check for messages
                    if not message_queue.empty():
                        message = message_queue.get_nowait()
                        
                        try:
                            tick_data = json.loads(message['data'])
                            logger.debug(f"Received tick: {tick_data}")
                            
                            # Filter for our symbol
                            if tick_data.get('symbol') == self.symbol:
                                # Process the tick
                                tick = {
                                    'timestamp': tick_data['ts_ms'],
                                    'price': tick_data['price'],
                                    'size': tick_data['size']
                                }
                                
                                await self.process_tick(tick)
                            else:
                                logger.debug(f"Ignoring tick for symbol {tick_data.get('symbol')}, watching {self.symbol}")
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse tick data: {e}")
                        except KeyError as e:
                            logger.error(f"Missing key in tick data: {e}")
                    else:
                        # No message, update GEX data based on market hours
                        await self._update_gex_if_needed()
                        
                        # Log status every minute
                        current_time = datetime.now(timezone.utc).timestamp()
                        if int(current_time) % 60 == 0:
                            logger.info(f"Status: Position={self.position}, P&L=${self.total_pnl:.2f}, Trades={self.total_trades}")
                        
                        # Small delay
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1.0)
                        
        except KeyboardInterrupt:
            logger.info("Shutting down live trading system...")
        finally:
            # Final status
            logger.info("Final Results:")
            logger.info(f"  Total P&L: ${self.total_pnl:.2f}")
            logger.info(f"  Total Trades: {self.total_trades}")
            logger.info(f"  Final Position: {self.position}")
            if self.total_trades > 0:
                logger.info(".1%")

    def _is_rth(self) -> bool:
        """Check if current time is during regular trading hours (9:30am-4pm ET)."""
        try:
            from zoneinfo import ZoneInfo
            eastern = ZoneInfo("America/New_York")
        except ImportError:
            # Fallback for older Python versions
            from datetime import timezone, timedelta
            eastern = timezone(timedelta(hours=-5))  # EST, not handling DST
        
        now = datetime.now(tz=eastern)
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now <= end

    def _is_rth(self) -> bool:
        """Check if current time is during regular trading hours (9:30am-4pm ET)."""
        try:
            from zoneinfo import ZoneInfo
            eastern = ZoneInfo("America/New_York")
        except ImportError:
            # Fallback for older Python versions
            from datetime import timezone, timedelta
            eastern = timezone(timedelta(hours=-5))  # EST, not handling DST
        
        now = datetime.now(tz=eastern)
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now <= end

    async def _update_gex_if_needed(self) -> None:
        """Update GEX data every second during RTH, every 5 seconds otherwise."""
        current_time = datetime.now(timezone.utc).timestamp()
        
        # Determine update interval based on market hours
        update_interval = 1.0 if self._is_rth() else 5.0
        
        # Check if enough time has passed since last update
        if current_time - self.last_gex_update >= update_interval:
            await self.update_gex_data()
            self.last_gex_update = current_time


async def main():
    """Main entry point."""
    print("Starting live trader main()")
    import argparse

    parser = argparse.ArgumentParser(description='Live trading with enhanced GEX LSTM model')
    parser.add_argument('--model', default='ml/models/enhanced_gex_model.pt',
                       help='Path to trained model')
    parser.add_argument('--symbol', default='MNQ', help='Trading symbol')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Prediction threshold for trades')
    parser.add_argument('--max-position', type=int, default=1,
                       help='Maximum position size')
    parser.add_argument('--commission', type=float, default=0.42,
                       help='Commission cost per trade')
    parser.add_argument('--api-endpoint', default='http://localhost:8877/ml-trade',
                       help='API endpoint for trade updates')

    args = parser.parse_args()

    trader = LiveTrader(
        model_path=args.model,
        symbol=args.symbol,
        prediction_threshold=args.threshold,
        max_position_size=args.max_position,
        commission_cost=args.commission,
        api_endpoint=args.api_endpoint
    )
    print("LiveTrader created, starting run()")
    await trader.run()


if __name__ == '__main__':
    asyncio.run(main())
