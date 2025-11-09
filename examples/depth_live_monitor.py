#!/usr/bin/env python3
"""Real-time Sierra Chart market depth (.depth) monitoring."""

import argparse
import time
import struct
import os
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market_ml.data import find_sierra_chart_depth_file

# Sierra Chart depth file constants
DEPTH_HEADER_SIZE = 64
DEPTH_RECORD_SIZE = 24

def parse_depth_record(record_bytes: bytes) -> Dict:
    """Parse a single depth record."""
    try:
        # Unpack: DateTime (8), Command (1), Flags (1), NumOrders (2), Price (4), Quantity (4), Reserved (4)
        dt_raw, cmd, flags, num_orders, price, qty, reserved = struct.unpack('<QBBHfII', record_bytes)
        
        # Convert Sierra Chart timestamp - it appears to be different format
        # Try different interpretations
        if dt_raw > 0:
            # Try as microseconds from epoch
            try:
                timestamp = pd.Timestamp(dt_raw, unit='us')
                if timestamp.year < 1900 or timestamp.year > 2100:
                    # Try as milliseconds
                    timestamp = pd.Timestamp(dt_raw, unit='ms')
                    if timestamp.year < 1900 or timestamp.year > 2100:
                        # Try as seconds  
                        timestamp = pd.Timestamp(dt_raw, unit='s')
            except:
                timestamp = pd.Timestamp.now()
        else:
            timestamp = pd.Timestamp.now()
            
        # Determine side based on command/flags (simplified interpretation)
        side = 'BUY' if cmd in [1, 2, 4] else 'SELL' if cmd in [3, 5] else 'UNKNOWN'
        
        return {
            'timestamp': timestamp,
            'price': price,
            'size': qty,
            'num_orders': num_orders,
            'side': side,
            'command': cmd,
            'flags': flags
        }
    except Exception as e:
        print(f"Error parsing depth record: {e}")
        return None

def get_current_depth_file(symbol: str) -> Path:
    """Get the current depth file path for a symbol."""
    discovered = find_sierra_chart_depth_file(symbol)
    if discovered is not None:
        return discovered

    today = datetime.now().strftime('%Y-%m-%d')
    depth_dir = Path(os.getenv("SIERRA_DEPTH_DIR", "/mnt/c/SierraChart/Data/MarketDepthData"))
    fallback_contracts = {
        'ES': 'ESZ25_FUT_CME',
        'MES': 'MESZ25_FUT_CME',
        'NQ': 'NQZ25_FUT_CME',
        'MNQ': 'MNQZ25_FUT_CME',
        'YM': 'YMZ25_FUT_CME',
        'MYM': 'MYMZ25_FUT_CME',
    }
    symbol_upper = symbol.upper()
    contract_name = fallback_contracts.get(symbol_upper, f"{symbol_upper}Z25_FUT_CME")
    return depth_dir / f"{contract_name}.{today}.depth"

def stream_depth_data(symbol: str, poll_interval: float = 1.0):
    """Stream market depth data for a symbol."""
    depth_file = get_current_depth_file(symbol)
    
    if not depth_file.exists():
        print(f"Warning: Depth file not found: {depth_file}")
        return
    
    print(f"=== Depth Monitor: {symbol} ===")
    print(f"File: {depth_file}")
    print(f"Poll interval: {poll_interval}s")
    print("Monitoring depth changes...\n")
    
    last_size = 0
    last_position = DEPTH_HEADER_SIZE
    
    # Order book tracking
    bids = {}  # price -> size
    asks = {}  # price -> size
    
    while True:
        try:
            current_size = os.path.getsize(depth_file)
            
            if current_size > last_size and current_size > DEPTH_HEADER_SIZE:
                # New data available
                with open(depth_file, 'rb') as f:
                    f.seek(last_position)
                    
                    # Read new records
                    while f.tell() < current_size:
                        record_bytes = f.read(DEPTH_RECORD_SIZE)
                        if len(record_bytes) < DEPTH_RECORD_SIZE:
                            break
                            
                        record = parse_depth_record(record_bytes)
                        if record and record['price'] > 0:
                            # Update order book
                            price = record['price']
                            size = record['size']
                            
                            # Simplified order book logic
                            if record['side'] == 'BUY' or record['command'] in [1, 2]:
                                if size > 0:
                                    bids[price] = size
                                else:
                                    bids.pop(price, None)
                            elif record['side'] == 'SELL' or record['command'] in [3, 4]:
                                if size > 0:
                                    asks[price] = size
                                else:
                                    asks.pop(price, None)
                            
                            # Display update
                            timestamp = record['timestamp'].strftime('%H:%M:%S.%f')[:-3]
                            print(f"[{timestamp}] {symbol} | Price: ${price:.2f} | Size: {size} | "
                                  f"Side: {record['side']} | Orders: {record['num_orders']}")
                
                # Update position
                last_position = current_size
                last_size = current_size
                
                # Show top of book
                if bids and asks:
                    best_bid = max(bids.keys())
                    best_ask = min(asks.keys())
                    spread = best_ask - best_bid
                    
                    print(f"   📊 Best Bid: ${best_bid:.2f} ({bids[best_bid]}) | "
                          f"Best Ask: ${best_ask:.2f} ({asks[best_ask]}) | "
                          f"Spread: ${spread:.2f}\n")
            
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print(f"\nStopping {symbol} depth monitor...")
            break
        except Exception as e:
            print(f"Error monitoring depth: {e}")
            time.sleep(poll_interval)

def main():
    parser = argparse.ArgumentParser(description="Real-time Sierra Chart depth monitoring")
    parser.add_argument("--symbol", required=True, help="Symbol to monitor (ES, MNQ, etc.)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds")
    
    args = parser.parse_args()
    
    try:
        stream_depth_data(args.symbol, args.poll_interval)
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
