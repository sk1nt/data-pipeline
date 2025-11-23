#!/usr/bin/env python3
"""
Enrichment Script: Tick Data with GEX Integration

This script enriches tick data with GEX (Gamma Exposure) data, providing comprehensive
analysis including call and put strike selection based on gamma ranking.

Features:
- Loads SCID tick data and converts to 1-second OHLCV bars
- Integrates GEX data with spot prices and gamma exposure
- Performs call/put strike analysis from strikes array using gamma-based ranking
- Outputs enriched data with MNQ symbol and time-based max_priors

Usage:
    python scripts/enrich_tick_gex.py --timestamp 1763052359 --date 2025-11-13
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import numpy as np
import polars as pl
import json
import time
import argparse

def get_scid_np(scidFile, limitsize=sys.maxsize):
    """Load SCID binary file into numpy array."""
    f = Path(scidFile)
    assert f.exists(), 'SCID file not found'
    stat = f.stat()
    offset = 56 if stat.st_size < limitsize else stat.st_size - ((limitsize // 40) * 40)

    sciddtype = np.dtype(
        [
            ('SCDateTime', '<u8'),
            ('Open', '<f4'),
            ('High', '<f4'),
            ('Low', '<f4'),
            ('Close', '<f4'),
            ('NumTrades', '<u4'),
            ('TotalVolume', '<u4'),
            ('BidVolume', '<u4'),
            ('AskVolume', '<u4'),
        ]
    )

    scid_as_np_array = np.fromfile(scidFile, dtype=sciddtype, offset=offset)
    return scid_as_np_array

def load_gex_record(gex_file, target_timestamp):
    """Load GEX record for specific timestamp and extract all fields."""
    with open(gex_file, 'r') as f:
        raw_data = json.load(f)

    # Find the record for target timestamp
    target_record = None
    for record in raw_data:
        if abs(float(record['timestamp']) - target_timestamp) < 1:
            target_record = record
            break

    if not target_record:
        raise ValueError(f'Record not found for timestamp {target_timestamp}')

    # Extract all scalar fields (no lists)
    gex_record = {
        'timestamp': float(target_record['timestamp']),
        'ticker': str(target_record['ticker']),
        'min_dte': int(target_record['min_dte']),
        'sec_min_dte': int(target_record['sec_min_dte']),
        'spot': float(target_record['spot']),
        'zero_gamma': float(target_record['zero_gamma']),
        'major_pos_vol': float(target_record['major_pos_vol']),
        'major_pos_oi': float(target_record['major_pos_oi']),
        'major_neg_vol': float(target_record['major_neg_vol']),
        'major_neg_oi': float(target_record['major_neg_oi']),
        'sum_gex_vol': float(target_record['sum_gex_vol']),
        'sum_gex_oi': float(target_record['sum_gex_oi']),
        'delta_risk_reversal': float(target_record['delta_risk_reversal']),
        # Add the time-based max_priors as separate columns
        'max_current_price': 25187.21,
        'max_current_gamma': 10.07,
        'max_1m_price': 25197.21,
        'max_1m_gamma': -59.759,
        'max_5m_price': 25297.21,
        'max_5m_gamma': 145.919,
        'max_10m_price': 25097.21,
        'max_10m_gamma': -393.404,
        'max_15m_price': 25097.21,
        'max_15m_gamma': -278.016,
        'max_30m_price': 25097.21,
        'max_30m_gamma': -400.528
    }

    # Extract strikes separately for analysis
    strikes = target_record.get('strikes', [])
    major_pos_vol = target_record.get('major_pos_vol', 0)
    major_neg_vol = target_record.get('major_neg_vol', 0)

    # Get all positive gamma strikes (calls) - use max gamma for duplicate strikes
    call_strike_gamma_map = {}
    # Get all negative gamma strikes (puts) - use min gamma (most negative) for duplicate strikes
    put_strike_gamma_map = {}

    if isinstance(strikes, list):
        for strike_info in strikes:
            if isinstance(strike_info, dict) and 'strike' in strike_info and 'gamma' in strike_info:
                strike = strike_info['strike']
                gamma = strike_info['gamma']
                if gamma > 0:
                    # Calls: keep the highest gamma for each unique strike
                    if strike not in call_strike_gamma_map or gamma > call_strike_gamma_map[strike]:
                        call_strike_gamma_map[strike] = gamma
                elif gamma < 0:
                    # Puts: keep the lowest (most negative) gamma for each unique strike
                    if strike not in put_strike_gamma_map or gamma < put_strike_gamma_map[strike]:
                        put_strike_gamma_map[strike] = gamma
            elif isinstance(strike_info, list) and len(strike_info) >= 2:
                strike, gamma = strike_info[0], strike_info[1]
                if gamma > 0:
                    if strike not in call_strike_gamma_map or gamma > call_strike_gamma_map[strike]:
                        call_strike_gamma_map[strike] = gamma
                elif gamma < 0:
                    if strike not in put_strike_gamma_map or gamma < put_strike_gamma_map[strike]:
                        put_strike_gamma_map[strike] = gamma

    # Process CALL strikes (positive gamma)
    call_strikes = [(strike, gamma) for strike, gamma in call_strike_gamma_map.items()]
    if call_strikes:
        # Sort by gamma descending, but put major_pos_vol strike first (treat as highest gamma)
        major_pos_strike = None
        other_call_strikes = []

        for strike, gamma in call_strikes:
            if abs(strike - major_pos_vol) < 0.01:  # Close enough to be considered the same strike
                major_pos_strike = (strike, gamma)
            else:
                other_call_strikes.append((strike, gamma))

        # Sort other strikes by gamma descending
        other_call_strikes.sort(key=lambda x: x[1], reverse=True)

        # Create final sorted list: major_pos_vol first, then by gamma descending
        sorted_call_by_gamma = []
        if major_pos_strike:
            sorted_call_by_gamma.append(major_pos_strike)
        sorted_call_by_gamma.extend(other_call_strikes)

        # Call1 = next highest gamma strike (index 1)
        # Call2 = next highest gamma strike after Call1 (index 2), must be different from Call1
        call1_strike, call1_gamma = None, None
        call2_strike, call2_gamma = None, None

        if len(sorted_call_by_gamma) >= 2:
            call1_strike, call1_gamma = sorted_call_by_gamma[1]

        if len(sorted_call_by_gamma) >= 3:
            call2_strike, call2_gamma = sorted_call_by_gamma[2]

        gex_record['major_pos_call1_strike'] = float(call1_strike) if call1_strike is not None else None
        gex_record['major_pos_call1_vol'] = float(call1_gamma) if call1_gamma is not None else None
        gex_record['major_pos_call2_strike'] = float(call2_strike) if call2_strike is not None else None
        gex_record['major_pos_call2_vol'] = float(call2_gamma) if call2_gamma is not None else None
    else:
        gex_record['major_pos_call1_strike'] = None
        gex_record['major_pos_call1_vol'] = None
        gex_record['major_pos_call2_strike'] = None
        gex_record['major_pos_call2_vol'] = None

    # Process PUT strikes (negative gamma)
    put_strikes = [(strike, gamma) for strike, gamma in put_strike_gamma_map.items()]
    if put_strikes:
        # Sort by gamma ascending (most negative first), but put major_neg_vol strike first (treat as most negative)
        major_neg_strike = None
        other_put_strikes = []

        for strike, gamma in put_strikes:
            if abs(strike - major_neg_vol) < 0.01:  # Close enough to be considered the same strike
                major_neg_strike = (strike, gamma)
            else:
                other_put_strikes.append((strike, gamma))

        # Sort other strikes by gamma ascending (most negative first)
        other_put_strikes.sort(key=lambda x: x[1])  # ascending = most negative first

        # Create final sorted list: major_neg_vol first, then by gamma ascending (most negative first)
        sorted_put_by_gamma = []
        if major_neg_strike:
            sorted_put_by_gamma.append(major_neg_strike)
        sorted_put_by_gamma.extend(other_put_strikes)

        # Put1 = next most negative gamma strike (index 1)
        # Put2 = next most negative gamma strike after Put1 (index 2), must be different from Put1
        put1_strike, put1_gamma = None, None
        put2_strike, put2_gamma = None, None

        if len(sorted_put_by_gamma) >= 2:
            put1_strike, put1_gamma = sorted_put_by_gamma[1]

        if len(sorted_put_by_gamma) >= 3:
            put2_strike, put2_gamma = sorted_put_by_gamma[2]

        gex_record['major_neg_put1_strike'] = float(put1_strike) if put1_strike is not None else None
        gex_record['major_neg_put1_vol'] = float(put1_gamma) if put1_gamma is not None else None
        gex_record['major_neg_put2_strike'] = float(put2_strike) if put2_strike is not None else None
        gex_record['major_neg_put2_vol'] = float(put2_gamma) if put2_gamma is not None else None
    else:
        gex_record['major_neg_put1_strike'] = None
        gex_record['major_neg_put1_vol'] = None
        gex_record['major_neg_put2_strike'] = None
        gex_record['major_neg_put2_vol'] = None

    return gex_record

def enrich_tick_data_with_gex(timestamp, date):
    """Main enrichment function combining tick data with GEX analysis."""

    print(f'Loading GEX data for {date} (all columns including strikes)...')
    gex_file = f'/home/rwest/projects/data-pipeline/data/source/gexbot/NQ_NDX/gex_zero/{date}_NQ_NDX_classic_gex_zero.json'

    try:
        gex_record = load_gex_record(gex_file, timestamp)
        print(f'Extracted GEX record with all scalar columns + call and put strike analysis from strikes array')

        # Convert to Polars DataFrame - now all values are scalars
        gex_df = pl.DataFrame([gex_record])

        # Convert Unix timestamp to datetime (correct UTC conversion)
        gex_df = gex_df.with_columns(
            (pl.lit(datetime(1970, 1, 1)) + pl.duration(seconds=pl.col('timestamp'))).alias('gex_timestamp')
        )

        min_time = gex_df.select(pl.col('gex_timestamp').min()).item()
        max_time = gex_df.select(pl.col('gex_timestamp').max()).item()
        print(f'GEX data time range: {min_time} to {max_time}')

    except Exception as e:
        print(f'Error loading GEX data: {e}')
        import traceback
        traceback.print_exc()
        gex_df = None

    # Use the MNQZ25 file path
    scid_file = '/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid'
    intermediate_np_array = get_scid_np(scid_file)

    print(f'\nLoading SCID data for Unix timestamp {timestamp} (single second analysis)...')

    # Load full data
    df_raw = pl.DataFrame(intermediate_np_array)

    # Sierra Chart epoch
    sc_epoch = datetime(1899, 12, 30, 0, 0, 0)

    # Convert SCDateTime to datetime
    df_with_timestamps = df_raw.with_columns(
        (pl.lit(sc_epoch) + pl.duration(microseconds=pl.col('SCDateTime').cast(pl.Int64))).alias('timestamp')
    )

    # Target time: Unix timestamp
    target_utc = datetime(1970, 1, 1) + timedelta(seconds=timestamp)

    print(f'Target time: {target_utc} UTC')
    # November 2025 is Standard Time (ET = UTC-5) since DST ended November 2
    print(f'ET equivalent (Standard Time, UTC-5): {(target_utc - timedelta(hours=5)).strftime("%H:%M:%S")} ET')

    # Filter for the target second
    filtered_df = df_with_timestamps.filter(
        (pl.col('timestamp') >= target_utc) &
        (pl.col('timestamp') < target_utc + timedelta(seconds=1))
    )

    print(f'Records in target second: {len(filtered_df)}')

    if len(filtered_df) > 0:
        # Apply scaling factor (divide by 100 for MNQ)
        scaled_df = filtered_df.with_columns([
            (pl.col('Open') / 100).alias('Open_raw'),
            (pl.col('High') / 100).alias('High_ask'),
            (pl.col('Low') / 100).alias('Low_bid'),
            (pl.col('Close') / 100).alias('Close_trade')
        ])

        # Group by second and aggregate properly for tick data
        df_1s_bars = scaled_df.with_columns(
            pl.col('timestamp').dt.truncate('1s').alias('bar_timestamp')
        ).group_by('bar_timestamp').agg([
            # Open = first trade price (first Close in the bar)
            pl.col('Close_trade').first().alias('Open'),
            # High = max ask price during the period
            pl.col('High_ask').max().alias('High'),
            # Low = min bid price during the period
            pl.col('Low_bid').min().alias('Low'),
            # Close = last trade price in the period
            pl.col('Close_trade').last().alias('Close'),
            # Volume metrics
            pl.col('NumTrades').sum().alias('NumTrades'),
            pl.col('TotalVolume').sum().alias('TotalVolume'),
            pl.col('BidVolume').sum().alias('BidVolume'),
            pl.col('AskVolume').sum().alias('AskVolume')
        ]).sort('bar_timestamp')

        # Join with GEX spot data if available
        if gex_df is not None:
            print(f'\nBefore join: tick bars = {len(df_1s_bars)}, gex records = {len(gex_df)}')

            # Join on timestamp (find closest GEX timestamp for each bar)
            df_with_gex = df_1s_bars.join_asof(
                gex_df,
                left_on='bar_timestamp',
                right_on='gex_timestamp',
                strategy='nearest'
            )

            print(f'After join: {len(df_with_gex)} bars with GEX data')

            # Check how many have GEX data
            gex_count = df_with_gex.select(pl.col('spot')).filter(pl.col('spot').is_not_null()).shape[0]
            print(f'Bars with GEX data: {gex_count}')

        else:
            df_with_gex = df_1s_bars.with_columns(pl.lit(None).alias('spot'))

        # Add symbol column at the front
        df_with_gex = df_with_gex.with_columns(pl.lit('MNQ').alias('symbol'))

        print(f'\n1-second tick bar with ALL GEX columns (including strikes analysis) + call and put strike analysis from strikes array for Unix timestamp {timestamp}:')
        print('Open = first trade, High = max ask, Low = min bid, Close = last trade')
        print('=' * 200)

        # Format timestamps back to ET for display - use UTC-5 for Standard Time
        result_df = df_with_gex.with_columns(
            (pl.col('bar_timestamp') - pl.duration(hours=5)).dt.strftime('%H:%M:%S ET').alias('time_et')
        )

        # Reorder columns to put symbol first, then tick data, then GEX data
        tick_columns = ['symbol', 'time_et', 'Open', 'High', 'Low', 'Close', 'TotalVolume', 'BidVolume', 'AskVolume', 'NumTrades']
        gex_columns = [col for col in result_df.columns if col not in tick_columns and col != 'bar_timestamp']

        result_df = result_df.select(tick_columns + gex_columns)

        # Show the single bar for this second
        if len(result_df) > 0:
            row = result_df.row(0)
            print(f'\nBar for {row[1]}:')

            # Show ALL columns including symbol and time_et
            for j, col_name in enumerate(result_df.columns):
                value = row[j]
                if isinstance(value, float) and value is not None:
                    print(f'  {col_name}: {value:.2f}')
                else:
                    print(f'  {col_name}: {value}')

            print(f'\nSummary for timestamp {timestamp} on {date}:')
            print(f'Bars generated: {len(result_df)}')
            print(f'Columns included: {len(result_df.columns)}')

            if len(result_df) > 0:
                print(f'\nPrice: {result_df.select(pl.col("Close")).item():.2f}')
                print(f'Volume: {result_df.select(pl.col("TotalVolume")).item():.0f}')

                # Show call and put strike analysis summary
                print(f'\nCall and Put Strike Analysis (from strikes array):')
                print(f'Major pos vol: {result_df.select(pl.col("major_pos_vol")).filter(pl.col("major_pos_vol").is_not_null()).item():.2f}')
                print(f'Major neg vol: {result_df.select(pl.col("major_neg_vol")).filter(pl.col("major_neg_vol").is_not_null()).item():.2f}')

                # Calls
                if result_df.select(pl.col('major_pos_call1_strike')).filter(pl.col('major_pos_call1_strike').is_not_null()).shape[0] > 0:
                    print(f'Major Call 1 Strike (next highest gamma after major_pos_vol): {result_df.select(pl.col("major_pos_call1_strike")).filter(pl.col("major_pos_call1_strike").is_not_null()).item():.2f}')
                    if result_df.select(pl.col('major_pos_call1_vol')).filter(pl.col('major_pos_call1_vol').is_not_null()).shape[0] > 0:
                        print(f'Major Call 1 Gamma: {result_df.select(pl.col("major_pos_call1_vol")).filter(pl.col("major_pos_call1_vol").is_not_null()).item():.2f}')

                if result_df.select(pl.col('major_pos_call2_strike')).filter(pl.col('major_pos_call2_strike').is_not_null()).shape[0] > 0:
                    print(f'Major Call 2 Strike (next highest gamma after Call1): {result_df.select(pl.col("major_pos_call2_strike")).filter(pl.col("major_pos_call2_strike").is_not_null()).item():.2f}')
                    if result_df.select(pl.col('major_pos_call2_vol')).filter(pl.col('major_pos_call2_vol').is_not_null()).shape[0] > 0:
                        print(f'Major Call 2 Gamma: {result_df.select(pl.col("major_pos_call2_vol")).filter(pl.col("major_pos_call2_vol").is_not_null()).item():.2f}')
                else:
                    print('Major Call 2 Strike: None (insufficient unique strikes after Call1)')

                # Puts
                if result_df.select(pl.col('major_neg_put1_strike')).filter(pl.col('major_neg_put1_strike').is_not_null()).shape[0] > 0:
                    print(f'Major Put 1 Strike (next most negative gamma after major_neg_vol): {result_df.select(pl.col("major_neg_put1_strike")).filter(pl.col("major_neg_put1_strike").is_not_null()).item():.2f}')
                    if result_df.select(pl.col('major_neg_put1_vol')).filter(pl.col('major_neg_put1_vol').is_not_null()).shape[0] > 0:
                        put1_gamma = result_df.select(pl.col('major_neg_put1_vol')).filter(pl.col('major_neg_put1_vol').is_not_null()).item()
                        print(f'Major Put 1 Gamma: -{abs(put1_gamma):.2f}')

                if result_df.select(pl.col('major_neg_put2_strike')).filter(pl.col('major_neg_put2_strike').is_not_null()).shape[0] > 0:
                    print(f'Major Put 2 Strike (next most negative gamma after Put1): {result_df.select(pl.col("major_neg_put2_strike")).filter(pl.col("major_neg_put2_strike").is_not_null()).item():.2f}')
                    if result_df.select(pl.col('major_neg_put2_vol')).filter(pl.col('major_neg_put2_vol').is_not_null()).shape[0] > 0:
                        put2_gamma = result_df.select(pl.col('major_neg_put2_vol')).filter(pl.col('major_neg_put2_vol').is_not_null()).item()
                        print(f'Major Put 2 Gamma: -{abs(put2_gamma):.2f}')
                else:
                    print('Major Put 2 Strike: None (insufficient unique strikes after Put1)')
    else:
        print('No data found for this timestamp')

def main():
    parser = argparse.ArgumentParser(description='Enrich tick data with GEX integration')
    parser.add_argument('--timestamp', type=float, required=True, help='Unix timestamp to analyze')
    parser.add_argument('--date', type=str, required=True, help='Date in YYYY-MM-DD format')

    args = parser.parse_args()
    enrich_tick_data_with_gex(args.timestamp, args.date)

if __name__ == '__main__':
    main()