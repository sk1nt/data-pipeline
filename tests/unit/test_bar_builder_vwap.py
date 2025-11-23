"""
Unit tests for VWAP calculations in bar_builder functions.
"""
import pandas as pd
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import numpy as np
from datetime import datetime

from ml.bar_builder import build_time_bars, build_volume_bars, build_dollar_bars


def test_vwap_time_bar():
    dt = datetime(2025, 11, 11, 10, 15, 0)
    ticks = pd.DataFrame({'timestamp': [dt, dt, dt], 'price': [100.0, 102.0, 98.0], 'volume': [1, 2, 3]})
    df = build_time_bars(ticks, seconds=1)
    # expected vwap
    expected_vwap = (100*1 + 102*2 + 98*3) / 6
    assert 'vwap' in df.columns
    assert np.isclose(df['vwap'].iloc[0], expected_vwap)


def test_vwap_volume_bar():
    dt = datetime(2025, 11, 11, 10, 15, 0)
    ticks = pd.DataFrame({'timestamp': [dt, dt, dt, dt], 'price': [100, 102, 100, 98], 'volume': [1, 1, 2, 3]})
    # bar size 3 -> two bars: first cum vol 1+1+2=4 meets size 3 -> first bar has (100*1 + 102*1 + 100*2)/4 = ...
    df = build_volume_bars(ticks, size=3)
    assert 'vwap' in df.columns
    expected_vwap_0 = (100*1 + 102*1) / 2
    assert np.isclose(df['vwap'].iloc[0], expected_vwap_0)


def test_vwap_dollar_bar():
    dt = datetime(2025, 11, 11, 10, 15, 0)
    ticks = pd.DataFrame({'timestamp': [dt, dt, dt], 'price': [100, 200, 50], 'volume': [1, 1, 2]})
    # trade_value = [100,200,100] cum = [100,300,400] size=250 -> first bar includes first two (100+200)/2 volume=2 -> vwap=150
    df = build_dollar_bars(ticks, size=250)
    assert 'vwap' in df.columns
    assert np.isclose(df['vwap'].iloc[0], 100.0)
