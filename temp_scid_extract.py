import numpy as np
import polars as pl
import datetime as dt
from pathlib import Path

# Load SCID
scid_file = Path('/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid')
sciddtype = np.dtype([
    ('SCDateTime','<u8'),
    ('Open','<f4'), ('High','<f4'), ('Low','<f4'), ('Close','<f4'),
    ('NumTrades','<u4'), ('TotalVolume','<u4'), ('BidVolume','<u4'), ('AskVolume','<u4')
])
arr = np.fromfile(scid_file, dtype=sciddtype, offset=56)

# Convert SCDateTime (microseconds since 1899-12-30) to Unix ms
SC_EPOCH_MS = 25569 * 86400 * 1000
ms = arr['SCDateTime'] // 1000
unix_ms = ms - SC_EPOCH_MS

# Example: filter a minute (13:30–13:30:59 UTC on 2025-10-13)
start = int(dt.datetime(2025, 10, 13, 13, 30).timestamp() * 1000)
end   = int(dt.datetime(2025, 10, 13, 13, 31).timestamp() * 1000)
mask = (unix_ms >= start) & (unix_ms < end)
sel = arr[mask]

if sel.size:
    print({
        "open": sel['Close'][0],
        "high": sel['Close'].max(),
        "low":  sel['Close'].min(),
        "close": sel['Close'][-1],
        "volume": int(sel['TotalVolume'].sum()),
        "num_trades": int(sel['NumTrades'].sum()),
        "bid_volume": int(sel['BidVolume'].sum()),
        "ask_volume": int(sel['AskVolume'].sum()),
        "first_unix_ms": int(unix_ms[mask].min()),
        "last_unix_ms":  int(unix_ms[mask].max())
    })
else:
    print("no rows for that minute")

# Example: count rows for a whole day
day_start = int(dt.datetime(2025,10,13).timestamp()*1000)
day_end   = int(dt.datetime(2025,10,14).timestamp()*1000)
day_mask = (unix_ms >= day_start) & (unix_ms < day_end)
print("rows for 2025-10-13 UTC:", int(day_mask.sum()))

# Create DataFrame for the day
if day_mask.sum() > 0:
    df = pl.DataFrame({
        'timestamp_ms': unix_ms[day_mask],
        'open': arr['Open'][day_mask],
        'high': arr['High'][day_mask],
        'low': arr['Low'][day_mask],
        'close': arr['Close'][day_mask],
        'num_trades': arr['NumTrades'][day_mask],
        'total_volume': arr['TotalVolume'][day_mask],
        'bid_volume': arr['BidVolume'][day_mask],
        'ask_volume': arr['AskVolume'][day_mask]
    })
    df.write_parquet('/home/rwest/projects/data-pipeline/mnq_2025_10_13_ticks.parquet')
    print("Saved to mnq_2025_10_13_ticks.parquet")
else:
    print("No data for the day")

# Check min and max timestamps
if arr.size > 0:
    min_ts = dt.datetime.fromtimestamp(unix_ms.min() / 1000)
    max_ts = dt.datetime.fromtimestamp(unix_ms.max() / 1000)
    print(f"Data from {min_ts} to {max_ts}")
else:
    print("No data in file")