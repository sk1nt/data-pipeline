# TastyTrade TRIN Streaming

This pipeline streams TRIN market indicators from TastyTrade's DXLink/dxFeed
market data feed. TRIN is a calculated market indicator, so it should be
handled as a live price-like indicator update rather than as a normal trade in
an exchange-traded instrument.

## Symbols

Use these dxFeed symbols in `TASTYTRADE_STREAM_SYMBOLS`:

| Symbol | Meaning | Primary use |
| --- | --- | --- |
| `$TRIN` | NYSE TRIN | Standard broad-market Arms Index |
| `$TRINSP` | S&P 500 TRIN | SPX / ES decisions |
| `$TRINND` | NASDAQ 100 TRIN | NDX / NQ decisions |
| `$TRIN/Q` | NASDAQ TRIN | Broader NASDAQ confirmation |

The default stream symbol list includes:

```env
TASTYTRADE_STREAM_SYMBOLS=MES,MNQ,NQ,SPY,QQQ,VIX,$TRIN,$TRINSP,$TRINND,$TRIN/Q
```

Do not use `$TRIN.NQ` for dxFeed. The documented NASDAQ symbols are `$TRIN/Q`
for broad NASDAQ TRIN and `$TRINND` for NASDAQ 100 TRIN.

## Event Type

TRIN symbols are subscribed through dxFeed `TimeAndSale` events. dxFeed describes
`TimeAndSale` as representing a trade or other market event with a price, which
matches calculated market indicators such as TRIN.

Do not use `Summary.day_close_price` as the live TRIN value. `Summary` is session
OHLC data for charting, and `day_close_price` represents close/settlement data,
not the current intraday indicator value.

The streamer still subscribes to `Trade` events for the full symbol list. For
symbols starting with `$TRIN`, it additionally subscribes to `TimeAndSale` and
persists those updates using the same RedisTimeSeries trade schema:

```text
ts:trade:price:{SYMBOL}:tastytrade
ts:trade:size:{SYMBOL}:tastytrade
```

Examples:

```text
ts:trade:price:$TRIN:tastytrade
ts:trade:price:$TRINSP:tastytrade
ts:trade:price:$TRINND:tastytrade
ts:trade:price:$TRIN/Q:tastytrade
```

Those live TRIN updates are also buffered into DuckDB and daily Parquet files for
historical replay:

```text
data/trin_history.db
data/parquet/trin/$TRIN/YYYYMMDD.parquet
```

You can query the persisted history through the app:

```text
/lookup/trin_history?symbol=$TRIN&limit=100
```

## Operational Notes

After changing `TASTYTRADE_STREAM_SYMBOLS` or deploying code changes, restart the
`tastytrade` service so DXLink resubscribes with the updated symbol/event set.

Quick Redis check:

```bash
redis-cli --scan --pattern 'ts:*TRIN*'
```

## References

- dxFeed market indicators symbol list:
  https://kb.dxfeed.com/en/data/calculated-data/access-to-dxfeed-market-indicators.html
- dxFeed market event semantics:
  https://kb.dxfeed.com/en/data-model/market-events/dxfeed-api-market-events.html
- TastyTrade streaming market data guide:
  https://developer.tastytrade.com/streaming-market-data/
