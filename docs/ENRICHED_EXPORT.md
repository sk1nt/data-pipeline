# Enriched MNQ 1s Bars (OHLCV + GEX)

Daily Parquet outputs combining MNQ ticks with GEX snapshots/strikes for `NQ_NDX`.

## Location

```
data/enriched/<symbol>/<YYYYMMDD>.parquet
```

Default symbol: `MNQ`.

## Session Window

- 09:30:00 to 16:00:00 America/New_York (inclusive second)
- Timestamps stored in UTC; `source_snapshot_ts` remains epoch milliseconds from the snapshot.

## Columns (schema)

| column | type | description |
| --- | --- | --- |
| `timestamp` | TIMESTAMP (UTC) | Second boundary for the bar |
| `open`,`high`,`low`,`close` | DOUBLE | OHLC derived from MNQ ticks (forward-filled when no trade in the second) |
| `volume` | BIGINT | Sum of volumes in the second (0 if no ticks) |
| `gap_filled` | BOOLEAN | True if tick second was empty or no snapshot at that second |
| `source_snapshot_ts` | BIGINT | Snapshot timestamp in epoch ms that supplied GEX fields |
| `spot` | DOUBLE | GEX spot price |
| `zero_gamma` | DOUBLE | Zero gamma level |
| `net_gex` | DOUBLE | Net gamma (also used for `net_gex_vol` and `sum_gex_vol`) |
| `net_gex_vol` | DOUBLE | Mirror of `net_gex` (per user requirement) |
| `sum_gex_vol` | DOUBLE | Mirror of `net_gex` (per user requirement) |
| `net_gex_oi` | DOUBLE | Sum of GEX open interest |
| `major_pos_vol` | DOUBLE | Primary positive strike gamma (millions proxy) |
| `major_pos_oi` | DOUBLE | OI at primary positive strike |
| `major_neg_vol` | DOUBLE | Primary negative strike gamma (millions proxy) |
| `major_neg_oi` | DOUBLE | OI at primary negative strike |
| `major_pos_strike` | DOUBLE | Strike price for primary positive strike |
| `major_neg_strike` | DOUBLE | Strike price for primary negative strike |
| `major_pos_vol_2/3` | DOUBLE | Next two positive strike gamma values (millions proxy) |
| `major_pos_strike_2/3` | DOUBLE | Strike prices for the above |
| `major_neg_vol_2/3` | DOUBLE | Next two negative strike gamma values (millions proxy) |
| `major_neg_strike_2/3` | DOUBLE | Strike prices for the above |
| `delta_risk_reversal` | DOUBLE | As provided by snapshots |
| `min_dte` | INTEGER | Minimum DTE |
| `sec_min_dte` | INTEGER | Secondary minimum DTE |
| `max_priors` | VARCHAR/JSON | Prior windows as stringified JSON |

## Generation

```
python scripts/export_enriched_bars.py --start-date 2025-10-20 --end-date 2025-10-21
```

Arguments:
- `--symbol` (default: MNQ)
- `--gex-ticker` (default: NQ_NDX)
- `--tick-root` (default: data/parquet/tick)
- `--gex-db` (default: data/gex_data.db)
- `--output-root` (default: data/enriched)

## Gap Policy

- A pre-open snapshot (previous day close OI) is expected and forward-filled from the first second.
- If a second has no tick trades, prices are forward-filled from the previous close and `gap_filled` is set to true. A per-run log line reports the count of such gaps.
- Snapshot gaps are also flagged via `gap_filled` (should be zero in normal operation).

## Consumer Example (DuckDB)

```sql
SELECT timestamp, open, close, spot, net_gex, major_pos_vol, major_neg_vol
FROM read_parquet('data/enriched/MNQ/20251020.parquet')
LIMIT 5;
```
