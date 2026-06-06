# Inter-repo contract (data-pipeline ↔ data-trading)

Canonical copy: keep identical to `data-trading/docs/INTER_REPO.md`.

## Redis

| Channel | Publisher | Consumer |
|---------|-----------|----------|
| `gex:snapshot:stream` | **this repo** | data-trading |
| `sweep:alert:{symbol}` | data-trading | intelligence.html |
| `market:dom/cvd:{symbol}` | data-trading | data-trading |

Files: `contracts/redis_channels.py`, `contracts/sweep_alert.py`, `contracts/CONTRACT_VERSION`.

## Parquet (server is source of truth)

| Path on server | Used by data-trading via |
|----------------|--------------------------|
| `data/parquet/tick/{SYMBOL}/{day}.parquet` | `TICK_PARQUET_REMOTE_DIR` |
| `data/parquet/candles/1m/{SYMBOL}/{day}.parquet` | `CANDLE_PARQUET_REMOTE_DIR` |
| `data/source/gexbot/` | `GEX_JSON_ROOT` |

## Verify

```bash
python scripts/verify_contracts.py
python scripts/verify_inter_repo_sync.py
```

Bump `CONTRACT_VERSION` in both repos when changing contracts.