# Data Sources & Storage Topology

This note captures how raw market feeds land in the repository, where they are stored (DuckDB/Parquet), and how complete each dataset is over the most recent **45 trading days** (weekdays only, observed on **2025‑11‑07**, covering the window **2025‑09‑08 → 2025‑11‑07**).

## Ingestion Overview

```mermaid
flowchart LR
    subgraph Sources
        TT[TastyTrade DXFeed<br/>real-time ticks]
        SCID[/Sierra Chart SCID<br/>(historical ticks)/]
        SCDD[/Sierra Chart SCDD<br/>(depth events)/]
        GEXBOT[[GEXBot API<br/>webhooks & history]]
        SCHWAB[[Schwab API<br/>OAuth + stream]]
    end

    TT -->|live ticks| TDDB[(data/tick_data.db)]
    SCID -->|batch ticks| TDDB
    SCDD -->|order book cmds| MBO[(data/tick_mbo_data.db)]
    MBO -->|daily export| DEPTH[[data/parquet/depth/YYYYMMDD/mnq_depth.parquet]]

    GEXBOT -->|snapshots + strikes| GEXDB[(data/gex_data.db)]
    GEXBOT -->|history queue| GEXHIST[(data/gex_data.db)]
    SCHWAB -->|auth + market data| TDDB
    SCHWAB -->|options/quotes| GEXDB

    TDDB -.->|future export| TPARQ[[data/parquet/ticks/YYYYMMDD/<symbol>.parquet]]
```

## Coverage Snapshot (last 45 trading days)

| Dataset | Storage | Primary Source | Observed Range (UTC) | Trading Days Present* | Coverage Notes |
|---------|---------|----------------|----------------------|----------------------|----------------|
| MNQ/NQ ticks | `data/tick_data.db::tick_data` | TastyTrade DXFeed, Sierra Chart SCID, Schwab | _None_ (table empty) | **0 / 45** | Ingestion not yet run after repo setup; DuckDB currently has zero rows. Need to hydrate from live feeds or by backfilling SCID files. |
| MNQ order-book depth (80 levels + last trade) | `data/parquet/depth/YYYYMMDD/mnq_depth.parquet` (via `data/tick_mbo_data.db`) | Sierra Chart SCDD exports | 2025‑09‑19 → 2025‑11‑07 | **36 / 45** | Missing 9 weekday sessions in the 45-day window (the kickoff week 2025‑09‑08→09‑12 plus 2025‑09‑26, 2025‑09‑29‑30). |
| NQ_NDX GEX snapshots & strikes | `data/parquet/gexbot/NQ_NDX/<endpoint>/<YYYYMMDD>.strikes.parquet` (mounted via DuckDB view `parquet_gex_strikes`) | GEXBot API/webhooks (runs Mon–Fri, 09:30 ET) | 2025‑09‑02 → 2025‑11‑07 | **45 / 45** | Complete coverage for the last 45 trading days (JSON source retained under `data/source/gexbot/`). |
| GEX import lineage/history | `data/gex_data.db` (`gex_history_queue`, `import_jobs`, `gex_snapshots`, `gex_strikes`) | GEXBot `/gex_history_url` queue | 2025‑09 onward | N/A (metadata) | Queue + canonical snapshots/strikes (JSON sources retained under `data/source/gexbot/`). |
| Tick/Depth metadata | `data/tick_mbo_data.db` tables (`mnq_depth_metadata`, `mnq_ticks`) | Sierra Chart | Metadata only | `mnq_depth_metadata` populated; `mnq_ticks` empty | Mirrors the depth export process; does not itself provide tick data. |

*Trading days = weekdays only (Mon–Fri). Weekends and market holidays are excluded from both the numerator and denominator.

> ⚠️ Completeness percentages were derived by scanning the DuckDB/Parquet assets in `data/` on 2025‑11‑07. Re-run the coverage script after new imports to keep this table current.

# Ticker Normalization - Quick Reference

## Summary

All GEX and futures code now uses canonical ticker formats with automatic normalization.

## Canonical Ticker Map

| Input Ticker | Canonical Format | File Prefix | API Format |
|--------------|------------------|-------------|------------|
| ES, ES_SPX, SPX | **ES_SPX** | `es` | ES_SPX |
| NQ, NQ_NDX, NDX | **NQ_NDX** | `nq` | NQ_NDX |
| MNQ, MNQ_NDX | **NQ_NDX** | `mnq` | NQ_NDX |
| MES, MES_SPX | **ES_SPX** | `mes` | ES_SPX |

## Usage Examples

## Next Steps

1. **Ticks**: Run the Schwab/TastyTrade ingestion jobs (or the SCID extractor) to populate `data/tick_data.db`, then export aligned Parquet files under `data/parquet/ticks/`.
2. **Depth**: Backfill the 9 missing MNQ depth trading sessions (2025‑09‑08→09‑12, 2025‑09‑26, 2025‑09‑29‑30) by re-running the SCDD extractor.
3. **GEX**: Keep `/gex_history_url` jobs scheduled so the next coverage snapshot continues to show a full 45-day window.
