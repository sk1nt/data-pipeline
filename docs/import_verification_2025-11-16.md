# Data Import Verification — 2025-11-16

Verified with `python scripts/verify_imports.py --start-date 2025-09-02 --depth-start-date 2025-09-19`.

## Highlights

- **All GEX artifacts are in millisecond epoch time.** `gex_snapshots`, `gex_strikes`, and the Parquet strikes exported under `data/parquet/gexbot/NQ_NDX/gex_zero/` all report `timestamp % 1000 = 0` across 1.26M snapshots and 123M strike rows spanning **2025-09-02T13:30:15Z → 2025-11-14T21:00:00Z**.
- **Tick data is only populated for 2025-09-18 → 2025-09-19 in DuckDB (`tick_data`).** The Parquet exports cover **2025-09-02 → 2025-10-06**, but two files (`20250929.parquet` and `20250930.parquet`) fail the Parquet magic-byte check, so downstream readers must skip or regenerate them.
- **Depth data exists for 2025-09-19 → 2025-10-06 only.** The `20250929.parquet` depth export is corrupt (missing footer), and there are no files after 2025-10-06.
- **Coverage percentages from 2025-09-02 onward (depth window trimmed to start on 2025-09-19) are shown below.** GEX is missing only 2025-09-30. Tick and depth exports are missing every weekday from 2025-10-07 onward.

## Timestamp precision checks

| Dataset | Rows | Epoch range (UTC) | ms remainder | Notes |
|---------|------|-------------------|--------------|-------|
| GEX snapshots (`data/gex_data.db::gex_snapshots`) | 1,261,402 | 2025-09-02T13:30:15Z → 2025-11-14T21:00:00Z | 0 → 0 | Stored as BIGINT epoch milliseconds.
| GEX strikes (`data/gex_data.db::gex_strikes`) | 125,249,384 | 2025-09-02T13:30:15Z → 2025-11-14T21:00:00Z | 0 → 0 | Stored as BIGINT epoch milliseconds.
| GEX strikes Parquet (`data/parquet/gexbot/NQ_NDX/gex_zero/*.parquet`) | 122,951,994 | 2025-09-02T13:30:15Z → 2025-11-14T21:00:00Z | 0 → 0 | Every Parquet partition passes the ms check.
| Tick DuckDB table (`data/tick_data.db::tick_data`) | 31,553 | 2025-09-18T16:12:04.194Z → 2025-09-19T13:29:59.424Z | 0 → 999 | Stored as DuckDB `TIMESTAMP`; fractional milliseconds exist, but dataset stops after Sep-19.
| Tick Parquet (`data/parquet/tick/MNQ/*.parquet`) | — | — | — | ❌ `20250930.parquet` is corrupt (`Parquet magic bytes not found`). Remove/re-export before querying.
| Depth Parquet (`data/parquet/depth/MNQ/*.parquet`) | — | — | — | ❌ `20250929.parquet` is corrupt (`Parquet magic bytes not found`).

## Coverage summary (Window: 2025-09-02 → 2025-11-14, depth starts 2025-09-19)

| Dataset | Trading days present / expected | Coverage | Missing weekdays | Comments |
|---------|--------------------------------|----------|------------------|----------|
| GEX strikes Parquet | 53 / 54 | **98.1%** | 2025-09-30 | Add the missing GEX job or re-export that session.
| Tick Parquet (MNQ) | 25 / 54 | **46.3%** | Every weekday from 2025-10-07 through 2025-11-14 | Need imports for the entire Oct–Nov window; also regenerate `20250929` and `20250930` because the Parquet files are unreadable.
| Depth Parquet (MNQ) | 12 / 41 | **29.3%** | Every weekday from 2025-10-07 through 2025-11-14 | No depth captures ran after 2025-10-06. Ignored dates before 2025-09-19 per request.

## Next steps

1. **Regenerate corrupt Parquets**
   - `data/parquet/tick/MNQ/20250929.parquet`
   - `data/parquet/tick/MNQ/20250930.parquet`
   - `data/parquet/depth/MNQ/20250929.parquet`
2. **Backfill missing sessions**
   - GEX: re-run import for 2025-09-30.
   - Tick: hydrate 2025-10-07 onward via SCID/Schwab in both DuckDB and Parquet.
   - Depth: rerun the SCDD exporter for 2025-10-07 onward.
3. **Rerun `python scripts/verify_imports.py` after each backfill to update the percentages in this report and in `datasources.md`.

