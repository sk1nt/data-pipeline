# /gex_history_url Pipeline Reference

## Purpose
The `/gex_history_url` FastAPI route exists to quickly queue up historical GEX imports from bespoke clients, download their payloads, hydrate DuckDB (`data/gex_data.db`), and publish canonical strike Parquet slices. This note enumerates every script/module involved so new contributors can reason about failures without spelunking through the codebase.

## Key Components
| File | Responsibility | Operational Notes |
| --- | --- | --- |
| `data-pipeline.py` (`ServiceManager` + FastAPI) | Hosts the `/gex_history_url` endpoint, validates loose payloads (any 3 string fields containing `url`, `gex_*`, and ticker), enqueues jobs via `gex_history_queue`, and triggers importer work in a background task. | Endpoint rejects non-HTTPS URLs or those that do not start with `https://hist.gex.bot/`. Ticker is inferred from payload or filename fragment like `2025-11-14_NQ_NDX_...`. |
| `src/lib/gex_history_queue.py` | Thin DuckDB queue helper with `enqueue_request`, `mark_job_*`, and `get_pending_jobs`. | Uses `data/gex_data.db::gex_history_queue`, deduplicates refreshes by base URL, and records status transitions (`pending → started → completed/failed`). |
| `src/lib/gex_database.py` | Connection factory for `data/gex_data.db`, ensuring `gex_snapshots`, `gex_strikes`, `gex_history_queue`, and helper views exist. | All helpers create short-lived DuckDB connections to avoid file locks. Settings in `src/config.py` decide the base data directory. |
| `src/import_gex_history.py` | Worker that processes queue records: downloads JSON (or copies `file://` sources) to staging, loads data with DuckDB `read_json_auto`, inserts into `gex_snapshots` / `gex_strikes`, and exports strikes to Parquet. | Adds per-job staging tables (`staging_raw_<ticker>_<ts>`), truncates that trading day’s rows before insert, and writes Parquet files via DuckDB `COPY` with `FORMAT 'parquet'` + `COMPRESSION 'zstd'`. |
| `scripts/purge_old_data.py` | Maintenance script to prune old DuckDB rows and Parquet slices once they fall outside the retention window. | Uses the same directory conventions as the importer (`data/parquet/gexbot/<ticker>/<endpoint>/<YYYYMMDD>.strikes.parquet`). |
| `scripts/migrate_parquet_layout.py` | One-off helper to migrate legacy `year=/month=` buckets into the flat ticker/endpoint layout if older data is reintroduced. | Point it at the desired root via `--root`; default is `data/parquet/gexbot`. |
| `tests/integration/test_history_import.py`, `tests/contract/test_history_endpoint.py` | Cover endpoint contract (loose schema, ticker inference, URL validation) and queue side effects. | No new tests requested right now, but these remain the reference for acceptable behavior. |

## Execution Flow
1. **Request intake** – `/gex_history_url` inspects the JSON body (must contain exactly three user fields, e.g., `{"url": "https://hist.gex.bot/...", "gex_type": "gex_zero", "ticker": "NQ_NDX"}`). It normalizes the endpoint based on the first `gex_*` looking value and uppercases the ticker (preferring filename inference).
2. **Queue persistence** – `gex_history_queue.enqueue_request` either refreshes an existing job (matching ticker+endpoint+base URL) or creates a new `pending` row in DuckDB. The FastAPI handler immediately schedules `_trigger_queue_processing` with `BackgroundTasks`.
3. **Worker loop** – `process_historical_imports()` instantiates `GEXHistoryImporter`, fetches up to 10 pending jobs, marks each one `started`, and downloads the JSON into `data/source/gexbot/<ticker>/<endpoint>/` (preserving the original filename). `file://` URLs are treated as local copies and never deleted.
4. **DuckDB staging** – The importer loads the JSON via `read_json_auto` into a transient `staging_raw_*` table. Using DuckDB SQL keeps conversion close to the data and avoids Python parsing overhead.
5. **Snapshot/strike writes** –
   - `gex_snapshots`: `DELETE` rows for that ticker + trade date, then `INSERT` from staging while coercing timestamps to epoch milliseconds.
   - `gex_strikes`: same day-level delete followed by `INSERT ... FROM staging, UNNEST(strikes)` to flatten strike arrays.
6. **Parquet export** – DuckDB executes `COPY (SELECT ...) TO 'data/parquet/gexbot/<ticker>/<endpoint>/<YYYYMMDD>.strikes.parquet'` (ZSTD compressed). Existing files for that day are replaced to keep imports idempotent.
7. **Job completion** – After successful writes, the worker drops the staging table and marks the queue record `completed`. Errors bubble up to `mark_job_failed` with the DuckDB message for triage.

## Data Storage Layout
```
data/
├── gex_data.db                 # DuckDB file containing queue + canonical tables
│   ├── gex_history_queue        # Pending/completed download jobs
│   ├── gex_snapshots            # Snapshot-level aggregates (timestamp=epoch ms)
│   └── gex_strikes              # Flattened strike rows with priors JSON
├── source/
│   └── gexbot/
│       └── <TICKER>/
│           └── <endpoint>/
│               └── YYYY-MM-DD_<TICKER>_classic_gex_zero.json
└── parquet/
    └── gexbot/
        └── <TICKER>/
            └── <endpoint>/
                └── YYYYMMDD.strikes.parquet
```
- **JSON staging** mirrors the URL taxonomy so investigators can map imports back to the original download. Files are retained for at least 14 days (see `docs/DATA_STRUCTURE_AND_GOVERNANCE.md`).
- **Parquet outputs** are the canonical downstream artifacts. Each file contains all strikes for a single trading day. Timestamps remain UTC epoch milliseconds; any display layer converts to `America/New_York`.

## Operational Commands
- Start API + background importer locally:
  ```bash
  python data-pipeline.py --host 0.0.0.0 --port 8877
  ```
- Manually kick the importer loop (useful when working with local `file://` jobs):
  ```bash
  python -m src.import_gex_history
  ```
- Inspect queue + verify rows:
  ```bash
  python - <<'PY'
  import duckdb
  con = duckdb.connect('data/gex_data.db')
  print(con.execute("SELECT id, ticker, endpoint, status, last_error FROM gex_history_queue ORDER BY id DESC").fetchall())
  con.close()
  PY
  ```

## Troubleshooting Checklist
1. **422 at the endpoint** – Confirm the client sends JSON with three string fields, the `url` begins with `https://hist.gex.bot/`, and at least one field starts with `gex_`.
2. **Queue stuck in `pending`** – Run `_trigger_queue_processing` manually or check logs for DuckDB lock contention. Make sure only one worker runs per `gex_data.db`.
3. **Binder errors (e.g., missing column)** – Usually caused by schema drift in staging JSON. Confirm the importer’s `INSERT` statement targets existing columns (`timestamp`, `ticker`, etc.).
4. **Parquet mismatch** – Delete the specific `YYYYMMDD.strikes.parquet` before rerunning the import to guarantee DuckDB writes a fresh file (the importer already unlinks existing files by default).
5. **Time-zone audits** – Timestamps are stored as epoch ms. To review in New York time without mutating storage, use DuckDB: `SELECT to_timezone(to_timestamp(timestamp/1000.0), 'America/New_York') FROM gex_snapshots LIMIT 5;`.

Keeping this reference up to date alongside code changes prevents repetition of the earlier regressions (missing columns, overwritten Parquet layouts, etc.).
