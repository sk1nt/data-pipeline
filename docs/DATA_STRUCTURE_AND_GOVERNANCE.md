DATA STRUCTURE & GOVERNANCE
==========================

Purpose
-------
This document describes the canonical data structure, import flow, and governance rules for the data-pipeline repository. The project must follow the rules below; these are the single source of truth for where data lives, how imports run, and who is responsible for changes.

Scope
-----
Covers staged source files (JSON), local DuckDB databases, canonical Parquet outputs, and import metadata. Does NOT cover monitoring containers or unrelated dev artifacts.

Canonical Directory Layout
--------------------------
- `data/source/gexbot/`
  - Staged JSON files downloaded from external sources (one file per import).
  - Example: `2025-10-22_NQ_NDX_classic_gex_zero.json`
  - Rule: files are read-only once imported; do not mutate in-place.
  - Retention & post-import policy: Staged JSON files are retained for 14 days after creation. A cleanup pass runs after imports to prune staged files older than 14 days. If you need to preserve raw JSON for audit/archival purposes, add an explicit archive step to copy files to an archive location (e.g., `data/archive/gex/`) before running imports.

- `data/gex_data.db`
  - DuckDB file used as the transient processing engine and to hold staging tables.
  - Table: `staging_strikes_<job_id>` used during import; final validated metadata recorded in `strikes` only as transient metadata.
  - Rule: DuckDB is a processing engine, not canonical storage. Parquet is canonical for downstream use.

- `data/gex_data.db` *(deprecated — history tables now live inside `data/gex_data.db`)*
  - Metadata store for import jobs (table: `import_jobs`). Tracks `id`, `url`, `checksum`, `ticker`, `status`, `records_processed`, `last_error`, `created_at`, `updated_at`.
  - Rule: This is the authoritative job log. Do not manually edit entries except through tooling.

- `data/source/gexbot/<ticker>/<endpoint>/YYYY-MM-DD_*.json`
  - Raw history snapshots downloaded from GEXBot (temporary staging).
- `data/parquet/gexbot/<ticker>/<endpoint>/<YYYYMMDD>.strikes.parquet`
  - Canonical strike history per trading day (mounted via DuckDB view `parquet_gex_strikes`).

> **Note (2025-11-11):** `gex_bridge_*` tables have been removed. Canonical data lives in
> `data/parquet/gexbot/<ticker>/<endpoint>/<YYYYMMDD>.strikes.parquet` with DuckDB views,
> while raw JSON remains under `data/source/gexbot/...`.
  - Rule: Parquet files are final outputs; downstream services read only from here. All writes must go through the import pipeline.

Import Flow (high-level)
------------------------
1. A job record is created or queued in the history tables (`gex_history_queue` inside `data/gex_data.db`) with `status='pending'` and a `url` pointing to the source JSON.
2. The import runner (manual CLI: `python scripts/import_gex_history.py --url ...` or the HTTP endpoint `/gex_history_url`) picks a `pending` job.
3. The runner downloads the JSON to `data/source/gexbot/<ticker>/<endpoint>/` and performs light validation of the payload.
4. `GEXHistoryImporter` streams the JSON through DuckDB, writing canonical `gex_snapshots` / `gex_strikes` rows (stored as UTC epoch milliseconds) and exporting strikes to Parquet.
5. Job store records `records_processed` and `status` (`completed`|`failed`).

Export Rule
-----------
- After validation, the pipeline MUST write or append the data into the canonical Parquet location by:
  - Ensuring the raw JSON download lands in `data/source/gexbot/<ticker>/<endpoint>/`.
  - Writing a single Parquet file for that date block (or appending / using partitioned Parquet tools). The project prefers partition-by-directory semantics.
- Parquet files are immutable once written for a given job (new jobs may create new files). If a rewrite is required, the operation must be recorded in the job history with reason and checksum.

Checksums & Idempotency
-----------------------
- The import job store computes a SHA256 checksum for each staged JSON and stores it in `checksum`.
- If a checksum matches an existing completed import, the job MUST be skipped and marked `skipped=True` in the API response.

Concurrency & Locks
-------------------
- DuckDB has file-level locks. Only one writer process should run against `data/gex_data.db` at a time. Use the job queue to serialize imports or run separate DuckDB instances with separate DB files and merge Parquet outputs.

Metadata & Provenance
---------------------
- Keep the original staged JSON file as the immutable source of truth for that import. Do not alter the filename.
- Job metadata in the history tables (now in `data/gex_data.db`) MUST include the original `url`, the computed `checksum`, and timestamps for `created_at` and `updated_at`.
- Any manual corrections to data must be accompanied by a follow-up job that documents the correction and references the `id` of the original job.

Access & Responsibilities
-------------------------
- Who can run imports: developers and ops with access to the project environment.
- Who can change governance: repository owners (name here) only. Changes to this document require PR review.

Retention & Backups
-------------------
- Staged JSON files: keep for 90 days by default, then archive to `archive/` if needed.
- Parquet files: canonical, keep indefinitely unless legal or storage constraints apply; changes should follow a documented rewrite process.
- DB backups: snapshot `data/gex_data.db` (covers both real-time and history tables) before mass import operations.

Troubleshooting & Verification
------------------------------
- Quick verifications:
  - Check job status: `SELECT * FROM import_jobs ORDER BY created_at DESC LIMIT 10` in `data/gex_data.db`.
  - Confirm staged file exists: `ls -la data/source/gexbot/`.
  - Confirm Parquet contents: use DuckDB to read Parquet and list distinct dates.
- If an import fails, `mark_failed` records `last_error`. Use that error to triage.

Change Management
-----------------
- Additions or structural changes to directories or schema must be proposed as a PR with migration steps:
  - Migration script to move Parquet files to new layout.
  - Update documentation and any consumers.
  - Run verification queries and snapshot metadata.

Appendix: Commands
------------------
- Run import CLI (process a specific URL without hitting FastAPI):

```bash
python scripts/import_gex_history.py --url https://hist.gex.bot/... --ticker SPX --endpoint gex_zero
```

- Run the server (FastAPI) locally:

```bash
python run_server.py
# or
python -m src.data_pipeline
```

- Inspect job history (quick python snippet):

```bash
python - <<'PY'
import duckdb
con = duckdb.connect('data/gex_data.db')
print(con.execute("SELECT id, url, status, records_processed, created_at FROM import_jobs ORDER BY created_at DESC LIMIT 20").fetchall())
con.close()
PY
```

Document maintenance
--------------------
- This file lives at `docs/DATA_STRUCTURE_AND_GOVERNANCE.md` and is the authoritative governance doc for the pipeline. Any divergence is not permitted without a reviewed PR.


---

Document created: AUTOMATIC (by developer tooling)

If you want, I can:
- Implement the Parquet export step inside `safe_import` now.
- Add a migration script to move the current Parquet files into the canonical partitioned layout.
- Add automated tests that verify idempotency and Parquet output.

Which of the above do you want next?
