# Scripts Agent Instructions

## Domain Scope
This folder contains batch processing scripts and CLI tools. Sub-agents working here should focus on:

### Orchestration & Batch Jobs
- `orchestrator.py` - Parallel date-range processing
- `worker_day.py` - Single-day tick/depth processing

### Data Import/Export
- `import_scid_ticks.py` - Sierra Chart SCID → internal format
- `export_scid_ticks_to_parquet.py` - SCID → Parquet conversion
- `import_gex_history.py` - Historical GEX imports
- `export_enriched_bars.py` - Aggregated bar exports

### Data Enrichment
- `enrich_tick_gex.py` - Merge GEX data with tick data
- `gex_regime_router.py` - GEX regime classification

### Maintenance
- `backup_db.py` / `restore_db.py` - DuckDB backups
- `purge_old_data.py` - Data retention cleanup
- `migrate_*.py` - Schema migrations

## Key Patterns

### Resumable Jobs
All long-running scripts should support:
```python
# Checkpoint pattern
def process_with_checkpoint(dates: List[date], checkpoint_file: Path):
    completed = load_checkpoint(checkpoint_file)
    for d in dates:
        if d in completed:
            continue
        process_date(d)
        save_checkpoint(checkpoint_file, d)
```

### Parallel Processing
Use `concurrent.futures` for parallelization:
```python
with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(process_date, d): d for d in dates}
    for future in concurrent.futures.as_completed(futures):
        date = futures[future]
        result = future.result()
```

## CME Contract Windows
SCID files are contract-specific. Reference:
```python
SCID_CONTRACT_WINDOWS = [
    (date(2025, 9, 2), date(2025, 9, 18), "MNQU25_FUT_CME.scid"),
    (date(2025, 9, 18), date(9999, 12, 31), "MNQZ25_FUT_CME.scid"),
]
```

## Output Locations
- Parquet: `data/parquet/{tick,depth}/symbol=X/date=YYYY-MM-DD/`
- DuckDB: `data/*.db`
- Logs: `logs/`
- Reports: `reports/`

## Testing
- Test scripts in `scripts/bench/`
- Use `--dry-run` flag where supported
- Validate output against known-good samples

## Do NOT
- Run destructive operations without `--dry-run` first
- Process more than 1 day without checkpointing
- Overwrite production databases without backup
