# Reconciliation: job-store vs Parquet

Generated: 2025-11-09T19:59:59.131221Z

## Completed jobs (local-file inspected where available)


## Parquet coverage (all files under `data/parquet/gexbot/<ticker>/<endpoint>/<YYYYMMDD>.strikes.parquet`)

### data/parquet/gexbot/NQ_NDX/gex_zero
  - Files / dates:
    - 20251027.strikes.parquet → 2025-10-27
    - 20251028.strikes.parquet → 2025-10-28
    - 20251029.strikes.parquet → 2025-10-29
    - 20251030.strikes.parquet → 2025-10-30
    - 20251031.strikes.parquet → 2025-10-31
    - 20251103.strikes.parquet → 2025-11-03
    - 20251104.strikes.parquet → 2025-11-04
    - 20251105.strikes.parquet → 2025-11-05
    - 20251106.strikes.parquet → 2025-11-06
    - 20251107.strikes.parquet → 2025-11-07
    - 20251110.strikes.parquet → 2025-11-10
    - 20251111.strikes.parquet → 2025-11-11
    - 20251112.strikes.parquet → 2025-11-12
    - 20251113.strikes.parquet → 2025-11-13
    - 20251114.strikes.parquet → 2025-11-14


## Comparison

- Dates inferred from completed job local files: []
- Dates present in Parquet files: ['2025-10-27', '2025-10-28', '2025-10-29', '2025-10-30', '2025-10-31', '2025-11-03', '2025-11-04', '2025-11-05', '2025-11-06', '2025-11-07', '2025-11-10', '2025-11-11', '2025-11-12', '2025-11-13', '2025-11-14']

- Date range determined from parquet: 2025-10-27 -> 2025-11-14 (weekdays only)
- Business days missing from Parquet in the range: []
- Parquet contains dates outside expected business days in range: []

## Jobs without local staged file or remote-only

## Jobs that failed to be inspected (errors reading file)
