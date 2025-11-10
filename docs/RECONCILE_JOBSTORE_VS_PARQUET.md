# Reconciliation: job-store vs Parquet

Generated: 2025-11-09T19:59:59.131221Z

## Completed jobs (local-file inspected where available)


## Parquet coverage (all files under `data/parquet/gex`)

### data/parquet/gex/2025/10/NQ_NDX/unknown/strikes.parquet
  - Dates:
    - 2025-10-28

### data/parquet/gex/2025/10/_legacy/data_0.parquet
  - Dates:
    - 2025-10-01
    - 2025-10-02
    - 2025-10-03
    - 2025-10-06
    - 2025-10-07
    - 2025-10-08
    - 2025-10-09
    - 2025-10-10
    - 2025-10-13
    - 2025-10-14
    - 2025-10-15
    - 2025-10-16
    - 2025-10-17
    - 2025-10-20
    - 2025-10-28
    - 2025-10-29
    - 2025-10-30
    - 2025-10-31

### data/parquet/gex/2025/11/NQ_NDX/gex_zero/strikes.parquet
  - Dates:
    - 2025-11-09

### data/parquet/gex/2025/11/_legacy/data_0.parquet
  - Dates:
    - 2025-11-04
    - 2025-11-06


## Comparison

- Dates inferred from completed job local files: []
- Dates present in Parquet files: ['2025-10-01', '2025-10-02', '2025-10-03', '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10', '2025-10-13', '2025-10-14', '2025-10-15', '2025-10-16', '2025-10-17', '2025-10-20', '2025-10-28', '2025-10-29', '2025-10-30', '2025-10-31', '2025-11-04', '2025-11-06', '2025-11-09']

- Date range determined from parquet: 2025-10-01 -> 2025-11-09 (weekends excluded)
- Business days missing from Parquet in the range: ['2025-10-21', '2025-10-22', '2025-10-23', '2025-10-24', '2025-10-27', '2025-11-03', '2025-11-05', '2025-11-07']
- Parquet contains dates outside expected business days in range: ['2025-11-09']

## Jobs without local staged file or remote-only

## Jobs that failed to be inspected (errors reading file)