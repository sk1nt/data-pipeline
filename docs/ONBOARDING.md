# Onboarding — data-pipeline (server)

**Repo:** https://github.com/sk1nt/data-pipeline  
**Role:** GEX ingest, tick/candle parquet, API, Discord, dashboard.  
**Not this repo:** `sweep_runner`, Sierra DOM, sweep ML → **data-trading** on the SC host.

## Start here

| Doc | Purpose |
|-----|---------|
| [REPO_OWNERSHIP.md](./REPO_OWNERSHIP.md) | What runs here vs data-trading |
| [INTER_REPO.md](./INTER_REPO.md) | Redis + contract sync |
| [SWEEP_MOVED_TO_DATA_TRADING.md](./SWEEP_MOVED_TO_DATA_TRADING.md) | Sweep deprecation |

## Run server

```bash
cd ~/projects/data-pipeline
pip install -e .
python data-pipeline.py
```

## Trading stack (other machine)

```bash
cd ~/projects/data-trading   # https://github.com/sk1nt/data-trading
python sweep_runner.py
```

## Contracts

```bash
python scripts/verify_contracts.py
python scripts/verify_inter_repo_sync.py   # if both repos cloned
```