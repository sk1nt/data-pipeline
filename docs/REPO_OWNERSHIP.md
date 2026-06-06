# Repository ownership — data-pipeline (server)

Mirror of trading-side doc: `data-trading/docs/REPO_OWNERSHIP.md`.

## This repo owns

- `data-pipeline.py` orchestrator
- GEX / UW / Schwab / TastyTrade ingest
- Tick & candle parquet under `data/parquet/`
- Redis **`gex:snapshot:stream`** publisher
- `frontend/intelligence.html` + `/ws/sweep` (Redis relay)
- Discord bot, API, alerts

## data-trading owns

- `sweep_runner.py`, classifier, position monitor
- ACSIL studies, `fast_moves.db`, sweep ML
- Redis **`sweep:*`** and **`market:dom/cvd:*`** publisher

## Coupling

Only **Redis** + optional **SCP** paths configured in data-trading `.env`.  
No Python imports across repos. See [INTER_REPO.md](./INTER_REPO.md).