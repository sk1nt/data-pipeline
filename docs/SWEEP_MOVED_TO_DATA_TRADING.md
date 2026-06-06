# Sweep stack moved to `data-trading`

Real-time sweep detection, Sierra DOM/CVD bridge, position monitor, and sweep ML **no longer run from this repo**.

## Run here (server)

```bash
cd ~/projects/data-pipeline
python data-pipeline.py
```

Publishes `gex:snapshot:stream`, ingests tick/candle parquet, serves `intelligence.html` (WebSocket `/ws/sweep` **relays** Redis only).

## Run on trading machine

```bash
cd ~/projects/data-trading
source .venv/bin/activate
python sweep_runner.py --dry-run   # then without --dry-run
./scripts/nightly_ml.sh            # after close
```

## Removed / stubbed in data-pipeline

| Former path | Status |
|-------------|--------|
| `sweep_runner.py` | Stub — exits with pointer |
| `src/services/sweep_classifier_service.py` | Stub |
| `src/services/position_monitor_service.py` | Stub |
| `src/services/sierra_dom_bridge_service.py` | Stub |
| `src/ml/sweep_feature_extractor.py` | Stub |
| `src/ml/sweep_trainer.py` | Stub |

ACSIL studies for sweep: use copies under **`data-trading/sc_studies/`** (`sweep_binary_bridge.cpp`, `sweep_dom_exporter.cpp`, `position_bridge.cpp`).

## Redis contract (unchanged)

Channel names are documented in `contracts/redis_channels.py` (synced with data-trading). Server must keep publishing GEX; trading box publishes sweep alerts.

## Rollback

Branch `feature/v2-sweep-system` in this repo still has full sweep code until deleted. Prefer fixing forward in `data-trading`.

## Docs

- Trading ops: `~/projects/data-trading/docs/REPO_OWNERSHIP.md`
- Canonical spec: `specs/002-sweep-classifier/plan.md`