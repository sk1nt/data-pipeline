#!/usr/bin/env python3
# ruff: noqa: E402
"""Sweep system entry point — runs on the trading machine (Windows + WSL).

This process owns the real-time sweep detection and position protection stack.
It is intentionally SEPARATE from data-pipeline.py (which runs on the Linux
server) because it needs direct, low-latency access to Sierra Chart's local
file I/O — dom_snapshot.json, trade_flow.json, danger_trigger.json.

╔══════════════════════════════════════════════════════════════════════╗
║                    MACHINE TOPOLOGY OVERVIEW                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Trading machine (Windows + WSL)                                     ║
║    ├── Sierra Chart                                                   ║
║    │     └── DTC Protocol Server  (port 11099, JSON Compact)         ║
║    │           Streams in real time:                                  ║
║    │             MARKET_DEPTH_SNAPSHOT/UPDATE_LEVEL2  (DOM)          ║
║    │             MARKET_DATA_UPDATE_TRADE              (CVD)         ║
║    │           Polls every 100 ms:                                    ║
║    │             C:/SierraChart/Data/danger_trigger.json             ║
║    │                                                                   ║
║    └── WSL (Ubuntu 22.04 recommended)                                ║
║          ├── Redis  (redis-server, port 6379)                        ║
║          └── sweep_runner.py  ← YOU ARE HERE                        ║
║                ├── SierraDOMBridgeService                            ║
║                │     • DTC TCP client → SC DTC server                ║
║                │     • publishes  market:dom:MNQ   to Redis          ║
║                │     • publishes  market:cvd:MNQ   to Redis          ║
║                │     • subscribes sweep:danger:MNQ from Redis        ║
║                │     • writes back danger_trigger.json → SC          ║
║                ├── SweepClassifierService                            ║
║                │     • subscribes market:dom:MNQ + market:cvd:MNQ   ║
║                │     • subscribes gex:snapshot:stream (from server)  ║
║                │     • publishes  sweep:alert:MNQ                    ║
║                │     • writes     data/fast_moves.db  (DuckDB)       ║
║                └── PositionMonitorService                            ║
║                      • subscribes sweep:alert:MNQ                   ║
║                      • reads      position_snapshot.json (SC file)  ║
║                      • publishes  sweep:danger:MNQ → triggers flatten║
║                      • subscribes sweep:ack:MNQ  (manual override)  ║
║                                                                       ║
║  Server / dev machine (Linux)                                        ║
║    └── data-pipeline.py                                              ║
║          ├── GEX pipeline  (publishes gex:snapshot:stream)           ║
║          ├── Schwab / TastyTrade streamers                           ║
║          ├── FastAPI surface                                         ║
║          └── Discord bot                                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  REDIS BRIDGE: point both machines at the same Redis instance.       ║
║  Simplest: run Redis on the trading machine (WSL), set REDIS_URL     ║
║  on the server to redis://<trading-machine-lan-ip>:6379/0            ║
╚══════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════
 FIRST-TIME TRADING MACHINE SETUP  (WSL / Ubuntu)
═══════════════════════════════════════════════════════════════════════

1. Install WSL 2 + Ubuntu 22.04 (once)
   ─────────────────────────────────────
   In Windows PowerShell (admin):
       wsl --install -d Ubuntu-22.04

2. Install system dependencies
   ─────────────────────────────
   Inside WSL:
       sudo apt update && sudo apt install -y python3.11 python3.11-venv \\
           python3-pip redis-server git

3. Clone the repo
   ─────────────────
       git clone https://github.com/<your-org>/data-pipeline.git
       cd data-pipeline
       git checkout feature/v2-sweep-system   # or main once merged

4. Create and activate Python venv
   ──────────────────────────────────
       python3.11 -m venv .venv
       source .venv/bin/activate

5. Install Python dependencies
   ──────────────────────────────
       pip install -e ".[sweep]"
   If pyproject.toml doesn't have a [sweep] extra yet:
       pip install redis aioredis duckdb pydantic python-dotenv

6. Configure .env  (NEVER overwrite — copy from .env.example)
   ──────────────────────────────────────────────────────────────
       cp .env.example .env   # then edit with your values

   Minimum required keys for sweep_runner:

       # Sierra Chart file paths (WSL maps C:/ to /mnt/c/)
       SC_DATA_DIR=/mnt/c/SierraChart/Data
       # Override danger trigger path only if SC Data dir is non-standard:
       # SC_DANGER_TRIGGER_PATH=/mnt/c/SierraChart/Data/danger_trigger.json

       # Full Sierra Chart symbol (update each quarterly roll)
       SC_DTC_SYMBOL=MNQM26_FUT_CME
       SC_DTC_HOST=       # leave blank to auto-detect from /etc/resolv.conf
       SC_DTC_PORT=11099

       # Symbol to trade
       SC_DOM_SYMBOL=MNQ

       # Redis — if running locally in WSL
       REDIS_URL=redis://localhost:6379/0
       # If Redis runs on the Linux server instead:
       # REDIS_URL=redis://192.168.1.xxx:6379/0

       # GEX snapshot channel (published by data-pipeline.py on the server)
       # No extra config needed — SweepClassifierService subscribes automatically.

       # Position file written by a second ACSIL study (position_bridge.cpp)
       # Leave blank to disable position-aware danger escalation
       SC_POSITIONS_PATH=/mnt/c/SierraChart/Data/position_snapshot.json

       # DuckDB for training data accumulation
       SWEEP_DB_PATH=data/fast_moves.db

       # SAFETY: keep false until system is fully validated
       SWEEP_LIVE_MODE=false
       # Set true to enable actual FlattenAndCancelAllOrders() via SC danger trigger

       # Stop thresholds (in MNQ ticks, 1 tick = 0.25 pts = $0.50/contract)
       SWEEP_WARNING_TICKS=15
       SWEEP_DANGER_TICKS=25
       SWEEP_CRITICAL_TICKS=35
       SWEEP_WARNING_CONFIDENCE=0.65
       SWEEP_DANGER_CONFIDENCE=0.78
       SWEEP_CRITICAL_CONFIDENCE=0.85
       SWEEP_LEVEL2_ACKNOWLEDGE_SECONDS=10

7. Enable Sierra Chart DTC Protocol Server
   ──────────────────────────────────────────
   In SC: Global Settings → Data/Trade Service Settings → DTC Protocol Server
     a. Enable DTC Protocol Server : Yes
     b. Listening Port             : 11099
     c. Require Authentication     : No
     d. Encoding (List)            : JSON Compact
   Click OK / Apply.  No ACSIL study or DLL compilation needed.

8. Start Redis
   ─────────────
       sudo service redis-server start
       redis-cli ping   # should return PONG

9. Start sweep runner (dry-run first)
   ──────────────────────────────────────
       python sweep_runner.py --dry-run --log-level DEBUG
   Watch logs/sweep-runner.log — you should see DOM snapshots arriving.

10. Arm live mode (only after thorough dry-run validation)
    ──────────────────────────────────────────────────────
    In .env:
        SWEEP_LIVE_MODE=true
    Then restart without --dry-run:
        python sweep_runner.py

═══════════════════════════════════════════════════════════════════════
 ML TRAINING CYCLE  (after accumulating ~200+ live triggers)
═══════════════════════════════════════════════════════════════════════
   # 1. Label outcomes from tick data (run after market close):
       python -m src.ml.sweep_feature_extractor

   # 2. Train XGBoost model:
       python -m src.ml.sweep_trainer

   # Model saves to src/ml/sweep_model.pkl and is loaded automatically
   # on next sweep_runner.py startup.

═══════════════════════════════════════════════════════════════════════
 HISTORICAL BACKTEST  (evaluate exit signal system on past trade data)
═══════════════════════════════════════════════════════════════════════
   # Requires tick parquet and trading/TradesList*.txt files.
   # Full depth data must run from the trading machine (has year of depth).
       python scripts/backtest_stop_system.py --help

═══════════════════════════════════════════════════════════════════════
 DESIGN PHILOSOPHY — INTELLIGENT EXITS, NOT STATIC STOPS
═══════════════════════════════════════════════════════════════════════
   This system does NOT implement traditional mechanical stop losses.
   Positions are closed when there is positive EVIDENCE that a move is
   real and directional — not simply because price reached a distance.

   The tick thresholds in PositionMonitorService are noise filters:
   they ensure the classifier's signal is at sufficient adverse excursion
   to be meaningful before acting on it.

   PLANNED: TOD-based position sizing
       Scale contract size by time-of-day volatility regime so that the
       same dollar risk is maintained in low-edge (midday) sessions.
       The system will adjust thresholds dynamically rather than using
       static configuration.

   PLANNED: GEX wall take-profit targets
       major_pos_call1_strike / major_neg_put1_strike from gex:snapshot:stream
       are natural structural TP levels.  A TP monitor will surface these
       as signals, giving the trader a reason to exit winners at a level
       that has physical meaning in the options market.

Services started
────────────────
    SierraDOMBridgeService   — polls SC JSON files, publishes to Redis
    SweepClassifierService   — detects fast moves, classifies sweep vs directional
    PositionMonitorService   — graduated danger escalation, optional flatten trigger

Flags
─────
    --dry-run       Force SWEEP_LIVE_MODE=false (no actual flatten triggers)
    --symbol SYM    Override SC_DOM_SYMBOL (default: MNQ)
    --redis URL     Override REDIS_URL (default: redis://localhost:6379/0)
    --log-level L   DEBUG | INFO | WARNING (default: INFO)

Configuration is also read from .env in the project root.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap paths before local imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass  # python-dotenv optional; env vars must be set externally

# ---------------------------------------------------------------------------
# Logging setup (mirrors data-pipeline.py conventions)
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_HANDLER_FILE = RotatingFileHandler(
    LOG_DIR / "sweep-runner.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
)
_HANDLER_FILE.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
)
_HANDLER_CONSOLE = logging.StreamHandler(sys.stdout)
_HANDLER_CONSOLE.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
)

logging.basicConfig(level=logging.INFO, handlers=[_HANDLER_FILE, _HANDLER_CONSOLE])
LOGGER = logging.getLogger("sweep_runner")

# ---------------------------------------------------------------------------
# Local service imports (deferred so logging is configured first)
# ---------------------------------------------------------------------------
from src.services.sierra_dom_bridge_service import SierraDOMBridgeService  # noqa: E402
from src.services.sweep_classifier_service import SweepClassifierService  # noqa: E402
from src.services.position_monitor_service import PositionMonitorService  # noqa: E402


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SweepRunner:
    """Owns the lifecycle of all sweep-system services."""

    def __init__(self) -> None:
        self._dom_bridge = SierraDOMBridgeService()
        self._classifier = SweepClassifierService()
        self._monitor = PositionMonitorService()
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        live_mode = os.getenv("SWEEP_LIVE_MODE", "false").lower() == "true"
        symbol = os.getenv("SC_DOM_SYMBOL", "MNQ")

        LOGGER.info("=" * 60)
        LOGGER.info("Sweep runner starting")
        LOGGER.info("  Symbol     : %s", symbol)
        LOGGER.info("  Live mode  : %s (flatten enabled: %s)", live_mode, live_mode)
        LOGGER.info("  Redis      : %s", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        LOGGER.info("  SC data dir: %s", os.getenv("SC_DATA_DIR", "/mnt/c/SierraChart/Data"))
        LOGGER.info("=" * 60)

        if not live_mode:
            LOGGER.warning(
                "SWEEP_LIVE_MODE is not 'true' — position flatten is DISABLED. "
                "Set SWEEP_LIVE_MODE=true in .env to arm the system."
            )

        await asyncio.gather(
            self._dom_bridge.start(),
            self._classifier.start(),
            self._monitor.start(),
        )

    async def stop(self) -> None:
        LOGGER.info("Sweep runner shutting down…")
        await asyncio.gather(
            self._dom_bridge.stop(),
            self._classifier.stop(),
            self._monitor.stop(),
            return_exceptions=True,
        )
        LOGGER.info("Sweep runner stopped.")
        self._stop_event.set()

    async def run_until_signal(self) -> None:
        loop = asyncio.get_running_loop()

        def _handle_signal() -> None:
            LOGGER.info("Signal received — initiating graceful shutdown")
            loop.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)

        await self.start()
        await self._stop_event.wait()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep system runner (trading machine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable flatten trigger (overrides SWEEP_LIVE_MODE)",
    )
    p.add_argument(
        "--symbol",
        default="",
        metavar="SYM",
        help="Override SC_DOM_SYMBOL (e.g. MNQ, MES)",
    )
    p.add_argument(
        "--redis",
        default="",
        metavar="URL",
        help="Override REDIS_URL (e.g. redis://192.168.1.100:6379/0)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Log verbosity (default: INFO)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # CLI flags override env (but never overwrite .env on disk)
    if args.dry_run:
        os.environ["SWEEP_LIVE_MODE"] = "false"
    if args.symbol:
        os.environ["SC_DOM_SYMBOL"] = args.symbol
    if args.redis:
        os.environ["REDIS_URL"] = args.redis

    runner = SweepRunner()
    try:
        asyncio.run(runner.run_until_signal())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
