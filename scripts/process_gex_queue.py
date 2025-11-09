#!/usr/bin/env python3
"""Queue worker for processing GEX historical data imports."""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lib.gex_history_queue import gex_history_queue


def process_queue_worker(limit: int = 10, sleep_seconds: int = 5):
    """Process pending jobs from the queue."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG = logging.getLogger("gex_queue_worker")

    LOG.info(f"Starting queue worker (limit={limit}, sleep={sleep_seconds}s)")

    while True:
        try:
            pending_jobs = gex_history_queue.get_pending_jobs(limit=limit)
            if not pending_jobs:
                LOG.info("No pending jobs, sleeping...")
                time.sleep(sleep_seconds)
                continue

            LOG.info(f"Found {len(pending_jobs)} pending jobs")

            for job in pending_jobs:
                job_id, url, ticker, endpoint, attempts = job
                LOG.info(f"Processing job {job_id}: {ticker} {endpoint}")

                # Import and run the import script logic
                try:
                    # Mark as started
                    gex_history_queue.mark_job_started(job_id)

                    # Run the import process (similar to import_gex_history.py)
                    import subprocess
                    result = subprocess.run([
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "import_gex_history.py"),
                        url,
                        ticker,
                        endpoint,
                        "--queue-id", str(job_id)
                    ], capture_output=True, text=True, cwd=PROJECT_ROOT)

                    if result.returncode == 0:
                        LOG.info(f"Job {job_id} completed successfully")
                    else:
                        LOG.error(f"Job {job_id} failed: {result.stderr}")
                        gex_history_queue.mark_job_failed(job_id, result.stderr[:1000])

                except Exception as exc:
                    LOG.error(f"Error processing job {job_id}: {exc}")
                    gex_history_queue.mark_job_failed(job_id, str(exc)[:1000])

        except KeyboardInterrupt:
            LOG.info("Shutting down queue worker")
            break
        except Exception as exc:
            LOG.error(f"Queue worker error: {exc}")
            time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(description="Process GEX history import queue")
    parser.add_argument("--limit", type=int, default=10, help="Max jobs to process per cycle")
    parser.add_argument("--sleep", type=int, default=5, help="Seconds to sleep between checks")
    parser.add_argument("--once", action="store_true", help="Process once and exit")
    args = parser.parse_args()

    if args.once:
        # Process one cycle and exit
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        LOG = logging.getLogger("gex_queue_worker")

        pending_jobs = gex_history_queue.get_pending_jobs(limit=args.limit)
        LOG.info(f"Found {len(pending_jobs)} pending jobs")

        for job in pending_jobs:
            job_id, url, ticker, endpoint, attempts = job
            LOG.info(f"Processing job {job_id}: {ticker} {endpoint}")

            try:
                gex_history_queue.mark_job_started(job_id)

                import subprocess
                result = subprocess.run([
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "import_gex_history.py"),
                    url,
                    ticker,
                    endpoint,
                    "--queue-id", str(job_id)
                ], capture_output=True, text=True, cwd=PROJECT_ROOT)

                if result.returncode == 0:
                    LOG.info(f"Job {job_id} completed successfully")
                else:
                    LOG.error(f"Job {job_id} failed: {result.stderr}")

            except Exception as exc:
                LOG.error(f"Error processing job {job_id}: {exc}")
                gex_history_queue.mark_job_failed(job_id, str(exc)[:1000])
    else:
        process_queue_worker(limit=args.limit, sleep_seconds=args.sleep)


if __name__ == "__main__":
    main()