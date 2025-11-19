#!/usr/bin/env python3
"""Queue worker for processing GEX historical data imports."""

import argparse
import logging
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_queue_helper():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.lib.gex_history_queue import gex_history_queue as queue_helper

    return queue_helper


gex_history_queue = _load_queue_helper()


def process_job(job_id: int, url: str, ticker: str, endpoint: str, log: logging.Logger) -> None:
    """Worker routine that runs a single job via import_gex_history.py."""
    try:
        gex_history_queue.mark_job_started(job_id)
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "import_gex_history.py"),
                url,
                ticker,
                endpoint,
                "--queue-id",
                str(job_id),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            log.info("Job %s completed successfully", job_id)
        else:
            log.error("Job %s failed: %s", job_id, result.stderr)
            gex_history_queue.mark_job_failed(job_id, result.stderr[:1000])
    except Exception as exc:
        log.error("Error processing job %s: %s", job_id, exc)
        try:
            gex_history_queue.mark_job_failed(job_id, str(exc)[:1000])
        except Exception as inner_exc:
            log.error("Failed to mark job %s as failed: %s", job_id, inner_exc)


def threaded_worker(job_queue: "queue.Queue[tuple]", log: logging.Logger) -> None:
    while True:
        try:
            job = job_queue.get()
        except Exception:
            job = None
        if job is None:
            job_queue.task_done()
            break
        job_id, url, ticker, endpoint = job
        process_job(job_id, url, ticker, endpoint, log)
        job_queue.task_done()


def run_parallel_once(limit: int, workers: int, log: logging.Logger) -> None:
    pending_jobs = gex_history_queue.get_pending_jobs(limit=limit)
    if not pending_jobs:
        log.info("No pending jobs found")
        return
    log.info("Dispatching %s jobs using %s workers", len(pending_jobs), workers)
    job_queue: "queue.Queue[tuple]" = queue.Queue()
    for job in pending_jobs:
        job_id, url, ticker, endpoint, attempts = job
        job_queue.put((job_id, url, ticker, endpoint))
    threads = []
    for _ in range(workers):
        t = threading.Thread(target=threaded_worker, args=(job_queue, log), daemon=True)
        t.start()
        threads.append(t)
    job_queue.join()
    for _ in threads:
        job_queue.put(None)
    for t in threads:
        t.join()


def process_queue_worker(limit: int, sleep_seconds: int, workers: int):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("gex_queue_worker")
    log.info("Starting queue worker (limit=%s, sleep=%ss, workers=%s)", limit, sleep_seconds, workers)
    while True:
        try:
            pending_jobs = gex_history_queue.get_pending_jobs(limit=limit)
            if not pending_jobs:
                log.info("No pending jobs, sleeping...")
                time.sleep(sleep_seconds)
                continue
            log.info("Found %s pending jobs", len(pending_jobs))
            run_parallel_once(limit=limit, workers=workers, log=log)
        except KeyboardInterrupt:
            log.info("Shutting down queue worker")
            break
        except Exception as exc:
            log.error("Queue worker error: %s", exc)
            time.sleep(sleep_seconds)


def main():
    parser = argparse.ArgumentParser(description="Process GEX history import queue")
    parser.add_argument("--limit", type=int, default=10, help="Max jobs to process per cycle")
    parser.add_argument("--sleep", type=int, default=5, help="Seconds to sleep between checks")
    parser.add_argument("--once", action="store_true", help="Process once and exit")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("gex_queue_worker")
    workers = max(1, args.workers)

    if args.once:
        run_parallel_once(limit=args.limit, workers=workers, log=log)
    else:
        process_queue_worker(limit=args.limit, sleep_seconds=args.sleep, workers=workers)


if __name__ == "__main__":
    main()
