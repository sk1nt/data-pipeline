from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
import duckdb


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ImportJobStore:
    def __init__(self, db_path: Path | str = "data/gex_data.db"):
        self.db_path = str(db_path)
        self._ensure_table()

    def _ensure_table(self) -> None:
        con = duckdb.connect(self.db_path)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS import_jobs (
                    id VARCHAR PRIMARY KEY,
                    url VARCHAR,
                    checksum VARCHAR,
                    ticker VARCHAR,
                    status VARCHAR,
                    records_processed BIGINT,
                    last_error VARCHAR,
                    created_at VARCHAR,
                    updated_at VARCHAR
                )
                """
            )
        finally:
            con.close()

    def _connect(self):
        return duckdb.connect(self.db_path)

    def compute_checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def find_by_checksum(self, checksum: str) -> dict | None:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT id, url, checksum, ticker, status, records_processed, last_error, created_at, updated_at FROM import_jobs WHERE checksum = ?",
                [checksum],
            ).fetchone()
            if not row:
                return None
            keys = [
                "id",
                "url",
                "checksum",
                "ticker",
                "status",
                "records_processed",
                "last_error",
                "created_at",
                "updated_at",
            ]
            return dict(zip(keys, row))
        finally:
            con.close()

    def create_job(self, url: str | None, checksum: str, ticker: str | None) -> str:
        job_id = uuid.uuid4().hex[:8]
        now = _now()
        con = self._connect()
        try:
            con.execute(
                "INSERT INTO import_jobs (id, url, checksum, ticker, status, records_processed, last_error, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [job_id, url, checksum, ticker, "pending", 0, None, now, now],
            )
            return job_id
        finally:
            con.close()

    def mark_started(self, job_id: str) -> None:
        now = _now()
        con = self._connect()
        try:
            con.execute(
                "UPDATE import_jobs SET status = ?, updated_at = ? WHERE id = ?",
                ["started", now, job_id],
            )
        finally:
            con.close()

    def mark_completed(self, job_id: str, records: int) -> None:
        now = _now()
        con = self._connect()
        try:
            con.execute(
                "UPDATE import_jobs SET status = ?, records_processed = ?, updated_at = ? WHERE id = ?",
                ["completed", records, now, job_id],
            )
        finally:
            con.close()

    def mark_failed(self, job_id: str, error: str) -> None:
        now = _now()
        con = self._connect()
        try:
            con.execute(
                "UPDATE import_jobs SET status = ?, last_error = ?, updated_at = ? WHERE id = ?",
                ["failed", error[:1024], now, job_id],
            )
        finally:
            con.close()
