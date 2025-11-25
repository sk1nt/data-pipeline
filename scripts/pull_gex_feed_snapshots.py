#!/usr/bin/env python3
"""
Utility to extract recent NQ_NDX GEX snapshots for feed simulations.

Creates a reusable pathway for the Discord GEX feed by hydrating the last
`N` snapshots directly from DuckDB (preferred) or the exported Parquet file.
The resulting rows are normalized into JSON-friendly dictionaries that can
be written to disk or streamed to stdout for offline testing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUCKDB_PATH = PROJECT_ROOT / "data" / "gex_data.db"
DEFAULT_PARQUET_PATH = PROJECT_ROOT / "data" / "exports" / "gex_snapshots_epoch.parquet"


@dataclass(frozen=True)
class SnapshotRecord:
    epoch_ms: int
    ticker: str
    spot_price: float | None
    zero_gamma: float | None
    net_gex: float | None
    min_dte: int | None
    sec_min_dte: int | None
    major_pos_vol: float | None
    major_pos_oi: float | None
    major_neg_vol: float | None
    major_neg_oi: float | None
    sum_gex_vol: float | None
    sum_gex_oi: float | None
    delta_risk_reversal: float | None
    max_priors: str | None
    source: str

    def to_dict(self) -> dict:
        """Render a JSON-serializable representation."""
        ts = datetime.fromtimestamp(self.epoch_ms / 1000, tz=timezone.utc)
        max_priors = self.max_priors
        if isinstance(max_priors, str):
            try:
                parsed = json.loads(max_priors)
            except json.JSONDecodeError:
                parsed = max_priors
        else:
            parsed = max_priors
        return {
            "timestamp_ms": self.epoch_ms,
            "timestamp_iso": ts.isoformat(),
            "timestamp": ts.isoformat(),
            "ticker": self.ticker,
            "spot_price": self.spot_price,
            "zero_gamma": self.zero_gamma,
            "net_gex": self.net_gex,
            "min_dte": self.min_dte,
            "sec_min_dte": self.sec_min_dte,
            "major_pos_vol": self.major_pos_vol,
            "major_pos_oi": self.major_pos_oi,
            "major_neg_vol": self.major_neg_vol,
            "major_neg_oi": self.major_neg_oi,
            "sum_gex_vol": self.sum_gex_vol,
            "sum_gex_oi": self.sum_gex_oi,
            "delta_risk_reversal": self.delta_risk_reversal,
            "max_priors": parsed,
            "_source": self.source,
        }


def _normalize_rows(rows: Sequence[Sequence], source: str) -> List[SnapshotRecord]:
    normalized: List[SnapshotRecord] = []
    for row in rows:
        normalized.append(
            SnapshotRecord(
                epoch_ms=int(row[0]),
                ticker=row[1],
                spot_price=row[2],
                zero_gamma=row[3],
                net_gex=row[4],
                min_dte=row[5],
                sec_min_dte=row[6],
                major_pos_vol=row[7],
                major_pos_oi=row[8],
                major_neg_vol=row[9],
                major_neg_oi=row[10],
                sum_gex_vol=row[11],
                sum_gex_oi=row[12],
                delta_risk_reversal=row[13],
                max_priors=row[14],
                source=source,
            )
        )
    return normalized


def fetch_from_duckdb(db_path: Path, ticker: str, limit: int) -> List[SnapshotRecord]:
    query = """
        SELECT
            timestamp AS epoch_ms,
            ticker,
            spot_price,
            zero_gamma,
            net_gex,
            min_dte,
            sec_min_dte,
            major_pos_vol,
            major_pos_oi,
            major_neg_vol,
            major_neg_oi,
            sum_gex_vol,
            sum_gex_oi,
            delta_risk_reversal,
            max_priors
        FROM gex_snapshots
        WHERE ticker = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute(query, [ticker, limit]).fetchall()
    finally:
        conn.close()
    return _normalize_rows(rows, source="duckdb")


def fetch_from_parquet(parquet_path: Path, ticker: str, limit: int) -> List[SnapshotRecord]:
    query = f"""
        SELECT
            epoch_ms,
            ticker,
            spot_price,
            zero_gamma,
            net_gex,
            min_dte,
            sec_min_dte,
            major_pos_vol,
            major_pos_oi,
            major_neg_vol,
            major_neg_oi,
            sum_gex_vol,
            sum_gex_oi,
            delta_risk_reversal,
            max_priors
        FROM read_parquet('{parquet_path.as_posix()}', union_by_name=true)
        WHERE ticker = ?
        ORDER BY epoch_ms DESC
        LIMIT ?
    """
    conn = duckdb.connect()
    try:
        rows = conn.execute(query, [ticker, limit]).fetchall()
    finally:
        conn.close()
    return _normalize_rows(rows, source="parquet")


def write_jsonl(records: Iterable[SnapshotRecord], out_path: Path, ascending: bool) -> None:
    ordered = list(records)
    if ascending:
        ordered = list(reversed(ordered))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record.to_dict(), separators=(",", ":")))
            handle.write("\n")


def _fmt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "-"
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value)


def replay_snapshots(
    records: Iterable[SnapshotRecord],
    sleep_seconds: float,
    ascending: bool,
    print_feed: bool,
    redis_writer: "RedisPublisher | None",
) -> None:
    ordered = list(records)
    if ascending:
        ordered = list(reversed(ordered))
    for rec in ordered:
        payload = rec.to_dict()
        if redis_writer:
            redis_writer.publish(payload)
        if print_feed:
            print(
                f"{payload['timestamp_iso']} "
                f"spot={_fmt(payload['spot_price'], 2)} "
                f"zero_gamma={_fmt(payload['zero_gamma'], 0)} "
                f"net_gex={_fmt(payload['net_gex'], 0)} "
                f"call_wall={_fmt(payload['major_pos_vol'], 0)} "
                f"put_wall={_fmt(payload['major_neg_vol'], 0)} "
                f"[source={payload['_source']}]"
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


class RedisPublisher:
    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        password: str | None,
        cache_key: str,
        snapshot_key: str | None,
        ttl: int,
    ) -> None:
        import redis

        self.client = redis.Redis(host=host, port=port, db=db, password=password)
        self.cache_key = cache_key
        self.snapshot_key = snapshot_key
        self.ttl = ttl

    def publish(self, payload: dict) -> None:
        now = datetime.now(timezone.utc)
        payload = dict(payload)
        payload["timestamp_iso"] = now.isoformat()
        payload["timestamp"] = payload["timestamp_iso"]
        payload["timestamp_ms"] = int(now.timestamp() * 1000)
        blob = json.dumps(payload, default=str)
        self.client.setex(self.cache_key, self.ttl, blob)
        if self.snapshot_key:
            self.client.setex(self.snapshot_key, self.ttl, blob)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract recent GEX snapshots for simulating the Discord GEX feed."
    )
    parser.add_argument("--ticker", default="NQ_NDX", help="Canonical ticker to query (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=300, help="Number of rows to pull (default: %(default)s)")
    parser.add_argument(
        "--source",
        choices=("auto", "duckdb", "parquet"),
        default="auto",
        help="Data source preference order",
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=DEFAULT_DUCKDB_PATH,
        help="Path to data/gex_data.db (default: %(default)s)",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=DEFAULT_PARQUET_PATH,
        help="Fallback Parquet export (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL path to persist results (default: stdout only)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Emit rows from oldest->newest (default: newest->oldest)",
    )
    parser.add_argument(
        "--simulate-feed",
        action="store_true",
        help="Print feed-friendly text lines for quick visual inspection",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Delay between simulated feed lines (seconds)",
    )
    parser.add_argument(
        "--push-redis",
        action="store_true",
        help="Write each snapshot to Redis (gex feed cache) as it replays",
    )
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--redis-db", type=int, default=int(os.getenv("REDIS_DB", "0")))
    parser.add_argument("--redis-password", default=os.getenv("REDIS_PASSWORD"))
    parser.add_argument(
        "--redis-cache-key",
        default="gex:snapshot:NQ_NDX",
        help="Key TradeBot polls first (default: %(default)s)",
    )
    parser.add_argument(
        "--redis-snapshot-key",
        default="gex:snapshot:NQ_NDX",
        help="Secondary snapshot key (default: %(default)s)",
    )
    parser.add_argument(
        "--redis-ttl",
        type=int,
        default=300,
        help="Expiration in seconds for cache entries written to Redis (default: %(default)s)",
    )
    return parser.parse_args(argv)


def resolve_records(args: argparse.Namespace) -> List[SnapshotRecord]:
    errors = []

    def _try_duck() -> List[SnapshotRecord]:
        return fetch_from_duckdb(args.duckdb_path, args.ticker, args.limit)

    def _try_parquet() -> List[SnapshotRecord]:
        return fetch_from_parquet(args.parquet_path, args.ticker, args.limit)

    sources = []
    if args.source == "auto":
        sources = [_try_duck, _try_parquet]
    elif args.source == "duckdb":
        sources = [_try_duck]
    else:
        sources = [_try_parquet]

    for fn in sources:
        try:
            records = fn()
        except Exception as exc:
            errors.append(str(exc))
            continue
        if records:
            return records
    joined = "; ".join(errors) or "no records found"
    raise RuntimeError(f"Failed to fetch snapshots: {joined}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = resolve_records(args)
    if args.output:
        write_jsonl(records, args.output, ascending=args.ascending)
        print(f"Wrote {len(records)} snapshots to {args.output}")
    redis_writer = None
    if args.push_redis:
        redis_writer = RedisPublisher(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db,
            password=args.redis_password or None,
            cache_key=args.redis_cache_key,
            snapshot_key=args.redis_snapshot_key,
            ttl=args.redis_ttl,
        )
        print(
            f"Streaming {len(records)} snapshots to Redis "
            f"({args.redis_cache_key}, snapshot={args.redis_snapshot_key or 'disabled'})"
        )
    print_feed = args.simulate_feed or not args.output
    if print_feed or redis_writer:
        replay_snapshots(
            records,
            sleep_seconds=args.sleep,
            ascending=args.ascending,
            print_feed=print_feed,
            redis_writer=redis_writer,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
