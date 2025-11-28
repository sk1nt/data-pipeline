#!/usr/bin/env python3
"""Verify timestamps in depth and tick parquet files.

- Checks that timestamp values are on the file's date
- Checks that `ts_ms` exists and matches the timestamp (if present)
- Detects corrupt/unreadable parquets
- Prints a summary and optional per-file details

Usage:
  python3 scripts/verify_timestamps.py --start 2025-09-02 --end 2025-11-16 --symbol MNQ

"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify timestamps in Parquet files")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--end", default=None, help="End date YYYY-MM-DD (defaults to today)"
    )
    parser.add_argument("--symbol", default="MNQ", help="Symbol (folder name) to scan")
    parser.add_argument(
        "--parquet-root", default="data/parquet", help="Parquet root folder"
    )
    parser.add_argument("--out-csv", default=None, help="Optional CSV output path")
    parser.add_argument("--kind", choices=["depth", "tick", "both"], default="both")
    parser.add_argument(
        "--timestamp-tz",
        default="UTC",
        help="Optional timezone to use when comparing ts_ms via 'AT TIME ZONE' (eg. America/New_York). Default: UTC.",
    )
    return parser.parse_args()


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


PARQUET_ROOT = Path("data/parquet")


def fmt_path(kind: str, symbol: str, dt: date) -> Path:
    return PARQUET_ROOT / kind / symbol / f"{dt.strftime('%Y%m%d')}.parquet"


def check_file(path: Path, kind: str, dt: date):
    """Return a dict of checks for a single Parquet file."""
    import duckdb

    con = duckdb.connect()
    result = {
        "kind": kind,
        "path": str(path),
        "date": dt.isoformat(),
        "exists": path.exists(),
        "readable": False,
        "rows": 0,
        "has_ts_ms": False,
        "mismatch_rows": 0,
        "ts_ms_mismatch_rows": 0,
        "min_ts": None,
        "max_ts": None,
        "min_ts_ms": None,
        "max_ts_ms": None,
    }
    if not path.exists():
        return result
    try:
        # try to read a single row to confirm schema/columns
        # fetch count, min/max, and mismatch metrics
        # DuckDB uses "date" function for timestamp -> date
        cols = (
            con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 1")
            .fetchdf()
            .columns.tolist()
        )

        # Basic stats
        rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
        result["rows"] = rows

        has_ts_ms = "ts_ms" in cols
        result["has_ts_ms"] = has_ts_ms

        # mismatch_rows = count of rows where DATE(timestamp) != expected
        mismatch_q = f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE CAST(timestamp AS DATE) != DATE('{dt.isoformat()}')"
        mismatch = con.execute(mismatch_q).fetchone()[0]
        result["mismatch_rows"] = mismatch

        # If ts_ms is present, compare conversions
        if has_ts_ms:
            # ts_ms vs CAST(EXTRACT(EPOCH FROM timestamp) * 1000)
            # Optionally apply AT TIME ZONE conversion when computing epoch in DuckDB
            tz_clause = f"AT TIME ZONE '{ARG_TS_TZ}'" if ARG_TS_TZ else ""
            ts_ms_mismatch_q = f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE ts_ms IS NOT NULL AND ABS(ts_ms - CAST(EXTRACT(epoch FROM timestamp {tz_clause}) * 1000 AS BIGINT)) > 5"
            tsms_mismatch = con.execute(ts_ms_mismatch_q).fetchone()[0]
            result["ts_ms_mismatch_rows"] = tsms_mismatch

            min_ts = con.execute(
                f"SELECT MIN(timestamp) FROM read_parquet('{path}')"
            ).fetchone()[0]
            max_ts = con.execute(
                f"SELECT MAX(timestamp) FROM read_parquet('{path}')"
            ).fetchone()[0]
            min_ts_ms = con.execute(
                f"SELECT MIN(ts_ms) FROM read_parquet('{path}')"
            ).fetchone()[0]
            max_ts_ms = con.execute(
                f"SELECT MAX(ts_ms) FROM read_parquet('{path}')"
            ).fetchone()[0]
            result["min_ts"] = min_ts
            result["max_ts"] = max_ts
            result["min_ts_ms"] = min_ts_ms
            result["max_ts_ms"] = max_ts_ms

        else:
            # min/max timestamp useful even if ts_ms missing
            min_ts = con.execute(
                f"SELECT MIN(timestamp) FROM read_parquet('{path}')"
            ).fetchone()[0]
            max_ts = con.execute(
                f"SELECT MAX(timestamp) FROM read_parquet('{path}')"
            ).fetchone()[0]
            result["min_ts"] = min_ts
            result["max_ts"] = max_ts

        result["readable"] = True
    except Exception as exc:
        result["readable"] = False
        result["error"] = str(exc)
    finally:
        try:
            con.close()
        except Exception:
            pass
    return result


def main():
    args = parse_args()
    global ARG_TS_TZ
    ARG_TS_TZ = args.timestamp_tz
    global PARQUET_ROOT
    PARQUET_ROOT = Path(args.parquet_root)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_dt = (
        datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
    )

    rows = []
    for dt in daterange(start_dt, end_dt):
        for kind in (
            ["depth"]
            if args.kind == "depth"
            else (["tick"] if args.kind == "tick" else ["depth", "tick"])
        ):
            path = fmt_path(kind, args.symbol, dt)
            if not path.exists():
                rows.append(
                    {
                        "kind": kind,
                        "path": str(path),
                        "date": dt.isoformat(),
                        "exists": False,
                    }
                )
                continue
            result = check_file(path, kind, dt)
            rows.append(result)

    # Print summary
    total_files = len(rows)
    corrupt = [
        r for r in rows if not r.get("readable", False) and r.get("exists", False)
    ]
    missing = [r for r in rows if not r.get("exists", False)]
    missing_ts_ms = [
        r
        for r in rows
        if r.get("exists") and r.get("readable") and not r.get("has_ts_ms")
    ]
    mismatch_date = [
        r
        for r in rows
        if r.get("exists") and r.get("readable") and r.get("mismatch_rows", 0) > 0
    ]
    ts_ms_mismatch = [
        r
        for r in rows
        if r.get("exists") and r.get("readable") and r.get("ts_ms_mismatch_rows", 0) > 0
    ]

    print(f"Total files scanned: {total_files}")
    print(f"Missing parquet (not present): {len(missing)}")
    print(f"Corrupt parquet (failed to read): {len(corrupt)}")
    print(f"Parquet missing ts_ms: {len(missing_ts_ms)}")
    print(f"Rows with DATE(timestamp) mismatch: {len(mismatch_date)}")
    print(f"Rows with ts_ms mismatch (>5ms): {len(ts_ms_mismatch)}")

    # Optionally CSV output
    if args.out_csv:
        out = Path(args.out_csv)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "kind",
                    "path",
                    "date",
                    "exists",
                    "readable",
                    "rows",
                    "has_ts_ms",
                    "mismatch_rows",
                    "ts_ms_mismatch_rows",
                    "min_ts",
                    "max_ts",
                    "min_ts_ms",
                    "max_ts_ms",
                    "error",
                ],
            )  # noqa: E501
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
        print(f"Wrote CSV to {out}")

    # Print some representative failures (head of lists)
    if corrupt:
        print("\nCorrupt samples:")
        for r in corrupt[:20]:
            print(r.get("path"), r.get("error"))
    if missing_ts_ms:
        print("\nMissing ts_ms samples:")
        for r in missing_ts_ms[:20]:
            print(
                r.get("path"),
                "rows:",
                r.get("rows"),
                "min_ts:",
                r.get("min_ts"),
                "max_ts:",
                r.get("max_ts"),
            )
    if mismatch_date:
        print("\nDATE(timestamp) mismatch samples:")
        for r in mismatch_date[:20]:
            print(
                r.get("path"),
                "mismatch_rows:",
                r.get("mismatch_rows"),
                "rows:",
                r.get("rows"),
                "min_ts:",
                r.get("min_ts"),
                "max_ts:",
                r.get("max_ts"),
            )
    if ts_ms_mismatch:
        print("\nTs_ms mismatch samples:")
        for r in ts_ms_mismatch[:20]:
            print(
                r.get("path"),
                "ts_ms_mismatch_rows:",
                r.get("ts_ms_mismatch_rows"),
                "rows:",
                r.get("rows"),
            )

    # exit code 0 if everything fine else 1
    if corrupt or missing_ts_ms or mismatch_date or ts_ms_mismatch:
        sys.exit(1)
    print("All files OK.")


if __name__ == "__main__":
    main()
