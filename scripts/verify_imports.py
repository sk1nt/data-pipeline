#!/usr/bin/env python3
"""Verify imported datasets for timestamp precision and coverage percentages.

This utility inspects the local DuckDB/Parquet artifacts under ``data/`` and
prints two sections:

1. Timestamp precision checks – ensures every dataset stores event times as
   epoch milliseconds (either as BIGINT or as TIMESTAMPs with millisecond
   resolution).
2. Coverage summary – counts trading days (weekdays) with data between the
   requested window, optionally overriding the depth start date (the depth
   exporter did not run until 2025-09-19).

Examples::

    # Default window (start=2025-09-02, end=latest date discovered)
    python scripts/verify_imports.py

    # Restrict the window to 2025-09-02 -> 2025-10-31
    python scripts/verify_imports.py --end-date 2025-10-31

"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import duckdb


@dataclasses.dataclass
class TimestampSpec:
    name: str
    source_sql: str
    timestamp_column: str
    kind: str  # ``epoch_ms`` (BIGINT) or ``timestamp`` (DuckDB TIMESTAMP)
    db_path: Optional[str] = None


@dataclasses.dataclass
class CoverageSpec:
    name: str
    path: Path
    glob: str
    coverage_start: Optional[dt.date] = None
    counts_toward_end: bool = True


def _connect(db_path: Optional[str]) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=db_path or ":memory:")


def _build_timestamp_query(spec: TimestampSpec) -> str:
    column = f'"{spec.timestamp_column}"'
    if spec.kind == "epoch_ms":
        epoch_expr = column
        remainder_expr = f"({column} % 1000)"
    elif spec.kind == "timestamp":
        epoch_expr = f"epoch_ms({column})"
        remainder_expr = f"(epoch_ms({column}) % 1000)"
    else:
        raise ValueError(f"Unknown timestamp kind: {spec.kind}")

    # Fractional-millisecond residue is only meaningful for TIMESTAMP columns.
    micro_expr = (
        "date_part('microseconds', {column}) % 1000".format(column=column)
        if spec.kind == "timestamp"
        else "NULL"
    )

    return f"""
        SELECT
            COUNT(*) AS total_rows,
            MIN({epoch_expr}) AS min_epoch_ms,
            MAX({epoch_expr}) AS max_epoch_ms,
            MIN({remainder_expr}) AS min_ms_remainder,
            MAX({remainder_expr}) AS max_ms_remainder,
            MIN({micro_expr}) AS min_micro_remainder,
            MAX({micro_expr}) AS max_micro_remainder
        FROM {spec.source_sql}
    """


def _epoch_to_iso(epoch_ms: Optional[int]) -> Optional[str]:
    if epoch_ms is None:
        return None
    return (
        dt.datetime.fromtimestamp(epoch_ms / 1000, tz=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_timestamp_checks(specs: Sequence[TimestampSpec]) -> List[dict]:
    results = []
    for spec in specs:
        try:
            with _connect(spec.db_path) as conn:
                query = _build_timestamp_query(spec)
                (
                    total_rows,
                    min_ms,
                    max_ms,
                    min_rem,
                    max_rem,
                    min_micro,
                    max_micro,
                ) = conn.execute(query).fetchone()
        except Exception as exc:  # noqa: BLE001 - bubble detailed error downstream
            results.append({"name": spec.name, "error": str(exc)})
            continue

        results.append(
            {
                "name": spec.name,
                "rows": total_rows,
                "min_epoch_ms": min_ms,
                "max_epoch_ms": max_ms,
                "min_epoch_iso": _epoch_to_iso(min_ms),
                "max_epoch_iso": _epoch_to_iso(max_ms),
                "ms_remainder_range": (min_rem, max_rem),
                "micro_remainder_range": (min_micro, max_micro),
            }
        )
    return results


def _extract_yyyymmdd(path: Path) -> Optional[dt.date]:
    match = re.search(r"(\d{8})", path.stem)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), "%Y%m%d").date()


def _business_days(start: dt.date, end: dt.date) -> List[dt.date]:
    days: List[dt.date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # Mon=0 .. Fri=4
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


def compute_coverage(
    specs: Sequence[CoverageSpec],
    start_date: dt.date,
    end_date: Optional[dt.date],
) -> Tuple[List[dict], dt.date]:
    all_dates: List[dt.date] = []
    per_spec_dates: List[List[dt.date]] = []

    for spec in specs:
        files = sorted(spec.path.glob(spec.glob))
        dates = [d for f in files if (d := _extract_yyyymmdd(f))]
        per_spec_dates.append(dates)
        if spec.counts_toward_end:
            all_dates.extend(dates)

    latest_date = max(all_dates) if all_dates else start_date
    window_end = end_date or latest_date

    reports: List[dict] = []
    for spec, dates in zip(specs, per_spec_dates):
        effective_start = max(start_date, spec.coverage_start or start_date)
        effective_dates = [
            d for d in dates if effective_start <= d <= window_end and d.weekday() < 5
        ]
        expected_days = _business_days(effective_start, window_end)
        if effective_start > window_end:
            expected_days = []
        missing = sorted(set(expected_days) - set(effective_dates))
        present = sorted(set(effective_dates))
        total = len(expected_days)
        pct = (len(present) / total * 100.0) if total else None
        reports.append(
            {
                "name": spec.name,
                "present_days": len(present),
                "expected_days": total,
                "coverage_pct": pct,
                "missing_dates": missing,
                "first_date": present[0] if present else None,
                "last_date": present[-1] if present else None,
            }
        )

    return reports, window_end


def format_date(d: Optional[dt.date]) -> str:
    return d.isoformat() if d else "N/A"


def print_timestamp_report(results: Sequence[dict]) -> None:
    print("Timestamp Precision Checks")
    print("---------------------------")
    for res in results:
        print(f"- {res['name']}")
        if res.get("error"):
            print(f"  error: {res['error']}")
            print()
            continue
        print(f"  rows: {res['rows']:,}")
        print(
            f"  epoch range: {res['min_epoch_ms']} ({res['min_epoch_iso']}) → "
            f"{res['max_epoch_ms']} ({res['max_epoch_iso']})"
        )
        min_rem, max_rem = res["ms_remainder_range"]
        if min_rem is not None and max_rem is not None:
            print(f"  ms remainder range: {min_rem} .. {max_rem}")
        micro_min, micro_max = res["micro_remainder_range"]
        if micro_min is not None and micro_max is not None:
            print(f"  micro remainder range: {micro_min} .. {micro_max}")
        print()


def print_coverage_report(
    reports: Sequence[dict],
    start_date: dt.date,
    end_date: dt.date,
    depth_start: dt.date,
) -> None:
    print("Coverage Summary")
    print("----------------")
    print(f"Window: {start_date.isoformat()} → {end_date.isoformat()} (weekdays)")
    print(f"Depth start override: {depth_start.isoformat()}")
    for res in reports:
        pct = res["coverage_pct"]
        pct_msg = f"{pct:.1f}%" if pct is not None else "N/A"
        print(
            f"- {res['name']}: {res['present_days']} / {res['expected_days']} trading days ({pct_msg})"
        )
        missing = res["missing_dates"]
        if missing:
            preview = ", ".join(d.isoformat() for d in missing[:10])
            extra = "" if len(missing) <= 10 else f" … +{len(missing) - 10} more"
            print(f"    missing: {preview}{extra}")
        else:
            print("    missing: none")
        print(
            f"    observed range: {format_date(res['first_date'])} → {format_date(res['last_date'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default="2025-09-02",
        help="Inclusive UTC date (YYYY-MM-DD) to start the coverage window",
    )
    parser.add_argument(
        "--end-date",
        help="Inclusive UTC date (YYYY-MM-DD) to stop the coverage window. "
        "Defaults to the latest file date discovered.",
    )
    parser.add_argument(
        "--depth-start-date",
        default="2025-09-19",
        help="Earliest date to consider for depth coverage (ignores older files)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = dt.date.fromisoformat(args.start_date)
    depth_start = dt.date.fromisoformat(args.depth_start_date)
    end_date = dt.date.fromisoformat(args.end_date) if args.end_date else None

    timestamp_specs = [
        TimestampSpec(
            name="GEX snapshots (data/gex_data.db::gex_snapshots)",
            db_path="data/gex_data.db",
            source_sql="gex_snapshots",
            timestamp_column="timestamp",
            kind="epoch_ms",
        ),
        TimestampSpec(
            name="GEX strikes (data/gex_data.db::gex_strikes)",
            db_path="data/gex_data.db",
            source_sql="gex_strikes",
            timestamp_column="timestamp",
            kind="epoch_ms",
        ),
        TimestampSpec(
            name="GEX strikes parquet",
            source_sql="read_parquet('data/parquet/gexbot/NQ_NDX/gex_zero/*.parquet')",
            timestamp_column="timestamp",
            kind="epoch_ms",
        ),
        TimestampSpec(
            name="Tick data (data/tick_data.db::tick_data)",
            db_path="data/tick_data.db",
            source_sql="tick_data",
            timestamp_column="timestamp",
            kind="timestamp",
        ),
        TimestampSpec(
            name="Tick parquet (data/parquet/tick/MNQ/*.parquet)",
            source_sql="read_parquet('data/parquet/tick/MNQ/*.parquet')",
            timestamp_column="timestamp",
            kind="timestamp",
        ),
        TimestampSpec(
            name="Depth parquet (data/parquet/depth/MNQ/*.parquet)",
            source_sql="read_parquet('data/parquet/depth/MNQ/*.parquet')",
            timestamp_column="timestamp",
            kind="timestamp",
        ),
    ]

    coverage_specs = [
        CoverageSpec(
            name="GEX strikes parquet",
            path=Path("data/parquet/gexbot/NQ_NDX/gex_zero"),
            glob="*.parquet",
        ),
        CoverageSpec(
            name="Tick parquet (MNQ)",
            path=Path("data/parquet/tick/MNQ"),
            glob="*.parquet",
        ),
        CoverageSpec(
            name="Depth parquet (MNQ)",
            path=Path("data/parquet/depth/MNQ"),
            glob="*.parquet",
            coverage_start=depth_start,
        ),
    ]

    ts_results = run_timestamp_checks(timestamp_specs)
    coverage_results, resolved_end = compute_coverage(
        coverage_specs, start_date=start_date, end_date=end_date
    )

    print_timestamp_report(ts_results)
    print_coverage_report(
        coverage_results,
        start_date=start_date,
        end_date=resolved_end,
        depth_start=depth_start,
    )


if __name__ == "__main__":
    main()
