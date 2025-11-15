"""
Data validation service for MNQ extracted data.
"""

from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
import duckdb
import polars as pl

from ..lib.logging_config import logger


class DataValidator:
    """
    Service for validating extracted MNQ tick and depth data.
    """

    def __init__(self):
        """Initialize validator."""
        pass

    def validate_timestamp(self, timestamp: datetime, start_date: datetime, end_date: datetime) -> bool:
        """
        Validate timestamp is within date range.

        Args:
            timestamp: Timestamp to validate
            start_date: Start of valid range
            end_date: End of valid range

        Returns:
            True if valid
        """
        return start_date <= timestamp.replace(hour=0, minute=0, second=0, microsecond=0) <= end_date

    def check_completeness(self, stats: Dict[str, Any]) -> bool:
        """
        Check if data is reasonably complete.

        Args:
            stats: Data statistics

        Returns:
            True if complete enough
        """
        ticks = stats.get('ticks', 0)
        expected_days = stats.get('expected_days', 1)
        actual_days = stats.get('actual_days', 0)

        # Basic heuristics
        min_ticks_per_day = 100  # Very conservative minimum
        expected_ticks = expected_days * min_ticks_per_day

        return ticks >= expected_ticks * 0.1 and actual_days >= expected_days * 0.5

    def find_duplicates(self, records: List[Dict], key_fields: List[str]) -> List[Dict]:
        """
        Find duplicate records based on key fields.

        Args:
            records: List of records
            key_fields: Fields to check for duplicates

        Returns:
            List of duplicate groups with counts
        """
        seen = {}
        duplicates = []

        for record in records:
            key = tuple(record.get(field) for field in key_fields)
            if key in seen:
                seen[key] += 1
            else:
                seen[key] = 1

        for key, count in seen.items():
            if count > 1:
                duplicates.append({
                    'key': key,
                    'count': count,
                    'fields': key_fields
                })

        return duplicates

    def validate_consistency(self, data: Dict[str, Any]) -> bool:
        """
        Validate consistency between tick and depth data.

        Args:
            data: Data to validate

        Returns:
            True if consistent
        """
        tick_timestamps = data.get('tick_timestamps', [])
        depth_timestamps = data.get('depth_timestamps', [])

        if not tick_timestamps or not depth_timestamps:
            return False

        # Check if timestamp ranges overlap reasonably
        tick_start = min(tick_timestamps)
        tick_end = max(tick_timestamps)
        depth_start = min(depth_timestamps)
        depth_end = max(depth_timestamps)

        # Allow some tolerance (1 hour)
        tolerance = timedelta(hours=1)

        return (tick_start - tolerance <= depth_end and
                depth_start - tolerance <= tick_end)

    def validate_dataset(self, db_path: str, parquet_dir: str,
                        start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Validate complete dataset.

        Args:
            db_path: Path to DuckDB file
            parquet_dir: Directory with Parquet files
            start_date: Start date
            end_date: End date

        Returns:
            Validation results
        """
        results = {
            'total_ticks': 0,
            'total_depth_records': 0,
            'duplicate_ticks': 0,
            'completeness_score': 0.0,
            'validation_errors': []
        }

        try:
            # Validate tick data
            tick_stats = self._validate_tick_data(db_path, start_date, end_date)
            results.update(tick_stats)

            # Validate depth data
            depth_stats = self._validate_depth_data(parquet_dir, start_date, end_date)
            results.update(depth_stats)

            # Calculate completeness
            expected_days = (end_date - start_date).days + 1
            tick_days = results.get('tick_days', 0)
            if tick_days == 0:
                results['completeness_score'] = 0.0
            else:
                actual_days = min(expected_days, tick_days)
                results['completeness_score'] = actual_days / expected_days

            logger.info(f"Validation completed: {results}")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            results['validation_errors'].append(str(e))

        return results

    def _validate_tick_data(self, db_path: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Validate tick data in DuckDB."""
        stats = {'tick_days': 0}

        try:
            conn = duckdb.connect(db_path)

            # Count total ticks
            result = conn.execute("SELECT COUNT(*) FROM mnq_ticks").fetchone()
            stats['total_ticks'] = result[0] if result else 0

            # Check date range
            end_date_next = end_date + timedelta(days=1)
            result = conn.execute("""
                SELECT COUNT(DISTINCT DATE(timestamp)) as days
                FROM mnq_ticks
                WHERE timestamp >= ? AND timestamp < ?
            """, [str(start_date), str(end_date_next)]).fetchone()
            stats['tick_days'] = result[0] if result else 0

            # Check for duplicates
            result = conn.execute("""
                SELECT COUNT(*) - COUNT(DISTINCT CAST(timestamp AS VARCHAR) || '_' || CAST(price AS VARCHAR)) as duplicates
                FROM mnq_ticks
            """).fetchone()
            stats['duplicate_ticks'] = result[0] if result else 0

            conn.close()

        except Exception as e:
            logger.error(f"Tick data validation failed: {e}")
            stats['tick_validation_error'] = str(e)

        return stats

    def _validate_depth_data(self, parquet_dir: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Validate depth data in Parquet files."""
        stats = {'depth_files': 0}

        try:
            parquet_path = Path(parquet_dir)
            total_records = 0
            file_count = 0

            # Check all parquet files
            for parquet_file in parquet_path.glob("*.parquet"):
                try:
                    df = pl.read_parquet(str(parquet_file))
                    total_records += len(df)
                    file_count += 1
                except Exception as e:
                    logger.warning(f"Failed to read {parquet_file}: {e}")

            stats['total_depth_records'] = total_records
            stats['depth_files'] = file_count

        except Exception as e:
            logger.error(f"Depth data validation failed: {e}")
            stats['depth_validation_error'] = str(e)

        return stats