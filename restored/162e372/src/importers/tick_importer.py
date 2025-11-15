import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from src.models.tick_record import TickRecord
from src.db.duckdb_utils import DuckDBUtils
from src.validation.data_validator import DataValidator
from src.lineage.lineage_tracker import LineageTracker

class TickImporter:
    def __init__(self, db_utils: DuckDBUtils, lineage_tracker: LineageTracker):
        self.db_utils = db_utils
        self.lineage_tracker = lineage_tracker
        self.table_name = "tick_records"

    def create_table(self):
        """Create tick records table in DuckDB."""
        schema = """
        timestamp TIMESTAMP,
        symbol VARCHAR,
        bid DOUBLE,
        ask DOUBLE,
        last DOUBLE,
        volume INTEGER
        """
        self.db_utils.create_table_if_not_exists(self.table_name, schema)

    def import_file(self, file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Import tick data from a single CSV file."""
        self.create_table()  # Ensure table exists
        try:
            # Read CSV with Polars
            df = pl.read_csv(file_path)

            valid_records = []
            errors = []

            for row in df.iter_rows(named=True):
                try:
                    # Map CSV columns to model fields
                    record_data = {
                        'timestamp': row.get('timestamp'),
                        'symbol': row.get('symbol'),
                        'bid': row.get('bid'),
                        'ask': row.get('ask'),
                        'last': row.get('last'),
                        'volume': row.get('last_size') or row.get('volume', 0)
                    }

                    record = TickRecord.from_dict(record_data)
                    validation_errors = record.validate()
                    if validation_errors:
                        errors.extend(validation_errors)
                        continue

                    # Validate with DataValidator
                    if not DataValidator.validate_timestamp(record.timestamp):
                        errors.append(f"Invalid timestamp: {record.timestamp}")
                        continue

                    valid_records.append(record)
                except Exception as e:
                    errors.append(f"Row processing error: {str(e)}")

            if not dry_run and valid_records:
                # Convert to dicts for insertion
                data_dicts = [record.to_dict() for record in valid_records]

                with self.db_utils:
                    self.db_utils.insert_data(self.table_name, data_dicts)

                # Record lineage
                self.lineage_tracker.record_import(
                    str(file_path), len(valid_records), 'tick'
                )

            return {
                'file': str(file_path),
                'total_rows': len(df),
                'valid_records': len(valid_records),
                'errors': errors,
                'dry_run': dry_run
            }

        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'dry_run': dry_run
            }

    def import_files(self, file_paths: List[Path], dry_run: bool = False) -> List[Dict[str, Any]]:
        """Import multiple tick files."""
        self.create_table()
        results = []
        for file_path in file_paths:
            result = self.import_file(file_path, dry_run)
            results.append(result)
        return results