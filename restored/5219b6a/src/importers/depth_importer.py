import json
from pathlib import Path
from typing import List, Dict, Any
from src.models.market_depth import MarketDepth
from src.db.duckdb_utils import DuckDBUtils
from src.validation.data_validator import DataValidator
from src.lineage.lineage_tracker import LineageTracker

class DepthImporter:
    def __init__(self, db_utils: DuckDBUtils, lineage_tracker: LineageTracker):
        self.db_utils = db_utils
        self.lineage_tracker = lineage_tracker
        self.table_name = "market_depth"

    def create_table(self):
        """Create market depth table in DuckDB."""
        schema = """
        timestamp TIMESTAMP,
        symbol VARCHAR,
        bids VARCHAR,
        asks VARCHAR
        """
        self.db_utils.create_table_if_not_exists(self.table_name, schema)

    def import_file(self, file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Import depth data from a single JSONL file."""
        self.create_table()  # Ensure table exists
        try:
            valid_records = []
            errors = []

            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())

                        # Map JSON fields to model
                        record_data = {
                            'timestamp': data.get('timestamp'),
                            'symbol': data.get('ticker', 'MNQ'),  # Assume MNQ
                            'bids': data.get('bids', []),
                            'asks': data.get('asks', [])
                        }

                        record = MarketDepth.from_dict(record_data)
                        validation_errors = record.validate()
                        if validation_errors:
                            errors.extend(validation_errors)
                            continue

                        # Validate timestamp
                        if not DataValidator.validate_timestamp(record.timestamp):
                            errors.append(f"Invalid timestamp at line {line_num}")
                            continue

                        valid_records.append(record)
                    except json.JSONDecodeError:
                        errors.append(f"Invalid JSON at line {line_num}")
                    except Exception as e:
                        errors.append(f"Error at line {line_num}: {str(e)}")

            if not dry_run and valid_records:
                # Convert to dicts for insertion
                data_dicts = [record.to_dict() for record in valid_records]

                with self.db_utils:
                    self.db_utils.insert_data(self.table_name, data_dicts)

                # Record lineage
                self.lineage_tracker.record_import(
                    str(file_path), len(valid_records), 'depth'
                )

            return {
                'file': str(file_path),
                'total_lines': line_num,
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
        """Import multiple depth files."""
        self.create_table()
        results = []
        for file_path in file_paths:
            result = self.import_file(file_path, dry_run)
            results.append(result)
        return results