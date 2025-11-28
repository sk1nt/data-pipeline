import pytest
import tempfile
import json
import csv
from pathlib import Path

from src.importers.gex_importer import GEXImporter
from src.importers.tick_importer import TickImporter
from src.importers.depth_importer import DepthImporter
from src.db.duckdb_utils import DuckDBUtils
from src.lineage.lineage_tracker import LineageTracker


class TestImportIntegration:
    @pytest.fixture
    def db_utils(self):
        import tempfile
        from pathlib import Path

        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = db_file.name
        db_file.close()
        # Delete the empty file so DuckDB can create it
        Path(db_path).unlink(missing_ok=True)
        yield DuckDBUtils(db_path)
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def lineage_tracker(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            lineage_file = f.name
        yield LineageTracker(lineage_file)
        Path(lineage_file).unlink(missing_ok=True)

    def test_gex_import_integration(self, db_utils, lineage_tracker):
        importer = GEXImporter(db_utils, lineage_tracker)

        # Create test JSON file
        test_data = {
            "timestamp": 1672574400,  # Unix timestamp for 2023-01-01T12:00:00
            "ticker": "NQ_NDX",
            "spot_price": 15000.0,
            "zero_gamma": 100.0,
            "strikes": [[15000, 10.0, 5.0, []]],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            json_file = Path(f.name)

        try:
            result = importer.import_file(json_file, dry_run=False)
            assert result["valid_snapshots"] == 1
            assert len(result["errors"]) == 0

            # Check database - need to use the same connection
            with db_utils:
                importer.create_table()  # Ensure table exists in this connection
                result = db_utils.execute_query(
                    "SELECT COUNT(*) as count FROM gex_snapshots"
                )
                count = result[0]["count"]
                assert count == 1
        finally:
            json_file.unlink()

    def test_tick_import_integration(self, db_utils, lineage_tracker):
        importer = TickImporter(db_utils, lineage_tracker)

        # Create test CSV file
        test_data = [
            ["timestamp", "symbol", "bid", "ask", "last", "volume"],
            ["2023-01-01T12:00:00", "MNQ", "15000", "15001", "15000.5", "10"],
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerows(test_data)
            csv_file = Path(f.name)

        try:
            result = importer.import_file(csv_file, dry_run=False)
            assert result["valid_records"] == 1
            assert len(result["errors"]) == 0

            # Check database
            with db_utils:
                result = db_utils.execute_query(
                    "SELECT COUNT(*) as count FROM tick_records"
                )
                count = result[0]["count"]
                assert count == 1
        finally:
            csv_file.unlink()

    def test_depth_import_integration(self, db_utils, lineage_tracker):
        importer = DepthImporter(db_utils, lineage_tracker)

        # Create test JSONL file
        test_data = {
            "timestamp": "2023-01-01T12:00:00",
            "ticker": "MNQ",
            "bids": [[15000, 10], [14999, 5]],
            "asks": [[15001, 10], [15002, 5]],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(test_data) + "\n")
            jsonl_file = Path(f.name)

        try:
            result = importer.import_file(jsonl_file, dry_run=False)
            assert result["valid_records"] == 1
            assert len(result["errors"]) == 0

            # Check database
            with db_utils:
                result = db_utils.execute_query(
                    "SELECT COUNT(*) as count FROM market_depth"
                )
                count = result[0]["count"]
                assert count == 1
        finally:
            jsonl_file.unlink()
