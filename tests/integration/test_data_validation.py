"""
Integration tests for data validation.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.lib.database import DatabaseConnection
from src.lib.parquet_handler import ParquetHandler
from src.models.depth_snapshot import DepthSnapshot
from src.services.data_validator import DataValidator


class TestDataValidation:
    """Integration tests for data validation pipeline."""

    def test_full_validation_pipeline(self):
        """Test complete validation of extracted data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup test data
            db_path = Path(temp_dir) / "test.db"
            db = DatabaseConnection(str(db_path))

            # Insert test tick data
            with db.connect() as conn:
                for i in range(100):
                    timestamp = datetime(2024, 1, 1, 12, 0, 0) + timedelta(seconds=i)
                    conn.execute("""
                        INSERT INTO mnq_ticks (timestamp, price, volume, tick_type)
                        VALUES (?, ?, ?, ?)
                    """, [str(timestamp), 4500.00 + i * 0.01, 10, 'buy'])

            # Create test depth data
            handler = ParquetHandler(temp_dir)
            depth_data = []
            for i in range(50):
                timestamp = datetime(2024, 1, 1, 12, 0, 0) + timedelta(seconds=i*2)
                depth_data.append(DepthSnapshot(
                    timestamp=timestamp,
                    bid_price_1=4500.00,
                    bid_size_1=100,
                    ask_price_1=4500.25,
                    ask_size_1=150
                ))

            # Save depth data
            handler.save_depth_data(depth_data, datetime(2024, 1, 1))

            # Run validation
            validator = DataValidator()
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 1)

            results = validator.validate_dataset(
                db_path=str(db_path),
                parquet_dir=temp_dir,
                start_date=start_date,
                end_date=end_date
            )

            # Check results
            assert results['total_ticks'] == 100
            assert results['total_depth_records'] == 50
            assert results['duplicate_ticks'] == 0
            assert results['completeness_score'] > 0.8  # Should be high for test data

    def test_validation_with_missing_data(self):
        """Test validation when data is missing."""
        validator = DataValidator()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "empty.db"

            # Empty database
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 1)

            results = validator.validate_dataset(
                db_path=str(db_path),
                parquet_dir=temp_dir,
                start_date=start_date,
                end_date=end_date
            )

            assert results['total_ticks'] == 0
            assert results['completeness_score'] == 0.0

    def test_validation_error_handling(self):
        """Test validation handles errors gracefully."""
        validator = DataValidator()

        # Test with invalid paths
        results = validator.validate_dataset(
            db_path="/nonexistent/db.db",
            parquet_dir="/nonexistent/dir",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1)
        )

        # Should return error indicators
        assert 'error' in results or results['total_ticks'] == 0
