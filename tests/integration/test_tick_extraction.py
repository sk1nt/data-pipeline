"""
Integration tests for tick data extraction.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from src.models.mnq_tick_record import MnqTickRecord
from src.lib.database import DatabaseConnection


class TestTickExtraction:
    """Integration tests for tick data extraction pipeline."""

    def test_tick_record_creation(self):
        """Test creating tick records from SCID data."""
        mock_record = {
            'timestamp': datetime(2024, 1, 1, 12, 0, 0),
            'close': 4500.12,
            'total_volume': 5,
            'bid_volume': 3,
            'ask_volume': 2
        }

        tick = MnqTickRecord.from_scid_record(mock_record)

        assert tick.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert tick.price == 4500.12
        assert tick.volume == 5
        assert tick.tick_type == 'buy'  # bid_volume > ask_volume

    def test_database_storage(self):
        """Test storing tick records in DuckDB."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = DatabaseConnection(str(db_path))

            with db.connect() as conn:
                # Insert test data
                tick = MnqTickRecord(
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    price=4500.12,
                    volume=5,
                    tick_type='buy'
                )

                conn.execute("""
                    INSERT INTO mnq_ticks (timestamp, price, volume, tick_type)
                    VALUES (?, ?, ?, ?)
                """, [tick.timestamp, tick.price, tick.volume, tick.tick_type])

                # Verify insertion
                result = conn.execute("SELECT COUNT(*) FROM mnq_ticks").fetchone()
                assert result[0] == 1

                # Verify data
                result = conn.execute("""
                    SELECT timestamp, price, volume, tick_type FROM mnq_ticks
                """).fetchone()
                assert result[0] == tick.timestamp
                assert result[1] == tick.price
                assert result[2] == tick.volume
                assert result[3] == tick.tick_type

    def test_tick_validation(self):
        """Test tick record validation."""
        # Valid tick
        MnqTickRecord(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            price=4500.12,
            volume=5
        )
        # Should not raise

        # Invalid price
        with pytest.raises(ValueError, match="Price must be positive"):
            MnqTickRecord(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                price=-100,
                volume=5
            )

        # Invalid volume
        with pytest.raises(ValueError, match="Volume must be positive"):
            MnqTickRecord(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                price=4500.12,
                volume=-1
            )
