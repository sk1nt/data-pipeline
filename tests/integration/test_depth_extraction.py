"""
Integration tests for depth data extraction.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from src.models.depth_snapshot import DepthSnapshot
from src.lib.parquet_handler import ParquetHandler


class TestDepthExtraction:
    """Integration tests for depth data extraction pipeline."""

    def test_depth_snapshot_creation_from_scid(self):
        """Test creating depth snapshots from SCID-like data."""
        mock_records = [
            {
                "timestamp": datetime(2024, 1, 1, 12, 0, 0),
                "low": 4500.00,  # Bid price
                "high": 4500.25,  # Ask price
                "bid_volume": 100,
                "ask_volume": 150,
            }
        ]

        snapshot = DepthSnapshot.from_scid_records(mock_records)

        assert snapshot.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert snapshot.bid_price_1 == 4500.00
        assert snapshot.bid_size_1 == 100
        assert snapshot.ask_price_1 == 4500.25
        assert snapshot.ask_size_1 == 150

    def test_parquet_storage(self):
        """Test storing depth data in Parquet format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ParquetHandler(temp_dir)

            snapshots = [
                DepthSnapshot(
                    timestamp=datetime(2024, 1, 1, 12, 0, 0),
                    bid_price_1=4500.00,
                    bid_size_1=100,
                    ask_price_1=4500.25,
                    ask_size_1=150,
                ),
                DepthSnapshot(
                    timestamp=datetime(2024, 1, 1, 12, 0, 1),
                    bid_price_1=4500.05,
                    bid_size_1=120,
                    ask_price_1=4500.30,
                    ask_size_1=140,
                ),
            ]

            date = datetime(2024, 1, 1)
            file_path = handler.save_depth_data(snapshots, date)

            # Verify file was created
            assert Path(file_path).exists()

            # Load and verify data
            df = handler.load_depth_data(date)
            assert len(df) == 2
            assert df["bid_price_1"][0] == 4500.00
            assert df["ask_size_1"][1] == 140

    def test_depth_validation(self):
        """Test depth snapshot validation."""
        # Valid depth
        DepthSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            bid_price_1=4500.00,
            bid_size_1=100,
            ask_price_1=4500.25,
            ask_size_1=150,
        )

        # Invalid: bid >= ask
        with pytest.raises(ValueError):
            DepthSnapshot(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                bid_price_1=4500.50,
                bid_size_1=100,
                ask_price_1=4500.25,
                ask_size_1=150,
            )
