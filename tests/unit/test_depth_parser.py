"""
Unit tests for depth data parsing.
"""

import pytest
from datetime import datetime
from src.models.depth_snapshot import DepthSnapshot


class TestDepthParsing:
    """Test depth data parsing and model creation."""

    def test_depth_snapshot_creation(self):
        """Test creating depth snapshot from basic data."""
        snapshot = DepthSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            bid_price_1=4500.00,
            bid_size_1=100,
            ask_price_1=4500.25,
            ask_size_1=150,
        )

        assert snapshot.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert snapshot.bid_price_1 == 4500.00
        assert snapshot.bid_size_1 == 100
        assert snapshot.ask_price_1 == 4500.25
        assert snapshot.ask_size_1 == 150

    def test_depth_validation(self):
        """Test depth snapshot validation."""
        # Valid snapshot
        DepthSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            bid_price_1=4500.00,
            bid_size_1=100,
            ask_price_1=4500.25,
            ask_size_1=150,
        )

        # Invalid: bid >= ask
        with pytest.raises(ValueError, match="Bid price must be less than ask price"):
            DepthSnapshot(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                bid_price_1=4500.50,
                bid_size_1=100,
                ask_price_1=4500.25,
                ask_size_1=150,
            )

        # Invalid: negative size
        with pytest.raises(ValueError, match="Sizes must be non-negative"):
            DepthSnapshot(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                bid_price_1=4500.00,
                bid_size_1=-100,
                ask_price_1=4500.25,
                ask_size_1=150,
            )

    def test_depth_with_additional_levels(self):
        """Test depth snapshot with additional price levels."""
        snapshot = DepthSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            bid_price_1=4500.00,
            bid_size_1=100,
            ask_price_1=4500.25,
            ask_size_1=150,
            bid_price_2=4499.75,
            bid_size_2=200,
            ask_price_2=4500.50,
            ask_size_2=175,
        )

        assert snapshot.bid_price_2 == 4499.75
        assert snapshot.ask_size_2 == 175
