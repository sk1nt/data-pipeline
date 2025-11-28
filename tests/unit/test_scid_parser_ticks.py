"""
Unit tests for SCID parser tick functionality.
"""

from datetime import datetime

from src.lib.scid_parser import is_tick_record


class TestScidTickParsing:
    """Test SCID record parsing for tick data."""

    def test_parse_tick_record(self):
        """Test parsing a basic tick record."""
        # Mock binary data for a tick record
        # This is a simplified test - in practice would need real SCID data
        mock_record = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "counter": 0,
            "open": 0.0,
            "high": 4500.25,  # Ask price
            "low": 4500.00,  # Bid price
            "close": 4500.12,  # Trade price
            "num_trades": 1,
            "total_volume": 5,
            "bid_volume": 3,
            "ask_volume": 2,
        }

        # Test tick identification
        assert is_tick_record(mock_record)

    def test_parse_non_tick_record(self):
        """Test parsing a non-tick record (OHLC bar)."""
        mock_record = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "counter": 0,
            "open": 4500.00,
            "high": 4500.50,
            "low": 4499.75,
            "close": 4500.25,
            "num_trades": 100,
            "total_volume": 500,
            "bid_volume": 250,
            "ask_volume": 250,
        }

        # Should not be identified as individual tick
        assert not is_tick_record(mock_record)

    def test_bundled_trade_detection(self):
        """Test detection of bundled trades."""
        from src.lib.scid_parser import is_bundled_trade

        # First in bundle
        record1 = {"open": -1.99900095e37}
        assert is_bundled_trade(record1)

        # Last in bundle
        record2 = {"open": -1.99900197e37}
        assert is_bundled_trade(record2)

        # Regular record
        record3 = {"open": 0.0}
        assert not is_bundled_trade(record3)
