"""
Unit tests for data validation functions.
"""

from datetime import datetime

from src.services.data_validator import DataValidator


class TestDataValidator:
    """Test data validation functionality."""

    def test_timestamp_range_validation(self):
        """Test timestamp range validation."""
        validator = DataValidator()

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Valid timestamp
        assert validator.validate_timestamp(datetime(2024, 1, 15), start_date, end_date)

        # Invalid: too early
        assert not validator.validate_timestamp(
            datetime(2023, 12, 31), start_date, end_date
        )

        # Invalid: too late
        assert not validator.validate_timestamp(
            datetime(2024, 2, 1), start_date, end_date
        )

    def test_data_completeness_check(self):
        """Test data completeness validation."""
        validator = DataValidator()

        # Mock complete data
        complete_data = {"ticks": 1000, "expected_days": 30, "actual_days": 30}

        assert validator.check_completeness(complete_data)

        # Mock incomplete data
        incomplete_data = {"ticks": 100, "expected_days": 30, "actual_days": 15}

        assert not validator.check_completeness(incomplete_data)

    def test_duplicate_detection(self):
        """Test duplicate record detection."""
        validator = DataValidator()

        # Mock records with duplicates
        records = [
            {"timestamp": datetime(2024, 1, 1, 12, 0, 0), "price": 4500.00},
            {
                "timestamp": datetime(2024, 1, 1, 12, 0, 0),
                "price": 4500.00,
            },  # Duplicate
            {"timestamp": datetime(2024, 1, 1, 12, 0, 1), "price": 4500.05},
        ]

        duplicates = validator.find_duplicates(records, ["timestamp", "price"])
        assert len(duplicates) == 1
        assert duplicates[0]["count"] == 2

    def test_data_consistency_validation(self):
        """Test data consistency checks."""
        validator = DataValidator()

        # Valid data
        valid_data = {
            "tick_timestamps": [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 0, 1),
            ],
            "depth_timestamps": [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 12, 0, 1),
            ],
        }

        assert validator.validate_consistency(valid_data)

        # Invalid: timestamp mismatch
        invalid_data = {
            "tick_timestamps": [datetime(2024, 1, 1, 12, 0, 0)],
            "depth_timestamps": [datetime(2024, 1, 1, 14, 0, 0)],  # 2 hours later
        }

        assert not validator.validate_consistency(invalid_data)
