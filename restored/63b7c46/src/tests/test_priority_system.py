"""
Basic tests for the GEX priority system.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from models.enums import PriorityLevel, JobStatus, GEXDataType
from models.priority_request import PriorityRequest
from models.data_source import DataSource


class TestPriorityRequest:
    """Test PriorityRequest model."""

    def test_create_priority_request(self):
        """Test creating a basic priority request."""
        request = PriorityRequest(
            data_type=GEXDataType.TICK_DATA,
            priority_level=PriorityLevel.HIGH,
            source_id=uuid4()
        )

        assert request.data_type == GEXDataType.TICK_DATA
        assert request.priority_level == PriorityLevel.HIGH
        assert request.status == JobStatus.PENDING
        assert request.priority_score == 0.0  # Not calculated yet

    def test_priority_request_with_deadline(self):
        """Test priority request with deadline."""
        deadline = datetime.utcnow() + timedelta(hours=1)
        request = PriorityRequest(
            data_type=GEXDataType.DEPTH_DATA,
            priority_level=PriorityLevel.URGENT,
            source_id=uuid4(),
            deadline=deadline
        )

        assert request.deadline == deadline
        assert request.is_overdue() is False

    def test_overdue_request(self):
        """Test overdue request detection."""
        past_deadline = datetime.utcnow() - timedelta(hours=1)
        request = PriorityRequest(
            data_type=GEXDataType.TICK_DATA,
            priority_level=PriorityLevel.NORMAL,
            source_id=uuid4(),
            deadline=past_deadline
        )

        assert request.is_overdue() is True


class TestDataSource:
    """Test DataSource model."""

    def test_create_data_source(self):
        """Test creating a basic data source."""
        source = DataSource(
            base_url="https://api.example.com",
            name="Example API"
        )

        assert source.base_url == "https://api.example.com"
        assert source.name == "Example API"
        assert source.reliability_score == 1.0
        assert source.total_requests == 0
        assert source.successful_requests == 0
        assert source.is_active is True

    def test_data_source_url_validation(self):
        """Test URL validation for data source."""
        # Valid HTTPS URL
        source = DataSource(
            base_url="https://api.example.com",
            name="Valid HTTPS"
        )
        assert source.base_url == "https://api.example.com"

        # Valid HTTP URL
        source = DataSource(
            base_url="http://api.example.com",
            name="Valid HTTP"
        )
        assert source.base_url == "http://api.example.com"

        # Invalid URL should raise ValueError
        with pytest.raises(ValueError, match="base_url must be a valid HTTP/HTTPS URL"):
            DataSource(
                base_url="ftp://api.example.com",
                name="Invalid Protocol"
            )

    def test_data_source_metrics(self):
        """Test data source metrics calculation."""
        source = DataSource(
            base_url="https://api.example.com",
            name="Test Source"
        )

        # Initial state
        assert source.success_rate == 1.0  # No requests yet
        assert source.reliability_score == 1.0

        # Record successful request
        source.record_request(success=True, response_time=timedelta(seconds=1))
        assert source.total_requests == 1
        assert source.successful_requests == 1
        assert source.success_rate == 1.0
        assert source.reliability_score == 1.0

        # Record failed request
        source.record_request(success=False, response_time=timedelta(seconds=2))
        assert source.total_requests == 2
        assert source.successful_requests == 1
        assert source.success_rate == 0.5
        assert source.reliability_score < 1.0  # Should decrease

    def test_data_source_reliability(self):
        """Test reliability threshold checking."""
        source = DataSource(
            base_url="https://api.example.com",
            name="Test Source"
        )

        # Initially reliable
        assert source.is_reliable() is True
        assert source.is_reliable(0.5) is True

        # Add some failures
        for _ in range(5):
            source.record_request(success=False)

        # Should become unreliable
        assert source.is_reliable() is False  # Default threshold 0.8
        assert source.is_reliable(0.5) is True  # But pass lower threshold


class TestEnums:
    """Test enum definitions."""

    def test_priority_level_values(self):
        """Test priority level enum values."""
        assert PriorityLevel.LOW.value == "low"
        assert PriorityLevel.NORMAL.value == "normal"
        assert PriorityLevel.HIGH.value == "high"
        assert PriorityLevel.URGENT.value == "urgent"

    def test_job_status_values(self):
        """Test job status enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"

    def test_gex_data_type_values(self):
        """Test GEX data type enum values."""
        assert GEXDataType.TICK_DATA.value == "tick_data"
        assert GEXDataType.DEPTH_DATA.value == "depth_data"
        assert GEXDataType.GEX_DATA.value == "gex_data"