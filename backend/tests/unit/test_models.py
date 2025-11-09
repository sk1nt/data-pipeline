import pytest
from backend.src.models.tick_data import TickData
from pydantic import ValidationError
from decimal import Decimal
from datetime import datetime

def test_tick_data_model():
    """Test TickData model validation."""
    tick = TickData(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=Decimal("150.25"),
        volume=100,
        tick_type="trade",
        source="tastyttrade"
    )
    assert tick.symbol == "AAPL"
    assert tick.price == Decimal("150.25")

def test_tick_data_validation():
    """Test TickData validation rules."""
    # Valid tick
    TickData(
        symbol="AAPL",
        timestamp=datetime.now(),
        price=Decimal("100.00"),
        tick_type="trade",
        source="sierra_chart"
    )

    # Invalid: negative price
    with pytest.raises(ValidationError):
        TickData(
            symbol="AAPL",
            timestamp=datetime.now(),
            price=Decimal("-1.00"),
            tick_type="trade",
            source="sierra_chart"
        )