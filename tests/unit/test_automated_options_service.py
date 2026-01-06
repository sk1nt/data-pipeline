"""Unit tests for AutomatedOptionsService.

Tests the create_entry_order method and integration with AlertParser, OptionsFillService.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from services.automated_options_service import AutomatedOptionsService
from services.options_fill_service import InsufficientBuyingPowerError
from services.tastytrade_client import TastytradeAuthError


@pytest.fixture
def mock_tastytrade_client():
    """Mock TastyTrade client."""
    client = Mock()
    client.ensure_authorized = AsyncMock()
    client.is_authorized = Mock(return_value=True)
    return client


@pytest.fixture
def service(mock_tastytrade_client):
    """AutomatedOptionsService with mocked dependencies."""
    svc = AutomatedOptionsService(tastytrade_client=mock_tastytrade_client)
    svc.fill_service.fill_options_order = AsyncMock()
    return svc


@pytest.mark.asyncio
async def test_create_entry_order_success_dry_run(service):
    """Test successful order creation in dry-run mode."""
    # Mock fill_options_order to return success (format: {order_id, entry_price})
    service.fill_service.fill_options_order.return_value = {
        "order_id": "dry-run-12345",
        "entry_price": "10.50",  # Note: fill service returns str
    }
    
    result = await service.create_entry_order(
        symbol="MNQ",
        strike=Decimal("17000"),
        option_type="CALL",
        expiry="2025-12-20",
        quantity=5,
        action="BTO",  # Use 'BTO' not 'BUY_TO_OPEN' - fill service expects short forms
        user_id="test_user",
        channel_id="test_channel",
        dry_run=True,
    )
    
    assert result is not None
    assert result["status"] == "success"
    assert result["order_id"] == "dry-run-12345"
    assert result["quantity"] == 5
    assert result["entry_price"] == Decimal("10.50")
    assert result["dry_run"] is True
    
    # Verify fill service was called correctly
    service.fill_service.fill_options_order.assert_called_once()
    call_args = service.fill_service.fill_options_order.call_args
    assert call_args[1]["symbol"] == "MNQ"
    assert call_args[1]["quantity"] == 5
    assert call_args[1]["action"] == "BTO"


@pytest.mark.asyncio
async def test_create_entry_order_auth_failure(service):
    """Test order creation when TastyTrade auth fails."""
    # Mock authorization to fail
    service.tastytrade_client.ensure_authorized.side_effect = TastytradeAuthError("Auth failed")
    
    result = await service.create_entry_order(
        symbol="SPX",
        strike=Decimal("5000"),
        option_type="PUT",
        expiry="2025-11-20",
        quantity=1,
        action="BTO",
        user_id="test_user",
        channel_id="test_channel",
        dry_run=False,
    )
    
    assert result is not None
    assert result["status"] == "error"
    assert result["reason"] == "authentication_failed"
    assert "auth" in result["error"].lower()


@pytest.mark.asyncio
async def test_create_entry_order_insufficient_buying_power(service):
    """Test order creation when buying power is insufficient."""
    # Mock fill service to raise buying power error
    service.fill_service.fill_options_order.side_effect = InsufficientBuyingPowerError(
        "Not enough buying power"
    )
    
    # Mock notify_operator as an async function
    with patch("services.automated_options_service.notify_operator", new_callable=AsyncMock):
        result = await service.create_entry_order(
            symbol="SPY",
            strike=Decimal("450"),
            option_type="CALL",
            expiry="2025-12-15",
            quantity=100,  # Large quantity
            action="BTO",
            user_id="test_user",
            channel_id="test_channel",
            dry_run=False,
        )
    
    assert result is not None
    assert result["status"] == "error"
    assert result["reason"] == "insufficient_buying_power"
    assert "buying power" in result["error"].lower()


@pytest.mark.asyncio
async def test_create_entry_order_with_initial_price(service):
    """Test order creation with explicitly provided initial price."""
    initial_price = Decimal("15.75")
    
    service.fill_service.fill_options_order.return_value = {
        "order_id": "order-67890",
        "entry_price": "15.75",
    }
    
    result = await service.create_entry_order(
        symbol="QQQ",
        strike=Decimal("400"),
        option_type="PUT",
        expiry="2025-11-30",
        quantity=2,
        action="BTO",
        user_id="test_user",
        channel_id="test_channel",
        initial_price=initial_price,
        dry_run=False,
    )
    
    assert result is not None
    assert result["status"] == "success"
    assert result["entry_price"] == Decimal("15.75")
    
    # Verify initial price was passed to fill service
    call_args = service.fill_service.fill_options_order.call_args
    assert call_args[1]["initial_price"] == initial_price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
