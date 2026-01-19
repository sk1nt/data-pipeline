"""Integration tests for UW message handling."""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.models.uw_message import (
    MarketAggData,
    MarketAggMessage,
    OptionTradeData,
    OptionTradeMessage,
    parse_uw_websocket_message,
)
from src.services.uw_message_service import UWMessageService
from src.lib.redis_client import RedisClient


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    client = MagicMock(spec=RedisClient)
    client.client = MagicMock()
    return client


@pytest.fixture
def uw_service(mock_redis_client):
    """Create UW message service with mock Redis."""
    return UWMessageService(mock_redis_client)


def test_parse_market_agg_message():
    """Test parsing market aggregation message from websocket array."""
    raw_message = [
        None,
        None,
        "market_agg_socket",
        "market-state",
        {
            "data": {
                "date": "2025-01-15",
                "call_premium": "1234567.89",
                "put_premium": "9876543.21",
                "call_volume": 10000,
                "put_volume": 15000,
                "put_call_ratio": "1.5",
            }
        },
    ]

    message = parse_uw_websocket_message(raw_message)
    assert message.message_type == "market_agg_socket"
    assert message.topic == "market-state"
    assert message.data.date == "2025-01-15"
    assert message.data.call_premium == "1234567.89"
    assert message.data.put_premium == "9876543.21"


def test_parse_option_trade_spx_message():
    """Test parsing SPX index option trade message."""
    raw_message = [
        None,
        None,
        "option_trades_super_algo:SPX",
        "new_msg",
        {
            "data": {
                "executed_at": "2025-01-15T14:30:00Z",
                "id": "trade_12345",
                "is_agg": False,
                "option_chain_id": "SPX250117C05000000",
                "premium": "5000.0",
                "price": "50.0",
                "size": 100,
                "underlying_price": "4950.0",
                "tags": ["sweep", "whale"],
                "delta": "0.5",
                "gamma": "0.01",
                "ask_vol": 50,
                "bid_vol": 50,
                "mid_vol": 0,
                "exchange": "CBOE",
                "open_interest": 10000,
            }
        },
    ]

    message = parse_uw_websocket_message(raw_message)
    assert message.message_type == "option_trades_super_algo"
    assert message.topic == "option_trades_super_algo:SPX"
    assert message.topic_symbol == "SPX"
    assert message.data.option_chain_id == "SPX250117C05000000"
    assert message.data.is_index_option is True  # Has greeks
    assert message.data.price == "50.0"


def test_parse_option_trade_stock_message():
    """Test parsing stock option trade message."""
    raw_message = [
        None,
        None,
        "option_trades_super_algo",
        "new_msg",
        {
            "data": {
                "executed_at": "2025-01-15T14:30:00Z",
                "id": "trade_98765",
                "is_agg": False,
                "option_chain_id": "AAPL250221P00175000",
                "premium": "2500.0",
                "price": "2.50",
                "size": 50,
                "underlying_price": "177.50",
                "tags": ["unusual_volume"],
                "sector": "Technology",
                "industry_type": "Consumer Electronics",
                "ask_vol": 25,
                "bid_vol": 25,
                "mid_vol": 0,
                "exchange": "CBOE",
            }
        },
    ]

    message = parse_uw_websocket_message(raw_message)
    assert message.message_type == "option_trades_super_algo"
    assert message.topic == "option_trades_super_algo"
    assert message.topic_symbol == "AAPL"  # Now extracted from option_chain_id
    assert message.data.option_chain_id == "AAPL250221P00175000"
    assert message.data.ticker == "AAPL"  # Ticker extracted from OCC symbol
    assert message.data.is_stock_option is True  # Has sector
    assert message.data.price == "2.50"


def test_service_process_market_agg(uw_service, mock_redis_client):
    """Test UWMessageService processing market aggregation message."""
    # Setup mock pipeline
    mock_pipeline = MagicMock()
    mock_redis_client.client.pipeline.return_value = mock_pipeline
    
    raw_message = [
        None,
        None,
        "market_agg_socket",
        "market-state",
        {
            "data": {
                "date": "2025-01-15",
                "call_premium": "1000000.0",
                "put_premium": "2000000.0",
                "call_volume": 10000,
                "put_volume": 20000,
                "put_call_ratio": "2.0",
            }
        },
    ]

    result = uw_service.process_raw_message(raw_message)

    assert result["status"] == "success"
    assert result["message_type"] == "market_agg"
    assert result["date"] == "2025-01-15"
    assert result["discord_notification"] is False

    # Verify Redis operations
    mock_redis_client.client.pipeline.assert_called()
    mock_pipeline.setex.assert_called()
    mock_pipeline.lpush.assert_called()
    mock_pipeline.ltrim.assert_called()
    mock_pipeline.execute.assert_called()


def test_service_process_option_trade_spx(uw_service, mock_redis_client):
    """Test UWMessageService processing SPX option trade."""
    raw_message = [
        None,
        None,
        "option_trades_super_algo:SPX",
        "new_msg",
        {
            "data": {
                "executed_at": "2025-01-15T14:30:00Z",
                "id": "trade_12345",
                "is_agg": False,
                "option_chain_id": "SPX250117C05000000",
                "premium": "5000.0",
                "price": "50.0",
                "size": 100,
                "underlying_price": "4950.0",
                "tags": [],
                "delta": "0.5",
                "gamma": "0.01",
            }
        },
    ]

    result = uw_service.process_raw_message(raw_message)

    assert result["status"] == "success"
    assert result["message_type"] == "option_trade"
    assert result["symbol"] == "SPX"
    assert result["is_index"] is True
    assert result["discord_notification"] is True
    assert result["discord_channel_id"] == 1429940127899324487  # SPX channel

    # Verify Redis operations
    mock_redis_client.client.pipeline.assert_called()


def test_service_process_option_trade_stock(uw_service, mock_redis_client):
    """Test UWMessageService processing stock option trade."""
    raw_message = [
        None,
        None,
        "option_trades_super_algo",
        "new_msg",
        {
            "data": {
                "executed_at": "2025-01-15T14:30:00Z",
                "id": "trade_98765",
                "is_agg": False,
                "option_chain_id": "AAPL250221P00175000",
                "premium": "2500.0",
                "price": "2.50",
                "size": 50,
                "underlying_price": "177.50",
                "tags": [],
                "sector": "Technology",
            }
        },
    ]

    result = uw_service.process_raw_message(raw_message)

    assert result["status"] == "success"
    assert result["message_type"] == "option_trade"
    assert result["symbol"] == "AAPL"  # Now extracted from option_chain_id
    assert result["is_index"] is False
    assert result["discord_notification"] is True
    assert result["discord_channel_id"] == 1425136266676146236  # General options channel

    # Verify Redis operations
    mock_redis_client.client.pipeline.assert_called()


def test_discord_channel_routing_spx(uw_service):
    """Test Discord channel routing for SPX trades."""
    from src.models.uw_message import OptionTradeMessage, OptionTradeData

    message = OptionTradeMessage(
        message_type="option_trades_super_algo",
        topic="option_trades_super_algo:SPX",
        topic_symbol="SPX",
        data=OptionTradeData(
            executed_at="2025-01-15T14:30:00Z",
            id="trade_12345",
            is_agg=False,
            option_chain_id="SPX250117C05000000",
            premium="5000.0",
            price="50.0",
            size=100,
            underlying_price="4950.0",
            tags=[],
        ),
    )

    channel_id = uw_service._get_discord_channel_for_trade(message)
    assert channel_id == 1429940127899324487


def test_discord_channel_routing_stock(uw_service):
    """Test Discord channel routing for stock option trades."""
    from src.models.uw_message import OptionTradeMessage, OptionTradeData

    message = OptionTradeMessage(
        message_type="option_trades_super_algo",
        topic="option_trades_super_algo",
        topic_symbol=None,
        data=OptionTradeData(
            executed_at="2025-01-15T14:30:00Z",
            id="trade_98765",
            is_agg=False,
            option_chain_id="AAPL250221P00175000",
            premium="2500.0",
            price="2.50",
            size=50,
            underlying_price="177.50",
            tags=[],
        ),
    )

    channel_id = uw_service._get_discord_channel_for_trade(message)
    assert channel_id == 1425136266676146236


def test_invalid_message_format(uw_service):
    """Test handling of invalid message format."""
    # Too short
    result = uw_service.process_raw_message([None, None])
    # Returns None when parsing fails
    assert result is None

    # Unknown channel
    result = uw_service.process_raw_message(
        [None, None, "unknown_channel", "event", {"data": {}}]
    )
    # Returns None when parsing fails
    assert result is None
