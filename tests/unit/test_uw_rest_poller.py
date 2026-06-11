from unittest.mock import MagicMock

from src.services.uw_rest_poller import (
    MARKET_AGG_HISTORY_KEY,
    MARKET_AGG_LATEST_KEY,
    MARKET_AGG_STREAM_CHANNEL,
    UWRestPoller,
    UWRestPollerSettings,
)


def test_market_tide_converts_to_market_agg_event():
    enriched = {
        "bars": [
            {
                "timestamp": "2026-06-10T16:10:00-04:00",
                "date": "2026-06-10",
                "net_call_premium": "-345976915.5000",
                "net_put_premium": "201892454.0000",
                "net_volume": -1396060,
                "cum_net_call_premium": -10076153552.5,
                "cum_net_put_premium": 7686615642.0,
                "bar_bias": "put",
            }
        ],
        "overall_bias": "neutral",
        "bar_count": 81,
        "_fetched_at": "2026-06-11T06:59:32.784506+00:00",
    }

    payload = UWRestPoller._market_agg_from_tide(enriched)

    assert payload["received_at"] == "2026-06-11T06:59:32.784506+00:00"
    assert payload["message_type"] == "market_agg_api"
    assert payload["topic"] == "market-tide"
    assert payload["data"]["timestamp"] == "2026-06-10T16:10:00-04:00"
    assert payload["data"]["net_call_premium"] == -345976915.5
    assert payload["data"]["net_put_premium"] == 201892454.0
    assert payload["data"]["put_call_ratio"] == 0.5835


def test_market_tide_publishes_market_agg_redis_surface():
    redis_client = MagicMock()
    redis_client.client = MagicMock()
    pipe = redis_client.client.pipeline.return_value
    poller = UWRestPoller(
        UWRestPollerSettings(api_key="fake"),
        redis_client=redis_client,
    )
    enriched = {
        "bars": [
            {
                "timestamp": "2026-06-10T16:10:00-04:00",
                "date": "2026-06-10",
                "net_call_premium": "10",
                "net_put_premium": "25",
                "net_volume": -3,
            }
        ],
        "_fetched_at": "2026-06-11T06:59:32.784506+00:00",
    }

    poller._publish_market_agg(enriched)

    pipe.setex.assert_called_once()
    assert pipe.setex.call_args.args[0] == MARKET_AGG_LATEST_KEY
    pipe.lpush.assert_called_once()
    assert pipe.lpush.call_args.args[0] == MARKET_AGG_HISTORY_KEY
    redis_client.client.publish.assert_called_once()
    assert redis_client.client.publish.call_args.args[0] == MARKET_AGG_STREAM_CHANNEL
