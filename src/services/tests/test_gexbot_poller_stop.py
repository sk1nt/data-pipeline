import asyncio

import pytest

from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings


class _NoopRedisClient:
    def __init__(self):
        self.client = self


@pytest.mark.asyncio
async def test_stop_cancels_running_task() -> None:
    poller = GEXBotPoller(
        GEXBotPollerSettings(api_key="apikey", symbols=["SPX"]),
        redis_client=_NoopRedisClient(),
        ts_client=None,
    )
    poller._task = asyncio.create_task(asyncio.sleep(3600))

    await poller.stop()

    assert poller._task is None
