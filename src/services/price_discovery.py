from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Tuple


def discovery_attempts(
    start_price: Decimal,
    market_price: Decimal | None,
    tick_size: Decimal,
    max_increments: int = 3,
    wait_seconds: int = 20,
    convert_to_market_if_remaining_ticks_leq: int = 1,
) -> Iterable[Tuple[Decimal, bool]]:
    """Yield price attempts.

    Yields tuples (price, convert_to_market_flag).
    - Start at start_price.
    - If market_price is provided, comparison with start_price to determine side.
    - After each wait, increment toward market by 1 tick.
    - If remaining difference to market is <= convert_to_market_if_remaining_ticks_leq * tick, yield market conversion flag True.
    """
    cur = Decimal(start_price)
    yield cur, False
    if market_price is None:
        # No market guidance — just yield increments up to max_increments
        for i in range(max_increments):
            cur = cur + tick_size
            yield cur, False
        return

    # Determine direction toward market
    # If market > cur -> we need to increment up, else down
    diff = Decimal(market_price) - cur
    if diff == 0:
        return
    step = tick_size if diff > 0 else -tick_size
    for i in range(max_increments):
        next_price = cur + step
        remaining_ticks = abs((Decimal(market_price) - next_price) / tick_size)
        if remaining_ticks <= convert_to_market_if_remaining_ticks_leq:
            # convert to market instead of making further limit attempts
            yield next_price, True
            return
        yield next_price, False
        cur = next_price
    return
