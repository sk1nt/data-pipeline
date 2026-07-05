"""Shared builder for Tastytrade atomic bracket (OTOCO/OTO) complex orders.

This module is intentionally dependency-free beyond the tastytrade SDK so it
can be imported from *both* runtimes without triggering heavy module-level
side effects:

* the Discord bot -> ``from services.complex_order_builder import ...``
  (``src/`` is on ``sys.path``)
* the REST API     -> ``from .complex_order_builder import ...``
  (relative import inside ``src.services``)

A *bracket* complex order submits the entry and its exit(s) atomically: the
exit legs are only activated once the entry (trigger) fills.  This removes the
cancel/poll/restore race conditions of placing the legs as independent orders.

* ``sl_price`` provided -> OTOCO: entry triggers (TP limit OR SL stop).
* ``sl_price`` is None   -> OTO:   entry triggers a single TP limit.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from tastytrade.order import (
    NewComplexOrder,
    NewOrder,
    OrderAction,
    OrderTimeInForce,
    OrderType,
)


def signed_limit_price(price: float, side: OrderAction) -> Decimal:
    """Encode a Tastytrade limit price with the correct debit/credit sign.

    Tastytrade encodes price *effect* via the sign of the limit price:
    negative = debit (we pay, e.g. ``BUY_TO_CLOSE``), positive = credit
    (we receive, e.g. ``SELL_TO_CLOSE``).
    """
    if side == OrderAction.BUY_TO_CLOSE:
        return Decimal(str(-abs(price)))
    return Decimal(str(abs(price)))


def build_bracket_complex_order(
    *,
    entry_leg,
    closing_leg,
    tp_price: float,
    tp_action: OrderAction,
    entry_price: Optional[float] = None,
    sl_price: Optional[float] = None,
) -> NewComplexOrder:
    """Build an atomic OTOCO or OTO complex order ready for ``place_complex_order``.

    Parameters
    ----------
    entry_leg, closing_leg:
        Pre-built :class:`tastytrade.order.Leg` objects.  ``entry_leg`` is the
        market trigger; ``closing_leg`` is reused for every exit (TP/SL), mirroring
        the previous inline OTOCO behaviour.
    tp_price:
        Take-profit limit price (absolute).  Sign is derived from ``tp_action``.
    tp_action:
        Close-side action for the TP leg (``SELL_TO_CLOSE`` / ``BUY_TO_CLOSE``).
    entry_price:
        Current market price for the entry leg.  Used to build a marketable
        limit trigger (required by TastyTrade — plain Market orders are rejected
        with ``invalid_oco_order_type_combo``).  When ``None`` falls back to
        ``tp_price`` as a rough proxy.
    sl_price:
        Optional stop-loss trigger price.  When ``None`` an OTO with a single TP
        exit is produced; otherwise an OTOCO with TP *and* SL exits.

    Returns
    -------
    NewComplexOrder
        ``trigger_order`` = DAY marketable-limit entry; ``orders`` = [TP limit] (OTO) or
        [TP limit, SL stop] (OTOCO), all GTC.
    """
    # TastyTrade rejects Market triggers in complex orders with
    # invalid_oco_order_type_combo — use Marketable Limit instead.
    _entry_price = entry_price if entry_price is not None else tp_price
    entry_signed = signed_limit_price(_entry_price, entry_leg.action)
    trigger_order = NewOrder(
        time_in_force=OrderTimeInForce.DAY,
        order_type=OrderType.MARKETABLE_LIMIT,
        price=entry_signed,
        legs=[entry_leg],
    )

    tp_order = NewOrder(
        time_in_force=OrderTimeInForce.GTC,
        order_type=OrderType.LIMIT,
        price=signed_limit_price(tp_price, tp_action),
        legs=[closing_leg],
    )

    if sl_price is None:
        return NewComplexOrder(trigger_order=trigger_order, orders=[tp_order])

    sl_order = NewOrder(
        time_in_force=OrderTimeInForce.GTC,
        order_type=OrderType.STOP,
        stop_trigger=Decimal(str(sl_price)),
        legs=[closing_leg],
    )
    return NewComplexOrder(trigger_order=trigger_order, orders=[tp_order, sl_order])
