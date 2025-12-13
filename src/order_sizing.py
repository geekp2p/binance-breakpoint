"""Utility helpers for exchange-aware order sizing."""

from __future__ import annotations

import math
from typing import Optional, Tuple


def normalize_quantity(
    quantity: float,
    *,
    min_qty: float = 0.0,
    step_size: Optional[float] = None,
    price: Optional[float] = None,
    min_notional: Optional[float] = None,
    allow_round_up: bool = False,
) -> Tuple[float, str]:
    """Normalize a raw quantity against exchange filters.

    Returns the adjusted quantity (floored to the step size) and a reason string.
    If the quantity cannot satisfy ``min_qty``/``min_notional`` constraints even
    after rounding, ``0.0`` is returned so the caller can clear local state
    without submitting an order.
    """

    qty = max(float(quantity), 0.0)
    if qty <= 0:
        return 0.0, "NON_POSITIVE"

    step = float(step_size) if step_size else None
    if step and step > 0:
        qty_steps = math.floor(qty / step)
        qty = qty_steps * step
    reason = "OK"

    if min_qty > 0 and qty < min_qty:
        if allow_round_up and step and step > 0:
            min_steps = math.ceil(min_qty / step)
            qty = min_steps * step
            reason = "ROUNDED_UP_MIN_QTY"
        else:
            return 0.0, "BELOW_MIN_QTY"

    if price and min_notional:
        notional = qty * price
        if notional < min_notional:
            if allow_round_up and step and step > 0:
                target_qty = min_notional / price
                steps = math.ceil(target_qty / step)
                qty = max(qty, steps * step)
                if qty * price < min_notional:
                    return 0.0, "BELOW_MIN_NOTIONAL"
                reason = "ROUNDED_UP_MIN_NOTIONAL"
            else:
                return 0.0, "BELOW_MIN_NOTIONAL"

    return float(qty), reason


def clear_position_state(state) -> None:
    """Zero out tracked position and micro state after an unsendable remainder."""

    state.Q = 0.0
    state.C = 0.0
    if hasattr(state, "micro_positions"):
        state.micro_positions = []
    state.round_active = False