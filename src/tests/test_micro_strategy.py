from typing import Any, Dict, List

import pandas as pd

from src.strategy import (
    StrategyState,
    BuyLadderConf,
    AdaptiveLadderConf,
    AnchorDriftConf,
    ProfitTrailConf,
    TimeMartingaleConf,
    TimeCapsConf,
    ScalpModeConf,
    MicroOscillationConf,
    BuyTheDipConf,
    SellAtHeightConf,
)


def make_state():
    return StrategyState(
        fees_buy=0.001,
        fees_sell=0.001,
        b_alloc=100.0,
        buy=BuyLadderConf(d_buy=0.02, m_buy=1.5, n_steps=3),
        adaptive=AdaptiveLadderConf(),
        anchor=AnchorDriftConf(),
        trail=ProfitTrailConf(
            p_min=0.01,
            s1=0.01,
            m_step=1.2,
            tau=0.2,
            p_lock_base=0.01,
            p_lock_max=0.05,
            tau_min=0.1,
        ),
        tmart=TimeMartingaleConf(W1_minutes=5, m_time=2, delta_lock=0.01, beta_tau=0.9),
        tcaps=TimeCapsConf(T_idle_max_minutes=60, p_idle=0.01, T_total_cap_minutes=120, p_exit_min=0.005),
        scalp=ScalpModeConf(),
        micro=MicroOscillationConf(
            enabled=True,
            take_profit_pct=0.01,
            stop_break_pct=0.005,
            order_pct_allocation=0.5,
            window=10,
        ),
        btd=BuyTheDipConf(),
        sah=SellAtHeightConf(),
        snapshot_every_bars=1,
        use_maker=False,
    )


def seed_micro_ready(state: StrategyState):
    # Build a tight band with multiple swings
    prices = [100.0, 100.2, 99.9, 100.25, 99.85, 100.15, 99.8]
    for p in prices:
        state._update_micro_window(p)
    state.micro_swings = max(state.micro_swings, state.micro.min_swings)
    state.bar_index = 10


def test_micro_buy_sell_and_reentry_guard():
    state = make_state()
    state.rebuild_ladder(100.0)
    seed_micro_ready(state)

    events = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=100.3, l=99.8, log_events=events)

    assert state.micro_positions, "expected a micro position to open"
    buy_event = next(e for e in events if e["event"] == "MICRO_BUY")
    entry_price = buy_event["price"]

    # Hit take-profit
    events.clear()
    state._check_micro_take_profit(ts, h=state.micro_positions[0]["target"], l=state.micro_positions[0]["target"], log_events=events)
    assert not state.micro_positions, "micro position should be closed on TP"
    exit_event = next(e for e in events if e["event"] == "MICRO_TP")
    assert exit_event["price"] == state.micro_last_exit_price
    assert state.micro_cooldown_until_bar > state.bar_index

    # Re-entry too soon near same price should be skipped
    events.clear()
    state.bar_index = state.micro_cooldown_until_bar + 1
    base = state.micro_last_exit_price
    state.micro_prices = [base * (1 + d) for d in (0.001, -0.0005, 0.0015, -0.0002, 0.0008)]
    state.micro_swings = state.micro.min_swings
    state.micro_last_direction = None
    state._maybe_micro_buy(ts, h=base * 1.001, l=base * 0.999, log_events=events)
    assert any(e.get("reason") == "NEED_PULLBACK" for e in events)
    assert not state.micro_positions

    # After a deeper pullback we can re-enter
    events.clear()
    state.bar_index = state.micro_cooldown_until_bar + 2
    pulled_back_base = entry_price * (1 - state.micro.reentry_drop_pct - 0.0015)
    state.micro_prices = [
        pulled_back_base * (1 + delta)
        for delta in (0.0002, -0.0004, 0.0007, -0.0001, 0.0005)
    ]
    state.micro_swings = state.micro.min_swings
    state.micro_last_direction = None
    state.ladder_next_idx = 0
    state._maybe_micro_buy(ts, h=pulled_back_base * 1.001, l=pulled_back_base * 0.999, log_events=events)
    assert any(e["event"] == "MICRO_BUY" for e in events)
    assert state.micro_positions


def test_micro_stop_clamped_and_no_oversell():
    state = make_state()
    state.rebuild_ladder(100.0)
    seed_micro_ready(state)

    events = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=100.3, l=99.7, log_events=events)

    position = state.micro_positions[0]
    breakeven = position["cost"] / position["qty"] / (1 - state.fees_sell)
    assert position["stop"] >= breakeven

    # Simulate some coins already sold elsewhere; ensure exit does not oversell
    state.Q -= position["qty"] * 0.5
    state.C -= position["cost"] * 0.5

    events.clear()
    state._check_micro_take_profit(
        ts,
        h=position["stop"],
        l=position["stop"],
        log_events=events,
    )

    exit_event = next(e for e in events if e["event"].startswith("MICRO_"))
    assert exit_event["qty"] <= position["qty"]
    assert exit_event["pnl"] >= -1e-9
    assert state.Q >= 0
    # Remaining qty should stay on the books if we sold only part of it
    assert any(pos["qty"] > 0 for pos in state.micro_positions) or not state.micro_positions


def test_micro_exit_price_never_below_breakeven():
    state = make_state()
    state.rebuild_ladder(100.0)
    seed_micro_ready(state)

    events = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=100.3, l=99.7, log_events=events)

    position = state.micro_positions[0]
    breakeven = position["cost"] / position["qty"] / (1 - state.fees_sell)

    # Force an unrealistically low stop to ensure it gets clamped
    state.micro_positions[0]["stop"] = position["entry"] * 0.95

    events.clear()
    low_stop = state.micro_positions[0]["stop"]
    state._check_micro_take_profit(
        ts,
        h=breakeven,
        l=low_stop,
        log_events=events,
    )

    exit_event = next(e for e in events if e["event"].startswith("MICRO_"))
    assert exit_event["price"] >= breakeven
    assert exit_event["pnl"] >= -1e-9


def test_micro_stop_not_triggered_if_above_range():
    state = make_state()
    state.rebuild_ladder(100.0)
    seed_micro_ready(state)

    events = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=100.3, l=99.7, log_events=events)

    position = state.micro_positions[0]
    assert position["stop"] > position["entry"]

    events.clear()
    # Stop above the bar range should not trigger an exit
    state._check_micro_take_profit(
        ts,
        h=position["entry"],
        l=position["entry"],
        log_events=events,
    )

    assert state.micro_positions, "position should remain open when stop is not in range"
    assert not any(e["event"].startswith("MICRO_") for e in events)


def test_micro_loss_recovery_boosts_next_take_profit():
    state = make_state()
    state.micro.loss_recovery_markup_pct = 0.002
    state.micro.loss_recovery_max_pct = 0.02
    state.rebuild_ladder(100.0)
    seed_micro_ready(state)

    events = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=100.3, l=99.7, log_events=events)
    position = state.micro_positions[0]

    # Simulate high taker fees so a stop exit realises a loss
    state.fees_sell = 0.01

    events.clear()
    # Force a stop exit so realised PnL is negative
    state._check_micro_take_profit(ts, h=position["stop"], l=position["stop"], log_events=events)
    assert state.micro_loss_recovery_pct == state.micro.loss_recovery_markup_pct

    # Allow cooldown to expire and rebuild a new ready window below last exit
    state.bar_index = state.micro_cooldown_until_bar + 2
    base = state.micro_last_exit_price * (1 - state.micro.reentry_drop_pct - 0.002)
    state.micro_prices = [base * (1 + d) for d in (0.0004, -0.0006, 0.0009, -0.0003, 0.0007)]
    state.micro_swings = state.micro.min_swings
    state.micro_last_direction = None
    state.ladder_next_idx = 0

    events.clear()
    state._maybe_micro_buy(ts, h=base * 1.001, l=base * 0.999, log_events=events)
    buy_event = next(e for e in events if e["event"] == "MICRO_BUY")
    assert buy_event["target"] >= buy_event["price"] * (1 + state.micro.take_profit_pct + state.micro_loss_recovery_pct)

    # A profitable exit should reset the recovery boost
    position = state.micro_positions[0]
    events.clear()
    state._check_micro_take_profit(ts, h=position["target"], l=position["target"], log_events=events)
    assert state.micro_loss_recovery_pct == 0.0


def test_micro_take_profit_respects_fees_and_margin():
    state = make_state()
    state.micro.take_profit_pct = 0.0005  # intentionally tiny
    state.micro.min_profit_pct = 0.0015
    state.rebuild_ladder(100.0)
    seed_micro_ready(state)

    events = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=100.2, l=99.7, log_events=events)

    position = state.micro_positions[0]
    break_even = position["cost"] / position["qty"] / (1 - state.fees_sell)
    effective_min_profit = max(state.micro.min_profit_pct, state.fees_buy + state.fees_sell)
    assert position["target"] >= break_even * (1 + effective_min_profit)


def test_micro_prunes_tiny_remainders():
    state = make_state()
    state.micro.min_exit_notional = 1e6  # force pruning for the test
    state.rebuild_ladder(100.0)
    seed_micro_ready(state)

    events = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=100.2, l=99.7, log_events=events)

    position = state.micro_positions[0]
    # Simulate that inventory was cleared elsewhere
    state.Q = 0.0
    state.C = 0.0

    events.clear()
    state._check_micro_take_profit(ts, h=position["stop"], l=position["stop"], log_events=events)

    assert not state.micro_positions, "stale micro position should be pruned"
    assert any(e["event"] == "MICRO_EXIT_PRUNED" for e in events)


def test_micro_atr_scales_thresholds():
    state = make_state()
    state.micro.take_profit_pct = 0.001
    state.micro.stop_break_pct = 0.001
    state.micro.reentry_drop_pct = 0.001
    state.micro.max_band_pct = 0.05
    state.micro.loss_recovery_max_pct = 0.05
    state.micro.atr_take_profit_mult = 2.5
    state.micro.atr_stop_mult = 2.5
    state.micro.atr_reentry_mult = 1.5
    state.rebuild_ladder(100.0)

    volatile_prices = [100 * (1 + delta) for delta in (0.0, 0.012, -0.01, 0.015, -0.012, 0.014, -0.011)]
    for price in volatile_prices:
        state._update_micro_window(price)
    state.micro_swings = state.micro.min_swings
    state.bar_index = 10
    atr_pct = state._micro_atr_pct()
    assert atr_pct is not None
    snapshot = state._micro_snapshot()
    volatility_pct = max(snapshot["band_pct"], atr_pct)
    scaled_reentry = max(
        state.micro.reentry_drop_pct,
        atr_pct * state.micro.atr_reentry_mult,
        volatility_pct * state.micro.volatility_reentry_mult,
    )
    entry_guess = snapshot["low"] + (snapshot["high"] - snapshot["low"]) * state.micro.entry_band_pct
    state.micro_last_exit_price = entry_guess * 0.5
    events = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=101.5, l=98.5, log_events=events)

    skip = [e for e in events if e.get("reason") == "NEED_PULLBACK"]
    assert skip, "expected pullback guard when re-entering too soon"
    assert skip[0]["reentry_drop_pct"] >= scaled_reentry

    # Allow re-entry after a deeper pullback and check thresholds scale
    events.clear()
    state.bar_index = state.micro_cooldown_until_bar + 1
    state.micro_last_exit_price = state.micro_last_exit_price * (1 + state.micro.reentry_drop_pct)
    deep_pullback_low = state.micro_last_exit_price * (1 - scaled_reentry - 0.005)
    state.micro_prices = [deep_pullback_low * (1 + delta) for delta in (0.002, -0.003, 0.004, -0.002, 0.003)]
    state.micro_swings = state.micro.min_swings
    state.micro_last_direction = None
    state.bar_index += 1

    current_atr_pct = state._micro_atr_pct()
    assert current_atr_pct is not None
    state._maybe_micro_buy(ts, h=deep_pullback_low * 1.01, l=deep_pullback_low * 0.995, log_events=events)
    buy_event = next(e for e in events if e.get("event") == "MICRO_BUY")
    current_snapshot = state._micro_snapshot()
    current_volatility_pct = max(current_snapshot["band_pct"], current_atr_pct)
    expected_stop_pct = max(
        state.micro.stop_break_pct,
        current_volatility_pct * state.micro.volatility_stop_mult,
        current_atr_pct * state.micro.atr_stop_mult,
    )
    expected_tp_pct = max(
        state.micro.take_profit_pct,
        current_volatility_pct * state.micro.volatility_take_profit_mult,
        current_atr_pct * state.micro.atr_take_profit_mult,
    )
    assert buy_event["target"] >= buy_event["price"] * (1 + expected_tp_pct)
    assert buy_event["stop"] >= buy_event["price"] * (1 - expected_stop_pct)


def test_micro_prunes_when_qty_guard_hits():
    state = make_state()
    state.micro.min_exit_qty = 0.001
    state.rebuild_ladder(100.0)
    seed_micro_ready(state)

    events: List[Dict[str, Any]] = []
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    state._maybe_micro_buy(ts, h=100.2, l=99.7, log_events=events)
    position = state.micro_positions[0]

    # Exhaust almost all inventory elsewhere so the exit has nothing meaningful to sell
    state.Q = state.micro.min_exit_qty * 0.5
    state.C = position["cost"] * (state.Q / position["qty"])

    events.clear()
    state._check_micro_take_profit(ts, h=position["stop"], l=position["stop"], log_events=events)

    assert not state.micro_positions
    pruned = next(e for e in events if e["event"] == "MICRO_EXIT_PRUNED")
    assert pruned["reason"] == "TINY_QTY"

    # A second check should no longer emit new micro stop/exit events
    events.clear()
    state._check_micro_take_profit(ts, h=position["stop"], l=position["stop"], log_events=events)
    assert not events