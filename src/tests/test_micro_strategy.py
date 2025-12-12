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
    assert buy_event["target"] == buy_event["price"] * (1 + state.micro.take_profit_pct + state.micro_loss_recovery_pct)

    # A profitable exit should reset the recovery boost
    position = state.micro_positions[0]
    events.clear()
    state._check_micro_take_profit(ts, h=position["target"], l=position["target"], log_events=events)
    assert state.micro_loss_recovery_pct == 0.0