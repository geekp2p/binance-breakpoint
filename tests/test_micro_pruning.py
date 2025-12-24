import math

import pandas as pd

from src.strategy import (
    AdaptiveLadderConf,
    AnchorDriftConf,
    BuyLadderConf,
    BuyTheDipConf,
    MicroOscillationConf,
    ProfitTrailConf,
    ScalpModeConf,
    SellAtHeightConf,
    StrategyState,
    TimeCapsConf,
    TimeMartingaleConf,
)


def _build_state_with_micro(min_exit_qty: float = 0.001) -> StrategyState:
    dummy_false = {"enabled": False}
    micro_cfg = MicroOscillationConf(
        enabled=True,
        window=6,
        max_band_pct=0.2,
        min_swings=1,
        min_swing_pct=0.0,
        entry_band_pct=0.15,
        take_profit_pct=0.01,
        stop_break_pct=0.005,
        min_exit_qty=min_exit_qty,
        min_exit_notional=0.0,
        cooldown_bars=0,
    )

    return StrategyState(
        fees_buy=0.001,
        fees_sell=0.001,
        b_alloc=100.0,
        buy=BuyLadderConf(d_buy=0.02, m_buy=1.0, n_steps=3),
        adaptive=AdaptiveLadderConf(**dummy_false),
        anchor=AnchorDriftConf(**dummy_false),
        trail=ProfitTrailConf(
            p_min=0.02,
            s1=0.01,
            m_step=1.6,
            tau=0.7,
            p_lock_base=0.0,
            p_lock_max=0.0,
            tau_min=0.3,
        ),
        tmart=TimeMartingaleConf(W1_minutes=5, m_time=2.0, delta_lock=0.0, beta_tau=0.9),
        tcaps=TimeCapsConf(T_idle_max_minutes=10, p_idle=0.0, T_total_cap_minutes=60, p_exit_min=0.0),
        scalp=ScalpModeConf(**dummy_false),
        micro=micro_cfg,
        btd=BuyTheDipConf(**dummy_false),
        sah=SellAtHeightConf(**dummy_false),
        snapshot_every_bars=1,
        use_maker=True,
    )


def test_micro_position_pruned_when_tiny_inventory_left():
    state = _build_state_with_micro(min_exit_qty=0.001)

    # Tiny inventory that cannot be sold under the exit threshold should be pruned
    state.Q = 0.0005
    state.C = 0.0005 * 100 * (1 + state.fees_buy)
    state.micro_positions = [
        {
            "entry": 100.0,
            "qty": 0.0005,
            "cost": state.C,
            "target": 100.0,
            "stop": 90.0,
            "entry_bar": 0,
        }
    ]

    state.bar_index = 10
    log_events = []

    state._check_micro_take_profit(
        pd.Timestamp("2025-01-01T00:00:00Z"),
        h=100.0,
        l=100.0,
        log_events=log_events,
    )

    assert not state.micro_positions, "Micro position should be pruned when only tiny qty remains"
    assert any(ev.get("event") == "MICRO_EXIT_PRUNED" for ev in log_events)


def test_p_be_uses_micro_holdings_when_present():
    state = _build_state_with_micro()
    # Two units at average price 100
    state.Q = 2.0
    state.C = state.Q * 100.0 * (1 + state.fees_buy)

    # Track a micro position inside the total inventory
    micro_cost = 0.5 * 100.0 * (1 + state.fees_buy)
    state.micro_positions = [
        {
            "entry": 100.0,
            "qty": 0.5,
            "cost": micro_cost,
            "target": 101.0,
            "stop": 98.0,
            "entry_bar": 0,
        }
    ]

    expected_p_be = state.C / (state.Q * (1 - state.fees_sell))
    assert math.isclose(state.P_BE(), expected_p_be, rel_tol=1e-9)


def test_micro_exit_nudges_ladder_progress():
    state = _build_state_with_micro()
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    log_events: list[dict] = []

    state.rebuild_ladder(base_price=100.0, ts=ts, log_events=log_events)

    # Hold a micro position that will stop out
    state.micro_positions = [
        {
            "entry": 97.0,
            "qty": 1.0,
            "cost": 97.0 * (1 + state.fees_buy),
            "target": 110.0,
            "stop": 96.0,
            "entry_bar": 0,
        }
    ]
    state.Q = 1.0
    state.C = 97.0 * (1 + state.fees_buy)

    state._check_micro_take_profit(ts=ts, h=96.0, l=95.5, log_events=log_events)

    assert state.ladder_next_idx >= 2, "Micro exit should pull the ladder forward"
    assert any(
        ev.get("event") == "LADDER_NUDGE" and ev.get("reason") == "MICRO_EXIT"
        for ev in log_events
    )