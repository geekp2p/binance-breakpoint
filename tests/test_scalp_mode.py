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


def _make_state(**kwargs):
    dummy_false = {"enabled": False}
    return StrategyState(
        fees_buy=0.001,
        fees_sell=0.001,
        b_alloc=100.0,
        buy=BuyLadderConf(d_buy=0.02, m_buy=1.0, n_steps=3),
        adaptive=AdaptiveLadderConf(enabled=False),
        anchor=AnchorDriftConf(enabled=False),
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
        scalp=ScalpModeConf(**kwargs),
        micro=MicroOscillationConf(**dummy_false),
        btd=BuyTheDipConf(**dummy_false),
        sah=SellAtHeightConf(**dummy_false),
        snapshot_every_bars=1,
        use_maker=True,
    )


def test_scalp_buy_respects_cooldown():
    state = _make_state(enabled=True, cooldown_bars=2, order_pct_allocation=1.0)
    log_events = []

    # First bar should trigger a scalp buy.
    state.on_bar(pd.Timestamp("2025-01-01T00:00:00Z"), 100.0, 100.0, 90.0, 95.0, 0, log_events)
    assert any(ev["event"] == "SCALP_BUY" for ev in log_events)
    assert state.scalp_trades_done == 1

    # Cooldown active: second bar under trigger should be skipped.
    log_events.clear()
    state.on_bar(pd.Timestamp("2025-01-01T00:01:00Z"), 99.0, 99.0, 90.0, 95.0, 0, log_events)
    assert any(ev.get("reason") == "COOLDOWN" for ev in log_events)
    assert state.scalp_trades_done == 1

    # After cooldown expires another scalp buy can proceed.
    log_events.clear()
    state.on_bar(pd.Timestamp("2025-01-01T00:03:00Z"), 98.0, 98.0, 90.0, 95.0, 0, log_events)
    assert sum(1 for ev in log_events if ev["event"] == "SCALP_BUY") == 1
    assert state.scalp_trades_done == 2