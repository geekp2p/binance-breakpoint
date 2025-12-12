import pytest

from src.utils import compute_ladder_amounts, compute_ladder_prices, geometric_base_allocation
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


def test_legacy_geometric_spacing_and_sizes():
    P0 = 100.0
    d_buy = 0.03
    steps = 3
    prices = compute_ladder_prices(P0, d_buy, steps)
    expected_prices = [P0 * ((1 - d_buy) ** k) for k in range(1, steps + 1)]
    assert prices == pytest.approx(expected_prices)

    total_alloc = 1000.0
    m_buy = 1.5
    base = geometric_base_allocation(total_alloc, m_buy, steps)
    expected_amounts = [base * (m_buy ** i) for i in range(steps)]
    amounts = compute_ladder_amounts(total_alloc, m_buy, steps)
    assert amounts == pytest.approx(expected_amounts)


def test_fibonacci_size_with_multiplicative_gaps_and_caps():
    P0 = 100.0
    prices = compute_ladder_prices(
        P0,
        d_buy=0.03,
        n_steps=4,
        gap_mode="multiplicative",
        gap_factor=1.2,
    )
    expected_prices = []
    prev = P0
    drops = [0.03, 0.03 * 1.2, 0.03 * (1.2 ** 2), 0.03 * (1.2 ** 3)]
    for drop in drops:
        prev *= (1 - drop)
        expected_prices.append(prev)
    assert prices == pytest.approx(expected_prices)

    amounts = compute_ladder_amounts(
        total_alloc=1000.0,
        m_buy=2.0,
        n_steps=4,
        size_mode="fibonacci",
        max_total_quote=600.0,
        max_quote_per_leg=200.0,
    )
    # Fibonacci weights [1,1,2,3]; effective allocation should be 600
    base = 600.0 / 7.0
    expected_amounts = [base, base, base * 2, min(base * 3, 200.0)]
    assert amounts == pytest.approx(expected_amounts)


def test_rebuild_preserves_remaining_allocation_when_progressed():
    buy_cfg = BuyLadderConf(d_buy=0.01, m_buy=2.0, n_steps=3)
    dummy = {
        "enabled": False,
    }
    state = StrategyState(
        fees_buy=0.001,
        fees_sell=0.001,
        b_alloc=100.0,
        buy=buy_cfg,
        adaptive=AdaptiveLadderConf(enabled=False),
        anchor=AnchorDriftConf(enabled=False),
        trail=ProfitTrailConf(p_min=0.02, s1=0.01, m_step=1.6, tau=0.7, p_lock_base=0.0, p_lock_max=0.0, tau_min=0.3),
        tmart=TimeMartingaleConf(W1_minutes=5, m_time=2.0, delta_lock=0.0, beta_tau=0.9),
        tcaps=TimeCapsConf(T_idle_max_minutes=10, p_idle=0.0, T_total_cap_minutes=60, p_exit_min=0.0),
        scalp=ScalpModeConf(**dummy),
        micro=MicroOscillationConf(**dummy),
        btd=BuyTheDipConf(**dummy),
        sah=SellAtHeightConf(**dummy),
        snapshot_every_bars=1,
        use_maker=True,
    )

    state._set_initial_d_buy()
    state.rebuild_ladder(100.0)

    # Simulate the first ladder leg being filled.
    state.ladder_next_idx = 1
    remaining_before = state._remaining_quote_allocation()

    # Rebuild while preserving progress; should only allocate the remaining capital.
    state.rebuild_ladder(100.0, preserve_progress=True)
    assert sum(state.ladder_amounts_quote) == pytest.approx(remaining_before)

    # Full rebuild (e.g., new round) should size for the full allocation again.
    state.rebuild_ladder(100.0, preserve_progress=False)
    assert sum(state.ladder_amounts_quote) == pytest.approx(state.b_alloc)