import pytest

from src.utils import compute_ladder_amounts, compute_ladder_prices, geometric_base_allocation


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