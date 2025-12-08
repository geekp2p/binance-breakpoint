
import pandas as pd

def load_price_csv(path):
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must have 'timestamp' column")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    if df['timestamp'].isna().any():
        raise ValueError("Invalid timestamps in CSV")
    for c in ['open','high','low','close']:
        if c not in df.columns:
            raise ValueError("CSV must have OHLC columns")
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    return df.sort_values('timestamp').reset_index(drop=True)

def geometric_base_allocation(B_alloc, m, n_steps):
    if n_steps == 1:
        return B_alloc
    if abs(m - 1.0) < 1e-12:
        return B_alloc / n_steps
    return B_alloc * (1 - m) / (1 - (m ** n_steps))

def _fibonacci_sequence(n: int) -> list[float]:
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

def compute_ladder_amounts(
    total_alloc: float,
    m_buy: float,
    n_steps: int,
    *,
    size_mode: str = "geometric",
    base_order_quote: float | None = None,
    max_quote_per_leg: float = 0.0,
    max_total_quote: float = 0.0,
):
    """Compute per-leg quote sizes for a ladder.

    size_mode:
        - "geometric" (default): classic martingale sizes using ``m_buy``.
        - "fibonacci": weights follow the Fibonacci sequence, optionally scaled
          by ``base_order_quote``; otherwise normalized to fit the effective
          allocation.
    Caps:
        - ``max_total_quote`` limits the capital allocated to the ladder.
        - ``max_quote_per_leg`` caps each leg individually.
    """

    steps = max(1, int(n_steps))
    effective_alloc = min(total_alloc, max_total_quote) if max_total_quote > 0 else total_alloc
    size_mode = (size_mode or "geometric").lower()

    if size_mode != "fibonacci":
        base = geometric_base_allocation(effective_alloc, m_buy, steps)
        amounts = [base * (m_buy ** i) for i in range(steps)]
    else:
        fib_weights = _fibonacci_sequence(steps)
        weight_sum = max(sum(fib_weights), 1.0)
        base = base_order_quote if base_order_quote and base_order_quote > 0 else effective_alloc / weight_sum
        if base * weight_sum > effective_alloc:
            base = effective_alloc / weight_sum
        amounts = [base * w for w in fib_weights]

    if max_quote_per_leg > 0:
        amounts = [min(a, max_quote_per_leg) for a in amounts]

    return amounts


def compute_ladder_prices(
    P0,
    d_buy,
    n_steps,
    *,
    spacing_mode: str = "geometric",
    d_multipliers=None,
    max_step_drop: float = 0.25,
):
    """
    Calculate ladder entry prices with optional non-linear spacing.

    spacing_mode:
        - "geometric" (default): fixed spacing using (1 - d_buy)^k.
        - "fibo": widening spacing using Fibonacci multipliers or a custom list
          (`d_multipliers`). Each step drop is `d_buy * multiplier`, capped by
          `max_step_drop`.
    """

    steps = max(1, int(n_steps))
    spacing_mode = (spacing_mode or "geometric").lower()
    gap_mode = (gap_mode or "additive").lower()
    gap_factor = float(gap_factor) if gap_factor is not None else 1.0
    max_step_drop = max(0.0, float(max_step_drop)) if max_step_drop is not None else 1.0

    multipliers = [1.0] * steps
    if spacing_mode == "fibo":
        multipliers = list(d_multipliers) if d_multipliers else _fibonacci_sequence(steps)

    prices = []
    prev_price = P0
    for idx, mult in enumerate(multipliers[:steps]):
        step_mult = float(mult)
        if gap_mode == "multiplicative":
            step_mult *= gap_factor ** idx
        drop = max(0.0, min(max_step_drop, d_buy * step_mult))
        prev_price = prev_price * (1 - drop)
        prices.append(prev_price)
    return prices
