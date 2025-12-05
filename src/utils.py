
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
    max_step_drop = max(0.0, float(max_step_drop)) if max_step_drop is not None else 1.0

    if spacing_mode != "fibo":
        return [P0 * ((1 - d_buy) ** k) for k in range(1, steps + 1)]

    multipliers = list(d_multipliers) if d_multipliers else _fibonacci_sequence(steps)
    prices = []
    prev_price = P0
    for mult in multipliers[:steps]:
        drop = max(0.0, min(max_step_drop, d_buy * float(mult)))
        prev_price = prev_price * (1 - drop)
        prices.append(prev_price)
    return prices
