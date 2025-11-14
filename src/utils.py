
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
    if m == 1.0:
        return B_alloc / n_steps
    return B_alloc * (1 - m) / (1 - (m ** n_steps))

def compute_ladder_prices(P0, d_buy, n_steps):
    return [P0 * ((1 - d_buy) ** k) for k in range(1, n_steps + 1)]
