
import time, requests, os
import pandas as pd
from datetime import datetime, timezone

INTERVAL_MS = {
    "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000, "30m": 1800000,
    "1h": 3600000, "2h": 7200000, "4h": 14400000, "6h": 21600000, "8h": 28800000, "12h": 43200000,
    "1d": 86400000, "3d": 259200000, "1w": 604800000, "1M": 2592000000
}

def _to_ms(x):
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        dt = datetime.fromisoformat(x.replace("Z","")).replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    return int(x.timestamp() * 1000)

def fetch_klines_binance(symbol, interval, base_url="https://api.binance.com",
                         start=None, end=None, lookback_days=None,
                         retry_attempts=4, backoff_sec=0.5, api_key=None):
    # Public klines ไม่จำเป็นต้องใช้คีย์
    if interval not in INTERVAL_MS:
        raise ValueError("Unsupported interval")
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    end_ms = _to_ms(end) or now_ms
    if lookback_days is not None and start is None:
        start_ms = end_ms - int(lookback_days * 86400000)
    else:
        start_ms = _to_ms(start)
        if start_ms is None:
            raise ValueError("Either lookback_days or start must be provided")
    step_ms = INTERVAL_MS[interval] * 1000  # 1000 bars/batch
    url = f"{base_url}/api/v3/klines"
    rows = []
    cur = start_ms
    headers = {"X-MBX-APIKEY": api_key} if api_key else None
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval, "limit": 1000,
                  "startTime": cur, "endTime": min(end_ms, cur + step_ms - 1)}
        for a in range(retry_attempts):
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff_sec * (a+1)); continue
            r.raise_for_status(); data = r.json(); break
        else:
            raise RuntimeError("Binance fetch failed")
        if not data: break
        for k in data:
            rows.append((int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        last_open = data[-1][0]
        cur = last_open + INTERVAL_MS[interval]
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"])[["timestamp","open","high","low","close","volume"]]
    return df.sort_values("timestamp").reset_index(drop=True)
