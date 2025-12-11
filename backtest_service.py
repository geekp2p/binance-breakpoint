import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory

from src.fetchers.binance import fetch_klines_binance

app = Flask(__name__, static_folder="static/backtest", static_url_path="")

WINDOW_LOOKBACK_DAYS: Dict[str, int] = {
    "1d": 1,
    "7d": 7,
    "30d": 30,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
}

CACHE_DIR = Path("data/backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def discover_usdt_pairs() -> List[Dict[str, Any]]:
    """Return a list of USDT pairs sorted by liquidity (quote volume)."""
    resp = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=30)
    resp.raise_for_status()
    tickers = resp.json()
    pairs = []
    for t in tickers:
        symbol = t.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue
        if any(symbol.endswith(suffix) for suffix in ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")):
            continue
        try:
            volume = float(t.get("quoteVolume", 0))
        except (TypeError, ValueError):
            volume = 0.0
        pairs.append({"symbol": symbol, "quoteVolume": volume})
    pairs.sort(key=lambda x: x["quoteVolume"], reverse=True)
    cutoff = max(1, len(pairs) // 2)
    return pairs[:cutoff]


def _cache_file(symbol: str, days: int) -> Path:
    return CACHE_DIR / f"{symbol}_{days}d.csv"


def _load_from_cache(symbol: str, days: int) -> pd.DataFrame | None:
    file = _cache_file(symbol, days)
    if not file.exists():
        return None
    age_hours = (time.time() - file.stat().st_mtime) / 3600
    if age_hours > 6:
        return None
    try:
        return pd.read_csv(file, parse_dates=["timestamp"])
    except Exception:
        return None


def _save_to_cache(symbol: str, days: int, df: pd.DataFrame) -> None:
    file = _cache_file(symbol, days)
    df.to_csv(file, index=False)


def fetch_history(symbol: str, days: int) -> pd.DataFrame:
    cached = _load_from_cache(symbol, days)
    if cached is not None:
        return cached
    df = fetch_klines_binance(symbol, "1d", lookback_days=days + 2)
    if not df.empty:
        _save_to_cache(symbol, days, df)
    return df


def compute_pnl(usdt: float, df: pd.DataFrame) -> Tuple[float, float]:
    if df is None or df.empty or len(df) < 2:
        return 0.0, 0.0
    start_price = float(df.iloc[0]["open"])
    end_price = float(df.iloc[-1]["close"])
    if start_price <= 0:
        return 0.0, 0.0
    qty = usdt / start_price
    final_usd = qty * end_price
    pnl_usd = final_usd - usdt
    pnl_pct = ((final_usd / usdt) - 1.0) * 100.0
    return pnl_pct, pnl_usd


def compute_levels(df: pd.DataFrame) -> Dict[str, float]:
    """Return simple buy-the-dip / sell-the-rip reference levels from recent price action."""
    if df is None or df.empty:
        return {}
    recent = df.tail(min(len(df), 14))
    current = float(recent.iloc[-1]["close"])
    buy_dip = float(recent["low"].min())
    sell_rip = float(recent["high"].max())
    return {"current": current, "buy_dip": buy_dip, "sell_rip": sell_rip}


def normalize_intervals(intervals: Iterable[str] | None) -> List[str]:
    if not intervals:
        return list(WINDOW_LOOKBACK_DAYS.keys())
    vals = []
    for iv in intervals:
        if iv in WINDOW_LOOKBACK_DAYS:
            vals.append(iv)
    return vals or list(WINDOW_LOOKBACK_DAYS.keys())


def normalize_symbols(symbols: Iterable[str] | None) -> List[str]:
    if not symbols:
        return [p["symbol"] for p in discover_usdt_pairs()]
    seen = []
    for s in symbols:
        sym = s.strip().upper()
        if not sym.endswith("USDT"):
            continue
        if sym not in seen:
            seen.append(sym)
    if seen:
        return seen
    return [p["symbol"] for p in discover_usdt_pairs()]


def prepare_top_lists(scores: List[Dict[str, Any]], limits: List[int]) -> Dict[str, List[Dict[str, Any]]]:
    sorted_scores = sorted(scores, key=lambda x: (x["pnl_usd"], x["pnl_pct"]), reverse=True)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for limit in limits:
        key = f"top{limit}"
        result[key] = sorted_scores[:limit]
    return result


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/backtest", methods=["POST"])
def backtest_endpoint():
    payload = request.get_json(force=True, silent=True) or {}
    usdt = float(payload.get("usdt", 1000.0))
    intervals = normalize_intervals(payload.get("intervals"))
    top_raw = payload.get("top_n") or [30, 100, 500]
    top_n: List[int] = []
    for x in top_raw:
        try:
            val = int(x)
        except (TypeError, ValueError):
            continue
        if val > 0:
            top_n.append(val)
    if not top_n:
        top_n = [30, 100, 500]
    symbols = normalize_symbols(payload.get("symbols"))

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    levels: Dict[str, Dict[str, float]] = {}
    scores: List[Dict[str, Any]] = []
    history_cache: Dict[Tuple[str, int], pd.DataFrame] = {}

    for sym in symbols:
        sym_res: Dict[str, Dict[str, float]] = {}
        total_usd = 0.0
        total_pct = 0.0
        df_for_levels: pd.DataFrame | None = None
        for iv in intervals:
            lookback_days = WINDOW_LOOKBACK_DAYS[iv]
            try:
                df = history_cache.get((sym, lookback_days))
                if df is None:
                    df = fetch_history(sym, lookback_days)
                    history_cache[(sym, lookback_days)] = df
            except Exception:
                continue
            if df_for_levels is None and df is not None and not df.empty:
                df_for_levels = df
            pnl_pct, pnl_usd = compute_pnl(usdt, df)
            sym_res[iv] = {"pnl_pct": pnl_pct, "pnl_usd": pnl_usd}
            total_usd += pnl_usd
            total_pct += pnl_pct
        if sym_res:
            results[sym] = sym_res
            scores.append({"symbol": sym, "pnl_usd": total_usd, "pnl_pct": total_pct})
            levels[sym] = compute_levels(df_for_levels)

    top_symbols = prepare_top_lists(scores, top_n)

    response = {
        "timestamp": _now_iso(),
        "params": {"usdt": usdt, "symbols": symbols, "intervals": intervals, "top_n": top_n},
        "results": results,
        "top_symbols": top_symbols,
        "levels": levels,
    }
    return jsonify(response)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": _now_iso()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8181)