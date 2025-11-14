import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

from main import load_config
from src.backtester import PairConfig, init_state_from_config, prepare_ladder
from src.fetchers.binance import INTERVAL_MS
from src.strategy import PHASE_ACCUMULATE
from src.binance_client import BinanceClient


def build_pair_config(raw: Dict, general: Dict) -> PairConfig:
    return PairConfig(
        symbol=raw["symbol"],
        quote=raw.get("quote", "QUOTE"),
        b_alloc=float(raw["b_alloc"]),
        source=raw.get("source", "binance"),
        interval=raw.get("interval", "1m"),
        lookback_days=float(raw.get("lookback_days", 1)) if raw.get("lookback_days") is not None else None,
        start=raw.get("start", ""),
        end=raw.get("end", ""),
        fees_taker=float(raw["fees"]["taker"]),
        fees_maker=float(raw["fees"]["maker"]),
        buy_d=float(raw["buy_ladder"]["d_buy"]),
        buy_m=float(raw["buy_ladder"]["m_buy"]),
        buy_n=int(raw["buy_ladder"]["n_steps"]),
        p_min=float(raw["profit_trail"]["p_min"]),
        s1=float(raw["profit_trail"]["s1"]),
        m_step=float(raw["profit_trail"]["m_step"]),
        tau=float(raw["profit_trail"]["tau"]),
        p_lock_base=float(raw["profit_trail"]["p_lock_base"]),
        p_lock_max=float(raw["profit_trail"]["p_lock_max"]),
        tau_min=float(raw["profit_trail"]["tau_min"]),
        no_loss_epsilon=float(raw["profit_trail"].get("no_loss_epsilon", 0.0005)),
        W1_minutes=float(raw["time_martingale"]["W1_minutes"]),
        m_time=float(raw["time_martingale"]["m_time"]),
        delta_lock=float(raw["time_martingale"]["delta_lock"]),
        beta_tau=float(raw["time_martingale"]["beta_tau"]),
        T_idle_max_minutes=float(raw["time_caps"]["T_idle_max_minutes"]),
        p_idle=float(raw["time_caps"]["p_idle"]),
        T_total_cap_minutes=float(raw["time_caps"]["T_total_cap_minutes"]),
        p_exit_min=float(raw["time_caps"]["p_exit_min"]),
        snapshot_every_bars=int(general.get("snapshot_every_bars", 1)),
        use_maker=bool(general.get("use_maker", True)),
        btd=raw.get("features", {}).get("buy_the_dip", {"enabled": False}),
        sah=raw.get("features", {}).get("sell_at_height", {"enabled": False}),
    )


def reset_round(state, price: float, ts: datetime) -> None:
    state.round_active = True
    state.phase = PHASE_ACCUMULATE
    state.P0 = price
    state.round_start_ts = ts
    state.last_high_ts = None
    state.TP_base = None
    state.H = None
    state.stage = 0
    state.cumulative_s = 0.0
    state.tau_cur = state.trail.tau
    state.p_lock_cur = state.trail.p_lock_base
    state.next_window_minutes = state.tmart.W1_minutes
    state.idle_checked = False
    state.Q = 0.0
    state.C = 0.0
    state.btd_armed = False
    state.sah_armed = False
    state.btd_last_bottom = None
    state.sah_last_top = None
    state.btd_last_arm_ts = None
    state.sah_last_arm_ts = None
    state.btd_cooldown_until = None
    state.sah_cooldown_until = None
    prepare_ladder(state, price)


def fetch_previous_closed(symbol: str, interval: str, base_url: str) -> Optional[Dict[str, float]]:
    url = f"{base_url.rstrip('/')}/api/v3/klines"
    resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": 2}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if len(data) < 2:
        return None
    kline = data[-2]
    open_time = int(kline[0])
    close_time = int(kline[6])
    ts = datetime.fromtimestamp(close_time / 1000.0, tz=timezone.utc)
    return {
        "open_time": open_time,
        "timestamp": ts,
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "volume": float(kline[5]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live trading loop against Binance")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbol", default="")
    parser.add_argument("--dry-run", action="store_true", help="Log trades without sending orders")
    parser.add_argument("--poll-seconds", type=float, default=10.0, help="Polling interval for klines")
    args = parser.parse_args()

    cfg = load_config(args.config)
    general = cfg.get("general", {})
    binance_cfg = (general.get("binance") or {})
    base_url = binance_cfg.get("base_url", "https://api.binance.com")

    pairs_raw = cfg.get("pairs", [])
    if not pairs_raw:
        raise SystemExit("No pairs defined in config")

    symbol = args.symbol.upper() if args.symbol else pairs_raw[0]["symbol"].upper()
    target_raw = next((p for p in pairs_raw if p["symbol"].upper() == symbol), None)
    if target_raw is None:
        raise SystemExit(f"Symbol {symbol} not found in config")

    pair_cfg = build_pair_config(target_raw, general)
    state = init_state_from_config(pair_cfg)

    latest = fetch_previous_closed(pair_cfg.symbol, pair_cfg.interval, base_url)
    if latest is None:
        raise SystemExit("Unable to seed state with latest kline")

    reset_round(state, latest["close"], latest["timestamp"])

    api_key = os.getenv("BINANCE_API_KEY", cfg.get("api", {}).get("key", ""))
    api_secret = os.getenv("BINANCE_API_SECRET", cfg.get("api", {}).get("secret", ""))

    client: Optional[BinanceClient] = None
    if not args.dry_run:
        if not api_key or not api_secret:
            raise SystemExit("Live trading requires BINANCE_API_KEY and BINANCE_API_SECRET")
        client = BinanceClient(api_key, api_secret, base_url=base_url)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting live trading for %s (%s)", pair_cfg.symbol, pair_cfg.interval)
    logging.info("Dry run mode: %s", "ON" if args.dry_run else "OFF")

    event_log: List[Dict] = []
    event_ptr = 0
    last_open_time = 0
    interval_ms = INTERVAL_MS[pair_cfg.interval]

    while True:
        try:
            kline = fetch_previous_closed(pair_cfg.symbol, pair_cfg.interval, base_url)
            if not kline:
                time.sleep(args.poll_seconds)
                continue
            if kline["open_time"] == last_open_time:
                time.sleep(args.poll_seconds)
                continue

            ts = kline["timestamp"]
            res = state.on_bar(ts, kline["open"], kline["high"], kline["low"], kline["close"], kline["volume"], event_log)
            new_events = event_log[event_ptr:]
            event_ptr = len(event_log)

            for ev in new_events:
                evt = ev.get("event")
                if evt == "BUY":
                    quote_amt = float(ev.get("amt_q", 0.0))
                    logging.info("BUY ladder fill at %.4f for quote %.2f", ev.get("price"), quote_amt)
                    if client and quote_amt > 0:
                        response = client.market_buy_quote(pair_cfg.symbol, quote_amt)
                        logging.info("Order response: %s", json.dumps(response))
                elif evt == "BTD_ORDER":
                    logging.info("BTD order suggested at %.4f for quote %.2f", ev.get("order_price"), ev.get("order_quote", 0.0))
                elif evt == "SELL":
                    logging.info("Strategy logged SELL event: %s", ev)

            if res is not None:
                qty = float(res.get("qty", 0.0))
                logging.info("Exit signal %s at %.4f (qty %.6f, pnl %.2f)", res.get("reason"), res.get("sell_price"), qty, res.get("pnl", 0.0))
                if client and qty > 0:
                    response = client.market_sell(pair_cfg.symbol, qty)
                    logging.info("Sell response: %s", json.dumps(response))
                reset_round(state, kline["close"], ts)
                event_log.clear()
                event_ptr = 0

            last_open_time = kline["open_time"]
            sleep_time = max(args.poll_seconds, interval_ms / 1000.0 / 2)
            time.sleep(sleep_time)
        except requests.RequestException as exc:
            logging.error("Network error: %s", exc)
            time.sleep(args.poll_seconds)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Unexpected error: %s", exc)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()