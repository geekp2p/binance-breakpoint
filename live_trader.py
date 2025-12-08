import argparse
import csv
import json
import logging
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from urllib.parse import parse_qs, urlparse

from main import load_config
from src.backtester import PairConfig, init_state_from_config, prepare_ladder
from src.fetchers.binance import INTERVAL_MS
from src.strategy import PHASE_ACCUMULATE, PHASE_TRAIL
from src.binance_client import BinanceClient
from src.savepoint import (
    DEFAULT_SAVEPOINT_DIR,
    apply_payload_to_state,
    load_savepoint,
    write_savepoint,
)


MAX_HISTORY_ENTRIES = 200


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
        buy_spacing=str(raw["buy_ladder"].get("spacing_mode", "geometric")),
        buy_multipliers=raw["buy_ladder"].get("d_multipliers"),
        buy_max_drop=float(raw["buy_ladder"].get("max_step_drop", 0.25)),
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
        scalp=raw.get("features", {}).get("scalp_mode", {"enabled": False}),
        btd=raw.get("features", {}).get("buy_the_dip", {"enabled": False}),
        sah=raw.get("features", {}).get("sell_at_height", {"enabled": False}),
        adaptive_ladder=raw.get("features", {}).get("adaptive_ladder", {"enabled": False}),
        anchor_drift=raw.get("features", {}).get("anchor_drift", {"enabled": False}),
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


class ControlCenter:
    def __init__(self, symbols: List[str]):
        self._lock = threading.Lock()
        self._status: Dict[str, Dict[str, object]] = {sym: {} for sym in symbols}
        self._paused: Dict[str, threading.Event] = {sym: threading.Event() for sym in symbols}

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        with self._lock:
            return {
                sym: {**status, "paused": self._paused[sym].is_set()}
                for sym, status in self._status.items()
            }

    def update_status(self, symbol: str, status: Dict[str, object]) -> None:
        with self._lock:
            self._status[symbol] = dict(status)

    def pause(self, symbol: Optional[str] = None) -> List[str]:
        targets = [symbol] if symbol else list(self._paused.keys())
        paused = []
        for sym in targets:
            ev = self._paused.get(sym)
            if ev:
                ev.set()
                paused.append(sym)
        return paused

    def resume(self, symbol: Optional[str] = None) -> List[str]:
        targets = [symbol] if symbol else list(self._paused.keys())
        resumed = []
        for sym in targets:
            ev = self._paused.get(sym)
            if ev:
                ev.clear()
                resumed.append(sym)
        return resumed

    def is_paused(self, symbol: str) -> bool:
        ev = self._paused.get(symbol)
        return bool(ev and ev.is_set())


def create_http_handler(control: ControlCenter):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _set_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                payload = {"status": "ok", "pairs": control.snapshot()}
                self._json(200, payload)
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            symbol = (params.get("symbol") or [None])[0]
            if symbol:
                symbol = symbol.upper()
            if parsed.path == "/pause":
                affected = control.pause(symbol)
                self._json(200, {"status": "paused", "pairs": affected})
                return
            if parsed.path == "/resume":
                affected = control.resume(symbol)
                self._json(200, {"status": "resumed", "pairs": affected})
                return
            self._json(404, {"error": "not found"})

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._set_cors_headers()
            self.end_headers()

    return Handler


def start_http_server(control: ControlCenter, port: int) -> Optional[ThreadingHTTPServer]:
    if port <= 0:
        return None
    handler = create_http_handler(control)
    httpd = ThreadingHTTPServer(("", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    logging.info("Control server listening on http://0.0.0.0:%s", port)
    return httpd


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
    parser.add_argument(
        "--savepoint-dir",
        default=os.getenv("SAVEPOINT_DIR", str(DEFAULT_SAVEPOINT_DIR)),
        help=(
            "Directory where live state snapshots will be stored; can be overridden "
            "with SAVEPOINT_DIR environment variable"
        ),
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port for health/pause/resume server (set 0 to disable)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    general = cfg.get("general", {})
    binance_cfg = (general.get("binance") or {})
    base_url = binance_cfg.get("base_url", "https://api.binance.com")

    pairs_raw = cfg.get("pairs", [])
    if not pairs_raw:
        raise SystemExit("No pairs defined in config")

    sell_scale_out_cfg = (general.get("sell_scale_out") or {})
    sell_chunks = max(int(sell_scale_out_cfg.get("chunks", 1)), 1)
    sell_chunk_delay = float(sell_scale_out_cfg.get("delay_seconds", 0.0))
    sell_profit_only = bool(sell_scale_out_cfg.get("profit_only", True))

    def run_pair(pair_raw: Dict) -> None:
        pair_cfg = build_pair_config(pair_raw, general)
        state = init_state_from_config(pair_cfg)
        pair_symbol = pair_cfg.symbol.upper()

        latest = fetch_previous_closed(pair_cfg.symbol, pair_cfg.interval, base_url)
        if latest is None:
            raise SystemExit("Unable to seed state with latest kline")

        savepoint_dir = Path(args.savepoint_dir) / pair_cfg.symbol.upper()
        savepoint_payload = load_savepoint(savepoint_dir, pair_cfg.symbol)

        event_log: List[Dict[str, object]] = []
        event_ptr = 0
        activity_history: List[Dict[str, object]] = []
        event_persist_count = 0
        activity_persist_count = 0
        realized_pnl_total = 0.0
        last_open_time = 0

        if savepoint_payload and savepoint_payload.get("state"):
            apply_payload_to_state(state, savepoint_payload["state"])
            event_log = list(savepoint_payload.get("event_log", []))
            event_ptr = len(event_log)
            activity_history = list(savepoint_payload.get("activity_history", []))
            if len(activity_history) > MAX_HISTORY_ENTRIES:
                activity_history = activity_history[-MAX_HISTORY_ENTRIES:]
            last_open_time = int(savepoint_payload.get("last_open_time") or 0)
            realized_pnl_total = float(savepoint_payload.get("realized_pnl_total") or 0.0)
            logging.info(
                "Loaded savepoint for %s from %s",
                pair_cfg.symbol,
                savepoint_dir.joinpath(f"{pair_cfg.symbol.upper()}.json"),
            )
            resume_status = savepoint_payload.get("status") or {}
            logging.info(
                "Resumed status | phase=%s | stage=%s | qty=%.6f | price=%.4f | realized_total=%.2f",
                resume_status.get("phase", "N/A"),
                resume_status.get("stage", "N/A"),
                float(resume_status.get("qty") or 0.0),
                float(resume_status.get("price") or 0.0),
                float(resume_status.get("realized_pnl_total") or 0.0),
            )
        else:
            reset_round(state, latest["close"], latest["timestamp"])

        history_jsonl = savepoint_dir / f"{pair_cfg.symbol.upper()}_event_log.jsonl"
        history_csv = savepoint_dir / f"{pair_cfg.symbol.upper()}_event_log.csv"
        activity_jsonl = savepoint_dir / f"{pair_cfg.symbol.upper()}_activity_history.jsonl"
        activity_csv = savepoint_dir / f"{pair_cfg.symbol.upper()}_activity_history.csv"

        def _normalise_event(event: Dict[str, object]) -> Dict[str, object]:
            normalised: Dict[str, object] = {}
            for key, val in event.items():
                if isinstance(val, datetime):
                    normalised[key] = val.isoformat()
                else:
                    normalised[key] = val
            return normalised

        def _count_lines(path: Path) -> int:
            if not path.exists():
                return 0
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return sum(1 for _ in fh)
            except OSError:
                return 0

        def _persist_records(records: List[Dict[str, object]], jsonl_path: Path, csv_path: Path) -> None:
            if not records:
                return
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not csv_path.exists()
            with jsonl_path.open("a", encoding="utf-8") as jf, csv_path.open(
                "a", encoding="utf-8", newline=""
            ) as cf:
                writer = csv.DictWriter(cf, fieldnames=["ts", "event", "details_json"])
                if write_header:
                    writer.writeheader()
                for rec in records:
                    norm = _normalise_event(rec)
                    jf.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    ts = norm.get("ts") or norm.get("timestamp") or norm.get("time")
                    evt = norm.get("event") or norm.get("type")
                    details = {
                        k: v
                        for k, v in norm.items()
                        if k
                        not in (
                            "ts",
                            "timestamp",
                            "time",
                            "event",
                            "type",
                        )
                    }
                    writer.writerow(
                        {
                            "ts": ts,
                            "event": evt,
                            "details_json": json.dumps(details, ensure_ascii=False),
                        }
                    )

        def _sync_history(
            records: List[Dict[str, object]], jsonl_path: Path, csv_path: Path
        ) -> int:
            existing_lines = _count_lines(jsonl_path)
            start_idx = min(existing_lines, len(records))
            _persist_records(records[start_idx:], jsonl_path, csv_path)
            return len(records)

        if state.P0 is None:
            reset_round(state, latest["close"], latest["timestamp"])

        api_key = os.getenv("BINANCE_API_KEY", cfg.get("api", {}).get("key", ""))
        api_secret = os.getenv("BINANCE_API_SECRET", cfg.get("api", {}).get("secret", ""))

        client: Optional[BinanceClient] = None
        base_asset: str = ""
        if not args.dry_run:
            if not api_key or not api_secret:
                raise SystemExit("Live trading requires BINANCE_API_KEY and BINANCE_API_SECRET")
            client = BinanceClient(api_key, api_secret, base_url=base_url)
            try:
                sym_info = client.get_symbol_info(pair_cfg.symbol)
                base_asset = str(sym_info.get("baseAsset", "")).upper()
            except Exception:
                logging.warning("Unable to fetch symbol metadata; commission adjustments disabled")

        logging.info("Starting live trading for %s (%s)", pair_cfg.symbol, pair_cfg.interval)
        logging.info("Dry run mode: %s", "ON" if args.dry_run else "OFF")

        interval_ms = INTERVAL_MS[pair_cfg.interval]

        event_persist_count = _sync_history(event_log, history_jsonl, history_csv)
        activity_persist_count = _sync_history(
            activity_history, activity_jsonl, activity_csv
        )

        def persist_event_log_delta() -> None:
            nonlocal event_persist_count
            delta = event_log[event_persist_count:]
            _persist_records(delta, history_jsonl, history_csv)
            event_persist_count = len(event_log)

        def record_history(event: Dict[str, object]) -> None:
            nonlocal activity_persist_count
            copy = dict(event)
            activity_history.append(copy)
            if len(activity_history) > MAX_HISTORY_ENTRIES:
                del activity_history[: len(activity_history) - MAX_HISTORY_ENTRIES]
            delta = activity_history[activity_persist_count:]
            _persist_records(delta, activity_jsonl, activity_csv)
            activity_persist_count = len(activity_history)

        def clamp_buy_to_available(
            ev: Dict[str, object],
            available_quote: Optional[float],
            ts: datetime,
            context: str,
        ) -> tuple[float, bool]:
            quote_amt = float(ev.get("amt_q", 0.0))
            if available_quote is None:
                return quote_amt, False

            cost_with_fees = quote_amt * (1 + state.fees_buy)
            if available_quote + 1e-12 >= cost_with_fees:
                return quote_amt, False

            if available_quote <= 0:
                rollback_failed_buy(ev, ts, RuntimeError("Insufficient quote balance"))
                record_history(ev)
                return 0.0, True

            adjusted_quote = available_quote / (1 + state.fees_buy)
            expected_qty = float(ev.get("q") or ev.get("qty") or 0.0)
            if quote_amt <= 0 or expected_qty <= 0:
                rollback_failed_buy(ev, ts, RuntimeError("Invalid buy sizing after clamping"))
                record_history(ev)
                return 0.0, True

            scale = max(min(adjusted_quote / quote_amt, 1.0), 0.0)
            adjusted_qty = expected_qty * scale
            delta_qty = expected_qty - adjusted_qty
            delta_cost = cost_with_fees - (adjusted_quote * (1 + state.fees_buy))
            if delta_qty > 0:
                state.Q = max(state.Q - delta_qty, 0.0)
            if delta_cost > 0:
                state.C = max(state.C - delta_cost, 0.0)
            ev["q"] = adjusted_qty
            ev["qty"] = adjusted_qty
            ev["amt_q"] = adjusted_quote
            logging.warning(
                "Clamped buy quote from %.2f to %.2f (%s) due to available %s balance %.2f",
                cost_with_fees,
                adjusted_quote * (1 + state.fees_buy),
                context,
                pair_cfg.quote,
                available_quote,
            )
            return adjusted_quote, False

        def rollback_failed_buy(ev: Dict[str, object], ts: datetime, exc: Exception) -> None:
            qty = float(ev.get("q") or ev.get("qty") or 0.0)
            quote_amt = float(ev.get("amt_q") or 0.0)
            if qty > 0:
                state.Q = max(state.Q - qty, 0.0)
            if quote_amt > 0:
                state.C = max(state.C - quote_amt * (1 + state.fees_buy), 0.0)
            if ev.get("event") == "BUY" and state.ladder_next_idx > 0:
                state.ladder_next_idx -= 1
            if ev.get("event") == "BTD_ORDER" and state.btd_orders_done > 0:
                state.btd_orders_done -= 1
            logging.error(
                "Market buy failed; rolled back expected position (qty %.8f, quote %.8f): %s",
                qty,
                quote_amt,
                exc,
            )
            record_history(
                {
                    "ts": ts,
                    "event": "BUY_FAILED",
                    "reason": str(exc),
                    "qty": qty,
                    "quote": quote_amt,
                }
            )

        def rollback_failed_sah(ev: Dict[str, object], ts: datetime, exc: Exception) -> None:
            qty = float(ev.get("qty") or ev.get("order_qty") or 0.0)
            cost_share = float(ev.get("cost_share") or 0.0)
            if qty > 0:
                state.Q += qty
            if cost_share > 0:
                state.C += cost_share
            logging.error(
                "SAH sell failed; restored qty %.8f (cost share %.8f): %s",
                qty,
                cost_share,
                exc,
            )
            record_history(
                {
                    "ts": ts,
                    "event": "SAH_FAILED",
                    "reason": str(exc),
                    "qty": qty,
                    "cost_share": cost_share,
                }
            )

        def build_status_snapshot(current_price: float, ts: datetime) -> Dict[str, object]:
            p_be = state.P_BE()
            floor_price = state.floor_price(p_be) if p_be is not None else None
            avg_price = (state.C / state.Q) if state.Q > 0 else None
            unrealized_pnl = None
            if state.Q > 0:
                proceeds = current_price * state.Q * (1 - state.fees_sell)
                unrealized_pnl = proceeds - state.C
            next_buy = (
                state.ladder_prices[state.ladder_next_idx]
                if state.ladder_next_idx < len(state.ladder_prices)
                else None
            )
            next_buy_quote = (
                state.ladder_amounts_quote[state.ladder_next_idx]
                if state.ladder_next_idx < len(state.ladder_amounts_quote)
                else None
            )
            next_sell_target = None
            if state.Q > 0:
                if state.phase == PHASE_TRAIL:
                    next_sell_target = floor_price
                else:
                    next_sell_target = state.TP_base
                    if next_sell_target is None and p_be is not None:
                        next_sell_target = p_be * (1 + state.trail.p_min)
            return {
                "timestamp": ts.isoformat(),
                "price": current_price,
                "phase": state.phase,
                "stage": state.stage,
                "qty": state.Q,
                "quote_spent": state.C,
                "avg_price": avg_price,
                "p_be": p_be,
                "floor_price": floor_price,
                "next_buy_price": next_buy,
                "next_buy_quote": next_buy_quote,
                "next_sell_price": next_sell_target,
                "ladder_index": state.ladder_next_idx,
                "ladder_total": len(state.ladder_prices),
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl_total": realized_pnl_total,
            }

        def serialise_status(status: Dict[str, object]) -> Dict[str, object]:
            serialised: Dict[str, object] = {}
            for key, val in status.items():
                if isinstance(val, datetime):
                    serialised[key] = val.isoformat()
                else:
                    serialised[key] = val
            return serialised

        def log_status(status: Dict[str, object]) -> None:
            parts = [
                f"price={status['price']:.4f}",
                f"phase={status['phase']}",
                f"stage={status['stage']}",
                f"qty={status['qty']:.6f}",
            ]
            if status["avg_price"] is not None:
                parts.append(f"avg={status['avg_price']:.4f}")
            if status["p_be"] is not None:
                parts.append(f"P_BE={status['p_be']:.4f}")
            if status["floor_price"] is not None:
                parts.append(f"floor={status['floor_price']:.4f}")
            if status["next_buy_price"] is not None:
                parts.append(f"next_buy={status['next_buy_price']:.4f}")
            if status["next_buy_quote"] is not None:
                parts.append(f"next_buy_q={status['next_buy_quote']:.2f}")
            if status["next_sell_price"] is not None:
                parts.append(f"sell_at={status['next_sell_price']:.4f}")
            if status["unrealized_pnl"] is not None:
                parts.append(f"unrealized={status['unrealized_pnl']:.2f}")
            parts.append(f"realized_total={status['realized_pnl_total']:.2f}")
            logging.info("Status | %s", " | ".join(parts))

        startup_status = build_status_snapshot(latest["close"], latest["timestamp"])
        log_status(startup_status)
        last_status = serialise_status(startup_status)
        control.update_status(pair_symbol, last_status)
        write_savepoint(
            savepoint_dir,
            pair_cfg.symbol,
            state,
            last_open_time=last_open_time,
            event_log=event_log,
            activity_history=activity_history,
            realized_pnl_total=realized_pnl_total,
            latest_price=latest["close"],
            status=startup_status,
        )

        while True:
            try:
                if control.is_paused(pair_symbol):
                    paused_status = {**last_status, "note": "paused"}
                    last_status = paused_status
                    control.update_status(pair_symbol, paused_status)
                    time.sleep(args.poll_seconds)
                    continue
                kline = fetch_previous_closed(pair_cfg.symbol, pair_cfg.interval, base_url)
                if not kline:
                    time.sleep(args.poll_seconds)
                    continue
                if kline["open_time"] == last_open_time:
                    time.sleep(args.poll_seconds)
                    continue

                ts = kline["timestamp"]
                res = state.on_bar(
                    ts,
                    kline["open"],
                    kline["high"],
                    kline["low"],
                    kline["close"],
                    kline["volume"],
                    event_log,
                )
                new_events = event_log[event_ptr:]
                event_ptr = len(event_log)
                persist_event_log_delta()

                for ev in new_events:
                    evt = ev.get("event")
                    if evt == "BUY":
                        quote_amt = float(ev.get("amt_q", 0.0))
                        logging.info("BUY ladder fill at %.4f for quote %.2f", ev.get("price"), quote_amt)
                        if client and quote_amt > 0:
                            try:
                                available_quote = client.get_free_balance(pair_cfg.quote)
                            except Exception as exc:  # pylint: disable=broad-except
                                logging.warning("Unable to fetch %s balance: %s", pair_cfg.quote, exc)
                                available_quote = None

                            quote_amt, aborted = clamp_buy_to_available(
                                ev, available_quote, ts, "initial check"
                            )
                            if aborted:
                                continue

                            try:
                                refreshed_available = client.get_free_balance(pair_cfg.quote)
                            except Exception as exc:  # pylint: disable=broad-except
                                logging.warning(
                                    "Unable to refresh %s balance before buy: %s",
                                    pair_cfg.quote,
                                    exc,
                                )
                                refreshed_available = None

                            quote_amt, aborted = clamp_buy_to_available(
                                ev, refreshed_available, ts, "pre-submit refresh"
                            )
                            if aborted:
                                continue

                            if quote_amt <= 0:
                                rollback_failed_buy(ev, ts, RuntimeError("No purchasable quote after clamping"))
                                record_history(ev)
                                continue

                            try:
                                response = client.market_buy_quote(pair_cfg.symbol, quote_amt)
                            except requests.RequestException as exc:
                                rollback_failed_buy(ev, ts, exc)
                                record_history(ev)
                                continue
                            logging.info("Order response: %s", json.dumps(response))
                            executed_qty = float(response.get("executedQty") or 0.0)
                            fills = response.get("fills") or []
                            commission_base = 0.0
                            if base_asset:
                                for f in fills:
                                    try:
                                        if str(f.get("commissionAsset", "")).upper() == base_asset:
                                            commission_base += float(f.get("commission", 0.0))
                                    except (TypeError, ValueError):
                                        continue
                            net_qty = max(executed_qty - commission_base, 0.0)
                            expected_qty = float(ev.get("q") or ev.get("qty") or 0.0)
                            if net_qty < expected_qty:
                                adjustment = net_qty - expected_qty
                                state.Q += adjustment
                                logging.info(
                                    "Adjusted position for commission: expected %.8f, net %.8f, delta %.8f",
                                    expected_qty,
                                    net_qty,
                                    adjustment,
                                )
                        record_history(ev)
                    else:
                        record_history(ev)
                        if evt == "BTD_ORDER":
                            logging.info(
                                "BTD order suggested at %.4f for quote %.2f",
                                ev.get("order_price"),
                                ev.get("order_quote", 0.0),
                            )
                        elif evt == "SELL":
                            logging.info("Strategy logged SELL event: %s", ev)

                if res is not None:
                    qty = float(res.get("qty") or 0.0)
                    sell_qty = qty
                    logging.info(
                        "Exit signal %s at %.4f (qty %.6f, pnl %.2f)",
                        res.get("reason"),
                        res.get("sell_price"),
                        sell_qty,
                        res.get("pnl", 0.0),
                    )
                    realized_pnl = float(res.get("pnl") or 0.0)
                    if client and sell_qty > 0:
                        if base_asset:
                            try:
                                available = client.get_free_balance(base_asset)
                                if available < sell_qty:
                                    logging.warning(
                                        "Requested sell qty %.6f exceeds available %.6f; clamping",
                                        sell_qty,
                                        available,
                                    )
                                    sell_qty = available
                            except Exception as exc:
                                logging.warning("Unable to fetch %s balance: %s", base_asset, exc)
                        if sell_qty > 0:
                            chunk_count = sell_chunks if (sell_chunks > 1 and (realized_pnl > 0 or not sell_profit_only)) else 1
                            chunk_count = max(chunk_count, 1)
                            chunk_size = sell_qty / chunk_count
                            sold_total = 0.0
                            for idx in range(chunk_count):
                                qty_chunk = chunk_size if idx < chunk_count - 1 else sell_qty - sold_total
                                if qty_chunk <= 0:
                                    continue
                                response = client.market_sell(pair_cfg.symbol, qty_chunk)
                                sold_total += qty_chunk
                                logging.info(
                                    "Sell response (%s/%s): %s",
                                    idx + 1,
                                    chunk_count,
                                    json.dumps(response),
                                )
                                if sell_chunk_delay > 0 and idx < chunk_count - 1:
                                    time.sleep(sell_chunk_delay)
                            sell_qty = sold_total
                    if qty > 0 and sell_qty != qty:
                        realized_pnl *= sell_qty / qty
                    realized_pnl_total += realized_pnl
                    record_history(
                        {
                            "ts": ts,
                            "event": "EXIT",
                            "reason": res.get("reason"),
                            "sell_price": res.get("sell_price"),
                            "qty": qty,
                            "pnl": realized_pnl,
                            "realized_pnl_total": realized_pnl_total,
                        }
                    )
                    reset_round(state, kline["close"], ts)
                    event_log.clear()
                    event_ptr = 0
                    event_persist_count = 0

                last_open_time = kline["open_time"]
                status_snapshot = build_status_snapshot(kline["close"], ts)
                log_status(status_snapshot)
                last_status = serialise_status(status_snapshot)
                control.update_status(pair_symbol, last_status)
                write_savepoint(
                    savepoint_dir,
                    pair_cfg.symbol,
                    state,
                    last_open_time=last_open_time,
                    event_log=event_log,
                    activity_history=activity_history,
                    realized_pnl_total=realized_pnl_total,
                    latest_price=kline["close"],
                    status=status_snapshot,
                )

                sleep_time = max(args.poll_seconds, interval_ms / 1000.0 / 2)
                time.sleep(sleep_time)
            except requests.RequestException as exc:
                logging.error("Network error: %s", exc)
                time.sleep(args.poll_seconds)
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Unexpected error: %s", exc)
                time.sleep(args.poll_seconds)

    symbol_filter = args.symbol.upper() if args.symbol else None
    selected_pairs = [p for p in pairs_raw if symbol_filter is None or p["symbol"].upper() == symbol_filter]
    if not selected_pairs:
        raise SystemExit(f"Symbol {symbol_filter} not found in config")

    control = ControlCenter([p["symbol"].upper() for p in selected_pairs])
    start_http_server(control, args.http_port)

    with ThreadPoolExecutor(max_workers=len(selected_pairs)) as executor:
        futures = [executor.submit(run_pair, pair_raw) for pair_raw in selected_pairs]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()