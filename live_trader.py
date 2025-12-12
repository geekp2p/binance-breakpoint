import argparse
import csv
import json
import logging
import math
import os
import time
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Mapping

import requests
from urllib.parse import parse_qs, urlparse

from main import load_config, fee_multiplier_from_general
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
from src.profit_allocator import ProfitAllocator, ProfitRecycleConfig


MAX_HISTORY_ENTRIES = 200


def build_pair_config(raw: Dict, general: Dict) -> PairConfig:
    fee_multiplier = fee_multiplier_from_general(general)
    return PairConfig(
        symbol=raw["symbol"],
        quote=raw.get("quote", "QUOTE"),
        b_alloc=float(raw["b_alloc"]),
        source=raw.get("source", "binance"),
        interval=raw.get("interval", "1m"),
        lookback_days=float(raw.get("lookback_days", 1)) if raw.get("lookback_days") is not None else None,
        start=raw.get("start", ""),
        end=raw.get("end", ""),
        fees_taker=float(raw["fees"]["taker"]) * fee_multiplier,
        fees_maker=float(raw["fees"]["maker"]) * fee_multiplier,
        buy_d=float(raw["buy_ladder"]["d_buy"]),
        buy_m=float(raw["buy_ladder"]["m_buy"]),
        buy_n=int(raw["buy_ladder"]["n_steps"]),
        buy_spacing=str(raw["buy_ladder"].get("spacing_mode", "geometric")),
        buy_multipliers=raw["buy_ladder"].get("d_multipliers"),
        buy_max_drop=float(raw["buy_ladder"].get("max_step_drop", 0.25)),
        buy_size_mode=str(raw["buy_ladder"].get("size_mode", "geometric")),
        buy_gap_mode=str(raw["buy_ladder"].get("gap_mode", "additive")),
        buy_gap_factor=float(raw["buy_ladder"].get("gap_factor", 1.0)),
        buy_base_order_quote=raw["buy_ladder"].get("base_order_quote"),
        buy_max_quote_per_leg=float(raw["buy_ladder"].get("max_quote_per_leg", 0.0)),
        buy_max_total_quote=float(raw["buy_ladder"].get("max_total_quote", 0.0)),
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
        micro=raw.get("features", {}).get("micro_oscillation", {"enabled": False}),
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
        self._sell_requested: Dict[str, threading.Event] = {sym: threading.Event() for sym in symbols}
        self._logs: deque[Dict[str, object]] = deque(maxlen=1000)

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        with self._lock:
            return {
                sym: {
                    **status,
                    "paused": self._paused[sym].is_set(),
                    "manual_exit": self._sell_requested[sym].is_set(),
                }
                for sym, status in self._status.items()
            }

    def logs(self, limit: int = 1000) -> List[Dict[str, object]]:
        if limit <= 0:
            return []
        limit = min(limit, 1000)
        with self._lock:
            items = list(self._logs)
        return items[-limit:]

    def add_log(
        self,
        message: str,
        *,
        symbol: Optional[str] = None,
        kind: str = "info",
        ts: Optional[datetime] = None,
    ) -> None:
        timestamp = ts or datetime.now(timezone.utc)
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now(timezone.utc)
        entry = {
            "ts": timestamp.isoformat(),
            "msg": message,
            "symbol": symbol,
            "kind": kind,
        }
        with self._lock:
            self._logs.append(entry)

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

    def request_sell(self, symbol: Optional[str] = None) -> List[str]:
        targets = [symbol] if symbol else list(self._sell_requested.keys())
        requested: List[str] = []
        for sym in targets:
            ev = self._sell_requested.get(sym)
            if ev:
                ev.set()
                requested.append(sym)
        return requested

    def consume_sell_request(self, symbol: str) -> bool:
        ev = self._sell_requested.get(symbol)
        if ev and ev.is_set():
            ev.clear()
            return True
        return False


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
            params = parse_qs(parsed.query)
            if parsed.path == "/health":
                snapshot = control.snapshot()

                def _as_float(val: object, default: Optional[float] = 0.0) -> Optional[float]:
                    try:
                        num = float(val)  # type: ignore[arg-type]
                        if math.isnan(num) or math.isinf(num):
                            return default
                        return num
                    except (TypeError, ValueError):
                        return default

                def _merge_other(target: Dict[str, float], source: Dict[str, object]) -> None:
                    for asset, amount in (source or {}).items():
                        try:
                            target[asset] = target.get(asset, 0.0) + float(amount)
                        except (TypeError, ValueError):
                            continue

                pairs: Dict[str, Dict[str, object]] = {}
                totals = {
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "pnl_total": 0.0,
                    "profit_reserve": {"pending_quote": 0.0, "holdings_qty": 0.0, "quote_spent": 0.0},
                    "bnb_reserve": {"pending_quote": 0.0, "holdings_qty": 0.0},
                    "fees_paid": {"quote": 0.0, "bnb": 0.0, "other": {}},
                }

                for sym, raw in snapshot.items():
                    profit_reserve = (
                        raw.get("profit_reserve")
                        or raw.get("profit_reserve_coin")
                        or {"pending_quote": 0.0, "holdings_qty": 0.0, "quote_spent": 0.0}
                    )
                    bnb_reserve = raw.get("bnb_reserve") or raw.get("bnb_reserve_for_fees") or {
                        "pending_quote": 0.0,
                        "holdings_qty": 0.0,
                    }
                    fees_paid = raw.get("fees_paid") or {"pair": {}, "totals": {}}

                    realized = _as_float(raw.get("realized_pnl"), _as_float(raw.get("realized_pnl_total")))
                    unrealized_raw = raw.get("unrealized_pnl")
                    unrealized = None
                    if isinstance(unrealized_raw, (int, float)):
                        unrealized = _as_float(unrealized_raw, default=None)  # type: ignore[arg-type]
                    pnl_total = realized + (unrealized or 0.0)

                    pair_payload = {
                        **raw,
                        "realized_pnl": realized,
                        "unrealized_pnl": unrealized,
                        "pnl_total": pnl_total,
                        "profit_reserve": profit_reserve,
                        "bnb_reserve": bnb_reserve,
                        "fees_paid": fees_paid,
                    }
                    pairs[sym] = pair_payload

                    totals["realized_pnl"] += realized
                    totals["unrealized_pnl"] += unrealized or 0.0
                    totals["profit_reserve"]["pending_quote"] += _as_float(profit_reserve.get("pending_quote"))
                    totals["profit_reserve"]["holdings_qty"] += _as_float(profit_reserve.get("holdings_qty"))
                    totals["profit_reserve"]["quote_spent"] += _as_float(profit_reserve.get("quote_spent"))
                    totals["bnb_reserve"]["pending_quote"] += _as_float(bnb_reserve.get("pending_quote"))
                    totals["bnb_reserve"]["holdings_qty"] += _as_float(bnb_reserve.get("holdings_qty"))

                    fee_totals = fees_paid.get("totals") if isinstance(fees_paid, dict) else {}
                    totals["fees_paid"]["quote"] += _as_float((fee_totals or {}).get("quote"))
                    totals["fees_paid"]["bnb"] += _as_float((fee_totals or {}).get("bnb"))
                    totals_other = totals["fees_paid"].setdefault("other", {})
                    _merge_other(totals_other, (fee_totals or {}).get("other", {}))

                totals["pnl_total"] = totals["realized_pnl"] + totals["unrealized_pnl"]
                log_limit_raw = (params.get("log_limit") or [None])[0]
                log_limit = 1000
                if log_limit_raw is not None:
                    try:
                        log_limit = max(0, min(1000, int(log_limit_raw)))
                    except ValueError:
                        log_limit = 1000

                payload = {
                    "status": "ok",
                    "pairs": pairs,
                    "totals": totals,
                    "logs": control.logs(log_limit),
                }
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
            if parsed.path == "/sell":
                affected = control.request_sell(symbol)
                self._json(200, {"status": "sell_requested", "pairs": affected})
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
    profit_cfg = ProfitRecycleConfig.from_dict(general.get("profit_recycling", {}))

    pairs_raw = cfg.get("pairs", [])
    if not pairs_raw:
        raise SystemExit("No pairs defined in config")

    sell_scale_out_cfg = (general.get("sell_scale_out") or {})
    sell_chunks = max(int(sell_scale_out_cfg.get("chunks", 1)), 1)
    sell_chunk_delay = float(sell_scale_out_cfg.get("delay_seconds", 0.0))
    sell_profit_only = bool(sell_scale_out_cfg.get("profit_only", True))

    profit_allocator = ProfitAllocator(profit_cfg, Path(args.savepoint_dir))

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
            expected_steps = state._effective_n_steps()
            existing_steps = len(state.ladder_prices)
            if existing_steps != expected_steps:
                base_price = state.anchor_base_price or state.P0 or latest["close"]
                state.rebuild_ladder(
                    base_price,
                    ts=latest["timestamp"],
                    log_events=event_log,
                    reason="SAVEPOINT_REBUILD",
                    preserve_progress=True,
                )
                logging.info(
                    "Rebuilt ladder from savepoint to %s steps (was %s) using base %.4f",
                    expected_steps,
                    existing_steps,
                    base_price,
                )
        else:
            reset_round(state, latest["close"], latest["timestamp"])

        baseline_qty = float(savepoint_payload.get("baseline_qty") or 0.0) if savepoint_payload else 0.0
        baseline_cost = float(savepoint_payload.get("baseline_cost") or 0.0) if savepoint_payload else 0.0

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
        discount_symbol = (profit_cfg.discount_symbol or pair_cfg.symbol).upper()
        last_close_price = float(latest["close"])

        def price_lookup(symbol: str) -> float:
            if client:
                try:
                    return client.get_ticker_price(symbol)
                except Exception as exc:  # pylint: disable=broad-except
                    logging.warning("Unable to fetch ticker for %s: %s", symbol, exc)
            if symbol.upper() == pair_symbol:
                return last_close_price
            return 0.0

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

        profit_allocator.process_pending(
            discount_symbol,
            price_lookup=price_lookup,
            client=None if args.dry_run else client,
            dry_run=args.dry_run,
            activity_logger=record_history,
        )

        def _weighted_fill_price(order_resp: Dict[str, object]) -> Optional[float]:
            fills = order_resp.get("fills") if isinstance(order_resp, dict) else None
            if not fills:
                return None
            total_qty = 0.0
            total_quote = 0.0
            for f in fills:
                try:
                    qty = float(f.get("qty") or f.get("quantity") or 0.0)
                    price = float(f.get("price") or 0.0)
                except (TypeError, ValueError):
                    continue
                total_qty += qty
                total_quote += qty * price
            if total_qty <= 0 or total_quote <= 0:
                return None
            return total_quote / total_qty

        def _normalise_buy_event(ev: Dict[str, object]) -> None:
            """Ensure buy events expose standard sizing keys.

            Ladder buys emit `amt_q`/`q`, while scalp/micro buys use
            `order_quote`/`qty`. Normalising here keeps downstream logic
            (clamping, logging, adjustments) consistent.
            """

            if "amt_q" not in ev:
                try:
                    ev["amt_q"] = float(ev.get("order_quote", 0.0))
                except (TypeError, ValueError):
                    ev["amt_q"] = 0.0

            if ev.get("q") is None and ev.get("qty") is None:
                try:
                    ev["q"] = float(ev.get("order_qty", 0.0))
                except (TypeError, ValueError):
                    ev["q"] = 0.0
            # Mirror back to `qty` for consistency with UI/history payloads
            if ev.get("qty") is None and ev.get("q") is not None:
                ev["qty"] = ev["q"]

        def _execute_micro_exit(ev: Dict[str, object]) -> None:
            nonlocal realized_pnl_total
            qty = float(ev.get("qty") or 0.0)
            if qty <= 0:
                record_history(ev)
                return

            target_price = float(ev.get("price") or 0.0)
            cost = float(ev.get("cost") or 0.0)
            sell_qty = qty
            if client and base_asset:
                try:
                    available = client.get_free_balance(base_asset)
                    if available < sell_qty:
                        logging.warning(
                            "Requested micro sell qty %.6f exceeds available %.6f; clamping",
                            sell_qty,
                            available,
                        )
                        sell_qty = max(available, 0.0)
                except Exception as exc:  # pylint: disable=broad-except
                    logging.warning("Unable to fetch %s balance: %s", base_asset, exc)

            if sell_qty <= 0:
                record_history(ev)
                return

            executed_qty = sell_qty
            fill_price = target_price
            fills: list[dict] = []
            if client:
                response = client.market_sell(pair_cfg.symbol, sell_qty)
                logging.info("Micro sell response: %s", json.dumps(response))
                fills = response.get("fills") or []
                profit_allocator.record_fees_from_fills(pair_symbol, fills, pair_cfg.quote)
                executed_qty = float(response.get("executedQty") or sell_qty)
                fill_price = _weighted_fill_price(response) or target_price or last_close_price
            else:
                fill_price = target_price or last_close_price

            executed_qty = max(executed_qty, 0.0)
            if executed_qty <= 0:
                record_history(ev)
                return

            if executed_qty < qty:
                unsold = qty - executed_qty
                state.Q += unsold
                state.C += cost * (unsold / qty)

            proceeds = executed_qty * fill_price * (1 - state.fees_sell)
            effective_cost = cost * (executed_qty / qty)
            realized_pnl = proceeds - effective_cost
            realized_pnl_total += realized_pnl

            ev["executed_qty"] = executed_qty
            ev["fill_price"] = fill_price
            ev["proceeds"] = proceeds
            ev["pnl"] = realized_pnl
            ev["realized_pnl_total"] = realized_pnl_total
            if fills:
                ev["fills"] = fills

            if realized_pnl > 0:
                profit_allocator.allocate_profit(
                    realized_pnl,
                    pair_symbol=pair_symbol,
                    discount_symbol=discount_symbol,
                    price_lookup=price_lookup,
                    client=None if args.dry_run else client,
                    dry_run=args.dry_run,
                    activity_logger=record_history,
                )

            record_history(ev)

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

        def net_position() -> tuple[float, float]:
            qty = max(state.Q - baseline_qty, 0.0)
            cost = max(state.C - baseline_cost, 0.0)
            # Avoid reporting a cost when there is no position (can happen if cost
            # bookkeeping drifts slightly after a full exit).
            if qty <= 0:
                return 0.0, 0.0
            return qty, cost

        def build_status_snapshot(current_price: float, ts: datetime) -> Dict[str, object]:
            net_qty, net_cost = net_position()
            p_be = None
            if net_qty > 0:
                p_be = net_cost / (net_qty * (1 - state.fees_sell))
            floor_price = state.floor_price(p_be) if p_be is not None else None
            avg_price = (net_cost / net_qty) if net_qty > 0 else None
            unrealized_pnl = None
            if net_qty > 0:
                proceeds = current_price * net_qty * (1 - state.fees_sell)
                unrealized_pnl = proceeds - net_cost
            realized_pnl = realized_pnl_total
            pnl_total = realized_pnl_total + (unrealized_pnl or 0.0)
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
            ladder_total = len(state.ladder_prices)
            ladder_progress = state.ladder_next_idx
            if net_qty <= 0 and net_cost <= 0:
                ladder_progress = 0
            ladder_progress = min(max(ladder_progress, 0), ladder_total)
            ladder_legs: List[Dict[str, object]] = []
            ladder_spent_quote = None
            if state.ladder_amounts_quote:
                ladder_spent_quote = sum(state.ladder_amounts_quote[: ladder_progress])
            for idx, (price, quote) in enumerate(zip(state.ladder_prices, state.ladder_amounts_quote)):
                ladder_legs.append(
                    {
                        "idx": idx + 1,
                        "price": price,
                        "quote": quote,
                        "filled": idx < ladder_progress,
                    }
                )
            btd_target = None
            if state.btd_armed and state.btd_last_bottom is not None:
                btd_target = state.btd_last_bottom * (1 + state.btd.limit_offset)

            micro_snapshot = state._micro_snapshot()
            micro_next_buy = None
            if micro_snapshot and micro_snapshot.get("ready"):
                band_span = micro_snapshot.get("high", 0.0) - micro_snapshot.get("low", 0.0)
                if band_span > 0:
                    micro_next_buy = micro_snapshot.get("low", 0.0) + band_span * max(
                        min(getattr(state.micro, "entry_band_pct", 0.0), 1.0), 0.0
                    )
            micro_position_qty = sum(float(pos.get("qty", 0.0)) for pos in state.micro_positions)
            micro_position_cost = sum(float(pos.get("cost", 0.0)) for pos in state.micro_positions)
            micro_targets = [float(pos.get("target", 0.0)) for pos in state.micro_positions if pos.get("target")]
            micro_stops = [float(pos.get("stop", 0.0)) for pos in state.micro_positions if pos.get("stop")]
            micro_next_sell = min(micro_targets) if micro_targets else None
            micro_next_stop = max(micro_stops) if micro_stops else None
            micro_state = "waiting_sell" if micro_position_qty > 0 else "idle"
            if micro_state == "idle" and state.bar_index < state.micro_cooldown_until_bar:
                micro_state = "cooldown"
            if micro_state == "idle" and micro_snapshot and micro_snapshot.get("ready"):
                micro_state = "ready_to_buy"

            next_sell_target = None
            if state.Q > 0:
                if state.phase == PHASE_TRAIL:
                    next_sell_target = floor_price
                else:
                    next_sell_target = state.TP_base
                    if next_sell_target is None and p_be is not None:
                        next_sell_target = p_be * (1 + state.trail.p_min)
            reserves = profit_allocator.reserves_snapshot(discount_symbol)
            fees = profit_allocator.fees_snapshot(pair_symbol)
            return {
                "timestamp": ts.isoformat(),
                "price": current_price,
                "phase": state.phase,
                "stage": state.stage,
                "qty": net_qty,
                "quote_spent": net_cost,
                "avg_price": avg_price,
                "p_be": p_be,
                "floor_price": floor_price,
                "next_buy_price": next_buy,
                "next_buy_quote": next_buy_quote,
                "next_sell_price": next_sell_target,
                "ladder_index": ladder_progress,
                "ladder_total": ladder_total,
                "ladder_legs": ladder_legs,
                "ladder_next_quote": next_buy_quote,
                "ladder_spent_quote": ladder_spent_quote,
                "btd_armed": state.btd_armed,
                "btd_target_price": btd_target,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "pnl_total": pnl_total,
                "realized_pnl_total": realized_pnl_total,
                "profit_reserve_coin": reserves.get("profit_reserve"),
                "bnb_reserve_for_fees": reserves.get("bnb_reserve"),
                "fees_paid": fees,
                "scalp_enabled": bool(state.scalp.enabled),
                "scalp_positions": len(state.scalp_positions),
                "micro_enabled": bool(state.micro.enabled),
                "micro_positions": len(state.micro_positions),
                "micro_position_qty": micro_position_qty,
                "micro_position_cost": micro_position_cost,
                "micro_swings": state.micro_swings,
                "micro_cooldown_until_bar": state.micro_cooldown_until_bar,
                "micro_last_exit_price": state.micro_last_exit_price,
                "micro_loss_recovery_pct": state.micro_loss_recovery_pct,
                "micro_next_buy_price": micro_next_buy,
                "micro_next_sell_price": micro_next_sell,
                "micro_next_stop_price": micro_next_stop,
                "micro_state": micro_state,
            }

        def serialise_status(status: Dict[str, object]) -> Dict[str, object]:
            serialised: Dict[str, object] = {}
            for key, val in status.items():
                if isinstance(val, datetime):
                    serialised[key] = val.isoformat()
                else:
                    serialised[key] = val
            return serialised

        last_status_line = ""
        last_pnl_compact = ""
        last_pnl_total = ""

        def log_status(status: Dict[str, object]) -> None:
            nonlocal last_status_line, last_pnl_compact, last_pnl_total
            parts = [
                f"p={status['price']:.4f}",
                f"ph={status['phase']}#{status['stage']}",
                f"q={status['qty']:.6f}",
            ]
            if status["avg_price"] is not None:
                parts.append(f"avg={status['avg_price']:.4f}")
            if status["p_be"] is not None:
                parts.append(f"be={status['p_be']:.4f}")
            if status["floor_price"] is not None:
                parts.append(f"flr={status['floor_price']:.4f}")
            if status["next_buy_price"] is not None:
                parts.append(f"nb={status['next_buy_price']:.4f}")
            if status["next_buy_quote"] is not None:
                parts.append(f"nbq={status['next_buy_quote']:.2f}")
            if status["next_sell_price"] is not None:
                parts.append(f"ns={status['next_sell_price']:.4f}")
            if status["unrealized_pnl"] is not None:
                parts.append(f"u={status['unrealized_pnl']:.2f}")
            parts.append(f"rT={status['realized_pnl_total']:.2f}")
            scalp_enabled = bool(status.get("scalp_enabled"))
            parts.append(
                "scalp={state}{count}".format(
                    state="on" if scalp_enabled else "off",
                    count=f"({status.get('scalp_positions', 0)})",
                )
            )

            micro_enabled = bool(status.get("micro_enabled"))
            parts.append(
                "micro={state} pos={pos} sw={swings} cd={cd} nb={nb} ns={ns} st={mode}".format(
                    state="on" if micro_enabled else "off",
                    pos=status.get("micro_positions", 0),
                    swings=status.get("micro_swings", 0),
                    cd=status.get("micro_cooldown_until_bar"),
                    nb=status.get("micro_next_buy_price"),
                    ns=status.get("micro_next_sell_price"),
                    mode=status.get("micro_state", "-"),
                )
            )
            profit_reserve = status.get("profit_reserve_coin") or {}
            bnb_reserve = status.get("bnb_reserve_for_fees") or {}
            fees_paid = status.get("fees_paid") or {}
            try:
                pending_coin = float(profit_reserve.get("pending_quote", 0.0))
                pending_bnb = float(bnb_reserve.get("pending_quote", 0.0))
                parts.append(f"reserve_coin={pending_coin:.2f}")
                parts.append(f"reserve_bnb={pending_bnb:.2f}")
            except Exception:
                logging.debug("Unable to format reserve snapshot")
            try:
                pair_fees = fees_paid.get("pair", {})
                total_fees = fees_paid.get("totals", {})
                parts.append(
                    "fees_pair=q={:.4f},bnb={:.4f},other={}".format(
                        float(pair_fees.get("quote", 0.0)),
                        float(pair_fees.get("bnb", 0.0)),
                        dict(pair_fees.get("other", {})),
                    )
                )
                parts.append(
                    "fees_total=q={:.4f},bnb={:.4f},other={}".format(
                        float(total_fees.get("quote", 0.0)),
                        float(total_fees.get("bnb", 0.0)),
                        dict(total_fees.get("other", {})),
                    )
                )
            except Exception:
                logging.debug("Unable to format fee snapshot")
            status_line = f"ST {pair_cfg.symbol} | " + " | ".join(parts)
            if status_line != last_status_line:
                logging.info(status_line)
                last_status_line = status_line

            def _format_pnl(val: Optional[float]) -> str:
                return f"{val:.2f}" if val is not None else "-"

            snapshot = control.snapshot()
            if not snapshot:
                return

            per_pair_parts = []
            total_realized = 0.0
            total_unrealized = 0.0
            for sym, snap in sorted(snapshot.items()):
                realized = float(snap.get("realized_pnl") or snap.get("realized_pnl_total") or 0.0)
                unrealized = snap.get("unrealized_pnl")
                total = realized + (unrealized or 0.0)
                per_pair_parts.append(
                    f"{sym}:r={_format_pnl(realized)},u={_format_pnl(unrealized)},t={_format_pnl(total)}"
                )
                total_realized += realized
                total_unrealized += unrealized or 0.0

            total_all = total_realized + total_unrealized
            compact_line = "PnL | " + " | ".join(per_pair_parts)
            total_line = "PnL sum | r=%s | u=%s | t=%s" % (
                _format_pnl(total_realized),
                _format_pnl(total_unrealized),
                _format_pnl(total_all),
            )

            if compact_line != last_pnl_compact or total_line != last_pnl_total:
                logging.info(compact_line)
                logging.info(total_line)
                last_pnl_compact = compact_line
                last_pnl_total = total_line

        def _extract_executed_quote(response: Mapping[str, object]) -> float:
            fills = response.get("fills") or []
            quote_spent = response.get("cummulativeQuoteQty")
            try:
                if quote_spent is not None:
                    return float(quote_spent)
            except (TypeError, ValueError):
                pass
            try:
                return sum(
                    float(f.get("price", 0.0)) * float(f.get("qty", 0.0)) for f in fills
                )
            except Exception:
                return 0.0

        def log_order_summary(side: str, response: Mapping[str, object]) -> None:
            """Log a concise summary of a Binance order response."""

            executed_qty = float(response.get("executedQty") or 0.0)
            quote_val = _extract_executed_quote(response)
            avg_price = quote_val / executed_qty if executed_qty else 0.0
            control.add_log(
                f"TXN {side.upper()} q={executed_qty:.4f} @ {avg_price:.4f} ({quote_val:.2f} {pair_cfg.quote}) {response.get('status')}",
                symbol=pair_cfg.symbol,
                kind="order",
            )
            logging.info(
                "TXN %s %s | qty=%.6f | avg=%.4f | quote=%.2f %s | status=%s",
                side.upper(),
                pair_cfg.symbol,
                executed_qty,
                avg_price,
                quote_val,
                pair_cfg.quote,
                response.get("status"),
            )

        def _coerce_ts(ts_val: object) -> datetime:
            if isinstance(ts_val, datetime):
                return ts_val
            if isinstance(ts_val, (int, float)):
                try:
                    return datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
                except Exception:
                    return datetime.now(timezone.utc)
            if isinstance(ts_val, str):
                try:
                    return datetime.fromisoformat(ts_val)
                except ValueError:
                    return datetime.now(timezone.utc)
            return datetime.now(timezone.utc)

        def _log_event(ev: Dict[str, object]) -> None:
            evt = str(ev.get("event", "")).upper()
            ts_val = _coerce_ts(ev.get("ts"))
            label = {
                "BUY": "BY",
                "SCALP_BUY": "SC",
                "MICRO_BUY": "MB",
                "SELL": "SL",
                "ENTER_TRAIL": "TRL",
                "STAGE_UP": "STG",
                "TIME_TIGHTEN": "TITE",
            }.get(evt, evt or "EVT")

            parts = [label]
            price = ev.get("price")
            qty = ev.get("qty") or ev.get("q")
            amt_q = ev.get("amt_q")
            pnl = ev.get("pnl") or ev.get("pnl_total")
            note = ev.get("note") or ev.get("reason")
            if isinstance(price, (int, float)):
                parts.append(f"@{price:.4f}")
            if isinstance(qty, (int, float)) and qty:
                parts.append(f"q{float(qty):.4f}")
            if isinstance(amt_q, (int, float)) and amt_q:
                parts.append(f"Q{float(amt_q):.2f}")
            if isinstance(pnl, (int, float)):
                parts.append(f"pnl={float(pnl):.2f}")
            if note:
                parts.append(str(note))

            control.add_log(" ".join(parts), symbol=pair_cfg.symbol, kind="event", ts=ts_val)

        startup_status = build_status_snapshot(latest["close"], latest["timestamp"])
        log_status(startup_status)
        last_status = serialise_status(startup_status)
        control.update_status(pair_symbol, last_status)
        control.add_log(
            f"START {pair_cfg.symbol} {pair_cfg.interval} dry={'Y' if args.dry_run else 'N'}",
            symbol=pair_cfg.symbol,
            kind="status",
            ts=latest["timestamp"],
        )
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
            baseline_qty=baseline_qty,
            baseline_cost=baseline_cost,
        )

        def process_manual_sell(ts: datetime) -> bool:
            nonlocal last_status, realized_pnl_total, last_open_time, event_log, event_ptr, event_persist_count
            if not control.consume_sell_request(pair_symbol):
                return False

            qty = state.Q
            if qty <= 0:
                record_history({"ts": ts, "event": "MANUAL_EXIT_SKIPPED", "reason": "no_position"})
                return True

            executed_qty = qty
            avg_price: Optional[float] = None
            sell_response: Optional[Dict[str, object]] = None
            if client:
                try:
                    available_qty = executed_qty
                    if base_asset:
                        try:
                            available_qty = min(executed_qty, client.get_free_balance(base_asset))
                        except Exception as exc:  # pylint: disable=broad-except
                            logging.warning("Unable to fetch %s balance before manual sell: %s", base_asset, exc)
                    if available_qty <= 0:
                        record_history({"ts": ts, "event": "MANUAL_EXIT_SKIPPED", "reason": "no_available_balance"})
                        return True

                    sell_response = client.market_sell(pair_cfg.symbol, available_qty)
                    logging.info("Manual sell response: %s", json.dumps(sell_response))
                    executed_qty = float(sell_response.get("executedQty") or available_qty)
                    avg_price = _weighted_fill_price(sell_response)
                    commission_base = 0.0
                    fills = sell_response.get("fills") if isinstance(sell_response, dict) else None
                    profit_allocator.record_fees_from_fills(pair_symbol, fills, pair_cfg.quote)
                    if base_asset and fills:
                        for f in fills:
                            try:
                                if str(f.get("commissionAsset", "")).upper() == base_asset:
                                    commission_base += float(f.get("commission", 0.0))
                            except (TypeError, ValueError):
                                continue
                    executed_qty = max(executed_qty - commission_base, 0.0)
                except Exception as exc:  # pylint: disable=broad-except
                    logging.error("Manual sell failed: %s", exc)
                    record_history({"ts": ts, "event": "MANUAL_EXIT_FAILED", "reason": str(exc)})
                    return True
            else:
                logging.info("Dry-run manual exit for %s", pair_cfg.symbol)

            if executed_qty <= 0:
                record_history({"ts": ts, "event": "MANUAL_EXIT_SKIPPED", "reason": "no_executed_qty"})
                return True

            if avg_price is None:
                try:
                    avg_price = client.get_ticker_price(pair_cfg.symbol) if client else None
                except Exception as exc:  # pylint: disable=broad-except
                    logging.warning("Unable to fetch ticker price for manual exit: %s", exc)
            if avg_price is None or avg_price <= 0:
                avg_price = float(last_status.get("price") or latest["close"])

            gross_proceeds = avg_price * executed_qty
            fee_rate = max(state.fees_sell, 0.001)
            net_proceeds = gross_proceeds * (1 - fee_rate)
            cost_share = state.C * (executed_qty / qty) if qty > 0 else 0.0
            realized_pnl = net_proceeds - cost_share

            state.Q = max(state.Q - executed_qty, 0.0)
            state.C = max(state.C - cost_share, 0.0)
            realized_pnl_total += realized_pnl

            if realized_pnl > 0:
                profit_allocator.allocate_profit(
                    realized_pnl,
                    pair_symbol=pair_symbol,
                    discount_symbol=discount_symbol,
                    price_lookup=price_lookup,
                    client=None if args.dry_run else client,
                    dry_run=args.dry_run,
                    activity_logger=record_history,
                )

            record_history(
                {
                    "ts": ts,
                    "event": "MANUAL_EXIT",
                    "qty": executed_qty,
                    "price": avg_price,
                    "gross_proceeds": gross_proceeds,
                    "net_proceeds": net_proceeds,
                    "fee_rate": fee_rate,
                    "pnl": realized_pnl,
                    "realized_pnl_total": realized_pnl_total,
                }
            )
            control.add_log(
                f"MANUAL EXIT q={executed_qty:.4f} @ {avg_price:.4f} pnl={realized_pnl:.2f}",
                symbol=pair_cfg.symbol,
                kind="event",
                ts=ts,
            )

            reset_round(state, avg_price, ts)
            event_log.clear()
            event_ptr = 0
            event_persist_count = 0
            status_snapshot = build_status_snapshot(avg_price, ts)
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
                latest_price=avg_price,
                status=status_snapshot,
                baseline_qty=baseline_qty,
                baseline_cost=baseline_cost,
            )
            return True

        while True:
            try:
                if process_manual_sell(datetime.now(timezone.utc)):
                    time.sleep(args.poll_seconds)
                    continue
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
                last_close_price = float(kline["close"])
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
                    _log_event(ev)
                    evt = ev.get("event")
                    if evt == "BTD_ORDER":
                        # Auto-submit Buy-the-Dip orders instead of just logging them.
                        # Map the scaffolded event into a standard BUY flow so cost/qty
                        # booking, balance clamping, and history logging stay consistent
                        # with ladder/scalp/micro purchases.
                        actionable = dict(ev)
                        _normalise_buy_event(actionable)
                        prev_qty = state.Q
                        qty = float(actionable.get("q") or actionable.get("qty") or 0.0)
                        amt_q = float(actionable.get("amt_q", 0.0))
                        cost = amt_q * (1 + state.fees_buy)
                        state.Q += qty
                        state.C += cost
                        if prev_qty <= 0 < state.Q:
                            state.round_start_ts = ts
                        actionable["event"] = "BUY"
                        actionable["reason"] = "BUY_THE_DIP"
                        actionable["hint_price"] = actionable.get("order_price")
                        actionable["source_event"] = "BTD_ORDER"
                        ev = actionable
                        evt = "BUY"
                    elif evt == "SAH_ORDER":
                        # Auto-execute Sell-the-Rip orders by placing a market sell
                        # for the suggested quantity, then reconciling our tracked
                        # inventory/cost basis with the actual fill.
                        hinted_price = float(ev.get("order_price") or 0.0) or last_close_price
                        requested_qty = float(ev.get("order_qty") or ev.get("qty") or 0.0)
                        if requested_qty <= 0 or state.Q <= 0:
                            record_history(ev)
                            continue

                        sell_qty = min(requested_qty, state.Q)
                        if client and base_asset:
                            try:
                                available = client.get_free_balance(base_asset)
                                if available < sell_qty:
                                    logging.warning(
                                        "Requested SAH sell qty %.6f exceeds available %.6f; clamping",
                                        sell_qty,
                                        available,
                                    )
                                    sell_qty = max(available, 0.0)
                            except Exception as exc:  # pylint: disable=broad-except
                                logging.warning("Unable to fetch %s balance: %s", base_asset, exc)

                        if sell_qty <= 0:
                            record_history(ev)
                            continue

                        fills: list[dict] = []
                        executed_qty = sell_qty
                        fill_price = hinted_price
                        if client:
                            try:
                                response = client.market_sell(pair_cfg.symbol, sell_qty)
                            except requests.RequestException as exc:
                                logging.error("SAH market sell failed: %s", exc)
                                record_history({**ev, "event": "SAH_ORDER_FAILED", "reason": str(exc)})
                                continue
                            logging.info("SAH sell response: %s", json.dumps(response))
                            fills = response.get("fills") or []
                            profit_allocator.record_fees_from_fills(pair_symbol, fills, pair_cfg.quote)
                            executed_qty = float(response.get("executedQty") or sell_qty)
                            fill_price = _weighted_fill_price(response) or hinted_price

                        prev_qty = state.Q
                        cost_share = state.C * (executed_qty / prev_qty) if prev_qty > 0 else 0.0
                        proceeds_gross = executed_qty * fill_price
                        proceeds_net = proceeds_gross * (1 - state.fees_sell)
                        realized_pnl = proceeds_net - cost_share

                        state.Q = max(prev_qty - executed_qty, 0.0)
                        state.C = max(state.C - cost_share, 0.0)
                        realized_pnl_total += realized_pnl

                        sah_event = {
                            **ev,
                            "event": "SAH_SELL",
                            "executed_qty": executed_qty,
                            "fill_price": fill_price,
                            "gross_proceeds": proceeds_gross,
                            "net_proceeds": proceeds_net,
                            "pnl": realized_pnl,
                            "realized_pnl_total": realized_pnl_total,
                        }

                        if realized_pnl > 0:
                            profit_allocator.allocate_profit(
                                realized_pnl,
                                pair_symbol=pair_symbol,
                                discount_symbol=discount_symbol,
                                price_lookup=price_lookup,
                                client=None if args.dry_run else client,
                                dry_run=args.dry_run,
                                activity_logger=record_history,
                            )

                        record_history(sah_event)
                        continue
                    if evt in {"BUY", "SCALP_BUY", "MICRO_BUY"}:
                        _normalise_buy_event(ev)
                        quote_amt = float(ev.get("amt_q", 0.0))
                        logging.info(
                            "%s fill at %.4f for quote %.2f",
                            evt,
                            ev.get("price"),
                            quote_amt,
                        )
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
                            log_order_summary("BUY", response)
                            executed_qty = float(response.get("executedQty") or 0.0)
                            fills = response.get("fills") or []
                            profit_allocator.record_fees_from_fills(pair_symbol, fills, pair_cfg.quote)
                            commission_base = 0.0
                            if base_asset:
                                for f in fills:
                                    try:
                                        if str(f.get("commissionAsset", "")).upper() == base_asset:
                                            commission_base += float(f.get("commission", 0.0))
                                    except (TypeError, ValueError):
                                        continue
                            net_qty = max(executed_qty - commission_base, 0.0)
                            executed_quote = _extract_executed_quote(response)
                            expected_qty = float(ev.get("q") or ev.get("qty") or 0.0)
                            expected_quote = float(ev.get("amt_q") or 0.0)
                            if net_qty < expected_qty:
                                adjustment = net_qty - expected_qty
                                state.Q += adjustment
                                logging.info(
                                    "Adjusted position for commission: expected %.8f, net %.8f, delta %.8f",
                                    expected_qty,
                                    net_qty,
                                    adjustment,
                                )
                            # Align tracked position + cost with actual fill for micro buys.
                            # Strategy books the expected qty/cost optimistically before the
                            # exchange fill; adjust the most recent micro position to avoid
                            # drifting inventory/cost when Binance returns slightly different
                            # fill sizes.
                            if evt == "MICRO_BUY" and state.micro_positions:
                                pos = state.micro_positions[-1]
                                try:
                                    if expected_qty > 0 and net_qty > 0:
                                        scale = net_qty / expected_qty
                                        pos["qty"] *= scale
                                        ev["qty"] = net_qty
                                        ev["q"] = net_qty
                                    if expected_quote > 0 and executed_quote > 0:
                                        cost_scale = executed_quote / expected_quote
                                        pos["cost"] *= cost_scale
                                        # Keep book cost in sync with realized spend
                                        expected_cost = expected_quote * (1 + state.fees_buy)
                                        actual_cost = executed_quote * (1 + state.fees_buy)
                                        state.C += actual_cost - expected_cost
                                except Exception:
                                    logging.debug("Unable to align micro fill with on-ledger position", exc_info=True)
                        record_history(ev)
                    elif evt and evt.startswith("MICRO_") and evt != "MICRO_BUY":
                        _execute_micro_exit(ev)
                    else:
                        record_history(ev)
                        if evt == "BTD_ORDER":
                            logging.info(
                                "BTD order suggested at %.4f for quote %.2f",
                                ev.get("order_price"),
                                ev.get("order_quote", 0.0),
                            )
                        elif evt == "SAH_ORDER":
                            profit_allocator.sell_on_rip(
                                discount_symbol,
                                price_lookup=price_lookup,
                                client=None if args.dry_run else client,
                                dry_run=args.dry_run,
                                activity_logger=record_history,
                                hinted_price=float(ev.get("order_price") or 0.0),
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
                            all_fills = []
                            for idx in range(chunk_count):
                                qty_chunk = chunk_size if idx < chunk_count - 1 else sell_qty - sold_total
                                if qty_chunk <= 0:
                                    continue
                                response = client.market_sell(pair_cfg.symbol, qty_chunk)
                                all_fills.extend(response.get("fills") or [])
                                sold_total += qty_chunk
                                logging.info(
                                    "Sell response (%s/%s): %s",
                                    idx + 1,
                                    chunk_count,
                                    json.dumps(response),
                                )
                                log_order_summary("SELL", response)
                                if sell_chunk_delay > 0 and idx < chunk_count - 1:
                                    time.sleep(sell_chunk_delay)
                            sell_qty = sold_total
                            profit_allocator.record_fees_from_fills(pair_symbol, all_fills, pair_cfg.quote)
                    if qty > 0 and sell_qty != qty:
                        realized_pnl *= sell_qty / qty
                    realized_pnl_total += realized_pnl
                    if realized_pnl > 0:
                        profit_allocator.allocate_profit(
                            realized_pnl,
                            pair_symbol=pair_symbol,
                            discount_symbol=discount_symbol,
                            price_lookup=price_lookup,
                            client=None if args.dry_run else client,
                            dry_run=args.dry_run,
                            activity_logger=record_history,
                        )
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
                    baseline_qty=baseline_qty,
                    baseline_cost=baseline_cost,
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