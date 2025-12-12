import argparse
import io
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, Response, jsonify, request

from live_trader import build_pair_config
from main import load_config
from src.backtester import PairConfig, load_data_for_pair, run_backtest_for_pair

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# Lazily populated when `main()` loads the config. Kept as module-level state so that
# the simulator can be restarted from HTTP endpoints without re-reading the config.
GENERAL_CFG: Dict = {}
API_CFG: Dict = {}
PAIRS_RAW: List[Dict] = []
DEFAULT_SYMBOL: Optional[str] = None


class SimulationRunner:
    def __init__(
        self,
        pair_cfg: PairConfig,
        general_cfg: Dict,
        api_cfg: Dict,
        speed_multiplier: float = 24.0,
        lookback_days: float = 3.0,
    ) -> None:
        self.pair_cfg = pair_cfg
        self.general_cfg = general_cfg
        self.api_cfg = api_cfg
        self.speed_multiplier = max(speed_multiplier, 1.0)
        self.lookback_days = lookback_days

        self._lock = threading.Lock()
        self._paused = threading.Event()
        self._stopped = threading.Event()
        self._logs: List[Dict[str, object]] = []
        self._status: Dict[str, object] = {}
        self._trades: pd.DataFrame = pd.DataFrame()
        self._equity: pd.DataFrame = pd.DataFrame()
        self._trade_ptr = 0
        self._realized = 0.0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        df = load_data_for_pair(self.pair_cfg, self.general_cfg, self.api_cfg)
        if self.lookback_days:
            df = df.tail(int(self.lookback_days * 24 * 60))
        result = run_backtest_for_pair(df, self.pair_cfg)
        trades = result.get("trades")
        equity = result.get("equity")
        self._trades = trades if trades is not None else pd.DataFrame()
        self._equity = equity if equity is not None else pd.DataFrame()
        self._trade_ptr = 0
        self._realized = 0.0
        self._logs.clear()
        self._status.clear()
        self._stopped.clear()
        self._paused.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _append_log(self, msg: str, kind: str = "event", ts: Optional[pd.Timestamp] = None) -> None:
        entry = {
            "ts": (ts or pd.Timestamp.utcnow()).isoformat(),
            "msg": msg,
            "symbol": self.pair_cfg.symbol,
            "kind": kind,
        }
        with self._lock:
            self._logs.append(entry)
            if len(self._logs) > 1000:
                self._logs = self._logs[-1000:]

    def _sleep_for_bar(self, current_ts: datetime, next_ts: Optional[datetime]) -> None:
        if next_ts is None:
            return
        delta = (next_ts - current_ts).total_seconds()
        pause = max(delta / self.speed_multiplier, 0.05)
        for _ in range(int(pause / 0.1)):
            if self._stopped.is_set():
                return
            time.sleep(0.1)
        remaining = pause % 0.1
        if remaining > 0 and not self._stopped.is_set():
            time.sleep(remaining)

    def _loop(self) -> None:
        if self._equity.empty:
            logging.warning("No equity data to simulate")
            return
        eq = self._equity.reset_index(drop=True)
        trades = self._trades.reset_index(drop=True)
        trade_ts = pd.to_datetime(trades["ts"]) if not trades.empty else pd.Series([], dtype="datetime64[ns]")

        for idx, row in eq.iterrows():
            if self._stopped.is_set():
                break
            while self._paused.is_set() and not self._stopped.is_set():
                time.sleep(0.2)

            ts = pd.to_datetime(row["ts"])
            price = float(row.get("price", 0.0))
            qty = float(row.get("Q", 0.0))
            cost = float(row.get("C", 0.0))
            avg_price = cost / qty if qty > 0 else None
            unrealized = price * qty - cost if qty > 0 else None

            # Apply trades that happened up to this bar
            while self._trade_ptr < len(trades) and trade_ts.iloc[self._trade_ptr] <= ts:
                tr = trades.iloc[self._trade_ptr]
                pnl = tr.get("pnl")
                if isinstance(pnl, (int, float)):
                    self._realized += float(pnl)
                side = tr.get("side", "?")
                reason = tr.get("reason", "")
                self._append_log(f"{side} {tr.get('qty', '')} @ {tr.get('price', '')} {reason}", ts=trade_ts.iloc[self._trade_ptr])
                self._trade_ptr += 1

            snapshot = {
                "symbol": self.pair_cfg.symbol,
                "timestamp": ts.isoformat(),
                "price": price,
                "qty": qty,
                "avg_price": avg_price,
                "unrealized_pnl": unrealized,
                "realized_pnl_total": self._realized,
                "pnl_total": self._realized + (unrealized or 0.0),
                "phase": row.get("phase") or "SIM", 
                "ladder_prices": [],
                "ladder_amounts_quote": [],
                "ladder_next_idx": int(row.get("next_ladder_idx") or 0),
                "ladder_total": int(row.get("next_ladder_idx") or 0),
                "profit_reserve": {"pending_quote": 0.0, "holdings_qty": 0.0, "quote_spent": 0.0},
                "bnb_reserve": {"pending_quote": 0.0, "holdings_qty": 0.0},
                "fees_paid": {"pair": {}, "totals": {"quote": 0.0, "bnb": 0.0, "other": {}}},
                "next_buy_price": None,
                "buy_the_dip": None,
                "next_sell_price": None,
                "micro_state": "idle",
                "micro_positions": 0,
                "micro_position_qty": 0.0,
                "micro_swings": 0,
                "micro_next_buy_price": None,
                "micro_next_sell_price": None,
                "micro_cooldown_until_bar": 0,
                "quote_spent": cost,
                "baseline_qty": 0.0,
                "baseline_cost": 0.0,
            }
            with self._lock:
                self._status = snapshot

            next_ts = None
            if idx + 1 < len(eq):
                next_ts = pd.to_datetime(eq.iloc[idx + 1]["ts"])
            self._sleep_for_bar(ts.to_pydatetime(), next_ts.to_pydatetime() if next_ts is not None else None)

        self._append_log("Simulation completed", kind="status")

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._stopped.set()
        self._paused.clear()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            pair = self._status.copy()
            logs = list(self._logs)
        totals = {
            "realized_pnl": pair.get("realized_pnl_total") or 0.0,
            "unrealized_pnl": pair.get("unrealized_pnl") if pair else 0.0,
            "pnl_total": pair.get("pnl_total") if pair else 0.0,
            "profit_reserve": {"pending_quote": 0.0, "holdings_qty": 0.0, "quote_spent": 0.0},
            "bnb_reserve": {"pending_quote": 0.0, "holdings_qty": 0.0},
            "fees_paid": {"quote": 0.0, "bnb": 0.0, "other": {}},
        }
        return {
            "status": "paused" if self._paused.is_set() else "running",
            "pairs": {self.pair_cfg.symbol: pair} if pair else {},
            "totals": totals,
            "logs": logs,
        }

    def export_csv(self) -> str:
        output = io.StringIO()
        realized = self._realized
        unrealized = 0.0
        with self._lock:
            unrealized = float(self._status.get("unrealized_pnl") or 0.0) if self._status else 0.0
        if not self._trades.empty:
            self._trades.to_csv(output, index=False)
        summary_line = f"total_realized,{realized}\ncurrent_unrealized,{unrealized}\npnl_total,{realized + unrealized}\n"
        output.write(summary_line)
        return output.getvalue()


SIMULATOR: Optional[SimulationRunner] = None


def start_simulator(symbol: Optional[str], lookback_days: float, speed: float) -> None:
    """Start or restart the simulator for the requested symbol."""

    if not PAIRS_RAW:
        raise RuntimeError("Config not loaded; restart the service with --config")

    target = (symbol or DEFAULT_SYMBOL or PAIRS_RAW[0]["symbol"]).upper()
    pair_raw = next((p for p in PAIRS_RAW if p.get("symbol", "").upper() == target), None)
    if pair_raw is None:
        raise ValueError(f"Unknown symbol: {target}")

    pair_cfg = build_pair_config(pair_raw, GENERAL_CFG)
    runner = SimulationRunner(
        pair_cfg,
        general_cfg=GENERAL_CFG,
        api_cfg=API_CFG,
        speed_multiplier=speed,
        lookback_days=lookback_days,
    )
    runner.start()

    global SIMULATOR
    SIMULATOR = runner


@app.after_request
def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/simulate", methods=["POST", "OPTIONS"])
def simulate() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)

    payload = request.get_json(silent=True) or {}
    symbol = payload.get("symbol")
    lookback = float(payload.get("lookback_days") or 3.0)
    speed = float(payload.get("speed") or 24.0)
    try:
        start_simulator(symbol, lookback_days=lookback, speed=speed)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to start simulator")
        return jsonify({"status": "error", "message": str(exc)}), 400

    snapshot = SIMULATOR.snapshot() if SIMULATOR else {"status": "unknown"}
    snapshot["meta"] = {
        "symbol": (symbol or DEFAULT_SYMBOL or "").upper(),
        "lookback_days": lookback,
        "speed": speed,
    }
    return jsonify(snapshot)


@app.route("/health")
def health() -> Response:
    if SIMULATOR is None:
        return jsonify({"status": "error", "message": "Simulator not started"}), 503
    return jsonify(SIMULATOR.snapshot())


@app.route("/pause", methods=["POST", "OPTIONS"])
def pause() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    if SIMULATOR:
        SIMULATOR.pause()
    return jsonify({"status": "paused"})


@app.route("/resume", methods=["POST", "OPTIONS"])
def resume() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    if SIMULATOR:
        SIMULATOR.resume()
    return jsonify({"status": "running"})


@app.route("/stop", methods=["POST", "OPTIONS"])
def stop() -> Response:
    if request.method == "OPTIONS":
        return Response(status=204)
    if SIMULATOR:
        SIMULATOR.stop()
    return jsonify({"status": "stopped"})


@app.route("/export")
def export_csv() -> Response:
    if SIMULATOR is None:
        return jsonify({"status": "error", "message": "Simulator not started"}), 503
    csv_body = SIMULATOR.export_csv()
    return Response(
        csv_body,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=simulation_trades.csv"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run accelerated simulation service")
    parser.add_argument("--config", default="config.yaml", help="Config file to reuse for pair settings")
    parser.add_argument("--symbol", default=None, help="Target symbol (default: first in config)")
    parser.add_argument("--lookback-days", type=float, default=3.0, help="Historical days to replay")
    parser.add_argument("--speed", type=float, default=24.0, help="Time acceleration factor")
    parser.add_argument("--http-port", type=int, default=8080)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pairs_raw = cfg.get("pairs", [])
    if not pairs_raw:
        raise SystemExit("No pairs configured")

    global GENERAL_CFG, API_CFG, PAIRS_RAW, DEFAULT_SYMBOL
    GENERAL_CFG = cfg.get("general", {})
    API_CFG = cfg.get("api", {})
    PAIRS_RAW = pairs_raw
    DEFAULT_SYMBOL = args.symbol.upper() if args.symbol else pairs_raw[0]["symbol"].upper()

    start_simulator(symbol=DEFAULT_SYMBOL, lookback_days=args.lookback_days, speed=args.speed)

    app.run(host="0.0.0.0", port=args.http_port)


if __name__ == "__main__":
    main()