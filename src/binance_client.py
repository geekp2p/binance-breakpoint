import time
import math
import hmac
import hashlib
import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import requests

from .order_sizing import normalize_quantity

class BinanceClient:
    """Minimal REST client for Binance spot trading."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.binance.com"):
        if not api_key or not api_secret:
            raise ValueError("API key and secret must be provided for live trading")
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": api_key})
        self._symbol_cache: Dict[str, Dict[str, Any]] = {}

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = urlencode(params, doseq=True)
        signature = hmac.new(self.api_secret, payload.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Dict[str, Any]:
        params = params.copy() if params else {}
        if signed:
            params.setdefault("timestamp", int(time.time() * 1000))
            params = self._sign(params)
        url = f"{self.base_url}{path}"
        resp = self._session.request(method, url, params=params if method == "GET" else None,
                                     data=params if method != "GET" else None, timeout=30)
        if resp.status_code >= 400:
            logging.error("Binance error %s: %s", resp.status_code, resp.text)
        resp.raise_for_status()
        data = resp.json()
        return data

    def get_exchange_info(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v3/exchangeInfo")

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper()
        if symbol not in self._symbol_cache:
            info = self.get_exchange_info()
            symbols = {s["symbol"]: s for s in info.get("symbols", [])}
            if symbol not in symbols:
                raise ValueError(f"Symbol {symbol} not found in exchange info")
            self._symbol_cache[symbol] = symbols[symbol]
        return self._symbol_cache[symbol]

    def get_ticker_price(self, symbol: str) -> float:
        data = self._request("GET", "/api/v3/ticker/price", params={"symbol": symbol.upper()})
        try:
            return float(data.get("price"))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _format_number(value: float, precision: int = 8) -> str:
        return f"{value:.{precision}f}".rstrip("0").rstrip(".") or "0"

    def _lot_filters(self, symbol: str, filter_type: str = "LOT_SIZE") -> Tuple[Optional[float], Optional[float]]:
        """Get lot size filters (minQty, stepSize) for a symbol.

        Args:
            symbol: Trading pair symbol
            filter_type: Filter type to check - "LOT_SIZE" for limit orders, "MARKET_LOT_SIZE" for market orders
        """
        info = self.get_symbol_info(symbol)
        for f in info.get("filters", []):
            if f.get("filterType") == filter_type:
                try:
                    step = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                    return min_qty, step
                except (TypeError, ValueError, KeyError):
                    return None, None
        # If MARKET_LOT_SIZE not found, fall back to LOT_SIZE
        if filter_type == "MARKET_LOT_SIZE":
            for f in info.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    try:
                        step = float(f["stepSize"])
                        min_qty = float(f["minQty"])
                        return min_qty, step
                    except (TypeError, ValueError, KeyError):
                        return None, None
        return None, None

    def _apply_lot_step(self, symbol: str, quantity: float, filter_type: str = "LOT_SIZE") -> float:
        """Apply lot size rounding to a quantity.

        Args:
            symbol: Trading pair symbol
            quantity: Raw quantity to round
            filter_type: Filter type - "LOT_SIZE" for limit orders, "MARKET_LOT_SIZE" for market orders
        """
        min_qty, step = self._lot_filters(symbol, filter_type)
        if step is None or step <= 0:
            return quantity
        qty_steps = math.floor(quantity / step)
        adjusted = qty_steps * step
        if min_qty is not None and adjusted < min_qty:
            return 0.0
        precision = max(len(str(step).split(".")[-1].rstrip("0")), 0)
        return float(self._format_number(adjusted, precision))

    def normalize_quantity(self, symbol: str, quantity: float, *, price: Optional[float] = None,
                            allow_round_up: bool = False) -> Tuple[float, str]:
        min_qty, step = self._lot_filters(symbol)
        min_notional = self.get_min_notional(symbol)
        qty, reason = normalize_quantity(
            quantity,
            min_qty=min_qty or 0.0,
            step_size=step,
            price=price,
            min_notional=min_notional,
            allow_round_up=allow_round_up,
        )
        if step:
            precision = max(len(str(step).split(".")[-1].rstrip("0")), 0)
            qty = float(self._format_number(qty, precision))
        return qty, reason

    def get_min_notional(self, symbol: str) -> Optional[float]:
        """Return the minimum notional requirement for the symbol, if present."""

        info = self.get_symbol_info(symbol)
        filters = info.get("filters", [])
        min_notional = None
        for f in filters:
            if f.get("filterType") in {"MIN_NOTIONAL", "NOTIONAL"}:
                try:
                    min_notional = float(f.get("minNotional") or f.get("notional"))
                except (TypeError, ValueError):
                    min_notional = None
                break
        return min_notional

    def get_free_balance(self, asset: str) -> float:
        """Return the available ("free") balance for the given asset."""

        account = self._request("GET", "/api/v3/account", signed=True)
        target = asset.upper()
        balances = account.get("balances") or []
        for bal in balances:
            if str(bal.get("asset", "")).upper() == target:
                try:
                    return float(bal.get("free") or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def market_buy_quote(self, symbol: str, quote_amount: float) -> Dict[str, Any]:
        params = {
            "symbol": symbol.upper(),
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": self._format_number(quote_amount, 8),
        }
        logging.info("Submitting MARKET BUY for %s with %s quote", symbol, params["quoteOrderQty"])
        return self._request("POST", "/api/v3/order", params=params, signed=True)

    def market_sell(self, symbol: str, quantity: float) -> Dict[str, Any]:
        # Use MARKET_LOT_SIZE filter for market orders
        adj_qty = self._apply_lot_step(symbol, quantity, filter_type="MARKET_LOT_SIZE")
        if adj_qty <= 0:
            raise ValueError("Quantity below minimum lot size")
        params = {
            "symbol": symbol.upper(),
            "side": "SELL",
            "type": "MARKET",
            "quantity": self._format_number(adj_qty, 8),
        }
        logging.info("Submitting MARKET SELL for %s with qty %s", symbol, params["quantity"])
        return self._request("POST", "/api/v3/order", params=params, signed=True)

    def limit_order(self, symbol: str, side: str, quantity: float, price: float, time_in_force: str = "GTC") -> Dict[str, Any]:
        adj_qty = self._apply_lot_step(symbol, quantity)
        if adj_qty <= 0:
            raise ValueError("Quantity below minimum lot size")
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": self._format_number(adj_qty, 8),
            "price": self._format_number(price, 8),
        }
        logging.info("Submitting LIMIT %s for %s qty %s @ %s", side.upper(), symbol, params["quantity"], params["price"])
        return self._request("POST", "/api/v3/order", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        params = {"symbol": symbol.upper(), "orderId": order_id}
        logging.info("Cancelling order %s for %s", order_id, symbol)
        return self._request("DELETE", "/api/v3/order", params=params, signed=True)