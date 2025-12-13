import hashlib
import hmac
import sys
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.binance_client import BinanceClient

def test_sign_uses_request_encoding_order():
    client = BinanceClient("key", "secret")
    params = {
        "symbol": "ZECUSDT",
        "side": "BUY",
        "type": "MARKET",
        "quoteOrderQty": "63.15789474",
        "timestamp": 1700000000000,
    }

    signed = client._sign(params.copy())

    payload = urlencode(params, doseq=True)
    expected_signature = hmac.new(b"secret", payload.encode(), hashlib.sha256).hexdigest()

    assert signed["signature"] == expected_signature
    # Ensure we didn't mutate parameter order by sorting; the signature must reflect the encoded payload.
    assert signed["signature"] == hmac.new(b"secret", urlencode(params, doseq=True).encode(), hashlib.sha256).hexdigest()


def test_get_free_balance_fetches_signed_account(monkeypatch):
    client = BinanceClient("key", "secret")

    calls = []

    def fake_request(method, path, *, params=None, signed=False):
        calls.append((method, path, signed))
        return {
            "balances": [
                {"asset": "ZEC", "free": "1.234", "locked": "0.0"},
                {"asset": "USDT", "free": "100.0", "locked": "0.0"},
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.get_free_balance("zec") == 1.234
    assert client.get_free_balance("BTC") == 0.0
    assert calls == [("GET", "/api/v3/account", True), ("GET", "/api/v3/account", True)]


def test_get_free_balance_handles_empty_balances(monkeypatch):
    client = BinanceClient("key", "secret")

    def fake_request(method, path, *, params=None, signed=False):
        assert signed is True
        return {"balances": []}

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.get_free_balance("usdt") == 0.0