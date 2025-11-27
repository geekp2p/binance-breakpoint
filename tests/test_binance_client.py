import hashlib
import hmac
from urllib.parse import urlencode

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