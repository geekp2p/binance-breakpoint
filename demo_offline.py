import argparse
import json
import os

import numpy as np
import pandas as pd

from src.backtester import PairConfig, run_backtest_for_pair
from src.report import save_outputs


def make_mock_ohlcv(
    n: int = 720,
    seed: int = 7,
    base_price: float = 1.0,
    pump_mult: float = 1.06,
    dip_mult: float = 0.92,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts0 = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    times = [ts0 + pd.Timedelta(minutes=i) for i in range(n)]
    price = base_price
    rows = []
    for i,t in enumerate(times):
        drift = 0.00012 * np.sin(i/48)
        shock = rng.normal(0, 0.0025)
        price = max(0.2, price * (1 + drift + shock))
        if i % 180 == 60:
            price *= pump_mult  # pump
        if i % 300 == 210:
            price *= dip_mult  # dip
        o = price*(1+rng.normal(0,0.0007))
        h = max(o, price*(1+abs(rng.normal(0,0.007))))
        l = min(o, price*(1-abs(rng.normal(0,0.007))))
        c = price
        v = 1_000 + rng.normal(0, 120)
        rows.append((t,o,h,l,c,v))
    return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])

def main(
    out_dir: str,
    symbol: str,
    base_price: float,
    seed: int,
    minutes: int,
    pump_mult: float,
    dip_mult: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = make_mock_ohlcv(
        n=minutes,
        seed=seed,
        base_price=base_price,
        pump_mult=pump_mult,
        dip_mult=dip_mult,
    )
    cfg = PairConfig(
        symbol=symbol, quote="USDT", b_alloc=200.0,
        source="binance", interval="1m", lookback_days=1, start="", end="",
        fees_taker=0.001, fees_maker=0.001,
        buy_d=0.03, buy_m=1.5, buy_n=3,
        p_min=0.02, s1=0.01, m_step=1.6, tau=0.7,
        p_lock_base=0.005, p_lock_max=0.02, tau_min=0.3, no_loss_epsilon=0.0005,
        W1_minutes=5, m_time=2.0, delta_lock=0.002, beta_tau=0.9,
        T_idle_max_minutes=45, p_idle=0.004,
        T_total_cap_minutes=180, p_exit_min=0.004,
        snapshot_every_bars=1, use_maker=True,
        scalp={"enabled": False}, btd={"enabled": False}, sah={"enabled": False},
        adaptive_ladder={"enabled": True}, anchor_drift={"enabled": True}
    )
    res = run_backtest_for_pair(df, cfg)
    events, trades, equity, summary = res["events"], res["trades"], res["equity"], res["summary"]
    events.to_csv(os.path.join(out_dir, "events.csv"), index=False)
    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    equity.to_csv(os.path.join(out_dir, "equity.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    save_outputs(symbol, out_dir, events, trades, equity, summary, plot=True)
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out/demo")
    ap.add_argument("--symbol", default="DEMOUSDT", help="Symbol name to embed in outputs")
    ap.add_argument("--base-price", type=float, default=1.0, help="Starting price level for the synthetic path")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for the synthetic generator")
    ap.add_argument("--minutes", type=int, default=720, help="Number of 1m bars to synthesize")
    ap.add_argument("--pump-mult", type=float, default=1.06, help="Multiplicative pump applied periodically")
    ap.add_argument("--dip-mult", type=float, default=0.92, help="Multiplicative dip applied periodically")
    args = ap.parse_args()
    main(
        args.out_dir,
        args.symbol.upper(),
        args.base_price,
        args.seed,
        args.minutes,
        args.pump_mult,
        args.dip_mult,
    )