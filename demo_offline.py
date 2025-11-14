
import argparse, os, json
import pandas as pd
import numpy as np
from src.backtester import PairConfig, run_backtest_for_pair
from src.report import save_outputs

def make_mock_ohlcv(n=720, seed=7):
    rng = np.random.default_rng(seed)
    ts0 = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    times = [ts0 + pd.Timedelta(minutes=i) for i in range(n)]
    price = 1.0; rows = []
    for i,t in enumerate(times):
        drift = 0.00012 * np.sin(i/48)
        shock = rng.normal(0, 0.0025)
        price = max(0.2, price * (1 + drift + shock))
        if i%180==60: price *= 1.06  # pump
        if i%300==210: price *= 0.92  # dip
        o = price*(1+rng.normal(0,0.0007))
        h = max(o, price*(1+abs(rng.normal(0,0.007))))
        l = min(o, price*(1-abs(rng.normal(0,0.007))))
        c = price
        v = 1_000 + rng.normal(0, 120)
        rows.append((t,o,h,l,c,v))
    return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = make_mock_ohlcv()
    cfg = PairConfig(
        symbol="DEMOUSDT", quote="USDT", b_alloc=200.0,
        source="binance", interval="1m", lookback_days=1, start="", end="",
        fees_taker=0.001, fees_maker=0.001,
        buy_d=0.07, buy_m=1.5, buy_n=7,
        p_min=0.02, s1=0.01, m_step=1.6, tau=0.7,
        p_lock_base=0.005, p_lock_max=0.02, tau_min=0.3, no_loss_epsilon=0.0005,
        W1_minutes=5, m_time=2.0, delta_lock=0.002, beta_tau=0.9,
        T_idle_max_minutes=45, p_idle=0.004,
        T_total_cap_minutes=180, p_exit_min=0.004,
        snapshot_every_bars=1, use_maker=True,
        btd={"enabled": False}, sah={"enabled": False}
    )
    res = run_backtest_for_pair(df, cfg)
    events, trades, equity, summary = res["events"], res["trades"], res["equity"], res["summary"]
    events.to_csv(os.path.join(out_dir, "events.csv"), index=False)
    trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    equity.to_csv(os.path.join(out_dir, "equity.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    save_outputs("DEMOUSDT", out_dir, events, trades, equity, summary, plot=True)
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out/demo")
    args = ap.parse_args()
    main(args.out_dir)
