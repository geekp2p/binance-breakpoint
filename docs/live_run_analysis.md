# Live run analysis (2025-12-05 configuration)

This document summarizes the observed behavior from the provided live trading logs using the updated configuration for `ZECUSDT` and `DCRUSDT` on 1m intervals.

## Configuration snapshot
- Two pairs configured (`ZECUSDT`, `DCRUSDT`) with quote `USDT`, base allocation 6,666 USDT, 1m interval, 2-day lookback, taker/maker fees at 0.1%.
- Core ladder: `d_buy=3%`, `m_buy=1.5`, `n_steps=3`.
- Profit trail: base take-profit around 2% with dynamic locking (`p_lock_base=0.005` to `p_lock_max=0.02`) and adaptive `tau`.
- Time-based martingale and caps enabled (idle cap 45m, total cap 180m).
- Features enabled: `scalp_mode`, `buy_the_dip`, `sell_at_height`, `adaptive_ladder`, `anchor_drift`; plotting enabled; `sell_scale_out` set to two chunks with 1s delay and profit-only exits.

## Timeline of DCRUSDT trades
- 02:16 UTC: Session started in `ACCUMULATE` stage 0, no position, next ladder buy at 21.0102 USDT.
- 04:48 UTC: First ladder fill triggered at 21.9128 USDT for ~1,403 USDT. Market buy executed; after commission, position was 63.742 DCR at average 22.0384. Break-even moved to 22.0605; next ladder buy at 21.4745.
- 05:06 UTC: Second ladder fill at 21.4745 USDT for ~2,105 USDT. Position increased to 160.764 DCR, average 21.8452, break-even 21.8671; next ladder buy planned at 21.0451.
- 07:20 UTC: Price recovered to 22.1900 USDT with unrealized gain ~51.86 USDT while still in `ACCUMULATE` stage 0.
- 07:21 UTC: Strategy emitted `TRAIL_PULLBACK` sell signal at 22.58082 USDT with PnL 114.62 USDT for qty 160.764. `sell_scale_out` executed two market sells of 80.382 DCR each one second apart. After fills, position returned to zero with realized total 114.62 USDT and next buy reset to 21.7571.

## ZECUSDT observations
- Throughout the log, `ZECUSDT` remained in `ACCUMULATE` stage 0 with zero position; `next_buy` stayed near 343–360 USDT. No ladder or feature-triggered trades occurred during the observed window.

## Notes
- The ladder trades stayed in stage 0 because only two ladder steps executed; no dip, scalp, or sell-at-height actions were logged. The exit came from the trailing profit pullback logic.
- Split sells reflect the configured `sell_scale_out` chunks; both were filled at market with slight price drift between orders.
- Commission adjustments were applied after each buy, reducing net quantity compared to the expected size (`delta` lines in the log).