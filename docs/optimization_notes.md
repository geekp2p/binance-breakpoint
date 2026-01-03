# Config tuning notes for ZECUSDT & DCRUSDT

These suggestions are derived from the current production-style config snapshot shown in the user log. They focus on raising realized PnL while keeping drawdowns contained and preventing the bot from getting stuck in endless micro-skips.

## Key observations
- **Micro-oscillation keeps skipping**: the log for both pairs shows thousands of `MICRO_BUY_SKIPPED` events. This implies the band/volatility gates are rarely satisfied, so micro trades do not add PnL.
- **Ladder is shallow but spreads are tight**: `d_buy = 1.5%` with Fibonacci spacing and `m_buy = 1.5` means inventory adds quickly but size grows aggressively. There is room to increase spacing and reduce size multiplier to smooth drawdown.
- **Profit trail starts early**: `p_min = 2%` and `p_lock_base = 0.5%` lock profits quickly; combined with micro stops this produces many small winners but leaves upside on momentum moves.
- **Time caps**: `T_idle_max_minutes = 45` and `p_idle = 0.8%` force exits during low-volatility drift; this can exit prematurely when trend follows through slowly.
- **Scalp mode allocation**: `order_pct_allocation = 0.33` per scalp trade limits ammo when many dips happen inside one trend.

## Recommended adjustments

### 1) Let micro-oscillation actually fire
- Increase `max_band_pct` from **0.008 → 0.012** so the band covers more intra-bar noise.
- Reduce `entry_band_pct` from **0.18 → 0.12** so the entry threshold is reachable more often.
- Lower `min_swing_pct` from **0.0012 → 0.0008** to accept smaller micro swings.
- Keep `min_profit_pct` at 0.25% but raise `take_profit_pct` slightly **0.003 → 0.004** to offset the more frequent entries.
- If skips persist, relax `cooldown_bars` from **8 → 5** to allow re-entries.

### 2) Smooth the ladder to reduce stuck inventory
- Widen primary spacing: `d_buy` **0.015 → 0.02** and cap adaptively with `adaptive_ladder.min_d_buy = 0.02`, `max_d_buy = 0.09`.
- Reduce size multiplier: `m_buy` **1.5 → 1.35** so later steps grow slower, lowering average entry price without exploding size.
- Consider limiting total exposure per leg: set `max_quote_per_leg` to **b_alloc * 0.08** (e.g., ~1,876 USDT for ZECUSDT) to avoid over-allocation in one cascade.
- Keep `n_steps = 10` but increase `gap_factor` **1.05 → 1.08** to widen later gaps.

### 3) Let winners run while still locking gains
- Raise `profit_trail.p_min` **0.02 → 0.025** and `s1` **0.01 → 0.012** so trailing starts slightly later and at a higher slope.
- Increase `p_lock_base` **0.005 → 0.007** and `p_lock_max` **0.02 → 0.03** to lock more on strong moves without cutting all upside.
- Decrease `no_loss_epsilon` **0.0018 → 0.0015** to flip to break-even protection sooner when moves fade.

### 4) Ease the time caps
- Extend idle timer: `T_idle_max_minutes` **45 → 70** to avoid exits during healthy consolidations.
- Reduce `p_idle` **0.008 → 0.006** so forced exits have a tighter profit requirement when triggered.
- Keep `T_total_cap_minutes` but increase `p_exit_min` **0.008 → 0.01** to ensure late exits are not too shallow.

### 5) Allow more scalp ammo when momentum persists
- Raise `scalp_mode.order_pct_allocation` **0.33 → 0.45** so each scalp uses more of the remaining allocation, improving realized gains in trending phases.
- Lower `cooldown_bars` **3 → 2** to rearm faster after quick fills.

### 6) Profit recycling (currently disabled)
- Enable `profit_recycling.enabled: true` and start with conservative splits: `discount_allocation_pct: 0.05`, `bnb_allocation_pct: 0.03`, and `min_order_quote: 20`. This compounds gains into discount buys and fee reduction without draining liquidity.

## Example overrides for ZECUSDT / DCRUSDT
Paste the following snippet into your config under each pair you want to tune:

```yaml
buy_ladder:
  d_buy: 0.02
  m_buy: 1.35
  gap_factor: 1.08
  max_quote_per_leg: 0.08  # fraction of b_alloc; replace with absolute if preferred
adaptive_ladder:
  min_d_buy: 0.02
  max_d_buy: 0.09
micro_oscillation:
  max_band_pct: 0.012
  entry_band_pct: 0.12
  min_swing_pct: 0.0008
  take_profit_pct: 0.004
  cooldown_bars: 5
profit_trail:
  p_min: 0.025
  s1: 0.012
  p_lock_base: 0.007
  p_lock_max: 0.03
  no_loss_epsilon: 0.0015
time_caps:
  T_idle_max_minutes: 70
  p_idle: 0.006
  p_exit_min: 0.01
features:
  scalp_mode:
    order_pct_allocation: 0.45
    cooldown_bars: 2
profit_recycling:
  enabled: true
  discount_allocation_pct: 0.05
  bnb_allocation_pct: 0.03
  min_order_quote: 20
```

> Apply gradually: start with micro-oscillation and profit trail tweaks, observe skip counts and PnL per symbol, then widen ladders if drawdown remains acceptable. Revert any change that raises max drawdown beyond your tolerance.
