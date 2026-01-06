# Config tuning notes for ZECUSDT & DCRUSDT

These suggestions are derived from the current production-style config snapshot shown in the user log. They focus on raising realized PnL while keeping drawdowns contained and preventing the bot from getting stuck in endless micro-skips.

## Key observations
- **Micro-oscillation keeps skipping**: the log for both pairs shows thousands of `MICRO_BUY_SKIPPED` events. This implies the band/volatility gates are rarely satisfied, so micro trades do not add PnL.
- **Ladder is shallow but spreads are tight**: `d_buy = 1.5%` with Fibonacci spacing and `m_buy = 1.5` means inventory adds quickly but size grows aggressively. There is room to increase spacing and reduce size multiplier to smooth drawdown.
- **Profit trail starts early**: `p_min = 2%` and `p_lock_base = 0.5%` lock profits quickly; combined with micro stops this produces many small winners but leaves upside on momentum moves.
- **Time caps**: `T_idle_max_minutes = 45` and `p_idle = 0.8%` force exits during low-volatility drift; this can exit prematurely when trend follows through slowly.
- **Scalp mode allocation**: `order_pct_allocation = 0.33` per scalp trade limits ammo when many dips happen inside one trend.

## Recommended adjustments

### Quick playbook when a pair gets stuck (ladder idle, micro skipping)
If a specific symbol (e.g., **DCRUSDT**) stops firing new ladder orders and the
activity log shows repeated `MICRO_BUY_SKIPPED OPEN_INVENTORY` entries for hours
while other pairs behave normally:

1. **Force a ladder refresh without changing global config**
   - In the UI, hit **"Refresh" → "Resume all"** to re-run health checks and
     trigger an immediate ladder rebuild for that pair.
   - If the pair remains in `ACCUMULATE` with ammo left but no pending orders,
     toggle **Pause → Resume** for just that symbol to reset its internal
     timers.

2. **Clear stale micro cooldowns**
   - Micro is suppressed while core inventory is open. If you want micro to
     resume before the ladder scales out, temporarily disable and re-enable the
     micro module for that symbol in the UI; this resets the
     `micro_cooldown_until_bar` gate and rechecks entry bands on the next bar.

3. **Check anchor shifts**
   - Frequent `LADDER_REBUILT ANCHOR_SHIFT` with no new orders usually means the
     anchor keeps drifting upward with price. Hit **"Simulate" → "Stop" →
     "Live"** to pin a fresh anchor from the latest close, then watch whether
     the next buy level moves into reach.

4. **Escalate only for the affected pair**
   - Apply the overrides in the next section to DCRUSDT only (do not change
     the base config). Start with the micro band tweaks and wider `d_buy`; keep
     the successful settings for pairs like ZECUSDT untouched.

> These steps are reversible and keep your global config intact. If the pair
> still idles after a full refresh, consider widening `d_buy` and loosening the
> micro bands for that symbol using the override snippet below.

**Automation helper**
- Generate and (optionally) write the staged overrides with:
  ```bash
  python -m src.stuck_pair_override DCRUSDT --stage baseline
  python -m src.stuck_pair_override DCRUSDT --stage follow-up --config config.yaml --write
  ```
  The first command prints the 2–3 day nudge snippet; the second escalates to the 7–14 day follow-up and patches `config.yaml`
  directly.
- To automate the same nudges inside the core strategy without touching files, enable the built-in stuck recovery guardrail:
  ```yaml
  features:
    stuck_recovery:
      enabled: true
      stage1_bars: 4320   # trigger after ~3 days of no trades
      stage2_bars: 10080  # trigger after ~7 days
  ```
  When active, the strategy widens ladder spacing and relaxes micro bands in two stages, temporarily allowing micro entries even
  with core inventory open. Any fill (ladder/micro/scalp) resets the overrides back to the baseline config.

**Time-boxed override cycle**
- If nothing fires for **2–3 days**, temporarily widen bands or `d_buy` for the stuck pair only, watch fills for another couple of days, then roll the override back to baseline.
- If after **7–14 days** the pair is still idle, repeat with a slightly larger step (e.g., another +0.005 to `d_buy` or +0.02 to `entry_band_pct`) and again revert once activity resumes.
- Keep successful pairs unchanged; only the idle symbol should get short-lived overrides.

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
