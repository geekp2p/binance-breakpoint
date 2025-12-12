
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import math
import sys

from .utils import compute_ladder_prices, compute_ladder_amounts

PHASE_ACCUMULATE = "ACCUMULATE"
PHASE_TRAIL = "TRAIL"

@dataclass
class BuyLadderConf:
    d_buy: float
    m_buy: float
    n_steps: int
    size_mode: str = "geometric"
    gap_mode: str = "additive"
    gap_factor: float = 1.0
    base_order_quote: Optional[float] = None
    spacing_mode: str = "geometric"
    d_multipliers: Optional[list] = None
    max_step_drop: float = 0.25
    max_quote_per_leg: float = 0.0
    max_total_quote: float = 0.0

@dataclass
class AdaptiveLadderConf:
    enabled: bool = False
    bootstrap_d_buy: float = 0.03
    bootstrap_steps: int = 3
    min_d_buy: float = 0.02
    max_d_buy: float = 0.08
    volatility_window: int = 120
    sensitivity: float = 0.6
    rebalance_threshold: float = 0.001

@dataclass
class AnchorDriftConf:
    enabled: bool = False
    structure_window: int = 40
    atr_period: int = 14
    breakout_atr_multiplier: float = 1.2
    stable_band_pct: float = 0.01
    dwell_bars: int = 15
    min_displacement_atr: float = 0.5
    min_displacement_pct: float = 0.005
    cooldown_bars: int = 20        

@dataclass
class ProfitTrailConf:
    p_min: float
    s1: float
    m_step: float
    tau: float
    p_lock_base: float
    p_lock_max: float
    tau_min: float
    no_loss_epsilon: float = 0.0005

@dataclass
class TimeMartingaleConf:
    W1_minutes: float
    m_time: float
    delta_lock: float
    beta_tau: float

@dataclass
class TimeCapsConf:
    T_idle_max_minutes: float
    p_idle: float
    T_total_cap_minutes: float
    p_exit_min: float
    
# --- Early scalp mode ---
@dataclass
class ScalpModeConf:
    enabled: bool = False
    max_trades: int = 10
    base_drop_pct: float = 0.03
    min_drop_pct: float = 0.02
    max_drop_pct: float = 0.06
    base_take_profit_pct: float = 0.01
    min_take_profit_pct: float = 0.008
    max_take_profit_pct: float = 0.02
    volatility_ref_pct: float = 0.04
    scale_strength: float = 0.6
    order_pct_allocation: float = 0.33

# --- Micro oscillation scalp ---
@dataclass
class MicroOscillationConf:
    enabled: bool = False
    window: int = 30
    max_band_pct: float = 0.006
    min_swings: int = 4
    min_swing_pct: float = 0.0008
    entry_band_pct: float = 0.15
    take_profit_pct: float = 0.0025
    stop_break_pct: float = 0.005
    order_pct_allocation: float = 0.15
    cooldown_bars: int = 5
    reentry_drop_pct: float = 0.002
    loss_recovery_enabled: bool = True
    loss_recovery_markup_pct: float = 0.001
    loss_recovery_max_pct: float = 0.01

# --- Scaffolds ---
@dataclass
class BuyTheDipConf:
    enabled: bool = False
    dip_threshold: float = 0.05
    rebound_min: float = 0.004
    rebound_max: float = 0.02
    cooldown_minutes: float = 10.0
    max_orders: int = 7
    rebase_ladder_from_dip: bool = True
    rebase_offset: float = 0.0
    order_quote: float = 0.0
    order_pct_remaining: float = 0.0
    limit_offset: float = 0.0
    cancel_on_miss: bool = True
    isolate_from_ladder: bool = True

@dataclass
class SellAtHeightConf:
    enabled: bool = False
    height_threshold: float = 0.04
    pullback_min: float = 0.004
    pullback_max: float = 0.02
    cooldown_minutes: float = 10.0
    order_pct_position: float = 1.0
    limit_offset: float = 0.0
    min_qty: float = 0.0
    cancel_on_miss: bool = True

@dataclass
class StrategyState:
    fees_buy: float
    fees_sell: float
    b_alloc: float
    buy: BuyLadderConf
    adaptive: AdaptiveLadderConf
    anchor: AnchorDriftConf    
    trail: ProfitTrailConf
    tmart: TimeMartingaleConf
    tcaps: TimeCapsConf
    scalp: ScalpModeConf
    btd: BuyTheDipConf
    sah: SellAtHeightConf
    micro: MicroOscillationConf
    snapshot_every_bars: int
    use_maker: bool

    # Core round state
    round_active: bool = True
    phase: str = PHASE_ACCUMULATE
    P0: Optional[float] = None
    ladder_prices: List[float] = field(default_factory=list)
    ladder_amounts_quote: List[float] = field(default_factory=list)
    ladder_next_idx: int = 0
    current_d_buy: float = 0.0
    adaptive_ready: bool = False
    volatility_samples: List[float] = field(default_factory=list)
    Q: float = 0.0
    C: float = 0.0
    TP_base: Optional[float] = None
    H: Optional[float] = None
    stage: int = 0
    cumulative_s: float = 0.0
    tau_cur: float = 0.0
    p_lock_cur: float = 0.0
    round_start_ts: Optional[pd.Timestamp] = None
    last_high_ts: Optional[pd.Timestamp] = None
    next_window_minutes: Optional[float] = None
    idle_checked: bool = False

    # Early scalp mode state
    scalp_anchor_price: Optional[float] = None
    scalp_trades_done: int = 0
    scalp_positions: List[Dict[str, float]] = field(default_factory=list)
    session_high: Optional[float] = None
    session_low: Optional[float] = None

    # Scaffolding flags (no auto-trading yet)
    btd_armed: bool = False
    btd_last_bottom: Optional[float] = None
    btd_last_arm_ts: Optional[pd.Timestamp] = None
    btd_cooldown_until: Optional[pd.Timestamp] = None
    btd_orders_done: int = 0

    sah_armed: bool = False
    sah_last_top: Optional[float] = None
    sah_last_arm_ts: Optional[pd.Timestamp] = None
    sah_cooldown_until: Optional[pd.Timestamp] = None

    # Anchor drift detection state
    anchor_window: List[Dict[str, float]] = field(default_factory=list)
    anchor_last_move_ts: Optional[pd.Timestamp] = None
    anchor_last_move_bar: int = 0
    anchor_base_price: Optional[float] = None
    bar_index: int = 0

    # Micro oscillation scalp state
    micro_prices: List[float] = field(default_factory=list)
    micro_last_direction: Optional[int] = None
    micro_swings: int = 0
    micro_cooldown_until_bar: int = 0
    micro_positions: List[Dict[str, float]] = field(default_factory=list)
    micro_last_exit_price: Optional[float] = None
    micro_loss_recovery_pct: float = 0.0

    def _remaining_quote_allocation(self) -> float:
        effective_alloc = min(self.b_alloc, self.buy.max_total_quote) if self.buy.max_total_quote > 0 else self.b_alloc
        spent = (
            sum(self.ladder_amounts_quote[:self.ladder_next_idx])
            + self._scalp_committed_quote()
            + self._micro_committed_quote()
        )
        remaining = max(effective_alloc - spent, 0.0)
        return remaining
    
    def _effective_n_steps(self) -> int:
        if self.buy.d_multipliers:
            return min(len(self.buy.d_multipliers), max(1, int(self.buy.n_steps)))
        if self.adaptive.enabled and not self.adaptive_ready:
            bootstrap_steps = max(1, int(self.adaptive.bootstrap_steps))
            return max(bootstrap_steps, max(1, int(self.buy.n_steps)))
        return max(1, int(self.buy.n_steps))

    def _set_initial_d_buy(self):
        if self.adaptive.enabled:
            self.current_d_buy = self.adaptive.bootstrap_d_buy
        else:
            self.current_d_buy = self.buy.d_buy

    def rebuild_ladder(self, base_price: float, ts=None, log_events: Optional[List[Dict[str, Any]]] = None,
                       reason: str = "RESET", preserve_progress: bool = False):
        steps = self._effective_n_steps()
        alloc = self.b_alloc
        if preserve_progress:
            alloc = self._remaining_quote_allocation()
        amounts = compute_ladder_amounts(
            alloc,
            self.buy.m_buy,
            steps,
            size_mode=self.buy.size_mode,
            base_order_quote=self.buy.base_order_quote,
            max_quote_per_leg=self.buy.max_quote_per_leg,
            max_total_quote=self.buy.max_total_quote,
        )
        prev_idx = self.ladder_next_idx if preserve_progress else 0
        self.ladder_amounts_quote = amounts
        self.ladder_prices = compute_ladder_prices(
            base_price,
            self.current_d_buy,
            steps,
            spacing_mode=self.buy.spacing_mode,
            d_multipliers=self.buy.d_multipliers,
            max_step_drop=self.buy.max_step_drop,
            gap_mode=self.buy.gap_mode,
            gap_factor=self.buy.gap_factor,
        )
        self.ladder_next_idx = min(prev_idx, steps)
        self.anchor_base_price = base_price
        if log_events is not None and ts is not None:
            log_events.append({
                "ts": ts,
                "event": "LADDER_REBUILT",
                "reason": reason,
                "steps": steps,
                "d_buy": self.current_d_buy,
            })

    def _push_volatility_sample(self, h: float, l: float, c: float):
        if not self.adaptive.enabled or c <= 0:
            return
        window = max(1, int(self.adaptive.volatility_window))
        sample = max((h - l) / c, 0.0)
        self.volatility_samples.append(sample)
        if len(self.volatility_samples) > window:
            self.volatility_samples = self.volatility_samples[-window:]

    def _max_anchor_window(self) -> int:
        return max(
            self.anchor.structure_window + self.anchor.dwell_bars + 5,
            self.anchor.atr_period * 3,
        )

    def _update_anchor_window(self, h: float, l: float, c: float):
        prev_close = self.anchor_window[-1]["close"] if self.anchor_window else c
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        self.anchor_window.append({"high": h, "low": l, "close": c, "tr": tr})
        max_len = self._max_anchor_window()
        if len(self.anchor_window) > max_len:
            self.anchor_window = self.anchor_window[-max_len:]

    def _anchor_atr(self) -> float:
        period = max(1, int(self.anchor.atr_period))
        if len(self.anchor_window) < period:
            return 0.0
        recent = self.anchor_window[-period:]
        tr_sum = sum(bar["tr"] for bar in recent)
        return tr_sum / period if period > 0 else 0.0

    def _stable_zone_midpoint(self) -> Optional[float]:
        if not self.anchor.enabled:
            return None
        dwell = max(1, int(self.anchor.dwell_bars))
        structure = max(self.anchor.structure_window, dwell + 1)
        if len(self.anchor_window) < structure + dwell:
            return None
        stable_slice = self.anchor_window[-dwell:]
        prior_slice = self.anchor_window[-(structure + dwell):-dwell]
        if not prior_slice:
            return None
        stable_high = max(bar["high"] for bar in stable_slice)
        stable_low = min(bar["low"] for bar in stable_slice)
        mid = (stable_high + stable_low) / 2 if stable_high + stable_low != 0 else None
        if mid is None or mid <= 0:
            return None
        band_width = stable_high - stable_low
        mid_ref = mid if mid != 0 else 1.0
        if band_width / mid_ref > self.anchor.stable_band_pct * 2:
            return None
        atr = self._anchor_atr()
        breakout = max(bar["close"] for bar in stable_slice) - max(bar["high"] for bar in prior_slice)
        if atr <= 0 or breakout < atr * self.anchor.breakout_atr_multiplier:
            return None
        current_anchor = self.anchor_base_price or self.P0 or mid
        displacement = abs(mid - current_anchor)
        min_disp = max(
            self.anchor.min_displacement_atr * atr,
            (self.anchor.min_displacement_pct * current_anchor) if current_anchor else 0.0,
        )
        if displacement < min_disp:
            return None
        return mid

    def _maybe_shift_anchor(self, ts, c: float, log_events: List[Dict[str, Any]]):
        if not self.anchor.enabled:
            return
        if self.bar_index - self.anchor_last_move_bar < max(1, int(self.anchor.cooldown_bars)):
            return
        mid = self._stable_zone_midpoint()
        if mid is None or mid <= 0:
            return
        if self.Q > 0:
            P_BE = self.P_BE()
            if P_BE is None or P_BE <= 0:
                return
            proposed_next_buy = mid * (1 - self.current_d_buy)
            safe_exit = P_BE * (1 + self.trail.no_loss_epsilon)
            if proposed_next_buy > safe_exit:
                log_events.append({
                    "ts": ts,
                    "event": "ANCHOR_REJECTED",
                    "reason": "WOULD_BUY_ABOVE_BE",
                    "proposed_next_buy": proposed_next_buy,
                    "P_BE": P_BE,
                })
                return
        prev_anchor = self.anchor_base_price
        self.rebuild_ladder(mid, ts, log_events, reason="ANCHOR_SHIFT", preserve_progress=self.Q > 0)
        self.anchor_last_move_ts = ts
        self.anchor_last_move_bar = self.bar_index
        if self.P0 is None:
            self.P0 = mid
        log_events.append({
            "ts": ts,
            "event": "ANCHOR_SHIFTED",
            "anchor": mid,
            "prev_anchor": prev_anchor,
            "bar_index": self.bar_index,
            "preserve_progress": self.Q > 0,
        })

    def _maybe_update_adaptive_spacing(self, ts, c, h, l, log_events: List[Dict[str, Any]]):
        if not self.adaptive.enabled:
            return
        self._push_volatility_sample(h, l, c)
        window = max(1, int(self.adaptive.volatility_window))
        if len(self.volatility_samples) < max(10, window // 2):
            return
        mean_range = sum(self.volatility_samples[-window:]) / min(len(self.volatility_samples), window)
        target = mean_range * self.adaptive.sensitivity
        target = min(max(target, self.adaptive.min_d_buy), self.adaptive.max_d_buy)
        if not self.adaptive_ready:
            self.adaptive_ready = True
        if abs(target - self.current_d_buy) < self.adaptive.rebalance_threshold:
            return
        self.current_d_buy = target
        base = c if c > 0 else self.P0 or c
        self.rebuild_ladder(base, ts, log_events, reason="ADAPTIVE_UPDATE", preserve_progress=True)

    def _compute_btd_quote_amount(self) -> float:
        if self.btd.order_quote > 0:
            return self.btd.order_quote
        if self.btd.order_pct_remaining > 0:
            remaining = self._remaining_quote_allocation()
            return remaining * self.btd.order_pct_remaining
        return 0.0

    def _compute_sah_qty(self) -> float:
        if self.Q <= 0:
            return 0.0
        qty = self.Q * max(self.sah.order_pct_position, 0.0)
        if self.sah.min_qty > 0 and qty < self.sah.min_qty:
            return 0.0
        return min(qty, self.Q)
    
    def _scalp_committed_quote(self) -> float:
        total = 0.0
        for pos in self.scalp_positions:
            total += pos.get("cost", 0.0) / (1 + self.fees_buy)
        return total

    def _micro_committed_quote(self) -> float:
        total = 0.0
        for pos in self.micro_positions:
            total += pos.get("cost", 0.0) / (1 + self.fees_buy)
        return total

    def _remaining_scalp_allocation(self) -> float:
        effective_alloc = min(self.b_alloc, self.buy.max_total_quote) if self.buy.max_total_quote > 0 else self.b_alloc
        target_alloc = effective_alloc * max(min(self.scalp.order_pct_allocation, 1.0), 0.0)
        spent_ladder = sum(self.ladder_amounts_quote[:self.ladder_next_idx])
        spent_scalp = self._scalp_committed_quote()
        remaining_quote = max(effective_alloc - spent_ladder - spent_scalp, 0.0)
        remaining_scalp_cap = max(target_alloc - spent_scalp, 0.0)
        ladder_cap = 0.0
        if 0 <= self.ladder_next_idx < len(self.ladder_amounts_quote):
            ladder_cap = max(self.ladder_amounts_quote[self.ladder_next_idx], 0.0)

        cap = remaining_scalp_cap
        if ladder_cap > 0:
            cap = min(cap, ladder_cap)

        return min(cap, remaining_quote)

    def _remaining_micro_allocation(self) -> float:
        effective_alloc = min(self.b_alloc, self.buy.max_total_quote) if self.buy.max_total_quote > 0 else self.b_alloc
        target_alloc = effective_alloc * max(min(self.micro.order_pct_allocation, 1.0), 0.0)
        spent_ladder = sum(self.ladder_amounts_quote[:self.ladder_next_idx])
        spent_scalp = self._scalp_committed_quote()
        spent_micro = self._micro_committed_quote()
        remaining_quote = max(effective_alloc - spent_ladder - spent_scalp - spent_micro, 0.0)
        remaining_micro_cap = max(target_alloc - spent_micro, 0.0)
        ladder_cap = 0.0
        if 0 <= self.ladder_next_idx < len(self.ladder_amounts_quote):
            ladder_cap = max(self.ladder_amounts_quote[self.ladder_next_idx], 0.0)
        cap = remaining_micro_cap
        if ladder_cap > 0:
            cap = min(cap, ladder_cap)
        return min(cap, remaining_quote)

    def _disarm_btd(self):
        self.btd_armed = False
        self.btd_last_bottom = None

    def _disarm_sah(self):
        self.sah_armed = False
        self.sah_last_top = None

    def _reset_btd_progress(self):
        self._disarm_btd()
        self.btd_cooldown_until = None
        self.btd_orders_done = 0 

    def reset_trail(self, now_ts, P_BE):
        self.TP_base = P_BE * (1 + self.trail.p_min)
        self.H = None
        self.stage = 1
        self.cumulative_s = 0.0
        self.tau_cur = self.trail.tau
        self.p_lock_cur = self.trail.p_lock_base
        self.last_high_ts = now_ts
        self.next_window_minutes = self.tmart.W1_minutes

    def s_for_stage(self, stage:int)->float:
        if stage <= 1:
            return self.trail.s1
        if self.trail.m_step <= 1:
            return self.trail.s1

        base = max(self.trail.s1, sys.float_info.min)
        m_step = max(self.trail.m_step, sys.float_info.min)
        max_power = math.log(sys.float_info.max / base, m_step)
        capped_power = min(stage - 1, max_power)

        log_base = math.log(base)
        log_m_step = math.log(m_step)
        max_log_value = math.log(sys.float_info.max)
        log_result = min(log_base + capped_power * log_m_step, max_log_value)
        return math.exp(log_result)

    def floor_price(self, P_BE):
        if self.Q <= 0 or P_BE is None:
            return None
        # no-loss guard
        base_floor = P_BE * (1 + max(self.p_lock_cur, self.trail.no_loss_epsilon))
        if self.H is None:
            return base_floor
        s_k = self.s_for_stage(self.stage)
        F1 = self.H * (1 - self.tau_cur * s_k)
        return max(F1, base_floor)

    def P_BE(self):
        if self.Q <= 0:
            return None
        return self.C / (self.Q * (1 - self.fees_sell))

    # --- Scalp helpers ---
    def _update_session_range(self, h: float, l: float):
        if self.session_high is None or h > self.session_high:
            self.session_high = h
        if self.session_low is None or l < self.session_low:
            self.session_low = l

    def _volatility_scale(self) -> float:
        if not self.scalp.enabled:
            return 1.0
        if self.session_high is None or self.session_low is None or self.session_high <= 0:
            return 1.0
        observed = (self.session_high - self.session_low) / self.session_high
        if self.scalp.volatility_ref_pct <= 0:
            return 1.0
        ratio = observed / self.scalp.volatility_ref_pct
        scale = 1.0 + (ratio - 1.0) * self.scalp.scale_strength
        return max(0.5, min(scale, 2.0))

    def _scalp_thresholds(self) -> Dict[str, float]:
        scale = self._volatility_scale()
        drop = self.scalp.base_drop_pct * scale
        tp = self.scalp.base_take_profit_pct * scale
        drop = min(max(drop, self.scalp.min_drop_pct), self.scalp.max_drop_pct)
        tp = min(max(tp, self.scalp.min_take_profit_pct), self.scalp.max_take_profit_pct)
        return {"drop_pct": drop, "take_profit_pct": tp}

    def _maybe_scalp_buy(self, ts, o, h, l, log_events: List[Dict[str, Any]]):
        if not self.scalp.enabled:
            return
        if self.scalp_trades_done >= self.scalp.max_trades:
            return
        anchor = self.scalp_anchor_price or self.P0
        if anchor is None or anchor <= 0:
            return
        thresholds = self._scalp_thresholds()
        trigger = anchor * (1 - thresholds["drop_pct"])
        if l > trigger:
            return
        order_quote = self._remaining_scalp_allocation()
        if order_quote <= 0:
            log_events.append({
                "ts": ts,
                "event": "SCALP_BUY_SKIPPED",
                "reason": "NO_CAPITAL",
                "trigger": trigger,
                "drop_pct": thresholds["drop_pct"],
            })
            self.scalp_trades_done += 1
            return
        price = trigger
        qty = order_quote / price
        cost = order_quote * (1 + self.fees_buy)
        prev_qty = self.Q
        self.Q += qty
        self.C += cost
        if prev_qty <= 0 < self.Q:
            self.round_start_ts = ts
        target = price * (1 + thresholds["take_profit_pct"])
        self.scalp_positions.append({
            "entry": price,
            "qty": qty,
            "cost": cost,
            "target": target,
        })
        self.scalp_trades_done += 1
        log_events.append({
            "ts": ts,
            "event": "SCALP_BUY",
            "price": price,
            "qty": qty,
            "order_quote": order_quote,
            "drop_pct": thresholds["drop_pct"],
            "take_profit_pct": thresholds["take_profit_pct"],
            "target": target,
            "trade_idx": self.scalp_trades_done,
        })

    def _check_scalp_take_profit(self, ts, h, log_events: List[Dict[str, Any]]):
        if not self.scalp_positions:
            return
        remaining_positions: List[Dict[str, float]] = []
        for pos in self.scalp_positions:
            target = pos["target"]
            if h >= target:
                qty = pos["qty"]
                proceeds = qty * target * (1 - self.fees_sell)
                pnl = proceeds - pos["cost"]
                self.Q = max(self.Q - qty, 0.0)
                self.C = max(self.C - pos["cost"], 0.0)
                log_events.append({
                    "ts": ts,
                    "event": "SCALP_TP",
                    "price": target,
                    "qty": qty,
                    "proceeds": proceeds,
                    "pnl": pnl,
                })
            else:
                remaining_positions.append(pos)
        self.scalp_positions = remaining_positions
    # --- End scalp helpers ---

        # --- Micro oscillation helpers ---
    def _update_micro_window(self, c: float):
        if not self.micro.enabled:
            return
        window = max(5, int(self.micro.window))
        if c > 0:
            self.micro_prices.append(c)
        if len(self.micro_prices) > window:
            self.micro_prices = self.micro_prices[-window:]
        if len(self.micro_prices) < 2:
            return
        prev = self.micro_prices[-2]
        change = (c - prev) / prev if prev > 0 else 0.0
        direction = 0
        if change > self.micro.min_swing_pct:
            direction = 1
        elif change < -self.micro.min_swing_pct:
            direction = -1
        if direction != 0:
            if self.micro_last_direction is not None and direction != self.micro_last_direction:
                self.micro_swings += 1
            self.micro_last_direction = direction

    def _micro_snapshot(self) -> Optional[Dict[str, float]]:
        if not self.micro.enabled or len(self.micro_prices) < max(5, int(self.micro.window) // 2):
            return None
        lo, hi = min(self.micro_prices), max(self.micro_prices)
        if hi <= 0:
            return None
        band_pct = (hi - lo) / hi
        ready = (
            band_pct <= self.micro.max_band_pct
            and self.micro_swings >= max(1, int(self.micro.min_swings))
        )
        return {
            "low": lo,
            "high": hi,
            "band_pct": band_pct,
            "ready": ready,
        }

    def _maybe_micro_buy(self, ts, h, l, log_events: List[Dict[str, Any]]):
        if not self.micro.enabled or self.bar_index < self.micro_cooldown_until_bar:
            return
        if self.micro_positions:
            log_events.append(
                {
                    "ts": ts,
                    "event": "MICRO_BUY_SKIPPED",
                    "reason": "OPEN_POSITION",
                    "open_positions": len(self.micro_positions),
                }
            )
            return        
        snap = self._micro_snapshot()
        if not snap or not snap["ready"]:
            return
        band_span = snap["high"] - snap["low"]
        if band_span <= 0:
            return
        entry = snap["low"] + band_span * max(min(self.micro.entry_band_pct, 1.0), 0.0)
        if l > entry:
            return
        
        if (
            self.micro_last_exit_price is not None
            and entry >= self.micro_last_exit_price * (1 - max(self.micro.reentry_drop_pct, 0.0))
        ):
            log_events.append(
                {
                    "ts": ts,
                    "event": "MICRO_BUY_SKIPPED",
                    "reason": "NEED_PULLBACK",
                    "entry": entry,
                    "last_exit": self.micro_last_exit_price,
                }
            )
            self.micro_cooldown_until_bar = self.bar_index + max(1, int(self.micro.cooldown_bars))
            return        

        order_quote = self._remaining_micro_allocation()
        if order_quote <= 0:
            log_events.append({
                "ts": ts,
                "event": "MICRO_BUY_SKIPPED",
                "reason": "NO_CAPITAL",
                "entry": entry,
                "band_pct": snap["band_pct"],
            })
            self.micro_cooldown_until_bar = self.bar_index + max(1, int(self.micro.cooldown_bars))
            return
        price = entry
        qty = order_quote / price
        cost = order_quote * (1 + self.fees_buy)
        prev_qty = self.Q
        self.Q += qty
        self.C += cost
        if prev_qty <= 0 < self.Q:
            self.round_start_ts = ts
        tp_pct = self.micro.take_profit_pct
        if self.micro.loss_recovery_enabled:
            tp_pct = min(tp_pct + self.micro_loss_recovery_pct, self.micro.loss_recovery_max_pct)
        target = price * (1 + tp_pct)
        break_even_price = cost / qty / max(1 - self.fees_sell, 1e-9)
        stop = max(price * (1 - self.micro.stop_break_pct), break_even_price)
        self.micro_positions.append({
            "entry": price,
            "qty": qty,
            "cost": cost,
            "target": target,
            "stop": stop,
        })
        self.micro_swings = 0
        self.micro_cooldown_until_bar = self.bar_index + max(1, int(self.micro.cooldown_bars))
        self.micro_last_exit_price = None
        log_events.append({
            "ts": ts,
            "event": "MICRO_BUY",
            "price": price,
            "qty": qty,
            "order_quote": order_quote,
            "target": target,
            "stop": stop,
            "band_pct": snap["band_pct"],
        })

        while self.ladder_next_idx < len(self.ladder_prices) and price <= self.ladder_prices[self.ladder_next_idx]:
            self.ladder_next_idx += 1
            log_events.append(
                {
                    "ts": ts,
                    "event": "LADDER_NUDGE",
                    "reason": "MICRO_BUY",
                    "ladder_idx": self.ladder_next_idx,
                    "ladder_price": self.ladder_prices[self.ladder_next_idx - 1],
                }
            )        

    def _check_micro_take_profit(self, ts, h, l, log_events: List[Dict[str, Any]]):
        if not self.micro_positions:
            return
        remaining_positions: List[Dict[str, float]] = []
        for pos in self.micro_positions:
            target = pos["target"]
            stop = pos["stop"]
            qty = pos["qty"]
            exit_price = None
            reason = None
            if h >= target:
                exit_price = target
                reason = "TP"
            elif l <= stop <= h:
                exit_price = stop
                reason = "STOP"
            if exit_price is not None:
                qty_to_sell = min(qty, self.Q)
                if qty_to_sell <= 0:
                    log_events.append(
                        {
                            "ts": ts,
                            "event": "MICRO_EXIT_SKIPPED",
                            "reason": "NO_QTY",
                            "qty": qty,
                        }
                    )
                    remaining_positions.append(pos)
                    continue

                cost_share = pos["cost"] * (qty_to_sell / qty)
                proceeds = qty_to_sell * exit_price * (1 - self.fees_sell)
                pnl = proceeds - cost_share
                self.Q = max(self.Q - qty_to_sell, 0.0)
                self.C = max(self.C - cost_share, 0.0)
                if qty_to_sell < qty:
                    remaining_positions.append(
                        {
                            **pos,
                            "qty": qty - qty_to_sell,
                            "cost": pos["cost"] - cost_share,
                        }
                    )
                self.micro_last_exit_price = exit_price
                self.micro_cooldown_until_bar = self.bar_index + max(1, int(self.micro.cooldown_bars))
                if self.micro.loss_recovery_enabled:
                    if pnl <= 0:
                        self.micro_loss_recovery_pct = min(
                            self.micro_loss_recovery_pct + self.micro.loss_recovery_markup_pct,
                            self.micro.loss_recovery_max_pct,
                        )
                    else:
                        self.micro_loss_recovery_pct = 0.0                
                log_events.append({
                    "ts": ts,
                    "event": f"MICRO_{reason}",
                    "price": exit_price,
                    "qty": qty_to_sell,
                    "proceeds": proceeds,
                    "pnl": pnl,
                    "cost": cost_share,
                })
            else:
                remaining_positions.append(pos)
        self.micro_positions = remaining_positions
    # --- End micro helpers ---

    # --- Scaffolding: detect/arm BTD/SAH, but do NOT auto-execute ---
    def _maybe_arm_btd(self, ts, low):
        if not self.btd.enabled or self.ladder_next_idx == 0:
            return None
        if self.btd_cooldown_until is not None and ts < self.btd_cooldown_until:
            return None    
        last_ladder = self.ladder_prices[self.ladder_next_idx-1]
        if low <= last_ladder * (1 - self.btd.dip_threshold):
            # overshoot → arm BTD and record bottom candidate
            self.btd_armed = True
            self.btd_last_bottom = low
            self.btd_last_arm_ts = ts
            return {"ts": ts, "event": "BTD_ARMED", "bottom": low, "from": last_ladder}
        return None

    def _maybe_arm_sah(self, ts, high):
        if not self.sah.enabled or self.H is None:
            return None
        if self.sah_cooldown_until is not None and ts < self.sah_cooldown_until:
            return None
        if high >= self.H * (1 + self.sah.height_threshold):
            self.sah_armed = True
            self.sah_last_top = high
            self.sah_last_arm_ts = ts
            return {"ts": ts, "event": "SAH_ARMED", "top": high, "from": self.H}
        return None

    def _rebase_ladder(self, new_base_price: float, ts, log_events: List[Dict[str, Any]]):
        if new_base_price <= 0:
            return
        self.rebuild_ladder(new_base_price, ts, log_events, reason="BTD_REBASE", preserve_progress=False)
        log_events.append({
            "ts": ts,
            "event": "BTD_LADDER_RESET",
            "base_price": new_base_price,
            "rebase_offset": self.btd.rebase_offset
        })

    def on_bar(self, ts, o, h, l, c, volume, log_events:List[Dict[str,Any]]):
        self.bar_index += 1
        if self.P0 is None:
            self.P0 = o
            self.round_start_ts = ts
            self.scalp_anchor_price = o

        self._update_session_range(h, l)
        self._update_anchor_window(h, l, c)
        self._update_micro_window(c)

        # --- scalp take-profit first (protect fast exits) ---
        self._check_scalp_take_profit(ts, h, log_events)
        self._check_micro_take_profit(ts, h, l, log_events)

        # --- early scalp buys ---
        self._maybe_scalp_buy(ts, o, h, l, log_events)
        self._maybe_micro_buy(ts, h, l, log_events)
        # --- adapt ladder spacing before consuming quote ---
        self._maybe_update_adaptive_spacing(ts, c, h, l, log_events)
        # --- dynamic anchor repositioning ---
        self._maybe_shift_anchor(ts, c, log_events)         

        # --- Arm BTD/SAH (scaffold only; no auto-action) ---
        ev = self._maybe_arm_btd(ts, l)
        if ev: log_events.append(ev)
        if self.phase == PHASE_TRAIL:
            ev2 = self._maybe_arm_sah(ts, h)
            if ev2: log_events.append(ev2)

        # --- BTD confirmation (priority before ladder) ---
        if self.btd_armed and self.btd_last_bottom is not None:
            bottom = self.btd_last_bottom
            rebound = (h - bottom) / bottom if bottom > 0 else 0.0
            if rebound >= self.btd.rebound_min:
                if rebound <= self.btd.rebound_max:
                    order_price = bottom * (1 + self.btd.limit_offset)
                    if self.btd_orders_done >= max(1, int(self.btd.max_orders)):
                        log_events.append({
                            "ts": ts,
                            "event": "BTD_ORDER_SKIPPED",
                            "reason": "BTD_MAX_REACHED",
                            "bottom": bottom,
                            "rebound": rebound
                        })
                    else:
                        quote_amt = self._compute_btd_quote_amount()
                        qty = (quote_amt / order_price) if order_price > 0 else 0.0
                        if qty > 0:
                            self.btd_orders_done += 1
                            log_events.append({
                                "ts": ts,
                                "event": "BTD_ORDER",
                                "bottom": bottom,
                                "rebound": rebound,
                                "order_price": order_price,
                                "order_quote": quote_amt,
                                "order_qty": qty,
                                "btd_order_idx": self.btd_orders_done,
                                "btd_max_orders": self.btd.max_orders
                            })
                        else:
                            log_events.append({
                                "ts": ts,
                                "event": "BTD_ORDER_SKIPPED",
                                "reason": "NO_CAPITAL",
                                "bottom": bottom,
                                "rebound": rebound
                            })
                    self.btd_cooldown_until = ts + pd.Timedelta(minutes=self.btd.cooldown_minutes)
                    if self.btd.rebase_ladder_from_dip and not self.btd.isolate_from_ladder:
                        new_base = bottom * (1 + self.btd.rebase_offset)
                        self._rebase_ladder(new_base, ts, log_events)
                    self._disarm_btd()
                elif self.btd.cancel_on_miss and rebound > self.btd.rebound_max:
                    log_events.append({
                        "ts": ts,
                        "event": "BTD_DISARMED",
                        "reason": "REBOUND_TOO_FAR",
                        "bottom": bottom,
                        "rebound": rebound
                    })
                    self._disarm_btd()

        # --- SAH confirmation (priority before ladder) ---
        if self.sah_armed and self.sah_last_top is not None and self.Q > 0:
            top = self.sah_last_top
            pullback = (top - l) / top if top > 0 else 0.0
            if pullback >= self.sah.pullback_min:
                if pullback <= self.sah.pullback_max:
                    order_price = top * (1 - self.sah.limit_offset)
                    qty = self._compute_sah_qty()
                    if qty > 0:
                        log_events.append({
                            "ts": ts,
                            "event": "SAH_ORDER",
                            "top": top,
                            "pullback": pullback,
                            "order_price": order_price,
                            "order_qty": qty,
                            "order_quote": order_price * qty
                        })
                    else:
                        log_events.append({
                            "ts": ts,
                            "event": "SAH_ORDER_SKIPPED",
                            "reason": "NO_POSITION",
                            "top": top,
                            "pullback": pullback
                        })
                    self.sah_cooldown_until = ts + pd.Timedelta(minutes=self.sah.cooldown_minutes)
                    self._disarm_sah()
                elif self.sah.cancel_on_miss and pullback > self.sah.pullback_max:
                    log_events.append({
                        "ts": ts,
                        "event": "SAH_DISARMED",
                        "reason": "PULLBACK_TOO_DEEP",
                        "top": top,
                        "pullback": pullback
                    })
                    self._disarm_sah()

        # --- ladder buys (after opportunistic actions) ---
        while self.ladder_next_idx < len(self.ladder_prices):
            trig = self.ladder_prices[self.ladder_next_idx]
            if l <= trig:
                amt_q = self.ladder_amounts_quote[self.ladder_next_idx]
                if amt_q > 0:
                    q = (amt_q / trig)
                    prev_qty = self.Q
                    self.Q += q
                    self.C += amt_q * (1 + self.fees_buy)
                    if prev_qty <= 0 < self.Q:
                        self.round_start_ts = ts
                    log_events.append({"ts": ts, "event":"BUY", "price": trig, "q": q, "amt_q": amt_q, "ladder_idx": self.ladder_next_idx+1})
                self.ladder_next_idx += 1
                continue
            break

        P_BE = self.P_BE()
        if P_BE is None or self.Q <= 0:
            return None

        # --- enter trail ---
        if self.phase == PHASE_ACCUMULATE:
            TP_base = P_BE * (1 + self.trail.p_min)
            self.TP_base = TP_base
            if h >= TP_base:
                self.phase = PHASE_TRAIL
                self.reset_trail(ts, P_BE)
                self.H = max(h, TP_base)
                self.last_high_ts = ts
                log_events.append({"ts": ts, "event":"ENTER_TRAIL", "TP_base": TP_base})

        # --- trail logic ---
        if self.phase == PHASE_TRAIL:
            if self.H is None or h > self.H:
                self.H = h
                self.last_high_ts = ts
                while True:
                    s_k = self.s_for_stage(self.stage)
                    thresh = P_BE * (1 + self.trail.p_min + self.cumulative_s + s_k)
                    if self.H >= thresh:
                        self.cumulative_s += s_k
                        self.stage += 1
                        log_events.append({"ts": ts, "event":"STAGE_UP", "stage": self.stage, "H": self.H})
                        continue
                    break
            mins_from_last_high = (ts - self.last_high_ts).total_seconds()/60.0 if self.last_high_ts is not None else 0.0
            if self.next_window_minutes is not None and mins_from_last_high >= self.next_window_minutes:
                self.p_lock_cur = min(self.trail.p_lock_max, self.p_lock_cur + self.tmart.delta_lock)
                self.tau_cur = max(self.trail.tau_min, self.tau_cur * self.tmart.beta_tau)
                self.next_window_minutes *= self.tmart.m_time
                log_events.append({"ts": ts, "event":"TIME_TIGHTEN", "p_lock_cur": self.p_lock_cur, "tau_cur": self.tau_cur})
            F = self.floor_price(P_BE)
            if F is not None and l <= F <= h:
                qty = self.Q
                proceeds = qty * F * (1 - self.fees_sell)
                pnl = proceeds - self.C
                log_events.append({"ts": ts, "event":"SELL", "price": F, "proceeds": proceeds, "pnl": pnl, "qty": qty})
                self.round_active = False
                self.Q = 0.0
                self.C = 0.0
                self._reset_btd_progress()
                return {"sell_price": F, "pnl": pnl, "reason":"TRAIL_PULLBACK", "qty": qty}

        # --- idle/total caps ---
        if self.phase == PHASE_ACCUMULATE and not self.idle_checked:
            minutes_since_round = (ts - self.round_start_ts).total_seconds()/60.0
            if minutes_since_round >= self.tcaps.T_idle_max_minutes:
                target = P_BE * (1 + max(self.tcaps.p_idle, self.trail.no_loss_epsilon))
                if h >= target:
                    qty = self.Q
                    proceeds = qty * target * (1 - self.fees_sell)
                    pnl = proceeds - self.C
                    log_events.append({"ts": ts, "event":"SELL", "price": target, "proceeds": proceeds, "pnl": pnl, "note":"IDLE_EXIT", "qty": qty})
                    self.round_active = False
                    self.Q = 0.0
                    self.C = 0.0
                    self._reset_btd_progress()
                    return {"sell_price": target, "pnl": pnl, "reason":"IDLE_EXIT", "qty": qty}
                self.idle_checked = True

        minutes_round = (ts - self.round_start_ts).total_seconds()/60.0
        if minutes_round >= self.tcaps.T_total_cap_minutes:
            target = P_BE * (1 + max(self.tcaps.p_exit_min, self.trail.no_loss_epsilon))
            if h >= target:
                qty = self.Q
                proceeds = qty * target * (1 - self.fees_sell)
                pnl = proceeds - self.C
                log_events.append({"ts": ts, "event":"SELL", "price": target, "proceeds": proceeds, "pnl": pnl, "note":"TOTAL_CAP", "qty": qty})
                self.round_active = False
                self.Q = 0.0
                self.C = 0.0
                self._reset_btd_progress()
                return {"sell_price": target, "pnl": pnl, "reason":"TOTAL_CAP", "qty": qty}
        return None
