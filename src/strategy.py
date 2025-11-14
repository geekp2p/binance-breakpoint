
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

PHASE_ACCUMULATE = "ACCUMULATE"
PHASE_TRAIL = "TRAIL"

@dataclass
class BuyLadderConf:
    d_buy: float
    m_buy: float
    n_steps: int

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

# --- Scaffolds ---
@dataclass
class BuyTheDipConf:
    enabled: bool = False
    dip_threshold: float = 0.05
    rebound_min: float = 0.004
    rebound_max: float = 0.02
    cooldown_minutes: float = 10.0
    rebase_ladder_from_dip: bool = True
    rebase_offset: float = 0.0

@dataclass
class SellAtHeightConf:
    enabled: bool = False
    height_threshold: float = 0.04
    pullback_min: float = 0.004
    pullback_max: float = 0.02
    cooldown_minutes: float = 10.0

@dataclass
class StrategyState:
    fees_buy: float
    fees_sell: float
    b_alloc: float
    buy: BuyLadderConf
    trail: ProfitTrailConf
    tmart: TimeMartingaleConf
    tcaps: TimeCapsConf
    btd: BuyTheDipConf
    sah: SellAtHeightConf
    snapshot_every_bars: int
    use_maker: bool

    # Core round state
    round_active: bool = True
    phase: str = PHASE_ACCUMULATE
    P0: Optional[float] = None
    ladder_prices: List[float] = field(default_factory=list)
    ladder_amounts_quote: List[float] = field(default_factory=list)
    ladder_next_idx: int = 0
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

    # Scaffolding flags (no auto-trading yet)
    btd_armed: bool = False
    btd_last_bottom: Optional[float] = None
    btd_last_arm_ts: Optional[pd.Timestamp] = None

    sah_armed: bool = False
    sah_last_top: Optional[float] = None
    sah_last_arm_ts: Optional[pd.Timestamp] = None

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
        return self.trail.s1 * (self.trail.m_step ** (stage - 1))

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

    # --- Scaffolding: detect/arm BTD/SAH, but do NOT auto-execute ---
    def _maybe_arm_btd(self, ts, low):
        if not self.btd.enabled or self.ladder_next_idx == 0:
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
        if high >= self.H * (1 + self.sah.height_threshold):
            self.sah_armed = True
            self.sah_last_top = high
            self.sah_last_arm_ts = ts
            return {"ts": ts, "event": "SAH_ARMED", "top": high, "from": self.H}
        return None

    def on_bar(self, ts, o, h, l, c, volume, log_events:List[Dict[str,Any]]):
        if self.P0 is None:
            self.P0 = o
            self.round_start_ts = ts

        # --- ladder buys ---
        while self.ladder_next_idx < len(self.ladder_prices):
            trig = self.ladder_prices[self.ladder_next_idx]
            if l <= trig:
                amt_q = self.ladder_amounts_quote[self.ladder_next_idx]
                if amt_q > 0:
                    q = (amt_q / trig)
                    self.Q += q
                    self.C += amt_q * (1 + self.fees_buy)
                    log_events.append({"ts": ts, "event":"BUY", "price": trig, "q": q, "amt_q": amt_q, "ladder_idx": self.ladder_next_idx+1})
                self.ladder_next_idx += 1
                continue
            break

        # --- Arm BTD/SAH (scaffold only; no auto-action) ---
        ev = self._maybe_arm_btd(ts, l)
        if ev: log_events.append(ev)
        if self.phase == PHASE_TRAIL:
            ev2 = self._maybe_arm_sah(ts, h)
            if ev2: log_events.append(ev2)

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
                proceeds = self.Q * F * (1 - self.fees_sell)
                pnl = proceeds - self.C
                log_events.append({"ts": ts, "event":"SELL", "price": F, "proceeds": proceeds, "pnl": pnl})
                self.round_active = False
                self.Q = 0.0
                self.C = 0.0
                return {"sell_price": F, "pnl": pnl, "reason":"TRAIL_PULLBACK"}

        # --- idle/total caps ---
        if self.phase == PHASE_ACCUMULATE and not self.idle_checked:
            minutes_since_round = (ts - self.round_start_ts).total_seconds()/60.0
            if minutes_since_round >= self.tcaps.T_idle_max_minutes:
                target = P_BE * (1 + max(self.tcaps.p_idle, self.trail.no_loss_epsilon))
                if h >= target:
                    proceeds = self.Q * target * (1 - self.fees_sell)
                    pnl = proceeds - self.C
                    log_events.append({"ts": ts, "event":"SELL", "price": target, "proceeds": proceeds, "pnl": pnl, "note":"IDLE_EXIT"})
                    self.round_active = False
                    self.Q = 0.0
                    self.C = 0.0
                    return {"sell_price": target, "pnl": pnl, "reason":"IDLE_EXIT"}
                self.idle_checked = True

        minutes_round = (ts - self.round_start_ts).total_seconds()/60.0
        if minutes_round >= self.tcaps.T_total_cap_minutes:
            target = P_BE * (1 + max(self.tcaps.p_exit_min, self.trail.no_loss_epsilon))
            if h >= target:
                proceeds = self.Q * target * (1 - self.fees_sell)
                pnl = proceeds - self.C
                log_events.append({"ts": ts, "event":"SELL", "price": target, "proceeds": proceeds, "pnl": pnl, "note":"TOTAL_CAP"})
                self.round_active = False
                self.Q = 0.0
                self.C = 0.0
                return {"sell_price": target, "pnl": pnl, "reason":"TOTAL_CAP"}
        return None
