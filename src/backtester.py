
import os, yaml
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .utils import geometric_base_allocation, compute_ladder_prices
from .strategy import (
    StrategyState,
    BuyLadderConf,
    AdaptiveLadderConf,
    AnchorDriftConf,
    ProfitTrailConf,
    TimeMartingaleConf,
    TimeCapsConf,
    ScalpModeConf,
    MicroOscillationConf,
    BuyTheDipConf,
    SellAtHeightConf,
)
from .fetchers.binance import fetch_klines_binance

@dataclass
class PairConfig:
    symbol: str
    quote: str
    b_alloc: float
    source: str
    interval: str
    lookback_days: float
    start: str
    end: str
    fees_taker: float
    fees_maker: float
    buy_d: float
    buy_m: float
    buy_n: int
    p_min: float
    s1: float
    m_step: float
    tau: float
    p_lock_base: float
    p_lock_max: float
    tau_min: float
    no_loss_epsilon: float
    W1_minutes: float
    m_time: float
    delta_lock: float
    beta_tau: float
    T_idle_max_minutes: float
    p_idle: float
    T_total_cap_minutes: float
    p_exit_min: float
    snapshot_every_bars: int
    use_maker: bool
    btd: Dict[str, Any]
    sah: Dict[str, Any]
    scalp: Dict[str, Any]
    micro: Dict[str, Any]
    adaptive_ladder: Dict[str, Any]
    anchor_drift: Dict[str, Any]
    buy_spacing: str = "geometric"
    buy_multipliers: Any = None
    buy_max_drop: float = 0.25
    buy_size_mode: str = "geometric"
    buy_gap_mode: str = "additive"
    buy_gap_factor: float = 1.0
    buy_base_order_quote: float | None = None
    buy_max_quote_per_leg: float = 0.0
    buy_max_total_quote: float = 0.0

def init_state_from_config(cfg: PairConfig):
    fees = cfg.fees_maker if cfg.use_maker else cfg.fees_taker
    return StrategyState(
        fees_buy=fees, fees_sell=fees, b_alloc=cfg.b_alloc,
        buy=BuyLadderConf(
            d_buy=cfg.buy_d,
            m_buy=cfg.buy_m,
            n_steps=cfg.buy_n,
            spacing_mode=cfg.buy_spacing,
            d_multipliers=cfg.buy_multipliers,
            max_step_drop=cfg.buy_max_drop,
            size_mode=cfg.buy_size_mode,
            gap_mode=cfg.buy_gap_mode,
            gap_factor=cfg.buy_gap_factor,
            base_order_quote=cfg.buy_base_order_quote,
            max_quote_per_leg=cfg.buy_max_quote_per_leg,
            max_total_quote=cfg.buy_max_total_quote,
        ),
        trail=ProfitTrailConf(p_min=cfg.p_min, s1=cfg.s1, m_step=cfg.m_step, tau=cfg.tau,
                              p_lock_base=cfg.p_lock_base, p_lock_max=cfg.p_lock_max,
                              tau_min=cfg.tau_min, no_loss_epsilon=cfg.no_loss_epsilon),
        tmart=TimeMartingaleConf(W1_minutes=cfg.W1_minutes, m_time=cfg.m_time,
                                 delta_lock=cfg.delta_lock, beta_tau=cfg.beta_tau),
        tcaps=TimeCapsConf(T_idle_max_minutes=cfg.T_idle_max_minutes, p_idle=cfg.p_idle,
                           T_total_cap_minutes=cfg.T_total_cap_minutes, p_exit_min=cfg.p_exit_min),
                           scalp=ScalpModeConf(**cfg.scalp),
        micro=MicroOscillationConf(**cfg.micro),
        btd=BuyTheDipConf(**cfg.btd), sah=SellAtHeightConf(**cfg.sah),
        adaptive=AdaptiveLadderConf(**cfg.adaptive_ladder),
        anchor=AnchorDriftConf(**cfg.anchor_drift),
        snapshot_every_bars=cfg.snapshot_every_bars, use_maker=cfg.use_maker
    )

def prepare_ladder(state: StrategyState, P0: float):
    state._set_initial_d_buy()
    state.adaptive_ready = False
    state.volatility_samples = []
    state.rebuild_ladder(P0)

def load_data_for_pair(pc: PairConfig, general: dict, api: dict)->pd.DataFrame:
    if pc.source.lower() == "binance":
        base_url = (general.get("binance") or {}).get("base_url", "https://api.binance.com")
        retry = (general.get("binance") or {}).get("retry", {})
        attempts = int(retry.get("attempts", 4))
        backoff = float(retry.get("backoff_sec", 0.5))
        api_key = os.getenv("BINANCE_API_KEY") or (api.get("key") if api else None)
        df = fetch_klines_binance(pc.symbol, pc.interval, base_url=base_url,
                                  start=pc.start or None, end=pc.end or None,
                                  lookback_days=pc.lookback_days, retry_attempts=attempts,
                                  backoff_sec=backoff, api_key=api_key)
        if (general.get("binance") or {}).get("save_csv", True):
            os.makedirs((general.get("binance") or {}).get("data_dir", "data"), exist_ok=True)
            out_file = os.path.join((general.get("binance") or {}).get("data_dir", "data"),
                                    f"{pc.symbol}_{pc.interval}.csv")
            df.to_csv(out_file, index=False)
        return df
    raise ValueError(f"Unsupported source: {pc.source}")

def run_backtest_for_pair(df: pd.DataFrame, cfg: PairConfig)->Dict[str, Any]:
    state = init_state_from_config(cfg)
    trades, equity_rows, log_events = [], [], []
    if len(df) == 0:
        return {"events": pd.DataFrame(), "trades": pd.DataFrame(), "equity": pd.DataFrame(), "summary": {}}
    P0 = df.iloc[0]['open']
    prepare_ladder(state, P0)
    bar_count = 0
    event_ptr = 0
    for _, row in df.iterrows():
        ts,o,h,l,c,v = row['timestamp'], row['open'],row['high'],row['low'],row['close'],row.get('volume',0.0)
        res = state.on_bar(ts,o,h,l,c,v, log_events)
        new_events = log_events[event_ptr:]
        for ev in new_events:
            if ev.get("event") == "BTD_ORDER":
                trades.append({
                    "ts": ev["ts"],
                    "side": "BUY",
                    "tag": "BTD",
                    "price": ev.get("order_price"),
                    "qty": ev.get("order_qty"),
                    "notional": ev.get("order_quote"),
                    "pnl": None,
                    "reason": "BUY_THE_DIP"
                })
            elif ev.get("event") == "SCALP_BUY":
                trades.append({
                    "ts": ev["ts"],
                    "side": "BUY",
                    "tag": "SCALP",
                    "price": ev.get("price"),
                    "qty": ev.get("qty"),
                    "notional": ev.get("order_quote"),
                    "pnl": None,
                    "reason": "SCALP_BUY"
                })
            elif ev.get("event") == "SCALP_TP":
                trades.append({
                    "ts": ev["ts"],
                    "side": "SELL",
                    "tag": "SCALP_TP",
                    "price": ev.get("price"),
                    "qty": ev.get("qty"),
                    "notional": ev.get("proceeds"),
                    "pnl": ev.get("pnl"),
                    "reason": "SCALP_TAKE_PROFIT"
                })    
            elif ev.get("event") == "SAH_ORDER":
                trades.append({
                    "ts": ev["ts"],
                    "side": "SELL",
                    "tag": "SAH",
                    "price": ev.get("order_price"),
                    "qty": ev.get("order_qty"),
                    "notional": ev.get("order_quote"),
                    "pnl": None,
                    "reason": "SELL_AT_HEIGHT"
                })
        event_ptr = len(log_events)        
        if bar_count % state.snapshot_every_bars == 0:
            P_BE = state.P_BE(); F = state.floor_price(P_BE) if P_BE is not None else None
            equity_rows.append({"ts": ts, "price": c, "Q": state.Q, "C": state.C, "P_BE": P_BE,
                                "phase": state.phase, "stage": state.stage, "H": state.H, "F": F,
                                "tau": state.tau_cur, "p_lock": state.p_lock_cur,
                                "next_ladder_idx": state.ladder_next_idx+1, "TP_base": state.TP_base,
                                "btd_armed": state.btd_armed, "sah_armed": state.sah_armed})
        bar_count += 1
        if res is not None:
            qty = res.get("qty")
            trades.append({
                "ts": ts,
                "side": "SELL",
                "tag": res["reason"],
                "price": res["sell_price"],
                "qty": qty,
                "notional": (res["sell_price"] * qty) if qty is not None else None,
                "pnl": res["pnl"],
                "reason": res["reason"]
            })
            break
    evdf = pd.DataFrame(log_events)
    tdf = pd.DataFrame(trades)
    eqdf = pd.DataFrame(equity_rows)
    realized = tdf[pd.notna(tdf.get("pnl"))] if not tdf.empty else tdf
    last_realized = realized.iloc[-1] if realized is not None and len(realized) > 0 else None
    summary = {
        "symbol": cfg.symbol, "quote": cfg.quote, "b_alloc": cfg.b_alloc,
        "n_buys": int((evdf["event"]=="BUY").sum()) if not evdf.empty else 0,
        "btd_orders": int((evdf["event"]=="BTD_ORDER").sum()) if not evdf.empty else 0,
        "scalp_buys": int((evdf["event"]=="SCALP_BUY").sum()) if not evdf.empty else 0,
        "scalp_tps": int((evdf["event"]=="SCALP_TP").sum()) if not evdf.empty else 0,
        "sah_orders": int((evdf["event"]=="SAH_ORDER").sum()) if not evdf.empty else 0,
        "sell_reason": last_realized["reason"] if last_realized is not None else None,
        "pnl": float(last_realized["pnl"]) if last_realized is not None else 0.0,
        "duration_minutes": float(((eqdf["ts"].iloc[-1] - eqdf["ts"].iloc[0]).total_seconds()/60.0)) if len(eqdf)>1 else 0.0
    }
    return {"events": evdf, "trades": tdf, "equity": eqdf, "summary": summary}
