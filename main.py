
import argparse, os, json, yaml
from src.backtester import PairConfig, run_backtest_for_pair, load_data_for_pair
from src.report import save_outputs

def load_config(path):
    # Expand ${ENV} in yaml values
    import re
    pattern = re.compile(r"\$\{([^}]+)\}")
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    def repl(m): return os.getenv(m.group(1), "")
    txt = pattern.sub(repl, txt)
    return yaml.safe_load(txt)

def run(cfg_path, out_dir):
    cfg = load_config(cfg_path)
    os.makedirs(out_dir, exist_ok=True)
    general = cfg.get("general", {})
    api = cfg.get("api", {})
    snapshot_every = int(general.get("snapshot_every_bars", 1))
    use_maker = bool(general.get("use_maker", True))

    for pc in cfg.get("pairs", []):
        paircfg = PairConfig(
            symbol=pc["symbol"], quote=pc.get("quote","QUOTE"), b_alloc=float(pc["b_alloc"]),
            source=pc.get("source","binance"), interval=pc.get("interval","1m"),
            lookback_days=float(pc.get("lookback_days", 1)) if pc.get("lookback_days") is not None else None,
            start=pc.get("start",""), end=pc.get("end",""),
            fees_taker=float(pc["fees"]["taker"]), fees_maker=float(pc["fees"]["maker"]),
            buy_d=float(pc["buy_ladder"]["d_buy"]), buy_m=float(pc["buy_ladder"]["m_buy"]), buy_n=int(pc["buy_ladder"]["n_steps"]),
            p_min=float(pc["profit_trail"]["p_min"]), s1=float(pc["profit_trail"]["s1"]),
            m_step=float(pc["profit_trail"]["m_step"]), tau=float(pc["profit_trail"]["tau"]),
            p_lock_base=float(pc["profit_trail"]["p_lock_base"]), p_lock_max=float(pc["profit_trail"]["p_lock_max"]),
            tau_min=float(pc["profit_trail"]["tau_min"]), no_loss_epsilon=float(pc["profit_trail"].get("no_loss_epsilon", 0.0005)),
            W1_minutes=float(pc["time_martingale"]["W1_minutes"]), m_time=float(pc["time_martingale"]["m_time"]),
            delta_lock=float(pc["time_martingale"]["delta_lock"]), beta_tau=float(pc["time_martingale"]["beta_tau"]),
            T_idle_max_minutes=float(pc["time_caps"]["T_idle_max_minutes"]), p_idle=float(pc["time_caps"]["p_idle"]),
            T_total_cap_minutes=float(pc["time_caps"]["T_total_cap_minutes"]), p_exit_min=float(pc["time_caps"]["p_exit_min"]),
            snapshot_every_bars=snapshot_every, use_maker=use_maker,
            btd=pc.get("features",{}).get("buy_the_dip", {"enabled": False}),
            sah=pc.get("features",{}).get("sell_at_height", {"enabled": False}),
            scalp=pc.get("features",{}).get("scalp_mode", {"enabled": False}),
            adaptive_ladder=pc.get("features",{}).get("adaptive_ladder", {"enabled": False}),
        )
        df = load_data_for_pair(paircfg, general, api)
        res = run_backtest_for_pair(df, paircfg)
        events, trades, equity, summary = res["events"], res["trades"], res["equity"], res["summary"]
        save_outputs(paircfg.symbol, out_dir, events, trades, equity, summary, plot=pc.get("plotting",{}).get("enable", True))
        print(f"[{paircfg.symbol}] summary:", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()
    run(args.config, args.out_dir)
