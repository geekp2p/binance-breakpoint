
import pandas as pd
import matplotlib.pyplot as plt
import os

def save_outputs(symbol, out_dir, events:pd.DataFrame, trades:pd.DataFrame, equity:pd.DataFrame, summary:dict, plot=True):
    os.makedirs(out_dir, exist_ok=True)
    events.to_csv(os.path.join(out_dir, f"{symbol}_events.csv"), index=False)
    trades.to_csv(os.path.join(out_dir, f"{symbol}_trades.csv"), index=False)
    equity.to_csv(os.path.join(out_dir, f"{symbol}_equity.csv"), index=False)
    with open(os.path.join(out_dir, f"{symbol}_summary.json"), "w", encoding="utf-8") as f:
        import json; json.dump(summary, f, ensure_ascii=False, indent=2)

    if plot and not equity.empty:
        if not pd.api.types.is_datetime64_any_dtype(equity["ts"]):
            equity["ts"] = pd.to_datetime(equity["ts"])
        if not events.empty and not pd.api.types.is_datetime64_any_dtype(events["ts"]):
            events["ts"] = pd.to_datetime(events["ts"])
        fig = plt.figure(figsize=(10,5))
        ax = plt.gca()
        equity.plot(x="ts", y="price", ax=ax, label="Price")
        evb = events[events["event"]=="BUY"] if not events.empty else pd.DataFrame()
        if not evb.empty:
            evb.plot(
                x="ts",
                y="price",
                ax=ax,
                kind="scatter",
                label="BUY",
                marker="^",
                color="tab:green",
            )
        evs = events[events["event"]=="SELL"] if not events.empty else pd.DataFrame()
        if not evs.empty:
            evs.plot(
                x="ts",
                y="price",
                ax=ax,
                kind="scatter",
                label="SELL",
                marker="v",
                color="tab:red",
            )
        ax.legend()
        fig.autofmt_xdate()
        fig.savefig(os.path.join(out_dir, f"{symbol}_plot.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
