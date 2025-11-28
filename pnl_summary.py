"""Quick PnL summary utility for live savepoints and backtest outputs."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _collect_savepoints(directory: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not directory.exists():
        return entries
    for json_path in directory.glob("**/*.json"):
        payload = _load_json(json_path)
        if not payload:
            continue
        state = payload.get("state") or {}
        symbol = str(payload.get("symbol") or json_path.stem).upper()
        qty = float(state.get("Q") or 0.0)
        cost = float(state.get("C") or 0.0)
        latest_price = float(payload.get("latest_price") or 0.0)
        realized = float(payload.get("realized_pnl_total") or 0.0)
        price_move = (qty * latest_price) - cost
        entries.append(
            {
                "symbol": symbol,
                "realized": realized,
                "price_move": price_move,
                "total": realized + price_move,
                "source": "live",
                "qty": qty,
                "latest_price": latest_price,
                "path": str(json_path),
                "timestamp": payload.get("saved_at"),
            }
        )
    return entries


def _collect_backtests(directory: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not directory.exists():
        return entries
    for json_path in directory.glob("**/*summary*.json"):
        payload = _load_json(json_path)
        if not payload:
            continue
        symbol = str(payload.get("symbol") or json_path.stem.replace("_summary", "")).upper()
        pnl = float(payload.get("pnl") or 0.0)
        entries.append(
            {
                "symbol": symbol,
                "realized": pnl,
                "price_move": 0.0,
                "total": pnl,
                "source": "backtest",
                "qty": 0.0,
                "latest_price": None,
                "path": str(json_path),
                "timestamp": None,
            }
        )
    return entries


def _merge_by_symbol(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for row in rows:
        symbol = str(row["symbol"])
        entry = merged.setdefault(
            symbol,
            {
                "symbol": symbol,
                "realized": 0.0,
                "price_move": 0.0,
                "total": 0.0,
                "sources": set(),
            },
        )
        entry["realized"] = float(entry["realized"]) + float(row["realized"])
        entry["price_move"] = float(entry["price_move"]) + float(row["price_move"])
        entry["total"] = float(entry["total"]) + float(row["total"])
        entry["sources"].add(str(row.get("source") or "?"))
    for entry in merged.values():
        entry["sources"] = ",".join(sorted(entry["sources"]))
    return sorted(merged.values(), key=lambda r: r["symbol"])


def _format_currency(value: float) -> str:
    return f"{value:,.2f}"


def _render_table(rows: List[Dict[str, object]]) -> str:
    headers = ["Symbol", "Realized", "Price Move", "Total", "Sources"]
    data = [
        [
            r["symbol"],
            _format_currency(float(r["realized"])),
            _format_currency(float(r["price_move"])),
            _format_currency(float(r["total"])),
            r.get("sources", r.get("source", "-")),
        ]
        for r in rows
    ]
    total_realized = sum(float(r["realized"]) for r in rows)
    total_price = sum(float(r["price_move"]) for r in rows)
    total_all = sum(float(r["total"]) for r in rows)
    data.append(["TOTAL", _format_currency(total_realized), _format_currency(total_price), _format_currency(total_all), ""])
    widths = [max(len(str(col)) for col in column) for column in zip(headers, *data)]

    def fmt_row(row: List[str]) -> str:
        return " | ".join(str(cell).ljust(width) for cell, width in zip(row, widths))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in data)
    return "\n".join(lines)


def _print_details(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    print("\nDetails:")
    for row in sorted(rows, key=lambda r: (r.get("source", ""), r["symbol"])):
        ts = row.get("timestamp")
        ts_str = ts if ts else "-"
        price_info = ""
        if row.get("source") == "live":
            price_info = f" | qty={row['qty']:.6f} @ last={row['latest_price']:.4f}"
        print(
            f"  [{row['source']}] {row['symbol']}: realized={row['realized']:.2f} "
            f"price_move={row['price_move']:.2f} total={row['total']:.2f}{price_info}"
            f" | saved={ts_str} | {row['path']}"
        )


def run(savepoint_dir: Path, out_dir: Path) -> None:
    savepoint_rows = _collect_savepoints(savepoint_dir)
    backtest_rows = _collect_backtests(out_dir)
    all_rows = savepoint_rows + backtest_rows
    merged_rows = _merge_by_symbol(all_rows)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n== PnL Summary @ {now} ==")
    if not merged_rows:
        print("(no data found)")
        return
    print(_render_table(merged_rows))
    _print_details(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PnL from savepoints and backtests")
    parser.add_argument("--savepoint-dir", type=Path, default=Path("savepoint"))
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=0.0,
        help="If >0, print summary repeatedly every N minutes",
    )
    args = parser.parse_args()

    interval = max(0.0, float(args.interval_minutes))
    while True:
        run(args.savepoint_dir, args.out_dir)
        if interval <= 0:
            break
        time.sleep(interval * 60)


if __name__ == "__main__":
    main()