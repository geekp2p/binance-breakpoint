"""Summarise realised/unrealised PnL from savepoints and backtests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class PnLRow:
    """Row representing realised/unrealised PnL for a trading pair."""

    symbol: str
    source: str
    realized: float
    price_move: float
    total: float
    qty: Optional[float]
    price: Optional[float]
    path: Path

    @classmethod
    def from_savepoint(cls, path: Path, payload: Dict[str, object]) -> "PnLRow":
        symbol = str(payload.get("symbol") or path.stem).upper()
        status = payload.get("status") if isinstance(payload.get("status"), dict) else {}
        realized = _to_float(status.get("realized_pnl_total") or payload.get("realized_pnl_total"))
        price_move = _to_float(status.get("unrealized_pnl"))
        qty = _to_float(status.get("qty"))
        price = _to_float(status.get("price") or payload.get("latest_price"))
        total = realized + price_move
        return cls(symbol, "savepoint", realized, price_move, total, qty, price, path)

    @classmethod
    def from_summary(cls, path: Path, payload: Dict[str, object]) -> "PnLRow":
        symbol = str(payload.get("symbol") or _symbol_from_name(path.stem)).upper()
        realized = _to_float(payload.get("pnl"))
        return cls(symbol, "backtest", realized, 0.0, realized, None, None, path)


def _symbol_from_name(name: str) -> str:
    if "_summary" in name:
        return name.split("_summary", 1)[0]
    return name


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_savepoint_rows(directory: Path) -> List[PnLRow]:
    if not directory.exists():
        return []
    rows: List[PnLRow] = []
    for path in sorted(directory.glob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        rows.append(PnLRow.from_savepoint(path, payload))
    return rows


def load_backtest_rows(directory: Path) -> List[PnLRow]:
    if not directory.exists():
        return []
    rows: List[PnLRow] = []
    for path in sorted(directory.glob("*summary*")):
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = _load_json(path)
            if not isinstance(payload, dict) or "pnl" not in payload:
                continue
            rows.append(PnLRow.from_summary(path, payload))
        elif suffix == ".csv":
            try:
                import csv

                with path.open("r", encoding="utf-8", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if "pnl" not in row:
                            continue
                        rows.append(PnLRow.from_summary(path, row))
            except OSError:
                continue
    return rows


def aggregate_by_symbol(rows: Iterable[PnLRow]) -> List[PnLRow]:
    totals: Dict[str, Dict[str, float]] = {}
    for row in rows:
        agg = totals.setdefault(row.symbol, {"realized": 0.0, "price_move": 0.0})
        agg["realized"] += row.realized
        agg["price_move"] += row.price_move
    out: List[PnLRow] = []
    for symbol, agg in sorted(totals.items()):
        realized = agg["realized"]
        price_move = agg["price_move"]
        out.append(
            PnLRow(
                symbol=symbol,
                source="combined",
                realized=realized,
                price_move=price_move,
                total=realized + price_move,
                qty=None,
                price=None,
                path=Path(symbol),
            )
        )
    return out


def render_table(rows: Sequence[PnLRow], *, include_path: bool = False) -> str:
    if not rows:
        return "(ไม่มีข้อมูล)"
    headers = ["Symbol", "Source", "Realized", "PriceMove", "Total", "Qty", "Price"]
    if include_path:
        headers.append("Path")

    def row_cells(row: PnLRow) -> List[str]:
        cells = [
            row.symbol,
            row.source,
            f"{row.realized:,.2f}",
            f"{row.price_move:,.2f}",
            f"{row.total:,.2f}",
            f"{row.qty:,.6f}" if row.qty else "-",
            f"{row.price:,.4f}" if row.price else "-",
        ]
        if include_path:
            cells.append(str(row.path))
        return cells

    table_rows = [headers] + [row_cells(r) for r in rows]
    widths = [max(len(str(col)) for col in column) for column in zip(*table_rows)]

    def fmt_line(values: Sequence[str]) -> str:
        return " | ".join(val.ljust(width) for val, width in zip(values, widths))

    lines = [fmt_line(headers), fmt_line(["-" * w for w in widths])]
    lines.extend(fmt_line(r) for r in table_rows[1:])
    return "\n".join(lines)


def render_summary_report(savepoint_dir: Path, out_dir: Path) -> str:
    detail_rows = load_savepoint_rows(savepoint_dir) + load_backtest_rows(out_dir)
    combined_rows = aggregate_by_symbol(detail_rows)
    realized_total = sum(r.realized for r in detail_rows)
    price_move_total = sum(r.price_move for r in detail_rows)
    grand_total = realized_total + price_move_total

    sections = [
        "รวม PnL รายไฟล์",
        render_table(detail_rows, include_path=True),
        "",
        "สรุปตามคู่",
        render_table(combined_rows),
        "",
        "รวมทั้งหมด",
        f"Realized: {realized_total:,.2f} | จากราคา: {price_move_total:,.2f} | รวม: {grand_total:,.2f}",
    ]
    return "\n".join(sections)


__all__ = [
    "PnLRow",
    "aggregate_by_symbol",
    "load_backtest_rows",
    "load_savepoint_rows",
    "render_summary_report",
    "render_table",
]