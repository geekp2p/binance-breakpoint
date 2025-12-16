import json
from pathlib import Path

import pytest

from src.pnl_summary import load_fees_snapshot, load_savepoint_rows, render_summary_report


def test_load_fees_snapshot_handles_missing_file(tmp_path: Path) -> None:
    assert load_fees_snapshot(tmp_path) is None


def test_load_fees_snapshot_parses_totals(tmp_path: Path) -> None:
    payload = {
        "fees_paid": {
            "totals": {
                "quote": "1.23",
                "bnb": 0.004,
                "other": {"ETH": "0.5", "BAD": "not-a-number"},
            }
        }
    }
    file_path = tmp_path / "profit_accumulation.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = load_fees_snapshot(tmp_path)

    assert snapshot == {"quote": 1.23, "bnb": 0.004, "other": {"ETH": 0.5}}


def test_render_summary_report_includes_fees(tmp_path: Path) -> None:
    savepoints = tmp_path / "savepoints"
    savepoints.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    payload = {"fees_paid": {"totals": {"quote": 12.5, "bnb": 0.000321, "other": {"BNBUSDC": 0.1}}}}
    (savepoints / "profit_accumulation.json").write_text(json.dumps(payload), encoding="utf-8")

    report = render_summary_report(savepoints, out_dir)

    assert "ค่าธรรมเนียมที่บันทึกไว้" in report
    assert "Quote: 12.500000" in report
    assert "BNB: 0.000321" in report
    assert "BNBUSDC:0.100000" in report


def test_render_summary_report_includes_net_after_quote_fees(tmp_path: Path) -> None:
    savepoints = tmp_path / "savepoints"
    savepoints.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    savepoint_payload = {
        "symbol": "zecusdt",
        "status": {"realized_pnl_total": 10, "unrealized_pnl": 5},
    }
    (savepoints / "ZECUSDT.json").write_text(json.dumps(savepoint_payload), encoding="utf-8")

    fees_payload = {"fees_paid": {"totals": {"quote": 3.0, "bnb": 0.0}}}
    (savepoints / "profit_accumulation.json").write_text(json.dumps(fees_payload), encoding="utf-8")

    report = render_summary_report(savepoints, out_dir)

    assert "รวมทั้งหมด (ก่อนหักค่าธรรมเนียม)" in report
    assert "Realized: 10.00 | จากราคา: 5.00 | รวม: 15.00" in report
    assert "สุทธิหลังหักค่าธรรมเนียมใน quote" in report
    assert "Realized: 7.00 | จากราคา: 5.00 | รวม: 12.00" in report


def test_render_summary_report_highlights_winners_and_losers(tmp_path: Path) -> None:
    savepoints = tmp_path / "savepoints"
    savepoints.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    winners_payload = {"symbol": "WIN", "status": {"realized_pnl_total": 12, "unrealized_pnl": 3}}
    losers_payload = {"symbol": "LOSE", "status": {"realized_pnl_total": -5, "unrealized_pnl": -7}}
    neutral_payload = {"symbol": "FLAT", "status": {"realized_pnl_total": 1, "unrealized_pnl": 0}}

    (savepoints / "win.json").write_text(json.dumps(winners_payload), encoding="utf-8")
    (savepoints / "lose.json").write_text(json.dumps(losers_payload), encoding="utf-8")
    (savepoints / "flat.json").write_text(json.dumps(neutral_payload), encoding="utf-8")

    report = render_summary_report(savepoints, out_dir)

    assert "ขาดทุนหนักที่สุด (ตาม Total)" in report
    assert "LOSE: -12.00 (Realized -5.00 | จากราคา -7.00" in report
    assert "กำไรสูงสุด (ตาม Total)" in report
    assert "WIN: 15.00 (Realized 12.00 | จากราคา 3.00" in report


def test_savepoint_realized_pnl_prefers_event_log(tmp_path: Path) -> None:
    savepoints = tmp_path / "savepoints"
    savepoints.mkdir()

    payload = {
        "symbol": "dcrusdt",
        "status": {"realized_pnl_total": 100.0, "unrealized_pnl": 0},
        "event_log": [
            {"event": "SELL", "pnl": 0.69},
            {"event": "SELL", "pnl": -0.10},
        ],
    }
    (savepoints / "DCRUSDT.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = load_savepoint_rows(savepoints)

    assert rows[0].realized == pytest.approx(0.59)