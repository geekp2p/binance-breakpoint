import json
from pathlib import Path

from src.pnl_summary import load_fees_snapshot, render_summary_report


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