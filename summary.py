"""CLI helper to summarise PnL across savepoints and backtest outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pnl_summary import render_summary_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate realised/unrealised PnL")
    parser.add_argument("--savepoint-dir", default="savepoint", help="Directory containing live savepoints")
    parser.add_argument("--out-dir", default="out", help="Directory containing backtest summaries")
    args = parser.parse_args()

    report = render_summary_report(Path(args.savepoint_dir), Path(args.out_dir))
    print(report)


if __name__ == "__main__":
    main()