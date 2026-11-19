"""Command-line interface.

Examples::

    fastbb import-excel Modele_FAST_Bund_Bobl_v10.xlsx --out data/raw
    fastbb run --data data/raw --config config/v10.yaml --out reports/v10
    fastbb run --data data/sample --config config/v10.yaml --out reports/sample
"""

from __future__ import annotations

import argparse
import sys

from .backtest import run_backtest
from .config import Config
from .data import MarketData
from .report import write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fastbb", description="FAST Bund/Bobl backtester")
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import-excel", help="extract the data sheets of the Excel model to CSV")
    imp.add_argument("workbook")
    imp.add_argument("--out", default="data/raw")

    run = sub.add_parser("run", help="run a backtest and write the report")
    run.add_argument("--data", default="data/raw", help="directory of CSV files")
    run.add_argument("--config", default="config/v10.yaml")
    run.add_argument("--out", default="reports/latest")

    args = parser.parse_args(argv)
    if args.command == "import-excel":
        md = MarketData.from_workbook(args.workbook)
        md.validate()
        md.to_csv(args.out)
        print(f"{len(md.futures)} sessions {md.futures.index[0]:%Y-%m-%d} -> {md.futures.index[-1]:%Y-%m-%d} written to {args.out}")
        return 0

    md = MarketData.from_csv(args.data)
    md.validate()
    cfg = Config.from_yaml(args.config)
    res = run_backtest(md, cfg)
    s = write_report(res, args.out)
    print(f"{cfg.name}: {s['cumulative_return']:+.1%} cumulative, {s['annualised_return']:.1%}/yr, "
          f"vol {s['annualised_vol']:.1%}, Sharpe {s['sharpe']:.2f}, max DD {s['max_drawdown']:.1%}")
    print(f"Report written to {args.out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
