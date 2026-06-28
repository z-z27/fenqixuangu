from __future__ import annotations

import argparse
import sys

import pandas as pd

from .backtester import run_full_history_backtest, run_history_backtest, run_top3_signal_backtest
from .config import get_data_config
from .data_acceptance import run_data_acceptance
from .failure_review import review_failed_data
from .history_samples import run_history_sample_generation
from .loaders import DataQualityError, MarketDataService, load_limitup_file
from .report import write_data_quality_reports, write_signal_reports
from .signal_engine import generate_signal


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "collect-limitups":
            return collect_limitups(args)
        if args.command == "collect-bars":
            return collect_bars(args)
        if args.command == "generate-signals":
            return generate_signals(args)
        if args.command == "validate-data":
            return validate_data(args)
        if args.command == "review-failed-data":
            return review_failed_data_command(args)
        if args.command == "backtest-top3":
            return backtest_top3(args)
        if args.command == "backtest-history":
            return backtest_history(args)
        if args.command == "backtest-run":
            return backtest_run(args)
        if args.command == "generate-history-samples":
            return generate_history_samples(args)
        if args.command == "run-daily":
            return run_daily(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="短线强势股分歧承接预案系统")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("collect-limitups", help="收集涨停池并主板过滤")
    p.add_argument("--date", default=None, help="交易日，例如 2026-06-25；默认今天")
    p.add_argument("--lookback-days", type=int, default=1, help="向前扫描自然日数量")
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("collect-bars", help="为涨停池候选补日线和 5 分钟线")
    p.add_argument("--limitup-file", default="data/processed/recent_limitups.csv")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("generate-signals", help="生成 D2 交易预案")
    p.add_argument("--limitup-file", default="data/processed/recent_limitups.csv")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("validate-data", help="validate cached daily/5m data and MA calculations")
    p.add_argument("--limitup-file", default="data/processed/recent_limitups.csv")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--reference-root", default=r"F:\dataaccept")

    p = sub.add_parser("review-failed-data", help="review failed data-quality rows and write exclusion list")
    p.add_argument("--quality-file", default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("backtest-top3", help="backtest daily top 3 ranked tradable signals")
    p.add_argument("--signals-file", default="reports/daily_signals/signals_2026-06-25.csv")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--include-small", action="store_true")
    p.add_argument("--fetch-through-date", default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--entry-price-mode", choices=["zone_max", "confirmation_close"], default="zone_max")

    p = sub.add_parser("backtest-history", help="backtest multiple daily signal files and write full records")
    p.add_argument("--signals-dir", default="reports/daily_signals")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--stop-loss-pct", type=float, default=3.0)
    p.add_argument("--include-all-allowed", action="store_true")
    p.add_argument("--include-small", action="store_true")
    p.add_argument("--entry-price-mode", choices=["zone_max", "confirmation_close"], default="zone_max")

    p = sub.add_parser("backtest-run", help="run full historical backtest from data collection to evaluation")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--stop-loss-pct", type=float, default=3.0)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--signal-days", type=int, default=None)
    p.add_argument("--eval-days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--include-all-allowed", action="store_true")
    p.add_argument("--include-small", action="store_true")
    p.add_argument("--entry-price-mode", choices=["zone_max", "confirmation_close"], default="zone_max")

    p = sub.add_parser("generate-history-samples", help="generate clean historical candidate samples without execution backtest")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--signal-days", type=int, default=None)
    p.add_argument("--eval-days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--secondary-target-return-pct", type=float, default=10.0)

    p = sub.add_parser("run-daily", help="涨停池、补数、信号一键执行")
    p.add_argument("--date", default=None)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")
    return parser


# (rest unchanged omitted for brevity in commit)
