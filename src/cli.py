from __future__ import annotations

import argparse
import sys

import pandas as pd

from .backtester import run_full_history_backtest, run_history_backtest, run_top3_signal_backtest
from .config import get_data_config
from .data_acceptance import run_data_acceptance
from .failure_review import review_failed_data
from .loaders import DataQualityError, MarketDataService, load_limitup_file
from .report import write_data_quality_reports, write_signal_reports
from .signal_engine import generate_signal
from .history_samples import run_history_sample_generation


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
        if args.command == "run-daily":
            return run_daily(args)
        if args.command == "generate-history-samples":
            return generate_history_samples(args)
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
    p.add_argument("--force-refresh", action="store_true", help="忽略缓存")

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

    p = sub.add_parser("run-daily", help="涨停池、补数、信号一键执行")
    p.add_argument("--date", default=None)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("generate-history-samples", help="generate clean historical candidate samples")
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

    return parser


# ---- commands ----

def collect_limitups(args) -> int:
    service = MarketDataService()
    frame = service.collect_limit_ups(args.date, args.lookback_days, args.force_refresh)
    print(f"limit-up rows: {len(frame)}")
    return 0


def collect_bars(args) -> int:
    service = MarketDataService()
    pool = load_limitup_file(args.limitup_file)
    bars = service.collect_bars_for_limitups(pool, args.days, args.max_codes, args.force_refresh)
    print(f"bars collected")
    return 0


def generate_signals(args) -> int:
    service = MarketDataService()
    pool = load_limitup_file(args.limitup_file)
    signals, quality_rows = service.build_signals(pool, args.days, args.max_codes, args.force_refresh)
    print(f"signals: {len(signals)}")
    return 0


def run_daily(args) -> int:
    service = MarketDataService()
    pool = service.collect_limit_ups(args.date, args.lookback_days, args.force_refresh)
    signals, quality_rows = service.build_signals(pool, args.days, args.max_codes, args.force_refresh)
    print(f"signals: {len(signals)}")
    return 0


def validate_data(args) -> int:
    frame, csv_path, md_path = run_data_acceptance(args.limitup_file, args.days, args.max_codes, args.reference_root)
    print(f"accepted: {len(frame)}")
    return 0


def review_failed_data_command(args) -> int:
    frame, csv_path, md_path, exclusion_path = review_failed_data(args.quality_file, args.days, args.force_refresh)
    print(f"failed: {len(frame)}")
    return 0


def backtest_top3(args) -> int:
    trades, summary, csv_path, md_path = run_top3_signal_backtest(args.signals_file, args.top_n, args.target_return_pct, args.include_small, args.fetch_through_date, args.days, args.force_refresh, args.entry_price_mode)
    print(f"selected: {len(trades)}")
    return 0


def backtest_history(args) -> int:
    trades, summary, factor_stats, trade_csv, summary_csv, factor_csv, md_path = run_history_backtest(args.signals_dir, args.start_date, args.end_date, args.top_n, args.hold_days, args.target_return_pct, args.stop_loss_pct, args.include_all_allowed, args.include_small, args.entry_price_mode)
    print(f"records: {len(trades)}")
    return 0


def backtest_run(args) -> int:
    res = run_full_history_backtest(args.start_date, args.end_date, args.top_n, args.hold_days, args.target_return_pct, args.stop_loss_pct, args.lookback_days, args.signal_days, args.eval_days)
    print(f"done")
    return 0


def generate_history_samples(args) -> int:
    from .history_samples import run_history_sample_generation
    res = run_history_sample_generation(args.start_date, args.end_date, args.lookback_days, args.signal_days, args.eval_days, args.max_codes, args.force_refresh, args.hold_days, args.target_return_pct, args.secondary_target_return_pct)
    print(f"history samples generated")
    return 0
