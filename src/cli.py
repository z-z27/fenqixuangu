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
from .ranking_backtest import run_ranking_backtest
from .report import write_data_quality_reports, write_signal_reports
from .research_models import run_factor_analysis
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
        if args.command == "analyze-factors":
            return analyze_factors(args)
        if args.command == "ranking-backtest":
            return ranking_backtest(args)
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

    p = sub.add_parser("collect-limitups")
    p.add_argument("--date", default=None)
    p.add_argument("--lookback-days", type=int, default=1)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("collect-bars")
    p.add_argument("--limitup-file", default="data/processed/recent_limitups.csv")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("generate-history-samples")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--target-return-pct", type=float, default=7.0)

    # NEW LAYER 2
    p = sub.add_parser("analyze-factors", help="pure factor analysis (no model generation)")
    p.add_argument("--samples-file", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--min-bucket-size", type=int, default=10)
    p.add_argument("--all-candidates", action="store_true")

    p = sub.add_parser("ranking-backtest")
    p.add_argument("--samples-file", required=True)
    p.add_argument("--model-file", required=True)

    return parser


def analyze_factors(args) -> int:
    result = run_factor_analysis(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        target_return_pct=args.target_return_pct,
        min_bucket_size=args.min_bucket_size,
        eligible_only=not args.all_candidates,
    )
    factor_summary, factor_buckets, daily_stability, pair_review, summary_csv, buckets_csv, stability_csv, pair_csv, md = result
    print(f"factor summary: {len(factor_summary)}")
    print(f"factor buckets: {len(factor_buckets)}")
    print(f"daily stability: {len(daily_stability)}")
    print(f"pair review: {len(pair_review)}")
    print(f"csv: {summary_csv}")
    print(f"markdown: {md}")
    return 0


def generate_research_model(args):
    raise RuntimeError("deprecated: use analyze-factors + manual model construction")
