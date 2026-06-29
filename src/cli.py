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


# ---------------------------
# main
# ---------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "collect-limitups":
            return collect_limitups(args)
        if args.command == "warmup-limitups":
            return warmup_limitups(args)
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


# ---------------------------
# parser
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="短线强势股分歧承接预案系统")
    sub = parser.add_subparsers(dest="command")

    # existing
    p = sub.add_parser("collect-limitups", help="收集涨停池并主板过滤")
    p.add_argument("--date", default=None)
    p.add_argument("--lookback-days", type=int, default=1)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--force-daily-refresh", action="store_true", help="强制日线反推刷新")

    # NEW: warmup
    p = sub.add_parser("warmup-limitups", help="预热涨停池缓存（不生成样本）")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--lookback-days", type=int, default=1)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--force-daily-refresh", action="store_true")

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

    p = sub.add_parser("validate-data", help="validate cached daily/5m data")
    p.add_argument("--limitup-file", default="data/processed/recent_limitups.csv")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--reference-root", default=r"F:\dataaccept")

    p = sub.add_parser("review-failed-data", help="review failed data-quality rows")
    p.add_argument("--quality-file", default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("backtest-top3", help="backtest top3")
    p.add_argument("--signals-file", default="reports/daily_signals/signals_2026-06-25.csv")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--include-small", action="store_true")
    p.add_argument("--fetch-through-date", default=None)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("backtest-history", help="history backtest")
    p.add_argument("--signals-dir", default="reports/daily_signals")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--stop-loss-pct", type=float, default=3.0)

    p = sub.add_parser("backtest-run", help="full pipeline")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--stop-loss-pct", type=float, default=3.0)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("generate-history-samples", help="generate samples")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--force-refresh", action="store_true")

    p = sub.add_parser("analyze-factors", help="factor analysis")
    p.add_argument("--samples-file", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--min-bucket-size", type=int, default=10)
    p.add_argument("--all-candidates", action="store_true")

    p = sub.add_parser("ranking-backtest", help="ranking backtest")
    p.add_argument("--samples-file", required=True)
    p.add_argument("--model-file", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--target-return-pct", type=float, default=None)

    return parser


# ---------------------------
# warmup implementation
# ---------------------------

def warmup_limitups(args) -> int:
    service = MarketDataService()
    rows = []

    for offset in pd.date_range(args.start_date, args.end_date, freq="D"):
        if offset.weekday() >= 5:
            continue

        date = offset.strftime("%Y-%m-%d")

        try:
            frame = service.collect_limit_ups(
                trade_date=date,
                lookback_days=args.lookback_days,
                force_refresh=args.force_refresh,
                write_processed=False,
                force_daily_refresh=(True if args.force_daily_refresh else None),
            )

            sources = frame["source"].value_counts().to_dict() if "source" in frame.columns else {}

            rows.append({
                "date": date,
                "rows": len(frame),
                "status": "ok",
                "sources": str(sources),
            })

            print(f"{date} ok rows={len(frame)}")

        except Exception as exc:
            rows.append({
                "date": date,
                "rows": 0,
                "status": "failed",
                "sources": "",
                "error": str(exc),
            })
            print(f"{date} failed: {exc}")

    df = pd.DataFrame(rows)
    out_dir = get_data_config().reports_dir / "warmup"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"limitup_warmup_{args.start_date}_{args.end_date}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"warmup done -> {out}")
    return 0


# existing function stubs unchanged

def collect_limitups(args):
    service = MarketDataService()
    frame = service.collect_limit_ups(
        trade_date=args.date,
        lookback_days=args.lookback_days,
        force_refresh=args.force_refresh,
        force_daily_refresh=(True if args.force_daily_refresh else None),
    )
    print(f"limit-up rows: {len(frame)}")
    print(f"saved: {get_data_config().processed_dir / 'recent_limitups.csv'}")
    return 0
