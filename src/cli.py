from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import get_data_config
from .loaders import MarketDataService, load_limitup_file
from .report import write_signal_reports
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

    p = sub.add_parser("run-daily", help="涨停池、补数、信号一键执行")
    p.add_argument("--date", default=None)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")
    return parser


def collect_limitups(args) -> int:
    service = MarketDataService()
    frame = service.collect_limit_ups(
        trade_date=args.date,
        lookback_days=args.lookback_days,
        force_refresh=args.force_refresh,
    )
    print(f"limit-up rows: {len(frame)}")
    print(f"saved: {get_data_config().processed_dir / 'recent_limitups.csv'}")
    return 0


def collect_bars(args) -> int:
    service = MarketDataService()
    pool = load_limitup_file(args.limitup_file)
    bars = service.collect_bars_for_limitups(
        pool,
        days=args.days,
        max_codes=args.max_codes,
        force_refresh=args.force_refresh,
    )
    print(f"collected bars for {len(bars)} codes")
    return 0


def generate_signals(args) -> int:
    service = MarketDataService()
    pool = load_limitup_file(args.limitup_file)
    signals = _build_signals(service, pool, args.days, args.max_codes, args.force_refresh)
    output_dir = get_data_config().reports_dir / "daily_signals"
    trade_date = _latest_trade_date(pool)
    csv_path, md_path = write_signal_reports(signals, output_dir, trade_date=trade_date)
    print(f"signals: {len(signals)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    return 0


def run_daily(args) -> int:
    service = MarketDataService()
    pool = service.collect_limit_ups(
        trade_date=args.date,
        lookback_days=args.lookback_days,
        force_refresh=args.force_refresh,
    )
    signals = _build_signals(service, pool, args.days, args.max_codes, args.force_refresh)
    output_dir = get_data_config().reports_dir / "daily_signals"
    trade_date = _latest_trade_date(pool)
    csv_path, md_path = write_signal_reports(signals, output_dir, trade_date=trade_date)
    print(f"limit-up rows: {len(pool)}")
    print(f"signals: {len(signals)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    return 0


def _build_signals(
    service: MarketDataService,
    pool: pd.DataFrame,
    days: int | None,
    max_codes: int | None,
    force_refresh: bool,
):
    codes = pool["code"].dropna().astype(str).drop_duplicates().tolist()
    if max_codes:
        codes = codes[: int(max_codes)]
    signals = []
    for code in codes:
        current = pool[pool["code"].astype(str) == code].tail(1)
        name = str(current.iloc[0].get("name") or "") if not current.empty else ""
        bars = service.get_stock_bars(code, days=days, force_refresh=force_refresh)
        signal = generate_signal(code, name, bars.daily, bars.minute_5m, pool)
        signals.append(signal)
    return signals


def _latest_trade_date(pool: pd.DataFrame) -> str:
    if pool is None or pool.empty or "trade_date" not in pool.columns:
        return pd.Timestamp.now().strftime("%Y-%m-%d")
    return str(pool["trade_date"].dropna().max())


if __name__ == "__main__":
    raise SystemExit(main())
