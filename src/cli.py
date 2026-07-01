from __future__ import annotations

import argparse
import sys

import pandas as pd

from .backtester import run_full_history_backtest, run_history_backtest, run_top3_signal_backtest
from .config import get_data_config
from .data_acceptance import run_data_acceptance
from .daily_ranking import DEFAULT_DAILY_RANKING_MODEL, DEFAULT_DAILY_TOP_N, apply_daily_research_ranking
from .failure_review import review_failed_data
from .history_samples import run_history_sample_generation
from .loaders import DataQualityError, MarketDataService, load_limitup_file
from .logistic_v003 import (
    DEFAULT_INITIAL_TRAIN_DAYS,
    DEFAULT_L2,
    DEFAULT_SAMPLES_FILE as DEFAULT_LOGISTIC_V003_SAMPLES_FILE,
    DEFAULT_TARGET_RETURN_PCT as DEFAULT_LOGISTIC_V003_TARGET_RETURN_PCT,
    DEFAULT_TOP_N as DEFAULT_LOGISTIC_V003_TOP_N,
    run_logistic_v003_l2_grid,
    run_logistic_v003_research,
)
from .ranking_backtest import DEFAULT_TARGET_COLUMN, run_ranking_backtest
from .report import write_data_quality_reports, write_signal_reports
from .research_models import run_factor_analysis
from .signal_engine import generate_signal
from .v004a import (
    DEFAULT_INITIAL_TRAIN_DAYS as DEFAULT_V004A_INITIAL_TRAIN_DAYS,
    DEFAULT_L2_GRID as DEFAULT_V004A_L2_GRID,
    DEFAULT_POSITIVE_WEIGHT_GRID as DEFAULT_V004A_POSITIVE_WEIGHT_GRID,
    DEFAULT_SAMPLES_FILE as DEFAULT_V004A_SAMPLES_FILE,
    DEFAULT_TARGET_RETURN_PCT as DEFAULT_V004A_TARGET_RETURN_PCT,
    DEFAULT_THRESHOLD_GRID as DEFAULT_V004A_THRESHOLD_GRID,
    DEFAULT_TOP_N as DEFAULT_V004A_TOP_N,
    run_v004a_research,
)


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
        if args.command == "train-logistic-v003":
            return train_logistic_v003(args)
        if args.command == "train-v004a":
            return train_v004a(args)
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
    p.add_argument("--force-refresh", action="store_true", help="忽略涨停池缓存")
    p.add_argument("--force-daily-refresh", action="store_true", help="当涨停池接口失败并进入日线反推时，同时忽略不复权日线缓存")
    p.add_argument("--workers", type=int, default=1, help="日线反推时的股票级并发数；collect 默认 1")

    p = sub.add_parser("warmup-limitups", help="按区间预热涨停池缓存；不生成样本、不补5分钟线")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--lookback-days", type=int, default=1)
    p.add_argument("--force-refresh", action="store_true", help="忽略涨停池缓存，重新抓取/反推并覆盖缓存")
    p.add_argument("--force-daily-refresh", action="store_true", help="日线反推时也忽略不复权日线缓存；一般不建议开启")
    p.add_argument("--workers", type=int, default=6, help="日线反推时的股票级并发数；warmup 默认 6，上限由 loaders 限制")

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
    p.add_argument("--ranking-model", default=str(DEFAULT_DAILY_RANKING_MODEL))
    p.add_argument("--top-n", type=int, default=DEFAULT_DAILY_TOP_N)

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
    p.add_argument("--workers", type=int, default=6, help="涨停池日线反推时的股票级并发数")
    p.add_argument("--hold-days", type=int, default=10)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--secondary-target-return-pct", type=float, default=10.0)

    p = sub.add_parser("analyze-factors", help="run pure factor analysis; does not generate ranking_model JSON")
    p.add_argument("--samples-file", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--target-return-pct", type=float, default=7.0)
    p.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    p.add_argument("--min-bucket-size", type=int, default=10)
    p.add_argument("--all-candidates", action="store_true", help="analyze all candidates with target return data instead of eligible_for_trade only")

    p = sub.add_parser("ranking-backtest", help="validate a manually constructed ranking model against history candidates")
    p.add_argument("--samples-file", required=True)
    p.add_argument("--model-file", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--target-return-pct", type=float, default=None)
    p.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)

    p = sub.add_parser("train-logistic-v003", help="train research-only logistic_v003 model for D2 open to D3 high target")
    p.add_argument("--samples-file", default=str(DEFAULT_LOGISTIC_V003_SAMPLES_FILE))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--top-n", type=int, default=DEFAULT_LOGISTIC_V003_TOP_N)
    p.add_argument("--initial-train-days", type=int, default=DEFAULT_INITIAL_TRAIN_DAYS)
    p.add_argument("--l2", type=float, default=DEFAULT_L2)
    p.add_argument("--l2-grid", default=None)
    p.add_argument("--target-return-pct", type=float, default=DEFAULT_LOGISTIC_V003_TARGET_RETURN_PCT)

    p = sub.add_parser("train-v004a", help="run research-only v004a weighted logistic walk-forward validation")
    p.add_argument("--samples-file", default=str(DEFAULT_V004A_SAMPLES_FILE))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--top-n", type=int, default=DEFAULT_V004A_TOP_N)
    p.add_argument("--initial-train-days", type=int, default=DEFAULT_V004A_INITIAL_TRAIN_DAYS)
    p.add_argument("--target-return-pct", type=float, default=DEFAULT_V004A_TARGET_RETURN_PCT)
    p.add_argument("--l2-grid", default=",".join(f"{float(value):g}" for value in DEFAULT_V004A_L2_GRID))
    p.add_argument("--positive-weight-grid", default=",".join(f"{float(value):g}" for value in DEFAULT_V004A_POSITIVE_WEIGHT_GRID))
    p.add_argument("--threshold-grid", default=",".join(f"{float(value):g}" for value in DEFAULT_V004A_THRESHOLD_GRID))

    p = sub.add_parser("run-daily", help="涨停池、补数、信号一键执行")
    p.add_argument("--date", default=None)
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--ranking-model", default=str(DEFAULT_DAILY_RANKING_MODEL))
    p.add_argument("--top-n", type=int, default=DEFAULT_DAILY_TOP_N)
    return parser


def collect_limitups(args) -> int:
    service = MarketDataService()
    frame = service.collect_limit_ups(
        trade_date=args.date,
        lookback_days=args.lookback_days,
        force_refresh=args.force_refresh,
        force_daily_refresh=True if args.force_daily_refresh else None,
        workers=args.workers,
    )
    print(f"limit-up rows: {len(frame)}")
    print(f"saved: {get_data_config().processed_dir / 'recent_limitups.csv'}")
    return 0


def warmup_limitups(args) -> int:
    service = MarketDataService()
    rows: list[dict[str, object]] = []

    for current in pd.date_range(args.start_date, args.end_date, freq="D"):
        if current.weekday() >= 5:
            continue
        date_text = current.strftime("%Y-%m-%d")
        row: dict[str, object] = {
            "date": date_text,
            "status": "failed",
            "rows": 0,
            "actual_date": "",
            "sources": "",
            "error": "",
        }
        try:
            frame = service.collect_limit_ups(
                trade_date=date_text,
                lookback_days=args.lookback_days,
                force_refresh=args.force_refresh,
                write_processed=False,
                force_daily_refresh=True if args.force_daily_refresh else False,
                workers=args.workers,
            )
            actual_date = _latest_trade_date(frame)
            source_counts = frame["source"].value_counts(dropna=False).to_dict() if "source" in frame.columns else {}
            row.update(
                {
                    "status": "ok" if actual_date == date_text else "skipped",
                    "rows": int(len(frame)),
                    "actual_date": actual_date,
                    "sources": str(source_counts),
                    "error": "" if actual_date == date_text else "exact date limit-up pool missing",
                }
            )
            print(f"{date_text} {row['status']} rows={len(frame)} actual={actual_date} sources={source_counts}")
        except Exception as exc:
            row["error"] = str(exc)
            print(f"{date_text} failed: {exc}", file=sys.stderr)
        rows.append(row)

    report = pd.DataFrame(rows)
    out_dir = get_data_config().reports_dir / "warmup"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"limitup_warmup_{args.start_date}_{args.end_date}.csv"
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    status_counts = report["status"].value_counts(dropna=False) if not report.empty else {}
    print(f"warmup dates: {len(report)}")
    for status, count in status_counts.items():
        print(f"{status}: {int(count)}")
    print(f"warmup csv: {out_path}")
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
    signals, quality_rows = _build_signals(service, pool, args.days, args.max_codes, args.force_refresh)
    ranked_signals, ranking_meta = apply_daily_research_ranking(signals, model_file=args.ranking_model, top_n=args.top_n)
    output_dir = get_data_config().reports_dir / "daily_signals"
    trade_date = _latest_trade_date(pool)
    csv_path, md_path = write_signal_reports(ranked_signals, output_dir, trade_date=trade_date)
    quality_csv_path, quality_md_path = write_data_quality_reports(
        quality_rows,
        get_data_config().reports_dir / "data_quality",
        trade_date=trade_date,
    )
    print(f"signals: {len(signals)}")
    print(f"ranking model: {ranking_meta['model_id']}")
    print(f"model top_n: {ranking_meta['top_n']}")
    print(f"data quality rows: {len(quality_rows)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print(f"quality csv: {quality_csv_path}")
    print(f"quality markdown: {quality_md_path}")
    return 0


def run_daily(args) -> int:
    service = MarketDataService()
    pool = service.collect_limit_ups(
        trade_date=args.date,
        lookback_days=args.lookback_days,
        force_refresh=args.force_refresh,
    )
    signals, quality_rows = _build_signals(service, pool, args.days, args.max_codes, args.force_refresh)
    ranked_signals, ranking_meta = apply_daily_research_ranking(signals, model_file=args.ranking_model, top_n=args.top_n)
    output_dir = get_data_config().reports_dir / "daily_signals"
    trade_date = _latest_trade_date(pool)
    csv_path, md_path = write_signal_reports(ranked_signals, output_dir, trade_date=trade_date)
    quality_csv_path, quality_md_path = write_data_quality_reports(
        quality_rows,
        get_data_config().reports_dir / "data_quality",
        trade_date=trade_date,
    )
    print(f"limit-up rows: {len(pool)}")
    print(f"signals: {len(signals)}")
    print(f"ranking model: {ranking_meta['model_id']}")
    print(f"model top_n: {ranking_meta['top_n']}")
    print(f"data quality rows: {len(quality_rows)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print(f"quality csv: {quality_csv_path}")
    print(f"quality markdown: {quality_md_path}")
    return 0


def validate_data(args) -> int:
    frame, csv_path, md_path = run_data_acceptance(
        limitup_file=args.limitup_file,
        days=args.days,
        max_codes=args.max_codes,
        reference_root=args.reference_root,
    )
    accepted = int((frame["status"] == "accepted").sum()) if not frame.empty else 0
    failed = int((frame["status"] == "failed").sum()) if not frame.empty else 0
    print(f"data acceptance rows: {len(frame)}")
    print(f"accepted: {accepted}")
    print(f"failed: {failed}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    return 0


def review_failed_data_command(args) -> int:
    frame, csv_path, md_path, exclusion_path = review_failed_data(
        quality_file=args.quality_file,
        days=args.days,
        force_refresh=args.force_refresh,
    )
    print(f"failed rows reviewed: {len(frame)}")
    if not frame.empty and "repair_status" in frame.columns:
        counts = frame["repair_status"].value_counts(dropna=False)
        for status, count in counts.items():
            print(f"{status}: {int(count)}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print(f"exclusions: {exclusion_path}")
    return 0


def backtest_top3(args) -> int:
    trades, summary, csv_path, md_path = run_top3_signal_backtest(
        signals_file=args.signals_file,
        top_n=args.top_n,
        target_return_pct=args.target_return_pct,
        include_small=args.include_small,
        fetch_through_date=args.fetch_through_date,
        days=args.days,
        force_refresh=args.force_refresh,
        entry_price_mode=args.entry_price_mode,
    )
    item = summary.iloc[0] if not summary.empty else {}
    print(f"selected: {len(trades)}")
    print(f"evaluable: {int(item.get('evaluable_count', 0)) if len(summary) else 0}")
    print(f"executed: {int(item.get('executed_count', 0)) if len(summary) else 0}")
    print(f"target hit rate: {item.get('target_hit_rate', '') if len(summary) else ''}")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    return 0


def backtest_history(args) -> int:
    trades, summary, factor_stats, trade_csv, summary_csv, factor_csv, md_path = run_history_backtest(
        signals_dir=args.signals_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        hold_days=args.hold_days,
        target_return_pct=args.target_return_pct,
        stop_loss_pct=args.stop_loss_pct,
        include_all_allowed=args.include_all_allowed,
        include_small=args.include_small,
        entry_price_mode=args.entry_price_mode,
    )
    item = summary.iloc[0] if not summary.empty else {}
    print(f"records: {len(trades)}")
    print(f"selected: {int(item.get('selected_count', 0)) if len(summary) else 0}")
    print(f"evaluable: {int(item.get('evaluable_count', 0)) if len(summary) else 0}")
    print(f"executed: {int(item.get('executed_count', 0)) if len(summary) else 0}")
    print(f"target hit rate: {item.get('target_hit_rate', '') if len(summary) else ''}")
    print(f"factor rows: {len(factor_stats)}")
    print(f"trades csv: {trade_csv}")
    print(f"summary csv: {summary_csv}")
    print(f"factor csv: {factor_csv}")
    print(f"markdown: {md_path}")
    return 0


def backtest_run(args) -> int:
    (
        trades,
        summary,
        factor_stats,
        run_log,
        trade_csv,
        summary_csv,
        factor_csv,
        md_path,
        run_log_csv,
        run_log_md,
        future_fetch_csv,
    ) = run_full_history_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        hold_days=args.hold_days,
        target_return_pct=args.target_return_pct,
        stop_loss_pct=args.stop_loss_pct,
        lookback_days=args.lookback_days,
        signal_days=args.signal_days,
        eval_days=args.eval_days,
        max_codes=args.max_codes,
        force_refresh=args.force_refresh,
        include_all_allowed=args.include_all_allowed,
        include_small=args.include_small,
        entry_price_mode=args.entry_price_mode,
    )
    item = summary.iloc[0] if not summary.empty else {}
    status_counts = run_log["status"].value_counts(dropna=False) if not run_log.empty and "status" in run_log.columns else {}
    print(f"run dates: {len(run_log)}")
    for status, count in status_counts.items():
        print(f"{status}: {int(count)}")
    print(f"records: {len(trades)}")
    print(f"selected: {int(item.get('selected_count', 0)) if len(summary) else 0}")
    print(f"evaluable: {int(item.get('evaluable_count', 0)) if len(summary) else 0}")
    print(f"executed: {int(item.get('executed_count', 0)) if len(summary) else 0}")
    print(f"target hit rate: {item.get('target_hit_rate', '') if len(summary) else ''}")
    print(f"factor rows: {len(factor_stats)}")
    print(f"trades csv: {trade_csv}")
    print(f"summary csv: {summary_csv}")
    print(f"factor csv: {factor_csv}")
    print(f"markdown: {md_path}")
    print(f"run log csv: {run_log_csv}")
    print(f"run log markdown: {run_log_md}")
    print(f"future fetch csv: {future_fetch_csv}")
    return 0


def generate_history_samples(args) -> int:
    (
        candidates,
        summary,
        run_log,
        future_fetch_log,
        candidates_csv,
        summary_csv,
        run_log_csv,
        future_fetch_csv,
        markdown_path,
    ) = run_history_sample_generation(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        signal_days=args.signal_days,
        eval_days=args.eval_days,
        max_codes=args.max_codes,
        force_refresh=args.force_refresh,
        workers=args.workers,
        hold_days=args.hold_days,
        target_return_pct=args.target_return_pct,
        secondary_target_return_pct=args.secondary_target_return_pct,
    )
    item = summary.iloc[0] if not summary.empty else {}
    print(f"run dates: {len(run_log)}")
    print(f"candidate rows: {len(candidates)}")
    print(f"eligible: {int(item.get('eligible_count', 0)) if len(summary) else 0}")
    print(f"candidate evaluable: {int(item.get('candidate_evaluable_count', 0)) if len(summary) else 0}")
    print(f"candidate target7 rate: {item.get('candidate_target7_rate', '') if len(summary) else ''}")
    print(f"candidates csv: {candidates_csv}")
    print(f"summary csv: {summary_csv}")
    print(f"run log csv: {run_log_csv}")
    print(f"future fetch csv: {future_fetch_csv}")
    print(f"markdown: {markdown_path}")
    return 0


def analyze_factors(args) -> int:
    (
        factor_summary,
        factor_buckets,
        daily_stability,
        pair_review,
        summary_csv,
        buckets_csv,
        stability_csv,
        pair_csv,
        markdown_path,
    ) = run_factor_analysis(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        target_return_pct=args.target_return_pct,
        min_bucket_size=args.min_bucket_size,
        eligible_only=not args.all_candidates,
        target_column=args.target_column,
    )
    print(f"factor summary rows: {len(factor_summary)}")
    print(f"factor bucket rows: {len(factor_buckets)}")
    print(f"daily stability rows: {len(daily_stability)}")
    print(f"pair review rows: {len(pair_review)}")
    print(f"target column: {args.target_column}")
    print(f"factor summary csv: {summary_csv}")
    print(f"factor buckets csv: {buckets_csv}")
    print(f"daily stability csv: {stability_csv}")
    print(f"pair review csv: {pair_csv}")
    print(f"markdown: {markdown_path}")
    return 0


def ranking_backtest(args) -> int:
    ranked, topn, daily, failures, summary, summary_csv, daily_csv, topn_csv, failures_csv, markdown_path = run_ranking_backtest(
        samples_file=args.samples_file,
        model_file=args.model_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        target_return_pct=args.target_return_pct,
        target_column=args.target_column,
    )
    item = summary.iloc[0] if not summary.empty else {}
    print(f"evaluable candidates: {int(item.get('evaluable_count', 0)) if len(summary) else 0}")
    print(f"eligible candidates: {len(ranked)}")
    print(f"topn rows: {len(topn)}")
    print(f"target column: {item.get('target_column', '') if len(summary) else ''}")
    print(f"daily hit rate: {item.get('daily_hit_rate', '') if len(summary) else ''}")
    print(f"topn target rate: {item.get('topn_target_rate', '') if len(summary) else ''}")
    print(f"summary csv: {summary_csv}")
    print(f"daily csv: {daily_csv}")
    print(f"topn csv: {topn_csv}")
    print(f"failures csv: {failures_csv}")
    print(f"markdown: {markdown_path}")
    return 0


def train_logistic_v003(args) -> int:
    if args.l2_grid:
        comparison, rankwise, top3_combo, comparison_csv, rankwise_csv, top3_combo_csv = run_logistic_v003_l2_grid(
            samples_file=args.samples_file,
            output_dir=args.output_dir,
            top_n=args.top_n,
            initial_train_days=args.initial_train_days,
            target_return_pct=args.target_return_pct,
            l2_grid=args.l2_grid,
        )
        print(f"grid comparison rows: {len(comparison)}")
        print(f"grid rankwise rows: {len(rankwise)}")
        print(f"grid top3 combo rows: {len(top3_combo)}")
        print(f"grid comparison csv: {comparison_csv}")
        print(f"grid rankwise csv: {rankwise_csv}")
        print(f"grid top3 combo csv: {top3_combo_csv}")
        return 0
    coefficients, comparison, rankwise, top3_combo, daily_top3, data_quality, markdown_path = run_logistic_v003_research(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        initial_train_days=args.initial_train_days,
        l2=args.l2,
        target_return_pct=args.target_return_pct,
    )
    print(f"coefficients: {len(coefficients)}")
    print(f"comparison rows: {len(comparison)}")
    print(f"rankwise rows: {len(rankwise)}")
    print(f"top3 combo rows: {len(top3_combo)}")
    print(f"daily top3 rows: {len(daily_top3)}")
    print(f"data quality rows: {len(data_quality)}")
    print(f"markdown: {markdown_path}")
    return 0


def train_v004a(args) -> int:
    comparison, rankwise, top3_combo, per_date, coefficients, data_quality, scored, report_path = run_v004a_research(
        samples_file=args.samples_file,
        output_dir=args.output_dir,
        top_n=args.top_n,
        initial_train_days=args.initial_train_days,
        target_return_pct=args.target_return_pct,
        l2_grid=args.l2_grid,
        positive_weight_grid=args.positive_weight_grid,
        threshold_grid=args.threshold_grid,
    )
    print(f"comparison rows: {len(comparison)}")
    print(f"rankwise rows: {len(rankwise)}")
    print(f"top3 combo rows: {len(top3_combo)}")
    print(f"per-date rows: {len(per_date)}")
    print(f"coefficients rows: {len(coefficients)}")
    print(f"data quality rows: {len(data_quality)}")
    print(f"scored candidate rows: {len(scored)}")
    print(f"markdown: {report_path}")
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
    as_of_date = _latest_trade_date(pool)
    signals = []
    quality_rows = []
    for code in codes:
        code_pool = pool[pool["code"].astype(str) == code].sort_values("trade_date")
        if code_pool.empty:
            continue
        d0_date = str(code_pool["trade_date"].iloc[-1])
        if d0_date > as_of_date:
            continue
        name = str(code_pool.iloc[-1].get("name") or "")
        try:
            bars = service.get_stock_bars(code, days=days, end_date=as_of_date, force_refresh=force_refresh)
            quality = dict(bars.quality)
            quality.update({"name": name, "trade_date": as_of_date, "d0_date": d0_date})
            quality_rows.append(quality)
            signal = generate_signal(code, name, bars.daily, bars.minute_5m, pool, d0_date=d0_date)
            signals.append(signal)
        except Exception as exc:
            if isinstance(exc, DataQualityError):
                quality = dict(exc.quality)
                quality.update({"name": name, "trade_date": as_of_date, "d0_date": d0_date, "error": str(exc)})
                quality_rows.append(quality)
            else:
                quality_rows.append(_failed_quality_row(code, name, as_of_date, d0_date, exc))
            print(f"WARN: skip {code} {name}: {exc}", file=sys.stderr)
    return signals, quality_rows


def _latest_trade_date(pool: pd.DataFrame) -> str:
    if pool is None or pool.empty or "trade_date" not in pool.columns:
        return pd.Timestamp.now().strftime("%Y-%m-%d")
    return str(pool["trade_date"].dropna().max())


def _failed_quality_row(code: str, name: str, trade_date: str, d0_date: str, exc: Exception) -> dict:
    return {
        "code": code,
        "name": name,
        "trade_date": trade_date,
        "d0_date": d0_date,
        "status": "failed",
        "daily_source": "",
        "minute_source": "",
        "from_cache": False,
        "daily_rows": 0,
        "daily_history_rows": 0,
        "daily_required_days": 0,
        "minute_rows": 0,
        "minute_trade_days": 0,
        "minute_required_days": 0,
        "daily_start": "",
        "daily_end": "",
        "minute_start": "",
        "minute_end": "",
        "daily_ma_coverage_ok": False,
        "missing_latest_daily_ma": "",
        "missing_daily_amount_count": 0,
        "missing_daily_volume_count": 0,
        "missing_minute_amount_count": 0,
        "missing_minute_volume_count": 0,
        "zero_minute_volume_count": 0,
        "daily_minute_close_matched_days": 0,
        "daily_minute_close_max_abs_diff": None,
        "daily_minute_close_check_ok": False,
        "warnings": "",
        "error": str(exc),
    }


if __name__ == "__main__":
    raise SystemExit(main())
