from __future__ import annotations

import argparse

from .v005_daily_selector import (
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_COEFFICIENT_PREDICT_DATE,
    DEFAULT_COEFFICIENTS_FILE,
    DEFAULT_GRID_ID,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RANKING_MODEL_FILE,
    DEFAULT_TOP_N,
    run_v005_daily_from_market,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run daily v2 signals plus the fixed-grid v005 research-watchlist shadow flow."
    )
    parser.add_argument("--date", default=None, help="Signal date, e.g. 2026-07-03; default latest available/today.")
    parser.add_argument("--lookback-days", type=int, default=5)
    parser.add_argument("--days", type=int, default=None, help="Bar lookback days passed to signal generation.")
    parser.add_argument("--max-codes", type=int, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--ranking-model", default=str(DEFAULT_RANKING_MODEL_FILE))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--coefficients-file", default=str(DEFAULT_COEFFICIENTS_FILE))
    parser.add_argument("--coefficient-predict-date", default=DEFAULT_COEFFICIENT_PREDICT_DATE)
    parser.add_argument("--grid-id", type=int, default=DEFAULT_GRID_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decisions, selections, combos, scored, report, signal_csv, quality_csv = run_v005_daily_from_market(
        date=args.date,
        lookback_days=args.lookback_days,
        days=args.days,
        max_codes=args.max_codes,
        force_refresh=args.force_refresh,
        workers=args.workers,
        output_root=args.output_root,
        ranking_model=args.ranking_model,
        top_n=args.top_n,
        candidate_top_k=args.candidate_top_k,
        coefficients_file=args.coefficients_file,
        coefficient_predict_date=args.coefficient_predict_date,
        grid_id=args.grid_id,
    )
    print(f"decision rows: {len(decisions)}")
    print(f"selection rows: {len(selections)}")
    print(f"combo rows: {len(combos)}")
    print(f"scored rows: {len(scored)}")
    print("status: RESEARCH ONLY / SHADOW FLOW; manual review required")
    if not decisions.empty:
        item = decisions.iloc[0]
        print(f"policy version: {item.get('policy_version', '')}")
        print(f"deployment status: {item.get('deployment_status', '')}")
        print(f"primary strategy: {item.get('final_strategy', '')}")
        print(f"action: {item.get('action', '')}")
        print(f"fallback_triggered: {item.get('fallback_triggered', '')}")
        print(f"research_watchlist_codes: {item.get('research_watchlist_codes', item.get('primary_buy_codes', ''))}")
        print(f"primary_buy_codes (legacy field): {item.get('primary_buy_codes', '')}")
        print(f"v005_baseline_codes: {item.get('v005_baseline_codes', '')}")
        print(f"v002_codes: {item.get('v002_codes', '')}")
        print(f"v004a_codes: {item.get('v004a_codes', '')}")
    print(f"v2 signals csv: {signal_csv}")
    print(f"quality csv: {quality_csv}")
    print(f"markdown: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
