# Research Run Manifest

## Purpose

This run is for factor discovery, not final TopN live-trading simulation.
It keeps success, failed, missed, ordinary, and data-issue samples for later comparison.

## Parameters

- start date: `2026-06-01`
- end date: `2026-06-25`
- top_n: `999`
- max_codes: `all`
- hold_days: `10`
- target_return_pct: `7.0`
- stop_loss_pct: `3.0`
- target_min_return_pct: `7.0`
- target_max_return_pct: `10.0`
- entry_price_mode: `zone_max`

## Command

```text
D:\python3.13.2\python.exe -m src.watch_backtest --start-date 2026-06-01 --end-date 2026-06-25 --top-n 999 --hold-days 10 --target-return-pct 7.0 --stop-loss-pct 3.0 --lookback-days 5 --interval-seconds 5 --entry-price-mode zone_max
```

## Output Files

- source backtest root: `reports\backtest_runs\2026-06-01_2026-06-25`
- research root: `reports\research_runs\2026-06-01_2026-06-25_research_top999_all`
- history trades: `reports\research_runs\2026-06-01_2026-06-25_research_top999_all\backtest_results\history_trades_2026-06-01_2026-06-25.csv`
- research samples: `reports\research_runs\2026-06-01_2026-06-25_research_top999_all\research_results\research_samples_2026-06-01_2026-06-25.csv`
- factor compare: `reports\research_runs\2026-06-01_2026-06-25_research_top999_all\research_results\factor_compare_2026-06-01_2026-06-25.csv`
- research review: `reports\research_runs\2026-06-01_2026-06-25_research_top999_all\research_results\research_review_2026-06-01_2026-06-25.md`

## Counts

- samples: **548**
- factor rows: **83**
