# Research Run Manifest

## Purpose

This run is for all-candidate return-distribution research, not final TopN live-trading simulation.
It keeps all useful rows and treats realized trade state, TopN, and execution fields as metadata rather than primary labels.
The main research target is profit/loss and opportunity distribution across the full sample universe.

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
- return_loss_cutoff_pct: `-3.0`
- return_quantiles: `5`
- entry_price_mode: `zone_max`
- with_factor_discovery: `False`

## Command

```text
D:\python3.13.2\python.exe -m src.watch_backtest --start-date 2026-06-01 --end-date 2026-06-25 --top-n 999 --hold-days 10 --target-return-pct 7.0 --stop-loss-pct 3.0 --lookback-days 5 --interval-seconds 5 --entry-price-mode zone_max
```

## Output Files

- source backtest root: `reports\backtest_runs\2026-06-01_2026-06-25`
- research root: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3`
- history trades: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\backtest_results\history_trades_2026-06-01_2026-06-25.csv`
- research samples: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\research_samples_2026-06-01_2026-06-25.csv`
- legacy factor compare: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\factor_compare_2026-06-01_2026-06-25.csv`
- legacy research review: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\research_review_2026-06-01_2026-06-25.md`
- return samples: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\return_samples_2026-06-01_2026-06-25.csv`
- return bucket compare: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\return_bucket_compare_2026-06-01_2026-06-25.csv`
- factor quantile report: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\factor_quantile_report_2026-06-01_2026-06-25.csv`
- profit/loss compare: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\profit_loss_compare_2026-06-01_2026-06-25.csv`
- daily return summary: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\daily_return_summary_2026-06-01_2026-06-25.csv`
- return distribution report: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix_v3\research_results\return_distribution_report_2026-06-01_2026-06-25.md`

## Counts

- samples: **2962**
- legacy factor rows: **83**
- return bucket rows: **28**
- factor quantile rows: **150**
- profit/loss groups: **9**
