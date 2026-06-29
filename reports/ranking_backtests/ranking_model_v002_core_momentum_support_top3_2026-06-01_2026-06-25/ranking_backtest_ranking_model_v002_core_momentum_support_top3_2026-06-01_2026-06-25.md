# Ranking Backtest: ranking_model_v002_core_momentum_support

## Scope

This report validates a research ranking model against clean history candidate samples. It does not simulate D2 execution and does not modify the formal daily strategy.

## Inputs

- samples file: `reports\history_samples\2026-06-01_2026-06-25\history_candidates_2026-06-01_2026-06-25.csv`
- model file: `reports\manual_models\ranking_model_v002_core_momentum_support.json`
- model type: **linear_score**

## Summary

- dates: **12**
- eligible candidates: **1548**
- top N: **3**
- topN row target7 rate: **66.67%**
- daily hit rate: **91.67%**
- avg top1 D3 max return: **10.2093%**
- avg topN D3 max return: **8.4611%**

## Failure Dates

| signal_date | top_n | codes | returns |
|---|---:|---|---|
| 2026-06-24 | 3 | 600353,603663,603823 | -1.7958,1.1463,2.4425 |

## Daily Snapshot

| date | hit | top codes | best D3 max% | avg D3 max% |
|---|---|---|---:|---:|
| 2026-06-08 | True | 000608,000520,603078 | 20.9622 | 9.8430 |
| 2026-06-09 | True | 603135,002931,603859 | 9.7046 | 5.0638 |
| 2026-06-10 | True | 603186,000608,600487 | 14.7563 | 5.7964 |
| 2026-06-11 | True | 002636,603186,603823 | 15.5660 | 11.7596 |
| 2026-06-12 | True | 002636,603823,600500 | 14.4860 | 12.4926 |
| 2026-06-15 | True | 001696,002971,002203 | 11.6238 | 6.6433 |
| 2026-06-16 | True | 002913,600176,600226 | 11.5429 | 9.9076 |
| 2026-06-17 | True | 002436,003036,001359 | 15.0628 | 8.7398 |
| 2026-06-18 | True | 603186,002824,603986 | 10.5222 | 9.5086 |
| 2026-06-23 | True | 600353,002805,000811 | 19.0127 | 10.3738 |
| 2026-06-24 | False | 600353,603663,603823 | 2.4425 | 0.5977 |
| 2026-06-25 | True | 600206,603986,600397 | 15.1175 | 10.8073 |