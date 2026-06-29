# Factor Analysis Report

## Scope

This report is a descriptive factor-analysis layer. It does not generate ranking_model_*.json.
A ranking model must be manually constructed after reviewing these outputs.

## Sample

- samples file: `reports\history_samples\2026-05-01_2026-06-25\history_candidates_2026-05-01_2026-06-25.csv`
- rows after filters: **1548**
- dates: **12**
- eligible only: **True**
- target7 rows: **524**
- target7 rate: **33.85%**

## Top Factor Signals

| feature | type | strength | corr target7 | corr D3 max | top q rate | bottom q rate | direction |
|---|---|---:|---:|---:|---:|---:|---|
| total_score | numeric | 0.5056 | 0.1475 | 0.1456 | 46.01% | 24.76% | higher_better |
| trend_hold_score | numeric | 0.3424 | 0.1207 | 0.1081 | 40.31% | 28.95% | higher_better |
| d1_low_ma10_pct | numeric | 0.3038 | 0.0976 | 0.0836 | 44.84% | 32.58% | higher_better |
| d1_close_ma10_pct | numeric | 0.2962 | 0.0898 | 0.0709 | 42.90% | 29.35% | higher_better |
| theme_score | numeric | 0.2806 | 0.0950 | 0.0913 | 37.22% | 27.80% | higher_better |
| active_cooling_score | numeric | 0.2617 | 0.0750 | 0.1025 | 37.22% | 28.80% | higher_better |
| graph_quality_score | numeric | 0.2402 | 0.0777 | 0.0684 | 40.06% | 30.64% | higher_better |
| active_money_score | numeric | 0.1837 | -0.0447 | -0.0723 | 28.27% | 34.94% | lower_better |
| d1_close_vwap_pct | numeric | 0.1805 | -0.0559 | -0.0859 | 35.48% | 39.35% | lower_better |
| consecutive_boards | numeric | 0.1412 | 0.0649 | 0.0664 | 33.85% | 32.86% | higher_better |
| invalid_distance_pct | numeric | 0.1396 | -0.0446 | -0.0466 | 30.00% | 34.84% | lower_better |
| low_absorb_width_pct | numeric | 0.1396 | -0.0446 | -0.0466 | 30.00% | 34.84% | lower_better |
| entry_width_score | numeric | 0.1167 | 0.0535 | 0.0610 | 33.73% | 33.94% | lower_better |
| support_score | numeric | 0.1072 | -0.0258 | -0.0505 | 32.02% | 35.11% | lower_better |
| days_since_d0 | numeric | 0.0750 | -0.0330 | -0.0054 | 32.05% | 35.71% | lower_better |
| support_type | categorical | 0.0561 |  |  |  |  | bucket_spread=0.0561 |
| signal_type | categorical | 0.0000 |  |  |  |  | bucket_spread=0.0000 |

## Strong Buckets

| feature | bucket | count | target7 rate | lift | avg D3 max% |
|---|---|---:|---:|---:|---:|
| trend_hold_score | (75.0, 95.0] | 250 | 47.20% | 13.35% | 7.0463 |
| total_score | (74.25, 88.75] | 293 | 46.76% | 12.91% | 7.2193 |
| d1_low_ma10_pct | (8.589, 33.548] | 310 | 44.84% | 10.99% | 6.8429 |
| d1_close_ma10_pct | (13.323, 39.123] | 310 | 42.90% | 9.05% | 6.6594 |
| entry_width_score | (65.0, 90.0] | 139 | 42.45% | 8.60% | 6.5184 |
| graph_quality_score | (70.0, 82.0] | 247 | 42.11% | 8.26% | 6.6044 |
| d1_close_vwap_pct | (-6.261, -1.662] | 310 | 39.35% | 5.50% | 6.6883 |
| support_type | B | 228 | 38.16% | 4.31% | 6.3582 |
| invalid_distance_pct | (4.123, 4.628] | 309 | 37.86% | 4.01% | 6.2798 |
| low_absorb_width_pct | (4.123, 4.628] | 309 | 37.86% | 4.01% | 6.2798 |
| graph_quality_score | (82.0, 90.0] | 165 | 37.58% | 3.73% | 5.6831 |
| active_money_score | (60.0, 72.0] | 315 | 37.46% | 3.61% | 5.8697 |
| theme_score | (60.0, 70.0] | 994 | 37.22% | 3.37% | 6.0370 |
| active_cooling_score | (85.0, 90.0] | 575 | 37.22% | 3.37% | 6.2341 |
| support_score | (39.0, 67.0] | 278 | 36.69% | 2.84% | 6.1702 |
| total_score | (66.75, 74.25] | 316 | 36.39% | 2.54% | 6.0003 |
| active_cooling_score | (70.0, 85.0] | 334 | 35.63% | 1.78% | 5.9265 |
| d1_close_vwap_pct | (1.189, 7.52] | 310 | 35.48% | 1.63% | 5.4856 |
| active_money_score | (72.0, 84.0] | 318 | 35.22% | 1.37% | 5.7829 |
| support_score | (-0.001, 21.0] | 487 | 35.11% | 1.26% | 5.8599 |
| active_money_score | (31.999, 50.0] | 395 | 34.94% | 1.09% | 5.9606 |
| invalid_distance_pct | (-0.001, 2.782] | 310 | 34.84% | 0.99% | 5.8093 |
| low_absorb_width_pct | (-0.001, 2.782] | 310 | 34.84% | 0.99% | 5.8093 |
| invalid_distance_pct | (3.489, 4.123] | 310 | 34.52% | 0.67% | 5.7093 |
| low_absorb_width_pct | (3.489, 4.123] | 310 | 34.52% | 0.67% | 5.7093 |
| graph_quality_score | (50.0, 70.0] | 499 | 34.47% | 0.62% | 5.8747 |
| trend_hold_score | (55.0, 65.0] | 325 | 34.46% | 0.61% | 5.7884 |
| days_since_d0 | (0.999, 2.0] | 1158 | 34.46% | 0.61% | 5.6432 |
| d1_low_ma10_pct | (4.887, 8.589] | 309 | 34.30% | 0.45% | 5.7334 |
| d1_close_vwap_pct | (-1.662, -0.672] | 309 | 33.98% | 0.13% | 5.4899 |
| entry_width_score | (24.999, 60.0] | 872 | 33.94% | 0.09% | 5.6486 |
| consecutive_boards | (0.999, 3.0] | 1548 | 33.85% | 0.00% | 5.6496 |
| signal_type | D2_LOW_ABSORB | 1548 | 33.85% | 0.00% | 5.6496 |
| trend_hold_score | (65.0, 75.0] | 266 | 33.83% | -0.02% | 5.7412 |
| active_money_score | (50.0, 60.0] | 228 | 33.77% | -0.08% | 5.9494 |
| support_type | C | 687 | 33.62% | -0.23% | 5.7324 |
| active_cooling_score | (65.0, 70.0] | 146 | 33.56% | -0.29% | 5.2340 |
| support_score | (67.0, 89.0] | 324 | 32.72% | -1.13% | 5.5888 |
| d1_close_ma10_pct | (8.574, 13.323] | 309 | 32.69% | -1.16% | 5.3897 |
| d1_close_ma10_pct | (5.073, 8.574] | 310 | 32.58% | -1.27% | 5.4524 |

## Pair Review

| left | right | rule | count | target7 rate | lift | corr |
|---|---|---|---:|---:|---:|---:|
| total_score | d1_close_ma10_pct | both_top_30pct | 332 | 44.58% | 10.73% | 0.7669 |
| total_score | theme_score | both_top_30pct | 344 | 44.19% | 10.34% | 0.1726 |
| total_score | trend_hold_score | both_top_30pct | 374 | 43.85% | 10.00% | 0.8481 |
| total_score | consecutive_boards | both_top_30pct | 468 | 43.80% | 9.95% | 0.2127 |
| theme_score | d1_low_ma10_pct | both_top_30pct | 327 | 43.73% | 9.88% | 0.1091 |
| total_score | active_money_score | both_top_30pct | 190 | 43.68% | 9.83% | 0.1317 |
| theme_score | trend_hold_score | both_top_30pct | 359 | 43.45% | 9.60% | 0.1078 |
| total_score | graph_quality_score | both_top_30pct | 435 | 43.45% | 9.60% | 0.8307 |
| total_score | d1_low_ma10_pct | both_top_30pct | 348 | 43.39% | 9.54% | 0.7953 |
| total_score | active_cooling_score | both_top_30pct | 188 | 43.09% | 9.23% | 0.1615 |
| total_score | entry_width_score | both_top_30pct | 217 | 42.86% | 9.01% | 0.1474 |
| total_score | low_absorb_width_pct | both_top_30pct | 119 | 42.86% | 9.01% | -0.1635 |
| total_score | invalid_distance_pct | both_top_30pct | 119 | 42.86% | 9.01% | -0.1635 |
| active_cooling_score | d1_close_vwap_pct | both_top_30pct | 124 | 42.74% | 8.89% | -0.2847 |
| graph_quality_score | trend_hold_score | both_top_30pct | 397 | 41.81% | 7.96% | 0.6314 |
| total_score | d1_close_vwap_pct | both_top_30pct | 199 | 41.71% | 7.86% | 0.1970 |
| graph_quality_score | d1_low_ma10_pct | both_top_30pct | 365 | 41.64% | 7.79% | 0.6383 |
| active_cooling_score | theme_score | both_top_30pct | 380 | 41.58% | 7.73% | -0.0024 |
| active_cooling_score | d1_low_ma10_pct | both_top_30pct | 118 | 41.53% | 7.68% | -0.1948 |
| trend_hold_score | d1_close_ma10_pct | both_top_30pct | 388 | 41.49% | 7.64% | 0.8602 |
| entry_width_score | d1_low_ma10_pct | both_top_30pct | 211 | 41.23% | 7.38% | 0.0515 |
| trend_hold_score | entry_width_score | both_top_30pct | 239 | 41.00% | 7.15% | 0.0532 |
| d1_low_ma10_pct | d1_close_ma10_pct | both_top_30pct | 371 | 40.97% | 7.12% | 0.9353 |
| active_cooling_score | trend_hold_score | both_top_30pct | 137 | 40.88% | 7.03% | -0.1710 |
| entry_width_score | d1_close_ma10_pct | both_top_30pct | 196 | 40.82% | 6.97% | 0.0461 |
| trend_hold_score | d1_low_ma10_pct | both_top_30pct | 465 | 40.43% | 6.58% | 0.9140 |
| d1_low_ma10_pct | consecutive_boards | both_top_30pct | 465 | 40.43% | 6.58% | 0.3388 |
| graph_quality_score | active_cooling_score | both_top_30pct | 223 | 40.36% | 6.51% | -0.1707 |
| trend_hold_score | consecutive_boards | both_top_30pct | 516 | 40.31% | 6.46% | 0.3067 |
| trend_hold_score | low_absorb_width_pct | both_top_30pct | 145 | 40.00% | 6.15% | -0.1068 |

## Daily Stability Snapshot

| feature | days | avg spread | avg corr |
|---|---:|---:|---:|
| total_score | 12 | 0.2359 | 0.1629 |
| d1_close_ma10_pct | 12 | 0.1238 | 0.1037 |
| d1_low_ma10_pct | 12 | 0.1167 | 0.1020 |
| graph_quality_score | 12 | 0.1128 | 0.0840 |
| trend_hold_score | 12 | 0.1071 | 0.1311 |
| active_cooling_score | 12 | 0.0940 | 0.0695 |
| theme_score | 12 | 0.0661 | 0.0732 |
| support_score | 12 | 0.0250 | 0.0089 |
| d1_close_vwap_pct | 12 | 0.0210 | 0.0052 |
| days_since_d0 | 12 | 0.0183 | 0.0175 |
| consecutive_boards | 12 | 0.0098 | 0.1146 |
| entry_width_score | 12 | 0.0007 | 0.0530 |
| low_absorb_width_pct | 12 | -0.0226 | -0.0405 |
| invalid_distance_pct | 12 | -0.0226 | -0.0405 |
| active_money_score | 12 | -0.0809 | -0.0265 |

## Boundary

These outputs are evidence for manual research. They are not a trading model and are not a validated ranking model.