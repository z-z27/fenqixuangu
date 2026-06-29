# Factor Analysis Report

## Scope

This report is a descriptive factor-analysis layer. It does not generate ranking_model_*.json.
A ranking model must be manually constructed after reviewing these outputs.

## Sample

- samples file: `reports\history_samples\2026-06-01_2026-06-25\history_candidates_2026-06-01_2026-06-25.csv`
- rows after filters: **2956**
- dates: **14**
- eligible only: **False**
- target7 rows: **1139**
- target7 rate: **38.53%**

## Top Factor Signals

| feature | type | strength | corr target7 | corr D3 max | top q rate | bottom q rate | direction |
|---|---|---:|---:|---:|---:|---:|---|
| total_score | numeric | 0.6212 | 0.1827 | 0.1821 | 54.38% | 28.74% | higher_better |
| d1_close_ma10_pct | numeric | 0.6195 | 0.1786 | 0.1706 | 55.74% | 28.72% | higher_better |
| days_since_d0 | numeric | 0.5861 | -0.1762 | -0.1941 | 27.37% | 48.94% | lower_better |
| d1_low_ma10_pct | numeric | 0.4662 | 0.1452 | 0.1402 | 54.22% | 36.15% | higher_better |
| graph_quality_score | numeric | 0.4607 | 0.1466 | 0.1376 | 48.92% | 31.27% | higher_better |
| trend_hold_score | numeric | 0.4145 | 0.1405 | 0.1304 | 48.53% | 34.16% | higher_better |
| invalid_distance_pct | numeric | 0.3557 | -0.1055 | -0.1151 | 34.97% | 48.48% | lower_better |
| low_absorb_width_pct | numeric | 0.3557 | -0.1055 | -0.1151 | 34.97% | 48.48% | lower_better |
| active_money_score | numeric | 0.2766 | -0.0663 | -0.1058 | 32.83% | 43.27% | lower_better |
| consecutive_boards | numeric | 0.2555 | 0.1204 | 0.1143 | 38.53% | 36.46% | higher_better |
| support_score | numeric | 0.2331 | 0.0680 | 0.0650 | 44.01% | 34.01% | higher_better |
| theme_score | numeric | 0.2259 | 0.0794 | 0.0779 | 41.31% | 34.45% | higher_better |
| entry_width_score | numeric | 0.2015 | 0.0791 | 0.0885 | 40.17% | 36.78% | higher_better |
| active_cooling_score | numeric | 0.1365 | 0.0232 | 0.0470 | 39.46% | 32.83% | higher_better |
| signal_type | categorical | 0.0996 |  |  |  |  | bucket_spread=0.0996 |
| d1_close_vwap_pct | numeric | 0.0996 | 0.0301 | 0.0222 | 41.05% | 36.32% | higher_better |
| support_type | categorical | 0.0665 |  |  |  |  | bucket_spread=0.0665 |

## Strong Buckets

| feature | bucket | count | target7 rate | lift | avg D3 max% |
|---|---|---:|---:|---:|---:|
| trend_hold_score | (75.0, 95.0] | 483 | 56.11% | 17.58% | 8.3594 |
| d1_close_ma10_pct | (15.298, 41.906] | 591 | 55.84% | 17.31% | 8.3128 |
| total_score | (74.25, 89.75] | 566 | 55.48% | 16.95% | 8.3769 |
| d1_low_ma10_pct | (8.392, 40.379] | 591 | 54.15% | 15.61% | 8.1130 |
| graph_quality_score | (82.0, 90.0] | 540 | 51.85% | 13.32% | 7.5390 |
| entry_width_score | (65.0, 90.0] | 552 | 49.64% | 11.11% | 7.6461 |
| invalid_distance_pct | (-0.001, 2.106] | 592 | 48.48% | 9.95% | 7.5427 |
| low_absorb_width_pct | (-0.001, 2.106] | 592 | 48.48% | 9.95% | 7.5427 |
| days_since_d0 | (-0.001, 1.0] | 1834 | 43.95% | 5.42% | 6.9889 |
| active_cooling_score | (50.0, 70.0] | 734 | 43.87% | 5.34% | 6.9300 |
| signal_type | WATCH_ONLY | 1408 | 43.75% | 5.22% | 7.0305 |
| graph_quality_score | (70.0, 82.0] | 480 | 43.75% | 5.22% | 6.9641 |
| active_money_score | (31.999, 50.0] | 617 | 43.27% | 4.74% | 7.1314 |
| d1_close_vwap_pct | (1.523, 2.909] | 591 | 42.64% | 4.11% | 6.6676 |
| total_score | (66.25, 74.25] | 607 | 42.34% | 3.81% | 6.7954 |
| theme_score | (60.0, 70.0] | 1760 | 41.31% | 2.78% | 6.6220 |
| d1_close_vwap_pct | (2.909, 10.272] | 591 | 40.95% | 2.42% | 6.5756 |
| support_score | (69.0, 91.0] | 1737 | 40.93% | 2.40% | 6.5368 |
| support_type | A | 1837 | 40.50% | 1.97% | 6.4927 |
| graph_quality_score | (58.0, 70.0] | 477 | 40.25% | 1.72% | 6.8910 |
| support_type | B | 283 | 39.58% | 1.04% | 6.6642 |
| active_cooling_score | (85.0, 90.0] | 1039 | 39.46% | 0.93% | 6.5825 |
| trend_hold_score | (65.0, 75.0] | 401 | 39.40% | 0.87% | 6.2803 |
| active_money_score | (50.0, 62.0] | 634 | 39.12% | 0.58% | 6.6649 |
| d1_close_ma10_pct | (9.607, 15.298] | 591 | 39.09% | 0.55% | 6.4015 |
| consecutive_boards | (0.999, 4.0] | 2956 | 38.53% | 0.00% | 6.3040 |
| active_money_score | (62.0, 74.0] | 576 | 38.37% | -0.16% | 6.2027 |
| invalid_distance_pct | (3.896, 4.554] | 591 | 37.90% | -0.63% | 6.3232 |
| low_absorb_width_pct | (3.896, 4.554] | 591 | 37.90% | -0.63% | 6.3232 |
| active_money_score | (74.0, 86.0] | 608 | 37.83% | -0.70% | 5.9598 |
| d1_close_vwap_pct | (0.309, 1.523] | 591 | 37.06% | -1.48% | 6.0820 |
| d1_low_ma10_pct | (4.153, 8.392] | 591 | 36.89% | -1.65% | 5.9854 |
| entry_width_score | (24.999, 60.0] | 1430 | 36.78% | -1.75% | 6.0209 |
| d1_close_vwap_pct | (-8.596, -0.97] | 592 | 36.32% | -2.21% | 6.0404 |
| support_score | (27.0, 69.0] | 625 | 36.16% | -2.37% | 6.2489 |
| d1_low_ma10_pct | (-26.172, -3.502] | 592 | 36.15% | -2.38% | 6.0243 |
| trend_hold_score | (55.0, 65.0] | 510 | 36.08% | -2.45% | 6.0939 |
| active_cooling_score | (70.0, 85.0] | 589 | 35.99% | -2.54% | 6.0054 |
| invalid_distance_pct | (3.148, 3.896] | 591 | 35.87% | -2.66% | 6.0014 |
| low_absorb_width_pct | (3.148, 3.896] | 591 | 35.87% | -2.66% | 6.0014 |

## Pair Review

| left | right | rule | count | target7 rate | lift | corr |
|---|---|---|---:|---:|---:|---:|
| total_score | d1_close_ma10_pct | both_top_30pct | 654 | 56.27% | 17.74% | 0.7541 |
| entry_width_score | d1_close_ma10_pct | both_top_30pct | 482 | 55.81% | 17.28% | 0.1006 |
| total_score | support_score | both_top_30pct | 466 | 55.79% | 17.26% | 0.2016 |
| support_score | d1_low_ma10_pct | both_top_30pct | 385 | 55.32% | 16.79% | 0.0225 |
| support_score | d1_close_ma10_pct | both_top_30pct | 537 | 54.56% | 16.03% | 0.3990 |
| total_score | d1_close_vwap_pct | both_top_30pct | 257 | 54.09% | 15.55% | 0.0421 |
| active_cooling_score | d1_close_ma10_pct | both_top_30pct | 213 | 53.52% | 14.99% | -0.2638 |
| total_score | entry_width_score | both_top_30pct | 506 | 53.16% | 14.63% | 0.1699 |
| total_score | d1_low_ma10_pct | both_top_30pct | 681 | 53.01% | 14.48% | 0.7644 |
| graph_quality_score | d1_low_ma10_pct | both_top_30pct | 497 | 52.92% | 14.39% | 0.5273 |
| graph_quality_score | support_score | both_top_30pct | 599 | 52.75% | 14.22% | 0.4584 |
| entry_width_score | d1_low_ma10_pct | both_top_30pct | 457 | 52.74% | 14.20% | 0.0502 |
| d1_close_ma10_pct | d1_close_vwap_pct | both_top_30pct | 364 | 52.47% | 13.94% | 0.2955 |
| graph_quality_score | d1_close_ma10_pct | both_top_30pct | 629 | 52.46% | 13.93% | 0.7091 |
| total_score | theme_score | both_top_30pct | 635 | 52.44% | 13.91% | 0.2012 |
| total_score | trend_hold_score | both_top_30pct | 838 | 52.39% | 13.85% | 0.8220 |
| d1_low_ma10_pct | d1_close_ma10_pct | both_top_30pct | 672 | 52.08% | 13.55% | 0.8694 |
| theme_score | d1_close_ma10_pct | both_top_30pct | 610 | 51.97% | 13.44% | 0.1207 |
| total_score | graph_quality_score | both_top_30pct | 648 | 51.85% | 13.32% | 0.7753 |
| d1_close_ma10_pct | consecutive_boards | both_top_30pct | 887 | 51.63% | 13.10% | 0.4061 |
| total_score | consecutive_boards | both_top_30pct | 920 | 51.41% | 12.88% | 0.2797 |
| d1_low_ma10_pct | d1_close_vwap_pct | both_top_30pct | 192 | 51.04% | 12.51% | -0.1437 |
| graph_quality_score | entry_width_score | both_top_30pct | 492 | 51.02% | 12.48% | 0.0304 |
| trend_hold_score | d1_close_ma10_pct | both_top_30pct | 809 | 50.93% | 12.40% | 0.7900 |
| total_score | active_money_score | both_top_30pct | 228 | 50.88% | 12.35% | -0.0631 |
| theme_score | d1_low_ma10_pct | both_top_30pct | 599 | 50.08% | 11.55% | 0.1453 |
| graph_quality_score | trend_hold_score | both_top_30pct | 672 | 49.85% | 11.32% | 0.5137 |
| graph_quality_score | d1_close_vwap_pct | both_top_30pct | 442 | 49.77% | 11.24% | 0.3687 |
| support_score | trend_hold_score | both_top_30pct | 575 | 49.74% | 11.21% | 0.0323 |
| trend_hold_score | d1_close_vwap_pct | both_top_30pct | 305 | 49.51% | 10.98% | -0.1239 |

## Daily Stability Snapshot

| feature | days | avg spread | avg corr |
|---|---:|---:|---:|
| total_score | 14 | 0.2994 | 0.2064 |
| d1_close_ma10_pct | 14 | 0.2442 | 0.2000 |
| d1_low_ma10_pct | 14 | 0.2181 | 0.1650 |
| graph_quality_score | 14 | 0.2128 | 0.1590 |
| trend_hold_score | 14 | 0.1643 | 0.1588 |
| support_score | 14 | 0.1406 | 0.0666 |
| entry_width_score | 14 | 0.0855 | 0.0805 |
| theme_score | 14 | 0.0520 | 0.0610 |
| d1_close_vwap_pct | 14 | 0.0472 | 0.0296 |
| active_cooling_score | 14 | 0.0374 | 0.0286 |
| consecutive_boards | 14 | 0.0192 | 0.1464 |
| active_money_score | 14 | -0.1014 | -0.0744 |
| low_absorb_width_pct | 14 | -0.1139 | -0.1109 |
| invalid_distance_pct | 14 | -0.1139 | -0.1109 |
| days_since_d0 | 14 | -0.1821 | -0.1777 |

## Boundary

These outputs are evidence for manual research. They are not a trading model and are not a validated ranking model.