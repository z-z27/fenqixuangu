# Factor Analysis Report

## Scope

This report is a descriptive factor-analysis layer. It does not generate ranking_model_*.json.
A ranking model must be manually constructed after reviewing these outputs.

## Sample

- samples file: `reports\history_samples\2026-06-01_2026-06-25\history_candidates_2026-06-01_2026-06-25.csv`
- rows after filters: **1548**
- dates: **12**
- eligible only: **True**
- target7 rows: **523**
- target7 rate: **33.79%**

## Top Factor Signals

| feature | type | strength | corr target7 | corr D3 max | top q rate | bottom q rate | direction |
|---|---|---:|---:|---:|---:|---:|---|
| total_score | numeric | 0.4986 | 0.1449 | 0.1444 | 45.69% | 24.76% | higher_better |
| trend_hold_score | numeric | 0.3373 | 0.1187 | 0.1069 | 40.12% | 28.95% | higher_better |
| d1_low_ma10_pct | numeric | 0.2954 | 0.0941 | 0.0820 | 44.52% | 32.58% | higher_better |
| d1_close_ma10_pct | numeric | 0.2875 | 0.0863 | 0.0689 | 42.58% | 29.35% | higher_better |
| theme_score | numeric | 0.2784 | 0.0942 | 0.0910 | 37.12% | 27.80% | higher_better |
| active_cooling_score | numeric | 0.2586 | 0.0740 | 0.1022 | 37.04% | 28.80% | higher_better |
| graph_quality_score | numeric | 0.2344 | 0.0758 | 0.0673 | 39.77% | 30.64% | higher_better |
| d1_close_vwap_pct | numeric | 0.1877 | -0.0579 | -0.0878 | 35.16% | 39.35% | lower_better |
| active_money_score | numeric | 0.1831 | -0.0440 | -0.0724 | 28.27% | 34.94% | lower_better |
| consecutive_boards | numeric | 0.1422 | 0.0653 | 0.0668 | 33.79% | 32.79% | higher_better |
| invalid_distance_pct | numeric | 0.1392 | -0.0437 | -0.0471 | 30.00% | 34.84% | lower_better |
| low_absorb_width_pct | numeric | 0.1392 | -0.0437 | -0.0471 | 30.00% | 34.84% | lower_better |
| entry_width_score | numeric | 0.1188 | 0.0531 | 0.0620 | 33.58% | 33.94% | lower_better |
| support_score | numeric | 0.1129 | -0.0275 | -0.0519 | 31.76% | 35.11% | lower_better |
| days_since_d0 | numeric | 0.0721 | -0.0317 | -0.0053 | 32.05% | 35.57% | lower_better |
| support_type | categorical | 0.0577 |  |  |  |  | bucket_spread=0.0577 |
| signal_type | categorical | 0.0000 |  |  |  |  | bucket_spread=0.0000 |

## Strong Buckets

| feature | bucket | count | target7 rate | lift | avg D3 max% |
|---|---|---:|---:|---:|---:|
| trend_hold_score | (75.0, 95.0] | 250 | 46.80% | 13.01% | 7.0176 |
| total_score | (74.25, 88.75] | 293 | 46.42% | 12.63% | 7.1948 |
| d1_low_ma10_pct | (8.589, 33.548] | 310 | 44.52% | 10.73% | 6.8197 |
| d1_close_ma10_pct | (13.323, 39.123] | 310 | 42.58% | 8.80% | 6.6332 |
| entry_width_score | (65.0, 90.0] | 139 | 42.45% | 8.66% | 6.5184 |
| graph_quality_score | (70.0, 82.0] | 247 | 42.11% | 8.32% | 6.5973 |
| d1_close_vwap_pct | (-6.261, -1.662] | 310 | 39.35% | 5.57% | 6.6883 |
| support_type | B | 228 | 38.16% | 4.37% | 6.3582 |
| invalid_distance_pct | (4.123, 4.628] | 309 | 37.86% | 4.08% | 6.2723 |
| low_absorb_width_pct | (4.123, 4.628] | 309 | 37.86% | 4.08% | 6.2723 |
| active_money_score | (60.0, 72.0] | 315 | 37.46% | 3.67% | 5.8578 |
| theme_score | (60.0, 70.0] | 994 | 37.12% | 3.34% | 6.0293 |
| active_cooling_score | (85.0, 90.0] | 575 | 37.04% | 3.26% | 6.2223 |
| graph_quality_score | (82.0, 90.0] | 165 | 36.97% | 3.18% | 5.6445 |
| support_score | (39.0, 67.0] | 278 | 36.69% | 2.91% | 6.1687 |
| total_score | (66.75, 74.25] | 316 | 36.39% | 2.61% | 5.9973 |
| active_cooling_score | (70.0, 85.0] | 334 | 35.63% | 1.84% | 5.9213 |
| active_money_score | (72.0, 84.0] | 318 | 35.22% | 1.43% | 5.7800 |
| d1_close_vwap_pct | (1.189, 7.52] | 310 | 35.16% | 1.38% | 5.4594 |
| support_score | (-0.001, 21.0] | 487 | 35.11% | 1.33% | 5.8599 |
| active_money_score | (31.999, 50.0] | 395 | 34.94% | 1.15% | 5.9606 |
| invalid_distance_pct | (-0.001, 2.782] | 310 | 34.84% | 1.05% | 5.8093 |
| low_absorb_width_pct | (-0.001, 2.782] | 310 | 34.84% | 1.05% | 5.8093 |
| invalid_distance_pct | (3.489, 4.123] | 310 | 34.52% | 0.73% | 5.7093 |
| low_absorb_width_pct | (3.489, 4.123] | 310 | 34.52% | 0.73% | 5.7093 |
| graph_quality_score | (50.0, 70.0] | 499 | 34.47% | 0.68% | 5.8738 |
| trend_hold_score | (55.0, 65.0] | 325 | 34.46% | 0.68% | 5.7884 |
| days_since_d0 | (0.999, 2.0] | 1158 | 34.37% | 0.58% | 5.6374 |
| d1_low_ma10_pct | (4.887, 8.589] | 309 | 34.30% | 0.52% | 5.7304 |
| d1_close_vwap_pct | (-1.662, -0.672] | 309 | 33.98% | 0.20% | 5.4885 |
| entry_width_score | (24.999, 60.0] | 872 | 33.94% | 0.16% | 5.6415 |
| trend_hold_score | (65.0, 75.0] | 266 | 33.83% | 0.05% | 5.7377 |
| consecutive_boards | (0.999, 3.0] | 1548 | 33.79% | 0.00% | 5.6431 |
| signal_type | D2_LOW_ABSORB | 1548 | 33.79% | 0.00% | 5.6431 |
| support_type | C | 687 | 33.62% | -0.16% | 5.7318 |
| active_cooling_score | (65.0, 70.0] | 146 | 33.56% | -0.22% | 5.2340 |
| active_money_score | (50.0, 60.0] | 228 | 33.33% | -0.45% | 5.9325 |
| support_score | (67.0, 89.0] | 324 | 32.72% | -1.07% | 5.5813 |
| d1_close_ma10_pct | (8.574, 13.323] | 309 | 32.69% | -1.10% | 5.3897 |
| d1_close_ma10_pct | (5.073, 8.574] | 310 | 32.58% | -1.20% | 5.4524 |

## Pair Review

| left | right | rule | count | target7 rate | lift | corr |
|---|---|---|---:|---:|---:|---:|
| total_score | d1_close_ma10_pct | both_top_30pct | 332 | 44.28% | 10.49% | 0.7669 |
| total_score | theme_score | both_top_30pct | 344 | 43.90% | 10.11% | 0.1726 |
| total_score | active_money_score | both_top_30pct | 190 | 43.68% | 9.90% | 0.1317 |
| total_score | consecutive_boards | both_top_30pct | 468 | 43.59% | 9.80% | 0.2127 |
| total_score | trend_hold_score | both_top_30pct | 374 | 43.58% | 9.80% | 0.8481 |
| theme_score | d1_low_ma10_pct | both_top_30pct | 327 | 43.43% | 9.64% | 0.1091 |
| total_score | graph_quality_score | both_top_30pct | 435 | 43.22% | 9.43% | 0.8307 |
| theme_score | trend_hold_score | both_top_30pct | 359 | 43.18% | 9.39% | 0.1078 |
| total_score | d1_low_ma10_pct | both_top_30pct | 348 | 43.10% | 9.32% | 0.7953 |
| total_score | low_absorb_width_pct | both_top_30pct | 119 | 42.86% | 9.07% | -0.1635 |
| total_score | invalid_distance_pct | both_top_30pct | 119 | 42.86% | 9.07% | -0.1635 |
| total_score | active_cooling_score | both_top_30pct | 188 | 42.55% | 8.77% | 0.1615 |
| total_score | entry_width_score | both_top_30pct | 217 | 42.40% | 8.61% | 0.1474 |
| active_cooling_score | d1_close_vwap_pct | both_top_30pct | 124 | 41.94% | 8.15% | -0.2847 |
| graph_quality_score | trend_hold_score | both_top_30pct | 397 | 41.56% | 7.78% | 0.6314 |
| graph_quality_score | d1_low_ma10_pct | both_top_30pct | 365 | 41.37% | 7.58% | 0.6383 |
| active_cooling_score | theme_score | both_top_30pct | 380 | 41.32% | 7.53% | -0.0024 |
| trend_hold_score | d1_close_ma10_pct | both_top_30pct | 388 | 41.24% | 7.45% | 0.8602 |
| total_score | d1_close_vwap_pct | both_top_30pct | 199 | 41.21% | 7.42% | 0.1970 |
| entry_width_score | d1_low_ma10_pct | both_top_30pct | 211 | 40.76% | 6.97% | 0.0515 |
| d1_low_ma10_pct | d1_close_ma10_pct | both_top_30pct | 371 | 40.70% | 6.92% | 0.9353 |
| active_cooling_score | d1_low_ma10_pct | both_top_30pct | 118 | 40.68% | 6.89% | -0.1948 |
| trend_hold_score | entry_width_score | both_top_30pct | 239 | 40.59% | 6.80% | 0.0532 |
| entry_width_score | d1_close_ma10_pct | both_top_30pct | 196 | 40.31% | 6.52% | 0.0461 |
| trend_hold_score | d1_low_ma10_pct | both_top_30pct | 465 | 40.22% | 6.43% | 0.9140 |
| d1_low_ma10_pct | consecutive_boards | both_top_30pct | 465 | 40.22% | 6.43% | 0.3388 |
| active_cooling_score | trend_hold_score | both_top_30pct | 137 | 40.15% | 6.36% | -0.1710 |
| trend_hold_score | consecutive_boards | both_top_30pct | 516 | 40.12% | 6.33% | 0.3067 |
| trend_hold_score | low_absorb_width_pct | both_top_30pct | 145 | 40.00% | 6.21% | -0.1068 |
| trend_hold_score | invalid_distance_pct | both_top_30pct | 145 | 40.00% | 6.21% | -0.1068 |

## Daily Stability Snapshot

| feature | days | avg spread | avg corr |
|---|---:|---:|---:|
| total_score | 12 | 0.2340 | 0.1615 |
| d1_close_ma10_pct | 12 | 0.1219 | 0.1017 |
| d1_low_ma10_pct | 12 | 0.1148 | 0.1000 |
| graph_quality_score | 12 | 0.1117 | 0.0830 |
| trend_hold_score | 12 | 0.1071 | 0.1300 |
| active_cooling_score | 12 | 0.0931 | 0.0688 |
| theme_score | 12 | 0.0656 | 0.0726 |
| support_score | 12 | 0.0233 | 0.0078 |
| days_since_d0 | 12 | 0.0195 | 0.0193 |
| d1_close_vwap_pct | 12 | 0.0190 | 0.0038 |
| consecutive_boards | 12 | 0.0098 | 0.1152 |
| entry_width_score | 12 | -0.0004 | 0.0525 |
| low_absorb_width_pct | 12 | -0.0207 | -0.0395 |
| invalid_distance_pct | 12 | -0.0207 | -0.0395 |
| active_money_score | 12 | -0.0809 | -0.0260 |

## Boundary

These outputs are evidence for manual research. They are not a trading model and are not a validated ranking model.