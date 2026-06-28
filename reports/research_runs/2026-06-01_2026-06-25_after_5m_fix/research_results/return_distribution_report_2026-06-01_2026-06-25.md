# Return Distribution Report

- source: `reports\research_runs\2026-06-01_2026-06-25_after_5m_fix\research_results\research_samples_2026-06-01_2026-06-25.csv`
- records: **2962**
- primary all-sample return: `candidate_d3_close_return_pct`
- opportunity all-sample return: `candidate_d3_max_return_pct`
- target return threshold: **7.00%**
- loss cutoff threshold: **-3.00%**

## Research Principle

This report treats every candidate as research data. It does not use TopN selection or execution status as the primary label.
The goal is to compare profit/loss and opportunity distributions so later factor changes can be based on data rather than trade-state labels.

## All-Sample Return Buckets

| bucket | count | avg_primary_d3_close_return | primary_target_rate | primary_loss_rate | avg_opportunity_d3_max_return | opportunity_target_rate | avg_total_score | avg_graph_quality_score | avg_support_score | avg_active_money_score |
|---|---|---|---|---|---|---|---|---|---|---|
| close_-3_to_0 | 485 | -1.5572 | 0.0000 | 0.0000 | 3.7756 | 0.1629 | 63.1546 | 56.6186 | 66.4227 | 68.5856 |
| close_-7_to_-3 | 591 | -4.8472 | 0.0000 | 1.0000 | 2.1978 | 0.0677 | 62.5811 | 56.5854 | 62.8274 | 70.0643 |
| close_0_to_3 | 431 | 1.4588 | 0.0000 | 0.0000 | 5.5016 | 0.2575 | 63.9614 | 59.4571 | 64.6450 | 69.1972 |
| close_3_to_7 | 442 | 4.9176 | 0.0000 | 0.0000 | 8.3234 | 0.6357 | 64.5552 | 59.8959 | 66.8439 | 68.7330 |
| close_7_to_10 | 223 | 8.4302 | 1.0000 | 0.0000 | 11.3255 | 1.0000 | 64.7610 | 63.3274 | 66.4709 | 65.3991 |
| close_<-7 | 413 | -10.1789 | 0.0000 | 1.0000 | 0.9176 | 0.0484 | 65.3525 | 61.6368 | 65.2760 | 71.8886 |
| close_>=10 | 367 | 15.1448 | 1.0000 | 0.0000 | 16.7365 | 1.0000 | 65.0093 | 63.8256 | 66.2289 | 64.9864 |
| data_issue | 10 |  | 0.0000 | 0.0000 |  | 0.0000 | 74.2200 | 67.2000 | 90.6000 | 78.0000 |

## Profit/Loss Comparison Groups

| universe | group | count | avg_primary_d3_close_return | avg_opportunity_d3_max_return | primary_target_rate | primary_loss_rate | opportunity_target_rate | avg_total_score | avg_graph_quality_score | avg_support_score | avg_active_money_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_candidates | all_d3_close_profit_ge_target | 590 | 12.6069 | 14.6913 | 1.0000 | 0.0000 | 1.0000 | 64.9154 | 63.6373 | 66.3203 | 65.1424 |
| all_candidates | all_d3_close_loss_le_cutoff | 1004 | -7.0404 | 1.6712 | 0.0000 | 1.0000 | 0.0598 | 63.7212 | 58.6633 | 63.8347 | 70.8147 |
| all_candidates | all_d3_opportunity_ge_target | 1121 | 7.6434 | 12.2182 | 0.5263 | 0.0535 | 1.0000 | 65.8967 | 63.8216 | 67.9545 | 67.2525 |
| all_candidates | all_d3_opportunity_lt_3 | 1063 | -5.3640 | 0.7347 | 0.0000 | 0.6952 | 0.0000 | 62.1639 | 55.9360 | 61.9003 | 70.3086 |
| all_candidates | all_d3_close_top20pct | 591 | 12.5975 | 14.6971 | 0.9983 | 0.0000 | 1.0000 | 64.9439 | 63.6819 | 66.3621 | 65.1472 |
| all_candidates | all_d3_close_bottom20pct | 591 | -8.9968 | 1.0945 | 0.0000 | 1.0000 | 0.0423 | 64.9103 | 60.8460 | 64.5262 | 71.8443 |
| all_candidates | all_d3_opportunity_top20pct | 591 | 11.8139 | 15.3374 | 0.8173 | 0.0017 | 1.0000 | 65.3340 | 63.9154 | 66.5195 | 66.2234 |
| executed_only | executed_realized_profit_ge_target | 49 | 4.7650 | 9.4088 | 0.3265 | 0.0816 | 0.6939 | 78.9592 | 81.6327 | 84.1429 | 77.9184 |
| executed_only | executed_realized_loss_le_cutoff | 42 | -7.4882 | 0.8187 | 0.0000 | 1.0000 | 0.0000 | 78.7060 | 80.8095 | 81.7143 | 79.1905 |

## Factor Quantile Report (5 buckets)

Use the CSV for full detail. The preview below shows all-sample D3 close-return quantiles.

| factor | quantile | count | factor_mean | target_mean | target_median | target7_rate | loss_rate | opportunity7_rate |
|---|---|---|---|---|---|---|---|---|
| active_money_score | 1 | 591 | 44.9543 | 2.0949 | 0.9227 | 0.2707 | 0.2792 | 0.4399 |
| active_money_score | 2 | 590 | 57.5627 | 1.2524 | 0.0614 | 0.2186 | 0.3271 | 0.3746 |
| active_money_score | 3 | 590 | 68.4237 | 0.7418 | 0.0000 | 0.1831 | 0.3305 | 0.3864 |
| active_money_score | 4 | 590 | 80.1051 | 0.7549 | 0.2875 | 0.1881 | 0.3390 | 0.3780 |
| active_money_score | 5 | 591 | 92.7885 | -0.7504 | -1.7257 | 0.1387 | 0.4247 | 0.3198 |
| consecutive_boards | 1 | 591 | 1.0000 | -0.6858 | -1.5775 | 0.1354 | 0.4095 | 0.3299 |
| consecutive_boards | 2 | 590 | 1.0000 | 3.0967 | 2.1073 | 0.2678 | 0.2153 | 0.4271 |
| consecutive_boards | 3 | 590 | 1.0000 | 1.3343 | 0.2593 | 0.2068 | 0.2881 | 0.3661 |
| consecutive_boards | 4 | 590 | 1.0000 | 0.1993 | -0.4342 | 0.2085 | 0.3797 | 0.3898 |
| consecutive_boards | 5 | 591 | 1.6684 | 0.1521 | -1.2925 | 0.1810 | 0.4078 | 0.3858 |
| daily_rank | 1 | 64 | 3.1875 | -0.3594 | -1.5998 | 0.1719 | 0.4219 | 0.4062 |
| daily_rank | 2 | 64 | 8.7969 | 0.8917 | -0.2804 | 0.2188 | 0.3281 | 0.3906 |
| daily_rank | 3 | 63 | 16.6032 | 1.3207 | 0.5128 | 0.1905 | 0.2698 | 0.4127 |
| daily_rank | 4 | 64 | 28.2031 | -0.4467 | -1.2320 | 0.2031 | 0.4062 | 0.3906 |
| daily_rank | 5 | 64 | 47.1562 | -1.3655 | -2.0432 | 0.0938 | 0.4531 | 0.2188 |
| days_since_d0 | 1 | 591 | 0.0000 | 2.4533 | 1.8033 | 0.2657 | 0.2826 | 0.4890 |
| days_since_d0 | 2 | 590 | 0.0831 | 1.2579 | 0.6164 | 0.2220 | 0.3102 | 0.4644 |
| days_since_d0 | 3 | 590 | 1.0000 | 0.6269 | 0.1019 | 0.1831 | 0.3288 | 0.3593 |
| days_since_d0 | 4 | 590 | 1.9695 | -0.1406 | -0.9813 | 0.1695 | 0.3966 | 0.3119 |
| days_since_d0 | 5 | 591 | 3.4027 | -0.1056 | -1.3055 | 0.1591 | 0.3824 | 0.2741 |
| graph_quality_score | 1 | 591 | 24.9882 | 0.1895 | -0.8117 | 0.1506 | 0.3418 | 0.2978 |
| graph_quality_score | 2 | 590 | 44.9559 | 0.4935 | -0.4243 | 0.1763 | 0.3610 | 0.3288 |
| graph_quality_score | 3 | 590 | 62.6305 | 0.8940 | -0.5251 | 0.2085 | 0.3559 | 0.3644 |
| graph_quality_score | 4 | 590 | 76.2610 | 1.3446 | 0.6003 | 0.2390 | 0.3254 | 0.4254 |
| graph_quality_score | 5 | 591 | 89.2826 | 1.1719 | 0.6866 | 0.2250 | 0.3164 | 0.4822 |
| invalid_distance_pct | 1 | 591 | 1.0808 | 2.3248 | 1.7428 | 0.2741 | 0.2927 | 0.4839 |
| invalid_distance_pct | 2 | 590 | 2.6907 | 0.4408 | -0.7322 | 0.1763 | 0.3458 | 0.3492 |
| invalid_distance_pct | 3 | 590 | 3.5282 | 0.7033 | -0.3515 | 0.1915 | 0.3390 | 0.3525 |
| invalid_distance_pct | 4 | 590 | 4.2184 | 0.6162 | -0.4230 | 0.1814 | 0.3458 | 0.3729 |
| invalid_distance_pct | 5 | 591 | 5.0059 | 0.0068 | -0.7647 | 0.1760 | 0.3773 | 0.3401 |
| low_absorb_width_pct | 1 | 591 | 1.0808 | 2.3248 | 1.7428 | 0.2741 | 0.2927 | 0.4839 |
| low_absorb_width_pct | 2 | 590 | 2.6907 | 0.4408 | -0.7322 | 0.1763 | 0.3458 | 0.3492 |
| low_absorb_width_pct | 3 | 590 | 3.5282 | 0.7033 | -0.3515 | 0.1915 | 0.3390 | 0.3525 |
| low_absorb_width_pct | 4 | 590 | 4.2184 | 0.6162 | -0.4230 | 0.1814 | 0.3458 | 0.3729 |
| low_absorb_width_pct | 5 | 591 | 5.0059 | 0.0068 | -0.7647 | 0.1760 | 0.3773 | 0.3401 |
| support_score | 1 | 591 | 12.3113 | 0.6215 | -0.6177 | 0.1895 | 0.3655 | 0.3350 |
| support_score | 2 | 590 | 47.9695 | 0.8576 | -0.2753 | 0.2119 | 0.3458 | 0.3593 |
| support_score | 3 | 590 | 84.3932 | -0.1054 | -1.2466 | 0.1593 | 0.3932 | 0.3475 |
| support_score | 4 | 590 | 91.0000 | 2.1772 | 1.6403 | 0.2424 | 0.2678 | 0.4492 |
| support_score | 5 | 591 | 91.0000 | 0.5430 | -0.2121 | 0.1963 | 0.3283 | 0.4078 |

## Daily Summary

| signal_date | count | avg_primary_d3_close_return | avg_opportunity_d3_max_return | primary_target_rate | primary_loss_rate | opportunity_target_rate | executed_rate |
|---|---|---|---|---|---|---|---|
| 2026-06-05 | 62 | 0.8218 | 6.6405 | 0.1935 | 0.3387 | 0.4355 | 0.0000 |
| 2026-06-08 | 95 | -0.5674 | 6.1366 | 0.2000 | 0.5158 | 0.3474 | 0.0211 |
| 2026-06-09 | 185 | -1.3785 | 5.6445 | 0.1243 | 0.4432 | 0.3243 | 0.0216 |
| 2026-06-10 | 188 | -1.7231 | 5.3853 | 0.0957 | 0.4787 | 0.3085 | 0.0372 |
| 2026-06-11 | 228 | 1.8957 | 6.0912 | 0.2018 | 0.2018 | 0.4079 | 0.0395 |
| 2026-06-12 | 289 | 3.9739 | 7.4137 | 0.2976 | 0.1903 | 0.4498 | 0.0519 |
| 2026-06-15 | 236 | 3.1503 | 7.0200 | 0.2881 | 0.2331 | 0.4322 | 0.0254 |
| 2026-06-16 | 268 | 1.6275 | 6.1936 | 0.1978 | 0.2537 | 0.3918 | 0.0522 |
| 2026-06-17 | 236 | 2.4696 | 6.3725 | 0.2542 | 0.2458 | 0.3729 | 0.0805 |
| 2026-06-18 | 286 | 0.4761 | 7.0387 | 0.2098 | 0.3601 | 0.4231 | 0.0629 |
| 2026-06-22 | 171 | 1.3027 | 6.3804 | 0.2456 | 0.3158 | 0.4211 | 0.0000 |
| 2026-06-23 | 181 | 1.7911 | 7.0187 | 0.2376 | 0.2597 | 0.4199 | 0.0276 |
| 2026-06-24 | 245 | -1.5501 | 6.2733 | 0.1592 | 0.5061 | 0.3796 | 0.0898 |
| 2026-06-25 | 292 | -2.1643 | 3.3483 | 0.0719 | 0.5205 | 0.2158 | 0.1507 |

## How to Use This Report

- First compare high-return buckets against low-return buckets.
- Then check whether factor quantiles show monotonic return differences.
- Treat execution and TopN fields as metadata, not primary labels.
- Do not modify strategy from a single run; use this output to decide which factors deserve deeper validation.