# Return Distribution Report

- source: `reports\research_runs\2026-06-01_2026-06-25_research_top999_all_v2\research_results\research_samples_2026-06-01_2026-06-25.csv`
- records: **548**
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
| close_-3_to_0 | 72 | -1.7036 | 0.0000 | 0.0000 | 3.6802 | 0.1389 | 71.0500 | 70.9167 | 74.3194 | 70.6389 |
| close_-7_to_-3 | 151 | -5.0175 | 0.0000 | 1.0000 | 1.9345 | 0.0397 | 62.7944 | 60.8742 | 58.7748 | 68.7152 |
| close_0_to_3 | 65 | 1.3062 | 0.0000 | 0.0000 | 6.1333 | 0.3538 | 72.8169 | 75.9385 | 73.6923 | 71.0154 |
| close_3_to_7 | 62 | 4.7551 | 0.0000 | 0.0000 | 8.9279 | 0.8226 | 72.1452 | 73.7097 | 75.6774 | 69.9355 |
| close_7_to_10 | 25 | 8.6583 | 1.0000 | 0.0000 | 11.1128 | 1.0000 | 68.1420 | 73.1200 | 65.9200 | 65.0400 |
| close_<-7 | 126 | -9.7232 | 0.0000 | 1.0000 | 1.2657 | 0.0476 | 66.6298 | 63.9365 | 66.9127 | 71.6190 |
| close_>=10 | 37 | 14.0578 | 1.0000 | 0.0000 | 15.8796 | 1.0000 | 70.7405 | 73.8919 | 73.1622 | 68.1622 |
| data_issue | 10 |  | 0.0000 | 0.0000 |  | 0.0000 | 74.2200 | 67.2000 | 90.6000 | 78.0000 |

## Profit/Loss Comparison Groups

| universe | group | count | avg_primary_d3_close_return | avg_opportunity_d3_max_return | primary_target_rate | primary_loss_rate | opportunity_target_rate | avg_total_score | avg_graph_quality_score | avg_support_score | avg_active_money_score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_candidates | all_d3_close_profit_ge_target | 62 | 11.8806 | 13.9575 | 1.0000 | 0.0000 | 1.0000 | 69.6927 | 73.5806 | 70.2419 | 66.9032 |
| all_candidates | all_d3_close_loss_le_cutoff | 277 | -7.1580 | 1.6303 | 0.0000 | 1.0000 | 0.0433 | 64.5390 | 62.2671 | 62.4765 | 70.0361 |
| all_candidates | all_d3_opportunity_ge_target | 158 | 5.8243 | 10.9989 | 0.3924 | 0.0759 | 1.0000 | 71.7462 | 75.6582 | 72.5253 | 69.0633 |
| all_candidates | all_d3_opportunity_lt_3 | 256 | -6.4122 | 0.7272 | 0.0000 | 0.8320 | 0.0000 | 64.3746 | 61.6562 | 62.3086 | 69.9531 |
| all_candidates | all_d3_close_top20pct | 108 | 9.0515 | 11.9667 | 0.5741 | 0.0000 | 0.9444 | 71.0130 | 73.4630 | 73.6574 | 68.2407 |
| all_candidates | all_d3_close_bottom20pct | 108 | -10.1298 | 1.1356 | 0.0000 | 1.0000 | 0.0463 | 67.2167 | 64.6667 | 68.1204 | 71.8519 |
| all_candidates | all_d3_opportunity_top20pct | 108 | 8.0036 | 12.5340 | 0.5556 | 0.0278 | 1.0000 | 72.1398 | 75.9815 | 72.6667 | 70.0741 |
| executed_only | executed_realized_profit_ge_target | 11 | 3.3392 | 8.9501 | 0.0909 | 0.0909 | 0.8182 | 78.7318 | 82.5455 | 83.7273 | 75.2727 |
| executed_only | executed_realized_loss_le_cutoff | 26 | -7.1531 | 0.6522 | 0.0000 | 1.0000 | 0.0000 | 78.2865 | 82.0000 | 80.4615 | 77.1538 |

## Factor Quantile Report (5 buckets)

Use the CSV for full detail. The preview below shows all-sample D3 close-return quantiles.

| factor | quantile | count | factor_mean | target_mean | target_median | target7_rate | loss_rate | opportunity7_rate |
|---|---|---|---|---|---|---|---|---|
| active_money_score | 1 | 108 | 46.1852 | -1.1037 | -3.2893 | 0.1574 | 0.5093 | 0.3426 |
| active_money_score | 2 | 107 | 59.0841 | -1.6787 | -3.0140 | 0.1215 | 0.5047 | 0.2804 |
| active_money_score | 3 | 108 | 69.2407 | -2.5551 | -3.8246 | 0.1019 | 0.5463 | 0.2593 |
| active_money_score | 4 | 107 | 81.4393 | -1.7946 | -3.0794 | 0.1028 | 0.5047 | 0.2991 |
| active_money_score | 5 | 108 | 93.3704 | -2.0585 | -3.1165 | 0.0926 | 0.5093 | 0.2870 |
| consecutive_boards | 1 | 108 | 1.0000 | -2.3742 | -3.9047 | 0.1296 | 0.5463 | 0.3704 |
| consecutive_boards | 2 | 107 | 1.0000 | -1.2193 | -3.0794 | 0.1869 | 0.5140 | 0.3364 |
| consecutive_boards | 3 | 108 | 1.0000 | -3.1997 | -3.8743 | 0.0370 | 0.6111 | 0.1296 |
| consecutive_boards | 4 | 107 | 1.0000 | -1.5810 | -2.7778 | 0.0935 | 0.4579 | 0.2523 |
| consecutive_boards | 5 | 108 | 1.9537 | -0.8103 | -1.5556 | 0.1296 | 0.4444 | 0.3796 |
| daily_rank | 1 | 24 | 6.5000 | -1.6365 | -3.0820 | 0.1250 | 0.5000 | 0.2500 |
| daily_rank | 2 | 24 | 18.5000 | -2.7014 | -2.3755 | 0.0417 | 0.4583 | 0.2083 |
| daily_rank | 3 | 23 | 30.2609 | -2.5369 | -2.8146 | 0.0870 | 0.4783 | 0.2609 |
| daily_rank | 4 | 24 | 42.0000 | -2.0730 | -3.2412 | 0.0833 | 0.5417 | 0.2083 |
| daily_rank | 5 | 24 | 57.0000 | -3.1243 | -2.8903 | 0.0000 | 0.5000 | 0.1667 |
| days_since_d0 | 1 | 108 | 0.0000 | 1.0233 | 0.5817 | 0.2315 | 0.3333 | 0.5278 |
| days_since_d0 | 2 | 107 | 0.4766 | -1.8677 | -2.7778 | 0.1215 | 0.4860 | 0.3551 |
| days_since_d0 | 3 | 108 | 1.1481 | -2.6187 | -3.8403 | 0.0833 | 0.5556 | 0.2315 |
| days_since_d0 | 4 | 107 | 2.0000 | -2.4817 | -4.1577 | 0.0935 | 0.5794 | 0.2336 |
| days_since_d0 | 5 | 108 | 2.7315 | -3.2540 | -3.8596 | 0.0463 | 0.6204 | 0.1204 |
| graph_quality_score | 1 | 108 | 29.7593 | -4.3697 | -5.2258 | 0.0556 | 0.6944 | 0.1296 |
| graph_quality_score | 2 | 107 | 58.3551 | -2.2578 | -3.8168 | 0.1215 | 0.5701 | 0.2523 |
| graph_quality_score | 3 | 108 | 74.1296 | -0.9658 | -2.6013 | 0.1481 | 0.4907 | 0.3333 |
| graph_quality_score | 4 | 107 | 86.3364 | -1.4020 | -2.6316 | 0.1215 | 0.4673 | 0.3832 |
| graph_quality_score | 5 | 108 | 90.0000 | -0.1971 | -0.5194 | 0.1296 | 0.3519 | 0.3704 |
| invalid_distance_pct | 1 | 108 | 1.6097 | -0.2229 | -1.2456 | 0.2222 | 0.4167 | 0.4167 |
| invalid_distance_pct | 2 | 107 | 3.0635 | -2.6741 | -3.8424 | 0.0748 | 0.5607 | 0.2150 |
| invalid_distance_pct | 3 | 108 | 3.8602 | -1.1140 | -2.2022 | 0.1111 | 0.4815 | 0.3796 |
| invalid_distance_pct | 4 | 107 | 4.4853 | -2.0384 | -3.7736 | 0.1121 | 0.5514 | 0.2710 |
| invalid_distance_pct | 5 | 108 | 5.1478 | -3.1528 | -3.7631 | 0.0556 | 0.5648 | 0.1852 |
| low_absorb_width_pct | 1 | 108 | 1.6097 | -0.2229 | -1.2456 | 0.2222 | 0.4167 | 0.4167 |
| low_absorb_width_pct | 2 | 107 | 3.0635 | -2.6741 | -3.8424 | 0.0748 | 0.5607 | 0.2150 |
| low_absorb_width_pct | 3 | 108 | 3.8602 | -1.1140 | -2.2022 | 0.1111 | 0.4815 | 0.3796 |
| low_absorb_width_pct | 4 | 107 | 4.4853 | -2.0384 | -3.7736 | 0.1121 | 0.5514 | 0.2710 |
| low_absorb_width_pct | 5 | 108 | 5.1478 | -3.1528 | -3.7631 | 0.0556 | 0.5648 | 0.1852 |
| support_score | 1 | 108 | 18.5000 | -3.3378 | -4.8074 | 0.0833 | 0.6574 | 0.1944 |
| support_score | 2 | 107 | 56.5327 | -2.4166 | -4.0896 | 0.1215 | 0.5888 | 0.2804 |
| support_score | 3 | 108 | 82.9259 | -2.3781 | -3.5796 | 0.0741 | 0.5648 | 0.2778 |
| support_score | 4 | 107 | 90.3084 | 0.1824 | 0.4660 | 0.2056 | 0.3832 | 0.4579 |
| support_score | 5 | 108 | 91.0000 | -1.2291 | -1.6189 | 0.0926 | 0.3796 | 0.2593 |

## Daily Summary

| signal_date | count | avg_primary_d3_close_return | avg_opportunity_d3_max_return | primary_target_rate | primary_loss_rate | opportunity_target_rate | executed_rate |
|---|---|---|---|---|---|---|---|
| 2026-06-15 | 4 |  |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2026-06-16 | 6 | 4.1863 | 10.4089 | 0.1667 | 0.1667 | 0.1667 | 0.0000 |
| 2026-06-22 | 1 | 11.1635 | 11.1635 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 2026-06-24 | 245 | -1.5501 | 6.2733 | 0.1592 | 0.5061 | 0.3796 | 0.0898 |
| 2026-06-25 | 292 | -2.1643 | 3.3483 | 0.0719 | 0.5205 | 0.2158 | 0.1507 |

## How to Use This Report

- First compare high-return buckets against low-return buckets.
- Then check whether factor quantiles show monotonic return differences.
- Treat execution and TopN fields as metadata, not primary labels.
- Do not modify strategy from a single run; use this output to decide which factors deserve deeper validation.