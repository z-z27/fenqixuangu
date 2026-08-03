# Stage 2 Nine-Day Model Behavior Diagnosis

## Scope and input identity

- Branch/HEAD: `research-sample-analysis` / `9f45af1f0ab8f506fb1cd41939e2c2985d0d9671`.
- Frozen samples SHA256: `f0a98512f1733f08b90d1555795b52fc9610b3ece754cfcf3141b6780ee001fc`.
- Holdout membership SHA256: `0165d84514b260d6bfaf1dbba49a7e51d5d5937d3b6e1e12faad004c07a0964b`.
- Holdout scored candidates SHA256: `839fff4cb12f55fe64d03431954ed566e863d37577d4a9e7292acb17a3d450cf`.
- Final Top3 SHA256: `c636af033dec2cade56790471e3a6d68be7076b1b98942415e718ec8073a9187`.
- Readiness remains `INSUFFICIENT_FORWARD_SAMPLE (9/30)`; this report is research-only.
- All returns and `target7_d2open_d3high` labels in this diagnosis come from the frozen history candidates, not copied from holdout outputs.

## Identity gates

- scorable rows/unique keys: **920 / 920**, duplicates: **0**.
- v004a Top15 / v002 Top15 / v005 candidate pool / final Top3: **135 / 135 / 135 / 27**.
- Each of nine dates contains **15 / 15 / 15 / 3** rows respectively.
- v005 candidate pool is exactly equal to v004a Top15 on every date. v005 therefore performs combination selection inside v004a's 15-stock recall set; it has no independent single-stock pool rank. `v005_pool_order` records the source v004a rank.

## Daily model behavior

| signal_date | scorable_count | target7_count_in_scorable | v004a_top15_hits | v002_top15_hits | v005_pool_hits | final_top3_hits | v004a_top3_hits | v002_top3_hits | final_top3_avg_return | v004a_top3_avg_return | v002_top3_avg_return | v004a_v002_top15_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-26 | 205 | 51 | 4 | 7 | 4 | 1 | 1 | 1 | 5.4181 | 3.0030 | 1.2651 | 7 |
| 2026-06-29 | 41 | 16 | 3 | 3 | 3 | 1 | 0 | 0 | 5.0408 | -2.3996 | 0.5040 | 14 |
| 2026-06-30 | 66 | 23 | 3 | 4 | 3 | 2 | 1 | 0 | 14.7513 | 10.7017 | -0.4293 | 14 |
| 2026-07-01 | 69 | 16 | 6 | 6 | 6 | 1 | 1 | 2 | 5.3383 | 5.5947 | 8.7477 | 14 |
| 2026-07-02 | 175 | 39 | 3 | 5 | 3 | 1 | 0 | 1 | 5.8553 | -1.7296 | 2.2917 | 11 |
| 2026-07-03 | 169 | 22 | 3 | 3 | 3 | 1 | 1 | 1 | 1.4265 | 1.8109 | 3.9745 | 7 |
| 2026-07-06 | 73 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | -3.2944 | -1.3830 | -1.3830 | 12 |
| 2026-07-07 | 49 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | -1.5079 | 0.7265 | -1.6269 | 13 |
| 2026-07-08 | 73 | 25 | 4 | 5 | 4 | 1 | 1 | 1 | 8.2921 | 5.3679 | 7.6138 | 13 |

## Failure attribution

The mutually exclusive precedence is: scorable hits <3 → `RECALL_FAILURE`; otherwise any Top15/pool hits <3 → `TOP15_RANKING_FAILURE`; otherwise final hits <3 → `TOP3_RANKING_FAILURE`; final hits=3 → `NO_FAILURE`.

| signal_date | failure_class | scorable_hit_count | v004a_top15_hit_count | v002_top15_hit_count | v005_pool_hit_count | final_top3_hit_count |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-26 | TOP3_RANKING_FAILURE | 51 | 4 | 7 | 4 | 1 |
| 2026-06-29 | TOP3_RANKING_FAILURE | 16 | 3 | 3 | 3 | 1 |
| 2026-06-30 | TOP3_RANKING_FAILURE | 23 | 3 | 4 | 3 | 2 |
| 2026-07-01 | TOP3_RANKING_FAILURE | 16 | 6 | 6 | 6 | 1 |
| 2026-07-02 | TOP3_RANKING_FAILURE | 39 | 3 | 5 | 3 | 1 |
| 2026-07-03 | TOP3_RANKING_FAILURE | 22 | 3 | 3 | 3 | 1 |
| 2026-07-06 | RECALL_FAILURE | 2 | 0 | 0 | 0 | 0 |
| 2026-07-07 | RECALL_FAILURE | 2 | 0 | 0 | 0 | 0 |
| 2026-07-08 | TOP3_RANKING_FAILURE | 25 | 4 | 5 | 4 | 1 |

Distribution: `{"RECALL_FAILURE":2,"TOP3_RANKING_FAILURE":7}`.

## Selection-path outcomes

| class | count |
| --- | --- |
| not_selected_non_hit | 705 |
| missed_hit_outside_candidate_pool | 170 |
| selected_false_positive | 19 |
| missed_hit_in_candidate_pool | 18 |
| selected_hit | 8 |
| v005_rescued_from_v004a | 0 |
| v005_rescued_from_v002 | 1 |
| v005_dropped_v004a_hit | 18 |
| v005_dropped_v002_hit | 26 |

`v005_rescued_from_*` means a target7 hit selected into final Top3 despite being outside that model's Top15. `v005_dropped_*_hit` means a target7 hit was in that model's Top15 but was not selected into final Top3. These flags may overlap the primary missed-hit classes.

## Baseline comparison

| baseline | ticket_count | single_ticket_target7_hit_rate | average_daily_hit_count | daily_top3_avg_return_pct | daily_positive_return_rate | top3_all_hit_rate | top3_at_least_one_hit_rate | worst_date | worst_daily_avg_return_pct | maximum_single_day_loss_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_top3 | 27 | 0.2963 | 0.8889 | 4.5911 | 0.7778 | 0.0000 | 0.7778 | 2026-07-06 | -3.2944 | 3.2944 |
| v004a_top3 | 27 | 0.1852 | 0.5556 | 2.4103 | 0.6667 | 0.0000 | 0.5556 | 2026-06-29 | -2.3996 | 2.3996 |
| v002_top3 | 27 | 0.2222 | 0.6667 | 2.3286 | 0.6667 | 0.0000 | 0.5556 | 2026-07-07 | -1.6269 | 1.6269 |
| scorable_pool | 920 | 0.2130 | 21.7778 | 2.7739 | 0.8889 |  |  | 2026-07-06 | -1.8425 | 1.8425 |
| random_top3_theoretical | 27 | 0.2203 | 0.6610 | 2.7739 | 0.8889 | 0.0184 | 0.4939 | 2026-07-06 | -1.8425 | 1.8425 |

- Final vs v004a daily comparison: `{"hit_losses":0,"hit_ties":6,"hit_wins":3,"return_losses":4,"return_ties":0,"return_wins":5}`.
- Final vs v002 daily comparison: `{"hit_losses":1,"hit_ties":6,"hit_wins":2,"return_losses":3,"return_ties":0,"return_wins":6}`.
- Final hit-rate/mean-daily-return deltas vs v004a: **+0.1111 / +2.1808 pct**.
- Final hit-rate/mean-daily-return deltas vs v002: **+0.0741 / +2.2625 pct**.
- Across the nine dates, v004a/v005 Top15 contains **26** target7 hits and v002 Top15 contains **33**; final Top3 contains **8** hits.
- Conclusion: **v005 shows a descriptive aggregate Top3 advantage over both Top3 baselines, but nine dates do not establish a stable gain.** Hit counts never lose to v004a Top3 but daily mean returns still lose on 4/9 dates; versus v002 Top3, hit counts lose on 1/9 date and daily mean returns lose on 3/9 dates. In addition, v005 has no independent recall gain because its candidate pool is exactly v004a Top15.

The random Top3 line is analytic, not simulated: expected hit rate and return equal the date's scorable-pool means; all-hit and at-least-one probabilities use the exact without-replacement hypergeometric formula. Its `daily_positive_return_rate` is the share of dates whose theoretical expected return is positive, not the probability that a realized random Top3 would be positive.

## Extreme-shape false positives

Group A is the unique union of final Top3 or v004a Top3 rows with `target7=False`. Group B is a v005-pool hit not selected into final Top3.

| feature | group_a_count | group_b_count | group_a_mean | group_b_mean | mean_difference_a_minus_b | group_a_median | group_b_median | daily_a_greater_than_b_rate | daily_direction_consistency_rate | persistent_extreme_signal_bool |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| total_score | 39 | 18 | 58.5385 | 56.3472 | 2.1912 | 58.7500 | 56.5000 | 1.0000 | 1.0000 | False |
| trend_hold_score | 39 | 18 | 90.2564 | 85.8333 | 4.4231 | 95.0000 | 92.5000 | 0.8571 | 1.0000 | False |
| active_money_score | 39 | 18 | 79.6410 | 78.2222 | 1.4188 | 80.0000 | 76.0000 | 0.5714 | 0.5714 | False |
| d1_low_ma10_pct | 39 | 18 | 13.7055 | 11.7710 | 1.9345 | 12.1911 | 11.3025 | 0.7143 | 0.7143 | False |
| d1_close_ma10_pct | 39 | 18 | 18.4032 | 18.9297 | -0.5265 | 15.7042 | 18.0788 | 0.5714 | 0.4286 | False |
| d1_close_vwap_pct | 39 | 18 | -0.2597 | 0.9874 | -1.2471 | -0.4138 | 1.3478 | 0.2857 | 0.7143 | False |
| candidate_base_price | 39 | 18 | 63.0785 | 85.6006 | -22.5221 | 26.0500 | 50.7700 | 0.5714 | 0.4286 | False |
| low_absorb_width_pct | 39 | 18 | 3.1052 | 3.1590 | -0.0538 | 3.3191 | 3.5724 | 0.4286 | 0.5714 | False |
| invalid_distance_pct | 39 | 18 | 3.1052 | 3.1590 | -0.0538 | 3.3191 | 3.5724 | 0.4286 | 0.5714 | False |
| days_since_d0 | 39 | 18 | 1.6923 | 1.7222 | -0.0299 | 1.0000 | 2.0000 | 0.0000 | 1.0000 | False |
| abs_d1_low_ma10_pct | 39 | 18 | 13.7055 | 11.7710 | 1.9345 | 12.1911 | 11.3025 | 0.7143 | 0.7143 | True |
| abs_d1_close_ma10_pct | 39 | 18 | 18.4032 | 18.9297 | -0.5265 | 15.7042 | 18.0788 | 0.5714 | 0.4286 | False |
| abs_d1_close_vwap_pct | 39 | 18 | 1.6607 | 2.4435 | -0.7828 | 1.3706 | 2.2403 | 0.4286 | 0.5714 | False |

Persistence rule for absolute MA10/VWAP measures: overall A−B mean >0, median >0, at least five comparable dates, and A>B on at least two-thirds of those dates.

Conclusion: **Only one of three absolute MA10/VWAP measures meets the persistence rule; there is no broad persistent extreme-shape pattern.**

This is a nine-date descriptive comparison, not a causal estimate. Group membership is selected by the same rankings being evaluated, observations repeat across dates, and no multiple-testing or out-of-sample correction is applied.

## Answers to the research questions

1. **Does v005 have stable gain?** No stable gain is established from 9/30 dates. The frozen slice shows an aggregate advantage over both Top3 baselines, but daily return comparisons remain mixed and v005 does not expand the v004a recall set.
2. **Where do failures occur?** Recall=2, Top15=0, Top3=7, no-failure=0. The detailed rank paths are in the failure CSV.
3. **Are extreme-shape false positives persistent?** Only one of three absolute MA10/VWAP measures meets the persistence rule; there is no broad persistent extreme-shape pattern.
4. **Which layer should be changed first?** Keep v004a as the provisional recall layer and focus any future experiment on v005 Top15-to-Top3 ordering, after more forward dates accrue.
5. **Are nine dates sufficient to change the model?** No. Readiness is explicitly 9/30; coefficients, grid and policy must remain frozen.
6. **What is exploratory only?** All performance deltas, daily win/loss counts, extreme-shape directions, and modification priority are exploratory until at least the frozen 30-date readiness threshold and an independent forward window are available.

## Reproducibility notes

- No history samples, holdout files, model parameters, source code, tests, configuration or cache were changed.
- No random simulation was run.
- The report preserves six-digit codes and uses pandas `float_precision="round_trip"` for every CSV read.
