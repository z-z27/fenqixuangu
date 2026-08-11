# v004c Stage1 Top3 Risk Information Foundation Audit v001

## 1. Experimental Contract

- Scope: strict June V4C_STAGE1 original Top3, information audit only.
- New model/features/thresholds/transformations: NO / NO / NO / NO.
- July result rows accessed: 0.
- Orientation is post-hoc June development diagnostics only; it is not a trading rule.

## 2. Strict Stage1 Top3 Population

- Dates: 17; rows: 51; unique event_id: 51.
- LOSS: 16; NONLOSS: 35; TARGET7: 16; POSITIVE_NON_TARGET: 19.
- Top3: 2.3096%; precision: 31.3725%; negative-date rate: 23.5294%; worst: -6.7688%.
- Population parity: PASS.

## 3. Existing 53-Feature Manifest

- Exact low-level count: 53; all D1-safe: YES.
- Coverage failures (<95%): 0.
| Group | Count | Coverage pass |
|---|---|---|
| BOARD_HISTORY | 6 | 6 |
| D0_CROSS_SECTION | 1 | 1 |
| D1_PRICE_ACTION | 14 | 14 |
| D1_VOLUME_ACTIVITY | 3 | 3 |
| D1_CHIP_DISTRIBUTION | 4 | 4 |
| D1_LATE_DAY_PRESSURE | 2 | 2 |
| D1_VWAP_POSITION | 1 | 1 |
| D1_MA_POSITION | 15 | 15 |
| POOL_MEMBERSHIP | 3 | 3 |
| RECENT_7D_PATH | 4 | 4 |

## 4. Failed 3-Feature Control Representation

| Feature | Raw AUC | Oriented AUC | LOSS/T7 AUC | Cliff | Date % | Boot P | LODO % |
|---|---|---|---|---|---|---|---|
| closing_completion_gap | 0.5179 | 0.5179 | 0.4902 | 0.0357 | 50.0000% | 0.5647 | 82.3529% |
| stage1_strength | 0.4768 | 0.5232 | 0.4863 | -0.0464 | 50.0000% | 0.6219 | 76.4706% |
| strength_x_gap | 0.5732 | 0.5732 | 0.5605 | 0.1464 | 58.3333% | 0.7539 | 100.0000% |

## 5. LOSS vs NONLOSS Univariate Evidence

No FOUNDATION_PASS or FOUNDATION_EXCEPTIONAL feature.

Strongest non-pass signals (ranked descriptively by oriented LOSS/NONLOSS AUC):

| Feature | Group | AUC | LOSS/T7 | Cliff | Date % | Boot P | LODO % | p | q | Failed |
|---|---|---|---|---|---|---|---|---|---|---|
| max_board_streak_20d | BOARD_HISTORY | 0.7125 | 0.7031 | 0.4250 | 58.3333% | 0.9986 | 100.0000% | 0.0317 | 1.0000 | DATE_CONSISTENCY|BH_FDR |
| recent_7d_limit_up_count | RECENT_7D_PATH | 0.6955 | 0.7070 | 0.3911 | 58.3333% | 0.9964 | 100.0000% | 0.0741 | 1.0000 | DATE_CONSISTENCY|BH_FDR |
| board_streak_is_3 | BOARD_HISTORY | 0.6500 | 0.6250 | 0.3000 | 50.0000% | 0.9825 | 100.0000% | 0.1479 | 1.0000 | DATE_CONSISTENCY|BH_FDR |
| pool_consecutive_count_last_board | POOL_MEMBERSHIP | 0.6500 | 0.6250 | 0.3000 | 50.0000% | 0.9825 | 100.0000% | 0.1479 | 1.0000 | DATE_CONSISTENCY|BH_FDR |
| break_volume_ratio_vs_board_days | D1_VOLUME_ACTIVITY | 0.6411 | 0.6914 | 0.2821 | 58.3333% | 0.9456 | 100.0000% | 0.0802 | 1.0000 | CLIFF|DATE_CONSISTENCY|BH_FDR |
| break_upper_shadow_ratio | D1_PRICE_ACTION | 0.6268 | 0.5859 | 0.2536 | 50.0000% | 0.9389 | 100.0000% | 0.2276 | 1.0000 | CLIFF|WINNER_SAFETY|DATE_CONSISTENCY|BH_FDR |
| recent_limit_up_count_10d | BOARD_HISTORY | 0.6259 | 0.6543 | 0.2518 | 58.3333% | 0.9265 | 100.0000% | 0.3280 | 1.0000 | CLIFF|DATE_CONSISTENCY|BH_FDR |
| recent_pool_appearance_count_10d | BOARD_HISTORY | 0.6259 | 0.6543 | 0.2518 | 58.3333% | 0.9265 | 100.0000% | 0.3280 | 1.0000 | CLIFF|DATE_CONSISTENCY|BH_FDR |
| d1_up_bar_volume_ratio | D1_VOLUME_ACTIVITY | 0.6214 | 0.6367 | 0.2429 | 66.6667% | 0.9520 | 100.0000% | 0.0728 | 1.0000 | CLIFF|BH_FDR |
| d1_high_to_close_drawdown_raw | D1_PRICE_ACTION | 0.6045 | 0.6172 | 0.2089 | 50.0000% | 0.9212 | 100.0000% | 0.3154 | 1.0000 | CLIFF|DATE_CONSISTENCY|BH_FDR |

## 6. LOSS vs Target7 Winner-Safety Evidence

Features with oriented LOSS/TARGET7 AUC >=0.60: 14.
The same full-sample LOSS/NONLOSS risk sign is used; no Target7-specific reorientation is permitted.

## 7. Date Stability and LODO

Features passing mixed-date support and >=65% direction consistency: 8.
Features with LODO Cliff-positive >=80%: 36.

## 8. Date-Block Bootstrap

Date resamples: 20000; seed: 20260811; unit: signal_date block.
Features with P(oriented Cliff>0) >=0.90: 11.

## 9. Stratified Permutation / FDR

Permutations: 10000; seed: 20260812; labels shuffled within signal_date.
Features with BH q<=0.10: 0.

## 10. Feature-Group Summary

| Group | Features | Pass | Exceptional | Best AUC | Best LOSS/T7 | Best abs Cliff | Best q |
|---|---|---|---|---|---|---|---|
| BOARD_HISTORY | 6 | 0 | 0 | 0.7125 | 0.7031 | 0.4250 | 1.0000 |
| D0_CROSS_SECTION | 1 | 0 | 0 | 0.5768 | 0.5820 | 0.1536 | 1.0000 |
| D1_CHIP_DISTRIBUTION | 4 | 0 | 0 | 0.5661 | 0.5703 | 0.1321 | 1.0000 |
| D1_LATE_DAY_PRESSURE | 2 | 0 | 0 | 0.5054 | 0.5352 | 0.0107 | 1.0000 |
| D1_MA_POSITION | 15 | 0 | 0 | 0.5804 | 0.5781 | 0.1607 | 1.0000 |
| D1_PRICE_ACTION | 14 | 0 | 0 | 0.6268 | 0.6172 | 0.2536 | 1.0000 |
| D1_VOLUME_ACTIVITY | 3 | 0 | 0 | 0.6411 | 0.6914 | 0.2821 | 1.0000 |
| D1_VWAP_POSITION | 1 | 0 | 0 | 0.5607 | 0.5977 | 0.1214 | 1.0000 |
| POOL_MEMBERSHIP | 3 | 0 | 0 | 0.6500 | 0.6250 | 0.3000 | 1.0000 |
| RECENT_7D_PATH | 4 | 0 | 0 | 0.6955 | 0.7070 | 0.3911 | 1.0000 |

## 11. Redundancy

Passing features: NONE.
No passing-feature pair available for redundancy assessment.

## 12. Rank / Board / Candidate-Count Context

| Context | Rows | LOSS rate | Target7 rate | Mean raw |
|---|---|---|---|---|
| STAGE1_RANK:Rank1 | 17 | 29.4118% | 17.6471% | 3.1832% |
| STAGE1_RANK:Rank2 | 17 | 23.5294% | 41.1765% | 5.2178% |
| STAGE1_RANK:Rank3 | 17 | 41.1765% | 35.2941% | 1.5696% |
| BOARD:board2 | 36 | 22.2222% | 33.3333% | 3.8394% |
| BOARD:board3 | 15 | 53.3333% | 26.6667% | 2.0856% |
| CANDIDATE_COUNT_BUCKET:5-9 | 27 | 29.6296% | 33.3333% | 2.7348% |
| CANDIDATE_COUNT_BUCKET:<5 | 6 | 33.3333% | 16.6667% | 3.6107% |
| CANDIDATE_COUNT_BUCKET:>=10 | 18 | 33.3333% | 33.3333% | 4.1110% |

All LOSS rows (the <=-5% flag is descriptive only):

| event_id | date | code | rank | raw | <=-5% |
|---|---|---|---|---|---|
| 600280_2026-06-03 | 2026-06-03 | 600280 | 1 | -1.2531% | False |
| 001210_2026-06-03 | 2026-06-03 | 001210 | 3 | -24.4558% | True |
| 600121_2026-06-04 | 2026-06-04 | 600121 | 1 | -0.7092% | False |
| 605162_2026-06-05 | 2026-06-05 | 605162 | 3 | -4.4992% | False |
| 000520_2026-06-08 | 2026-06-08 | 000520 | 1 | -1.0941% | False |
| 600293_2026-06-08 | 2026-06-08 | 600293 | 2 | -1.7284% | False |
| 002421_2026-06-09 | 2026-06-09 | 002421 | 3 | -14.2105% | True |
| 000048_2026-06-12 | 2026-06-12 | 000048 | 2 | -2.1505% | False |
| 000510_2026-06-16 | 2026-06-16 | 000510 | 3 | -1.4531% | False |
| 605198_2026-06-17 | 2026-06-17 | 605198 | 1 | -2.1277% | False |
| 002741_2026-06-18 | 2026-06-18 | 002741 | 2 | -2.0413% | False |
| 600110_2026-06-18 | 2026-06-18 | 600110 | 3 | -4.4457% | False |
| 600851_2026-06-24 | 2026-06-24 | 600851 | 2 | -3.6816% | False |
| 000070_2026-06-24 | 2026-06-24 | 000070 | 3 | -2.2503% | False |
| 600379_2026-06-25 | 2026-06-25 | 600379 | 3 | -1.1318% | False |
| 603989_2026-06-26 | 2026-06-26 | 603989 | 1 | -4.1964% | False |

Target7 winner-safety rows:

| event_id | date | code | rank | raw |
|---|---|---|---|---|
| 600255_2026-06-05 | 2026-06-05 | 600255 | 2 | 8.9655% |
| 603500_2026-06-09 | 2026-06-09 | 603500 | 2 | 9.3627% |
| 603929_2026-06-10 | 2026-06-10 | 603929 | 1 | 8.5054% |
| 603186_2026-06-10 | 2026-06-10 | 603186 | 3 | 12.8968% |
| 002636_2026-06-11 | 2026-06-11 | 002636 | 1 | 8.9010% |
| 605589_2026-06-11 | 2026-06-11 | 605589 | 3 | 8.7288% |
| 002409_2026-06-12 | 2026-06-12 | 002409 | 3 | 7.7108% |
| 603014_2026-06-15 | 2026-06-15 | 603014 | 3 | 11.1852% |
| 001359_2026-06-16 | 2026-06-16 | 001359 | 2 | 12.0841% |
| 603256_2026-06-17 | 2026-06-17 | 603256 | 2 | 12.0025% |
| 002515_2026-06-17 | 2026-06-17 | 002515 | 3 | 16.5372% |
| 002491_2026-06-22 | 2026-06-22 | 002491 | 3 | 7.1119% |
| 002297_2026-06-23 | 2026-06-23 | 002297 | 2 | 13.2492% |
| 600909_2026-06-25 | 2026-06-25 | 600909 | 1 | 7.0000% |
| 605069_2026-06-25 | 2026-06-25 | 605069 | 2 | 10.8333% |
| 002990_2026-06-26 | 2026-06-26 | 002990 | 2 | 8.6384% |

## 13. Risk Information Foundation Decision

- Q1 material LOSS/NONLOSS separation: **YES**.
- Q2 winner-safe separation: **YES**.
- Q3 stable across dates: **PARTIAL**.
- Q4 survives permutation/FDR: **NO**.
- Q5 at least two non-redundant sources: **NO**.
- Q6 prior 3-feature representation discarded established low-level information: **INCONCLUSIVE**.
- RISK_INFORMATION_FOUNDATION: **NOT_ESTABLISHED**.
- LIMITED_RISK_REPRESENTATION_EXPERIMENT_WARRANTED: **NO**.
- No risk model, veto threshold, feature transform, or trading counterfactual was tested.
- Recommended next action: stop the existing D1-only risk-protector route; oracle headroom remains theoretical, but this audited feature set did not establish a stable information foundation.
