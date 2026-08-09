# v004c Top10 Target-Information Benchmark v001

## 1. Experimental Contract

- Learner: same-date pairwise weighted Ridge, no intercept on pair differences.
- Predictors: stage1_strength, closing_completion_gap, strength_x_gap.
- L2: 0.30. Pair date weight: 1 / pair_count_on_date.
- Pair orientation: event_id ascending; pair_x = X_left - X_right.
- Same pair keys, X, weights, learner and chronology; only pair_y changes.
- BINARY7 utility: 1[raw_return >= 0.07].
- CAPPED7 utility: min(raw_return, 0.07), with no lower floor.
- RAW_RETURN utility: raw_return, with no clipping or winsorization.
- Hyperparameter search: NO. Feature selection: NO. July: NOT ACCESSED.

## 2. Frozen Stage1 Parity

- Identity exact: True.
- Score exact at frozen precision: True.
- Rank exact: True.
- Parity gate: PASS.

## 3. Leakage Audit

- SELF_LABEL_LEAKAGE_ROWS: 0.
- CURRENT_TEST_DATE_LEAKAGE: 0.
- Historical meta method: DATE-CROSSFITTED STAGE1 TOP10.
- Leakage gate: PASS.

## 4. Pair and Target Information Audit

| Metric | Count | Rate |
| --- | --- | --- |
| Total training pairs | 14752 | 100.0000% |
| Binary cross-threshold | 6873 | 46.5903% |
| Binary same-class | 7879 | 53.4097% |
| Within non-target | 5633 | NA |
| Within Target7 | 2246 | NA |
| Binary ties separated by CAPPED | 5633 | 71.4938% |
| Binary ties separated by RAW | 7879 | 100.0000% |

## 5. Binary vs Capped vs Raw Practical Performance

| Model | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Excess | Precision | Beat | Negative | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNIVERSE | 2.7596% | 2.7596% | 2.6575% | 2.7596% | 2.6575% | 0.0000% | 32.9456% | NA | NA | NA |
| CONTROL_V4A | 2.8393% | 3.2031% | -0.3025% | 3.0212% | 1.8539% | -0.8036% | 28.3333% | 40.0000% | 30.0000% | -6.7688% |
| PAIR_BINARY7 | 1.1385% | 2.7872% | 2.2460% | 1.9629% | 1.9626% | -0.6949% | 31.6667% | 35.0000% | 20.0000% | -6.7688% |
| PAIR_CAPPED7 | 3.5170% | 2.7166% | 3.0677% | 3.1168% | 3.0443% | 0.3868% | 33.3333% | 55.0000% | 15.0000% | -2.4797% |
| PAIR_RAW | 2.2906% | 1.4500% | 2.1995% | 1.8703% | 1.8823% | -0.7752% | 31.6667% | 35.0000% | 15.0000% | -6.8535% |
| TOP10_ORACLE | 6.9721% | 5.7770% | 5.2819% | 6.3745% | 6.0627% | 3.4052% | 63.3333% | 90.0000% | 0.0000% | 1.3323% |

## 6. Ordering Diagnostics

| Model | All | Cross | Within non-target | Within Target7 |
| --- | --- | --- | --- | --- |
| PAIR_BINARY7 | 0.5224 | 0.5499 | 0.5787 | 0.6663 |
| PAIR_CAPPED7 | 0.5217 | 0.5058 | 0.5509 | 0.5127 |
| PAIR_RAW | 0.5204 | 0.5253 | 0.5730 | 0.6940 |

## 7. LOW / MID / HIGH Opportunity

| Bucket | Model | Rank1 | Top2 | Top3 | Excess | Precision | Negative | Median | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOW | CONTROL_V4A | -0.3685% | 1.7422% | -0.9211% | -1.7133% | 9.5238% | 71.4286% | -0.6462% | -6.7688% |
| LOW | PAIR_BINARY7 | -2.7019% | -1.4368% | -0.9950% | -1.7872% | 4.7619% | 57.1429% | -0.2688% | -6.7688% |
| LOW | PAIR_CAPPED7 | 2.0890% | 1.4782% | 0.5891% | -0.2032% | 0.0000% | 42.8571% | 1.3323% | -2.4797% |
| LOW | PAIR_RAW | -0.2026% | -1.7667% | -1.4177% | -2.2099% | 0.0000% | 42.8571% | 0.7630% | -6.8535% |
| LOW | UNIVERSE | 0.7922% | 0.7922% | 0.7922% | 0.0000% | 11.1451% | NA | NA | NA |
| MID | CONTROL_V4A | 3.2762% | 3.6640% | 2.2631% | -0.4464% | 33.3333% | 14.2857% | 2.9590% | -0.2540% |
| MID | PAIR_BINARY7 | 0.6721% | 2.4183% | 2.2225% | -0.4871% | 42.8571% | 0.0000% | 0.9343% | 0.3430% |
| MID | PAIR_CAPPED7 | 2.9685% | 3.3263% | 3.5356% | 0.8260% | 47.6190% | 0.0000% | 4.2491% | 0.3430% |
| MID | PAIR_RAW | 1.6289% | 2.4183% | 2.2225% | -0.4871% | 42.8571% | 0.0000% | 0.9343% | 0.3430% |
| MID | UNIVERSE | 2.7096% | 2.7096% | 2.7096% | 0.0000% | 35.0907% | NA | NA | NA |
| HIGH | CONTROL_V4A | 5.6103% | 3.6574% | 4.6139% | -0.1589% | 44.4444% | 0.0000% | 4.8157% | 3.4284% |
| HIGH | PAIR_BINARY7 | 5.4455% | 4.9071% | 5.1099% | 0.3371% | 50.0000% | 0.0000% | 4.9792% | 3.4284% |
| HIGH | PAIR_CAPPED7 | 5.4936% | 4.5459% | 5.3354% | 0.5626% | 55.5556% | 0.0000% | 5.6253% | 3.4284% |
| HIGH | PAIR_RAW | 5.4455% | 4.9593% | 5.3354% | 0.5626% | 55.5556% | 0.0000% | 5.6253% | 3.4284% |
| HIGH | UNIVERSE | 4.7770% | 4.7770% | 4.7728% | 0.0000% | 52.6010% | NA | NA | NA |

## 8. Daily Robustness

| Comparison | Mean | Median | + | - | 0 | Bootstrap 95% | Bootstrap P50 | P(>0) | LODO + | LODO Min | LODO Median | LODO Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAPPED_MINUS_BINARY | 1.0302% | 0.0000% | 6 | 3 | 12 | [-0.0270%, 2.2183%] | 1.0077% | 97.2350% | 100.0000% | 0.6741% | 1.0817% | 1.2674% |
| RAW_MINUS_BINARY | -0.0765% | 0.0000% | 2 | 1 | 18 | [-0.8762%, 0.5497%] | -0.0447% | 43.0450% | 4.7619% | -0.2616% | -0.0803% | 0.2489% |
| RAW_MINUS_CAPPED | -1.1066% | 0.0000% | 3 | 6 | 12 | [-2.4155%, 0.0842%] | -1.0843% | 3.5200% | NA | NA | NA | NA |

| Membership Comparison | Changed Dates | Changed Slots |
| --- | --- | --- |
| BINARY_VS_CAPPED | 12 | 17 |
| BINARY_VS_RAW | 3 | 4 |
| CAPPED_VS_RAW | 13 | 16 |

## 9. Target-Information Decision

| Comparison | Top3 Delta | Precision Delta | Negative-Rate Delta | Gate/Status |
| --- | --- | --- | --- | --- |
| PAIR_CAPPED7 - PAIR_BINARY7 | 1.0817% | 1.6667% | -5.0000% | YES |
| PAIR_RAW - PAIR_BINARY7 | -0.0803% | 0.0000% | -5.0000% | NO |
| PAIR_RAW - PAIR_CAPPED7 | -1.1620% | -1.6667% | 0.0000% | HARMFUL |

- Q1 Does CAPPED materially beat BINARY: **YES**.
- Q2 Does RAW materially beat BINARY: **NO**.
- Q3 Does preserving sub-7% severity improve Top10→Top3: **YES**.
- Q4 Does preserving information above 7% add value: **NO**.
- Q5 Did detailed targets improve within-nontarget ordering: **NO**.
- Q6 Did detailed targets improve within-Target7 ordering: **YES**.
- Q7 Did finer ordering improve practical Top3: **YES**.
- TARGET_INFORMATION_SIGNAL: **SUPPORTED**.
- ABOVE7_INFORMATION_SIGNAL: **HARMFUL**.

## 10. Trading-Candidate Decision

- Selected detailed variant by frozen rule: **PAIR_CAPPED7**.
- TARGET_OBJECTIVE_TRADING_CANDIDATE: **YES**.
- Trading gates: {'top3_control': True, 'top3_universe': True, 'precision': True, 'risk': True, 'low': True, 'high': True, 'cross_date': True}.
- July: NOT ACCESSED. Production ready: NO.
- Another target-objective variant immediately: NO.
