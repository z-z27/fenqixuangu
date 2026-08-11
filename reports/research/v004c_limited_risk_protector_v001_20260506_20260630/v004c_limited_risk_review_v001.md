# v004c Limited Stage1 Risk Protector v001

## 1. Experimental Contract

- Task: LIMITED STAGE1 RISK PROTECTOR
- Development data: May/June only
- July result rows accessed: 0
- Stage1 owns selection; risk model has one-pass veto authority over original Top3 only.
- Backfill: frozen Stage1 Rank4 through Rank10; no recursive veto.

## 2. Strict Temporal Reconstruction

- Strict June dates: 17
- Dates: 2026-06-03|2026-06-04|2026-06-05|2026-06-08|2026-06-09|2026-06-10|2026-06-11|2026-06-12|2026-06-15|2026-06-16|2026-06-17|2026-06-18|2026-06-22|2026-06-23|2026-06-24|2026-06-25|2026-06-26
- Unavailable: 2026-06-01|2026-06-02|2026-06-29|2026-06-30
- SELF_LABEL_LEAKAGE_ROWS: 0
- CURRENT_TEST_LEAKAGE_ROWS: 0
- Stage1 parity: PASS

## 3. Historical Stage1 Top3 Risk Training Set

- Fold-aggregated meta rows: 1243
- Fold-aggregated meta dates: 442
- Loss / non-loss / Target7 rows: 294 / 949 / 495
- Weighted loss prevalence: 22.1719%
- Predictors: stage1_strength, closing_completion_gap, strength_x_gap
- L2: 0.30; class weighting: none; threshold: 0.50 fixed.

## 4. Loss-Probability Model

- OOF original Stage1 Top3 rows: 51
- Actual loss / non-loss / Target7: 16 / 35 / 16
- ROC AUC: 0.4054
- Average precision: 0.3339
- Mean p_loss LOSS / NONLOSS / Target7: 0.2147 / 0.2212 / 0.2257
- Coefficient stability:

| Coefficient | Mean | Median | Min | Max | Positive Folds |
| --- | --- | --- | --- | --- | --- |
| intercept | -1.2498 | -1.2234 | -1.4828 | -1.1075 | 0.0000% |
| stage1_strength | -0.0143 | -0.0176 | -0.0320 | 0.0150 | 17.6471% |
| closing_completion_gap | -0.0254 | -0.0268 | -0.0410 | -0.0072 | 0.0000% |
| strength_x_gap | -0.0107 | -0.0130 | -0.0239 | 0.0025 | 11.7647% |

## 5. Fixed 0.50 Veto Classification

- Veto count: 0
- TP / FP / FN / TN: 0 / 0 / 16 / 35
- Loss veto precision / recall / F1: NA / 0.0000% / NA
- Stage1 base loss prevalence: 31.3725%
- Vetoed-stock loss rate / lift: NA / NAx
- Target7 false veto count / rate: 0 / 0.0000%

## 6. Veto / Backfill Attribution

- Executed vetoes / backfills: 0 / 0
- Backfill Target7 / positive non-target / loss / severe loss: 0 / 0 / 0 / 0
- Net Target7 slots gained: 0
- Correct-loss contribution: 0.0000%
- False-Target7 contribution: 0.0000%
- False-positive-nontarget contribution: 0.0000%
- Attribution closure: PASS

## 7. Practical Top3 Performance

| Policy | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Top3 Precision | Negative Dates | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNIVERSE | 2.6786% | 2.6786% | 2.6786% | 2.6786% | 2.6786% | 31.9160% | NA | NA |
| STRICT_STAGE1 | 2.9829% | 3.6804% | 0.2655% | 3.3316% | 2.3096% | 31.3725% | 23.5294% | -6.7688% |
| STRICT_CAPPED | 2.2733% | 3.4467% | -0.0977% | 2.8600% | 1.8741% | 29.4118% | 17.6471% | -6.8535% |
| LIMITED_RISK_PROTECTOR_P050 | 2.9829% | 3.6804% | 0.2655% | 3.3316% | 2.3096% | 31.3725% | 23.5294% | -6.7688% |
| ORACLE_LOSS_VETO_STAGE1_BACKFILL | 5.4650% | 4.8283% | 1.1841% | 5.1466% | 3.8258% | 39.2157% | 0.0000% | 0.2546% |

- Raw downside:

| Policy | Stock Mean | Stock Median | Worst Stock | Worst Top3 Date |
| --- | --- | --- | --- | --- |
| STRICT_STAGE1 | 3.3236% | 4.5483% | -24.4558% | -6.7688% |
| STRICT_CAPPED | 2.8129% | 3.2295% | -24.4558% | -6.8535% |
| LIMITED_RISK_PROTECTOR_P050 | 3.3236% | 4.5483% | -24.4558% | -6.7688% |
| ORACLE_LOSS_VETO_STAGE1_BACKFILL | 5.1268% | 5.6447% | -12.2694% | 0.2546% |
- LIMITED minus STRICT_STAGE1 Top3: 0.0000%
- LIMITED minus STRICT_CAPPED Top3: 0.4355%
- Oracle headroom recovered: 0.0000% of 1.5162% (0.0000%).

## 8. LOW / MID / HIGH

### LOW

| Policy | Top3 | Precision | Negative Dates | Worst |
| --- | --- | --- | --- | --- |
| UNIVERSE | 0.6195% | 10.6217% | NA | NA |
| STRICT_STAGE1 | -0.1668% | 11.1111% | 50.0000% | -6.7688% |
| STRICT_CAPPED | -1.1737% | 0.0000% | 33.3333% | -6.8535% |
| LIMITED_RISK_PROTECTOR_P050 | -0.1668% | 11.1111% | 50.0000% | -6.7688% |
| ORACLE_LOSS_VETO_STAGE1_BACKFILL | 2.6225% | 16.6667% | 0.0000% | 0.2546% |

### MID

| Policy | Top3 | Precision | Negative Dates | Worst |
| --- | --- | --- | --- | --- |
| UNIVERSE | 2.6364% | 32.1270% | NA | NA |
| STRICT_STAGE1 | 2.8944% | 40.0000% | 20.0000% | -0.2540% |
| STRICT_CAPPED | 2.3103% | 46.6667% | 20.0000% | -0.2798% |
| LIMITED_RISK_PROTECTOR_P050 | 2.8944% | 40.0000% | 20.0000% | -0.2540% |
| ORACLE_LOSS_VETO_STAGE1_BACKFILL | 3.4839% | 46.6667% | 0.0000% | 0.5769% |

### HIGH

| Policy | Top3 | Precision | Negative Dates | Worst |
| --- | --- | --- | --- | --- |
| UNIVERSE | 4.7728% | 53.0345% | NA | NA |
| STRICT_STAGE1 | 4.2986% | 44.4444% | 0.0000% | 3.4284% |
| STRICT_CAPPED | 4.5584% | 44.4444% | 0.0000% | 3.4284% |
| LIMITED_RISK_PROTECTOR_P050 | 4.2986% | 44.4444% | 0.0000% | 3.4284% |
| ORACLE_LOSS_VETO_STAGE1_BACKFILL | 5.3140% | 55.5556% | 0.0000% | 3.5722% |

## 9. Robustness

- Daily delta mean / median: 0.0000% / 0.0000%
- Positive / negative / zero dates: 0 / 0 / 17
- Bootstrap 95%: [0.0000%, 0.0000%]
- Bootstrap P(delta > 0): 0.0000%
- LODO positive: 0.0000%

## 10. Risk-Protector Decision

| Gate | Result |
| --- | --- |
| top3_delta_50bp | FAIL |
| top3_above_universe | FAIL |
| precision_preserved | PASS |
| net_target7_nonnegative | PASS |
| negative_rate_no_worse | PASS |
| worst_no_worse | PASS |
| loss_veto_precision_50 | FAIL |
| loss_veto_recall_25 | FAIL |
| winner_false_veto_25 | PASS |
| low_delta_50bp | FAIL |
| low_negative_no_worse | PASS |
| high_preserved_50bp | PASS |
| positive_dates_gt_negative | FAIL |
| bootstrap_probability_65 | FAIL |
| lodo_positive_70 | FAIL |

- RISK_IDENTIFICATION_SIGNAL: **ABSENT**
- RISK_PROTECTOR_SIGNAL: **ABSENT**
- JULY_CONFIRMATION_CANDIDATE: **NO**
- Existing three-feature representation did not identify Stage1 losses under the frozen natural-probability threshold.
- The learned policy executed no veto, recovered none of the oracle headroom, and produced no practical or downside change.
- No threshold, learner, or feature variant was tested.
