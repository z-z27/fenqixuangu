# v004c Veto-Only Backfill Audit v001

## 1. Input and Frozen Artifact Parity

- Source: `reports/research/v004c_v4a_top10_residual_reranker_v001_20260506_20260630/v004c_v4a_top10_reranker_oof_v001.csv`
- June rows / dates: **173 / 21**
- Previous artifact parity: **PASS**
- SELF_LABEL_LEAKAGE_ROWS / current-test leakage: **0 / 0**
- July accessed: **NO**
- Candidate-count < 3 dates: 2026-06-02

## 2. Veto Policy Definition

- Veto iff frozen CONTROL member has `stage2_rank_within_top10 > 3`.
- Backfill strictly follows frozen `stage1_rank` from Rank4 through Rank10.
- Stage2 score never controls replacement order.
- No training, tuning, new feature, threshold search, or July access.

## 3. Vetoed Member Quality

- Veto dates / members: **4 / 4**
- Target7 / non-target: **0 / 4**
- Loss / severe loss: **2 / 0**
- Non-target rate: **100.0000%**
- Raw-return mean / median: **1.6852% / 1.8061%**

## 4. Stage1-Order Backfill Quality

- Backfilled total / Target7 / non-target: **4 / 1 / 3**
- Backfill Target7 rate: **25.0000%**
- FULL_RERANKER promoted Target7 rate: **25.0000%**
- Backfill minus FULL promotion: **+0.0000pp**
- Bad→winner / bad→bad / winner→winner / winner→bad: **1 / 3 / 0 / 0**
- Successful / failed / possible / capture: **1 / 3 / 21 / 4.7619%**

## 5. Practical June Performance

| Model | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Top3 Precision | Negative Dates | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNIVERSE | 2.7596% | 2.7596% | 2.6575% | 2.7596% | 2.6575% | 32.9456% | NA | NA |
| CONTROL | 2.8393% | 3.2031% | -0.3025% | 3.0212% | 1.8539% | 28.3333% | 30.0000% | -6.7688% |
| FULL_RERANKER | 1.8495% | 1.2598% | 2.5358% | 1.5547% | 1.7735% | 30.0000% | 20.0000% | -6.7688% |
| VETO_ONLY | 2.4178% | 2.7941% | 0.4489% | 2.6059% | 1.8137% | 30.0000% | 25.0000% | -6.7688% |
| TOP10_ORACLE | 6.9721% | 5.7770% | 5.2819% | 6.3745% | 6.0627% | 63.3333% | 0.0000% | 1.3323% |
| FULL_ORACLE | 6.9721% | 5.7886% | 5.4347% | 6.3803% | 6.1177% | 68.3333% | 0.0000% | 1.3323% |

- VETO_ONLY − CONTROL: **-0.0402pp**
- VETO_ONLY − FULL_RERANKER: **+0.0403pp**
- VETO_ONLY − UNIVERSE: **-0.8438pp**
- TOP10 Oracle Top3: **6.0627%**

## 6. LOW / MID / HIGH Opportunity

| Bucket | Model | Rank1 | Top2 | Top3 | Top3 Excess | Top3 Precision | Negative Dates | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOW | UNIVERSE | 0.7922% | 0.7922% | 0.7922% | 0.0000% | 11.1451% | NA | NA |
| LOW | CONTROL | -0.3685% | 1.7422% | -0.9211% | -1.7133% | 9.5238% | 71.4286% | -6.7688% |
| LOW | FULL_RERANKER | -1.8042% | -1.3637% | -0.7077% | -1.4999% | 9.5238% | 57.1429% | -6.7688% |
| LOW | VETO_ONLY | -0.3685% | 1.0265% | -0.5281% | -1.3204% | 14.2857% | 57.1429% | -6.7688% |
| MID | UNIVERSE | 2.7096% | 2.7096% | 2.7096% | 0.0000% | 35.0907% | NA | NA |
| MID | CONTROL | 3.2762% | 3.6640% | 2.2631% | -0.4464% | 33.3333% | 14.2857% | -0.2540% |
| MID | FULL_RERANKER | 2.0117% | 1.7576% | 1.8199% | -0.8897% | 38.0952% | 0.0000% | 0.3430% |
| MID | VETO_ONLY | 2.0117% | 3.1339% | 1.7554% | -0.9542% | 33.3333% | 14.2857% | -0.0957% |
| HIGH | UNIVERSE | 4.7770% | 4.7770% | 4.7728% | 0.0000% | 52.6010% | NA | NA |
| HIGH | CONTROL | 5.6103% | 3.6574% | 4.6139% | -0.1589% | 44.4444% | 0.0000% | 3.4284% |
| HIGH | FULL_RERANKER | 5.3412% | 4.2702% | 4.6139% | -0.1589% | 44.4444% | 0.0000% | 3.4284% |
| HIGH | VETO_ONLY | 5.6103% | 3.6574% | 4.6139% | -0.1589% | 44.4444% | 0.0000% | 3.4284% |

## 7. Changed-Date Attribution

- Changed / better / worse / equal dates: **4 / 2 / 2 / 0**
- Changed-date delta mean / median / best / worst: **-0.2010pp / -0.0526pp / +3.0138pp / -3.7125pp**

| Date | CONTROL Top3 | Vetoed | Veto Target7 | Backfilled | Backfill Target7 | FULL promoted | CONTROL | FULL | VETO_ONLY | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-18 | 603186<br>002741<br>600110 | 002741 | 0 | 603002 | 1 | 603115 | -0.6462% | 1.1107% | 2.3675% | +3.0138pp |
| 2026-06-24 | 603757<br>600851<br>000070 | 603757 | 0 | 600172 | 0 | 000783 | -0.2540% | 0.3560% | -0.0957% | +0.1583pp |
| 2026-06-25 | 605069<br>600909<br>600379 | 600379 | 0 | 002580 | 0 | 002580 | 4.2894% | 0.5769% | 0.5769% | -3.7125pp |
| 2026-06-30 | 601133<br>603650<br>603163 | 603650 | 0 | 603956 | 0 | 603956 | -1.4689% | -1.7323% | -1.7323% | -0.2634pp |

## 8. Downside Risk Comparison

- All-date mean / median delta: **-0.0383pp / +0.0000pp**
- Positive / negative / zero dates: **2 / 2 / 17**
- CONTROL negative rate / p25 / median / worst: **30.0000% / -0.2577% / 2.5742% / -6.7688%**
- VETO_ONLY negative rate / p25 / median / worst: **25.0000% / 0.2333% / 2.2785% / -6.7688%**

## 9. Formal Decision

- Q1: **YES**
- Q2: **YES**
- Q3: **YES**
- Q4: **NO**
- Q5: **NO**
- Q6: **NO**
- Q7: **YES**
- Q8: **NO**
- Q9: **NO**
- Q10: **NO**
- Q11: **NO**
- Q12: **YES**
- Q13: **NO**
- Q14: **YES**
- Q15: **YES**
- Q16: **YES**

VETO_ONLY_SIGNAL: **ABSENT**

FORWARD_STRESS_CANDIDATE: **NO**

- VETO_ONLY Top3 > CONTROL: **NO**
- VETO_ONLY Top3 > Universe: **NO**
- Top3 precision improved: **YES**
- LOW downside materially improved: **NO**
- HIGH upside preserved: **YES**
- Closing-completion / veto route: **STOP**
