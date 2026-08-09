# Did the Two-Stage Reranker Produce a Better Tradable Top3?

| Model | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Top3 Excess vs Universe | Top3 Precision | Beat-Universe | Negative Days | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DAILY_UNIVERSE | 2.7596% | 2.7596% | 2.6575% | 2.7596% | 2.6575% | 0.0000% | 32.9456% | NA | NA | NA |
| CONTROL | 2.8393% | 3.2031% | -0.3025% | 3.0212% | 1.8539% | -0.8036% | 28.3333% | 40.0000% | 30.0000% | -6.7688% |
| RERANKER | 1.8495% | 1.2598% | 2.5358% | 1.5547% | 1.7735% | -0.8840% | 30.0000% | 35.0000% | 20.0000% | -6.7688% |
| TOP10_ORACLE | 6.9721% | 5.7770% | 5.2819% | 6.3745% | 6.0627% | 3.4052% | 63.3333% | 90.0000% | 0.0000% | 1.3323% |

# Did It Actually Replace the Wrong Stocks?

- Changed / unchanged dates: **4 / 17**
- Demoted false positives / winners: **4 / 0**
- Promoted winners / false positives: **1 / 3**
- Promotion Target7 rate: **25.0000%**
- False-positive veto rate: **100.0000%**
- Net Target7 slots gained: **1**
- Successful replacement slots / possible: **1 / 21**
- Replacement capture: **4.7619%**

# Did Downside Improve Without Killing Upside?

| Bucket | Model | Rank1 | Top2 | Top3 | Top3 Excess | Negative Rate | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LOW | CONTROL | -0.3685% | 1.7422% | -0.9211% | -1.7133% | 71.4286% | -6.7688% |
| LOW | RERANKER | -1.8042% | -1.3637% | -0.7077% | -1.4999% | 57.1429% | -6.7688% |
| MID | CONTROL | 3.2762% | 3.6640% | 2.2631% | -0.4464% | 14.2857% | -0.2540% |
| MID | RERANKER | 2.0117% | 1.7576% | 1.8199% | -0.8897% | 0.0000% | 0.3430% |
| HIGH | CONTROL | 5.6103% | 3.6574% | 4.6139% | -0.1589% | 0.0000% | 3.4284% |
| HIGH | RERANKER | 5.3412% | 4.2702% | 4.6139% | -0.1589% | 0.0000% | 3.4284% |

# Was the Result Broad or Driven by One Date?

- Daily delta mean / median: **-0.0766% / 0.0000%**
- Positive / negative / zero dates: **2 / 2 / 17**
- Bootstrap 95%: **[-0.5264%, 0.2510%]**
- P(delta > 0): **38.1100%**
- Leave-one-date-out positive: **4.7619%**
- Best 3 improvement dates: 2026-06-18 (1.7569%), 2026-06-24 (0.6101%), 2026-06-01 (0.0000%)
- Worst 3 deterioration dates: 2026-06-25 (-3.7125%), 2026-06-30 (-0.2634%), 2026-06-01 (0.0000%)

# Was Stage2 Truly Leakage-Free?

- SELF_LABEL_LEAKAGE_ROWS = **0**
- Current test date leakage rows = **0**
- July accessed = **NO**
- Stage1 control parity = **PASS**
- Historical Stage1 meta predictions use H−{s}; Stage2 labels are applied only after each date's self-excluded Stage1 Top10 is formed.

## Stage2 Coefficient Audit

| Feature | Median Beta | Positive Fold % | Negative Fold % | IQR |
| --- | --- | --- | --- | --- |
| stage1_strength | 0.076570 | 100.0000% | 0.0000% | 0.036010 |
| closing_completion_gap | -0.000607 | 47.6190% | 52.3810% | 0.016141 |
| strength_x_gap | 0.038115 | 100.0000% | 0.0000% | 0.010106 |

## Oracle Gap

- CONTROL Top3: **1.8539%**
- RERANKER Top3: **1.7735%**
- Top10 Oracle Top3: **6.0627%**
- Full Oracle Top3: **6.1177%**
- Reranker recovery ratio: **-1.9114%**

## Formal Decision

- Q1: **YES**
- Q2: **YES**
- Q3: **NO**
- Q4: **NO**
- Q5: **NO**
- Q6: **NO**
- Q7: **YES**
- Q8: **YES**
- Q9: **NO**
- Q10: **NO**

RERANKER_SIGNAL: **ABSENT**

FORWARD_STRESS_CANDIDATE: **NO**

- Production ready: **NO**
- July accessed: **NO**
- Feature selection: **NO**
- Hyperparameter search: **NO**

This tiny residual reranker did not convert v4a Top10 retrieval strength into a reliable tradable Top3 edge.
