# v004c Regime-Aware Stage1 August OOT v001

## Q1. August candidate pool本身难不难？

- Universe Target7 0.317, LOSS 0.267, capped 0.031; complete-date mature 60 rows / 8 dates (row-level labels available: 97).

## Q2. close/VWAP × board4plus 条件关系在August是否继续存在？

- LOW 0.333; HIGH 0.611; delta 0.278; direction consistent = False.

## Q3. Baseline S2的August Rank1/Rank2/Rank3/Top3表现怎样？

- Rank1 capped 0.045, T7 0.500, LOSS 0.250; Rank2 capped 0.032, T7 0.375, LOSS 0.375; Rank3 capped 0.026, T7 0.250, LOSS 0.250; Top3 capped 0.034.

## Q4. Challenger相比Baseline改善了什么？

- Top3 capped delta -0.007; Rank2 delta -0.001; Rank3 delta -0.011; paired bootstrap 95% CI [-0.025, 0.010].

## Q5. Target7增加了吗？

- NO. Delta -0.125.

## Q6. LOSS减少了吗？

- NO. LOSS delta 0.042; severe LOSS delta 0.042.

## Q7. Rank2/Rank3 winner-loss confusion改善了吗？

- NO/MIXED.

## Q8. 改善是不是少数日期造成？

- NOT_APPLICABLE: challenger没有总体改善；paired dates为 3 positive / 4 negative / 1 zero.

## Q9. 单一regime interaction是否通过August forward test？

- AUGUST_SAMPLE_INSUFFICIENT.

REGIME_AWARE_STAGE1_STATE = AUGUST_SAMPLE_INSUFFICIENT

AUGUST_HOLDOUT_STATUS = CONSUMED

NEXT_ACTION = STOP_AND_REVIEW

EVALUATION_ASOF = 2026-09-01
