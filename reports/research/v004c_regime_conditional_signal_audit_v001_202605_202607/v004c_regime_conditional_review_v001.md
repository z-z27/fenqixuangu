# v004c Regime-Conditional Signal Audit v001

## Contract

- Fixed matrix: 8 existing stock signals × 11 existing D1-close local regime variables.
- One preregistered split only: all-60-date median; LOW <= median, HIGH > median.
- Same-date Target7-vs-LOSS information audit only; no model, feature/threshold/group search, interaction construction, score/rank change, or August outcome.
- Mature population: 485 rows / 60 dates; max signal date 2026-07-29.

## Q1. Can local board/break ecology explain why the same signal changes direction across months?

- LOCAL_REGIME_EXPLAINS_SIGNAL_FLIP. Formal explanation relationships: 4; preregistered conditional-flip candidates: 6.

## Q2. Which local regime variable most clearly explains a signal flip?

- board4plus_count for d1_close_to_vwap_raw. Largest absolute HIGH-minus-LOW concordance gap: 0.276.

## Q3. Is the relationship a regime condition or only a May/June/July proxy?

- Formal non-month-proxy explanations: 4. Month-confounded flip candidates: 0; month-proxy relationships by fixed Cramer's-V gate: 0.

## Q4. Can local ecology explain the May vs June/July d1_open_to_close_return_raw flip?

- NO. Formal open/close explanations: 0; regime splits where the same state still materially flips by month: 8.

## Q5. Does the conditional structure remain in the fixed S2 Rank2-6 error region?

- YES. Relationships with the same supported conditional direction in Rank2-6: 13.

## Q6. Is local board/break ecology sufficient for now?

- BROADER_MARKET_CONTEXT_NEEDED = NO_FOR_NOW.

## Reported Bootstrap Relationships

- #1 d1_close_to_vwap_raw × board4plus_count: delta -0.276, 95% CI [-0.457, -0.091]; #2 d1_open_to_close_return_raw × board4plus_count: delta -0.266, 95% CI [-0.450, -0.080]; #3 d1_afternoon_return × first_break_candidate_count: delta 0.231, 95% CI [0.009, 0.445]

REGIME_CONDITIONAL_STATE = LOCAL_REGIME_EXPLAINS_SIGNAL_FLIP

BROADER_MARKET_CONTEXT_NEEDED = NO_FOR_NOW

NEXT_ACTION = STOP_AND_REVIEW

AUGUST_SIGNAL_OUTCOME_ACCESSED = NO
