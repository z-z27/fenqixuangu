# v004c Close/VWAP × Board4plus Confirmation v001

## Frozen Contract

- Single stock signal: d1_close_to_vwap_raw.
- Single regime: board4plus_count; fixed all-60-date median = 1; LOW <= 1, HIGH > 1.
- No model, feature/threshold search, score/rank change, second relation, or August outcome.

## Q1. Can the prior 0.591 vs 0.315 relationship be reproduced exactly?

- YES. LOW 0.591; HIGH 0.315; delta -0.276; max parity error 5.55e-17.

## Direction in Plain Language

- LOW board4plus: Target7 names tend to close higher versus D1 VWAP than LOSS names.
- HIGH board4plus: the direction reverses; Target7 names tend to close lower versus D1 VWAP than LOSS names.

## Q2. Is the relationship visible in May, June, and July?

- MAY: LOW 0.646; HIGH 0.324. JUNE: LOW 0.554; HIGH 0.408. JULY: LOW 0.617; HIGH 0.154.

## Q3. Does it remain after omitting any one month?

- YES. HIGH-minus-LOW deltas: omit MAY -0.275; omit JUNE -0.353; omit JULY -0.217.

## Q4. Does it remain in the fixed S2 Rank2-6 error region?

- YES. LOW 0.557; HIGH 0.236; delta -0.321.

## Q5. Is the effect driven by only a few dates?

- NO. LODO delta range [-0.314, -0.256], sign flips 0. Largest five date influences: 2026-06-12 (|change|=0.038); 2026-06-29 (|change|=0.030); 2026-06-03 (|change|=0.030); 2026-06-17 (|change|=0.025); 2026-07-29 (|change|=0.020).

## Q6. After correcting for selecting the strongest of 88 relationships, is it still abnormal?

- NO. Selection-adjusted permutation p = 0.3879; null p95 = 0.349; observed |delta| = 0.276.

## Fixed-Relation Bootstrap

- Delta median -0.278; 95% CI [-0.455, -0.092]; P(delta < 0) = 0.998.

SINGLE_RELATION_STATE = SINGLE_RELATION_SUPPORTED_BUT_SELECTION_RISK

AUGUST_SIGNAL_OUTCOME_ACCESSED = NO

NEXT_ACTION = STOP_AND_REVIEW
