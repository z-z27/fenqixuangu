# v004c D1 Unfinished Repair Information Audit v001

## Contract

- Information audit only: four preregistered existing D1-close-safe fields.
- No model fit, parameter/feature/window/threshold search, reranking, or August signal-date outcome access.
- Mature population: 485 rows / 60 dates; max signal date 2026-07-29.
- Existing case-pack semantic parity max absolute error: 9.367506770274758e-17.

## Q1. Does the case-pack unfinished-repair pattern exist in the full candidate universe?

- Full-population T7-low-vs-LOSS concordance: d1_afternoon_return=52.02%; d1_open_to_close_return_raw=59.19%; d1_close_to_vwap_raw=53.89%; d1_low_to_close_recovery=54.05%.
- Formal supported fields: NONE; case-conditional fields: d1_afternoon_return, d1_open_to_close_return_raw, d1_close_to_vwap_raw, d1_low_to_close_recovery.

## Q2. Do any fields carry directionally stable T7-vs-LOSS information across May/June/July?

- Stable-direction fields: d1_close_to_vwap_raw.

## Q3. Do June and July replicate the unfinished-repair direction?

- Fields with T7-low concordance above 0.5 in both June and July: d1_afternoon_return, d1_open_to_close_return_raw, d1_close_to_vwap_raw, d1_low_to_close_recovery.

## Q4. Is the information concentrated near S2 Rank2-6?

- Rank2-6 fields with T7-low concordance >= 0.55: d1_afternoon_return.

## Q5. Is Rank3 failure associated with preferring D1-completed-looking LOSS names?

- Rank3 LOSS vs lower-ranked Target7, Target7<LOSS proportions: d1_afternoon_return=90.62%; d1_open_to_close_return_raw=90.62%; d1_close_to_vwap_raw=84.38%; d1_low_to_close_recovery=87.50%.

## Q6. Does this mechanism authorize one structural-correction experiment?

- NO. This audit itself does not create a score, select a field, or train a challenger.

UNFINISHED_REPAIR_STATE = CASE_CONDITIONAL_ONLY

NEXT_ACTION = STOP_AND_REVIEW

AUGUST_SIGNAL_OUTCOME_ACCESSED = NO
