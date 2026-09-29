# v004c S2 Replaceable-Loss Case Pack v001

## Contract

- Data preparation only; no model fit, parameter search, feature search, or August signal-date outcome access.
- Signal-date boundary: 2026-05-06 through 2026-07-29.
- Pair rules: highest-ranked outside-Top3 Target7 and nearest-score outside-Top3 Target7; duplicate A/B selections retained once.

## Q1. How many replaceable LOSS cases were prepared?

- Unique HARD_FALSE_POSITIVE_LOSS cases: 25.
- Pair rows after fixed-rule A/B de-duplication: 25.
- Archived one-to-one replacement capacity: 25 slots; the case identities match this authoritative attribution exactly.
- May / June / July cases: 6 / 8 / 11.

## Q2. Where are the errors concentrated?

- Rank1 / Rank2 / Rank3 LOSS cases: 5 / 10 / 10.
- Board2 / Board3 LOSS cases: 20 / 5.

## Q3. Are D1 daily, 5-minute, 7F, and 53F data complete?

- D1 5-minute source use by unique event: {'baostock_5m': 26, 'sina_5m': 19}.
- Every case side has 48 D1 bars: YES.
- 53F feature count: 53; missing target/loss value cells: 0.
- Formula-reconstructed 53F cells: 89; every such cell carries explicit provenance and no archived value was overwritten.
- Largest archived-vs-local reconstruction difference on overlap: 1.0 at 600378_2026-07-01 / board_day_volume_rank (18.0 archived vs 19.0 reconstructed); the archived value was retained.
- Frozen S2 7F and their existing raw parents are included in the 7F comparison table.

## Q4. Which existing 53F fields contain intraday/path semantics?

- d1_high_to_close_drawdown_raw, d1_low_to_close_recovery, d1_open_to_close_return_raw, d1_close_location, d1_intraday_range, d1_afternoon_return, d1_last_hour_return, d1_up_bar_volume_ratio, down_bar_volume_ratio, volume_above_d1_close_ratio, amount_above_d1_close_ratio, high_zone_volume_ratio, high_zone_amount_ratio, late_day_sell_volume_ratio, late_day_sell_amount_ratio, d1_close_to_vwap_raw.
- These names and semantics are listed only as coverage metadata; no predictive-effect ranking was performed.

## Q5. Are there missing-data or time-boundary issues?

- Integrity failures: 0.
- 5-minute exports contain D1 only; no D2/D3 bar is included.
- AUGUST_SIGNAL_OUTCOME_ACCESSED = NO.

CASE_PACK_STATE = READY_FOR_MECHANISM_REVIEW

NEXT_ACTION = MANUAL_MECHANISM_REVIEW
