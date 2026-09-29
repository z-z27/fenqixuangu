# v004c Board-Break Regime Audit v001

## Contract

- Descriptive ecology audit only; no model fit, weighting/feature/regime search, or composite regime score.
- Mature candidate outcomes only: 2026-05-06 through 2026-07-29; label_available_date < 2026-08-01.
- Board ecology uses existing main-board final-limit-up caches only; no external data was connected.
- True failed-limit-up (炸板失败池) count/rate: [DATA_NOT_AVAILABLE_NO_FAILED_LIMIT_UP_POOL]. open_board_count is not substituted for this endpoint.
- August signal-date outcome accessed: NO.

## Q1. Are the May, June and July board-break repair environments clearly different?

- LOCAL_REGIME_DIFFERENCE_SUPPORTED. Date-equal Target7 rates May/June/July: 41.60% / 32.95% / 25.26%; LOSS: 18.63% / 26.69% / 35.62%; universe capped: 3.74% / 2.76% / 1.83%.

## Q2. What most clearly distinguishes May from June/July?

- May has the strongest realized repair opportunity (highest Target7, lowest LOSS, highest universe capped return). Board/path ecology is not a simple May-vs-later split: final-limit-up pool mean is 69.44 / 82.48 / 63.33, promotion rate is 19.39% / 17.32% / 20.25%, and June—not May—is the strongest D1 path-state month (afternoon median -0.78% / -0.14% / -0.72%; close/VWAP -1.22% / -0.59% / -1.38%).

## Q3. Where are June and July most similar?

- Limited similarity only. Metrics with June-July absolute distance no larger than May-June: target7_rate_date_equal, candidate_universe_capped_return_date_equal, d1_afternoon_return_cohort_median_date_equal; the four D1 cohort-path measures do not form a broad June/July-identical pattern.

## Q4. How does July deterioration appear?

- Simultaneous descriptive changes: Target7 opportunity lower than May, LOSS higher than May, final limit-up pool breadth lower than June, candidate-cohort D1 afternoon state weakened from June, candidate-cohort D1 close/VWAP state weakened from June. These are concurrent observations, not causal claims.

## Q5. Does the board/break pool alone show a sufficiently clear local regime difference?

- YES. Opportunity shift=True; LOSS shift=True; board ecology shift=True; D1 path-state shift=True.

## Q6. Is broader-market context needed next?

- NO. This task stops at local ecology and does not infer causality.

LOCAL_REGIME_STATE = LOCAL_REGIME_DIFFERENCE_SUPPORTED

BROADER_MARKET_CONTEXT_NEEDED = NO

NEXT_ACTION = STOP_AND_REVIEW

AUGUST_SIGNAL_OUTCOME_ACCESSED = NO
