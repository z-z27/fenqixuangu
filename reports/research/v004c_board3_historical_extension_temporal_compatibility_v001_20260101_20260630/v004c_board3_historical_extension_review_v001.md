# v004c Board3 Historical Extension & Temporal Compatibility Audit v001

## 1. Frozen Experimental Contract

- HISTORICAL_BACKCAST_REPLICATION: YES
- PRISTINE_OOT: NO
- Frozen features / learner / L2 / weighting: PASS
- July outcome rows accessed: 0

## 2. Historical Data Coverage

- data status: COMPLETE
- raw candidates / reconstructed: 482 / 482
- missing critical inputs: 0

## 3. Existing May-Jun Parity

- 319 rows / 39 dates / Board2 261 / Board3 58: PASS

## 4. Jan-Jun Authoritative Candidate Population

- matured Board3 rows / dates: 146 / 80

## 5. Monthly Board3 Opportunity Base Rates

| month   |   candidate_dates |   all_candidates |   board2_candidates |   board3_candidates |   matured_board3_rows |   target7 |   pnt |   loss |   target7_rate |   loss_rate |   raw_mean |   raw_median |   capped_mean |   worst_raw |   oof_dates |   oof_rows |   nonloss_auc |   pair_concordance |   bottom_loss_capture |   bottom_winner_removal |   bottom_selectivity_gap | month_support_weak   |
|:--------|------------------:|-----------------:|--------------------:|--------------------:|----------------------:|----------:|------:|-------:|---------------:|------------:|-----------:|-------------:|--------------:|------------:|------------:|-----------:|--------------:|-------------------:|----------------------:|------------------------:|-------------------------:|:---------------------|
| 2026-01 |                20 |              158 |                 122 |                  36 |                    36 |        13 |    12 |     11 |       0.361111 |    0.305556 |  0.0432594 |    0.0431319 |     0.0165938 |  -0.100204  |           0 |          0 |    nan        |         nan        |                nan    |              nan        |               nan        | True                 |
| 2026-02 |                14 |               75 |                  64 |                  11 |                    11 |         5 |     1 |      5 |       0.454545 |    0.454545 |  0.0416965 |    0.0553122 |     0.0166133 |  -0.0772164 |           5 |          6 |      0.5      |         nan        |                  0.5  |              nan        |               nan        | True                 |
| 2026-03 |                21 |              142 |                 118 |                  24 |                    24 |         8 |     6 |     10 |       0.333333 |    0.416667 |  0.0286141 |    0.0360171 |     0.0143416 |  -0.120755  |          13 |         24 |      0.664286 |           0.8      |                  0.75 |                0.333333 |                 0.416667 | False                |
| 2026-04 |                21 |              107 |                  87 |                  20 |                    20 |        12 |     5 |      3 |       0.6      |    0.15     |  0.0908117 |    0.0964253 |     0.0397927 |  -0.0723077 |          14 |         20 |      0.862745 |         nan        |                nan    |                0.333333 |               nan        | False                |
| 2026-05 |                18 |              146 |                 116 |                  30 |                    30 |        14 |    11 |      5 |       0.466667 |    0.166667 |  0.0621246 |    0.0593358 |     0.0377559 |  -0.0594228 |          14 |         30 |      0.48     |           0.333333 |                  0.5  |                0.636364 |                -0.136364 | False                |
| 2026-06 |                21 |              173 |                 145 |                  28 |                    25 |         9 |     8 |      8 |       0.36     |    0.32     |  0.0444378 |    0.0322946 |     0.0279265 |  -0.0444573 |          14 |         25 |      0.889706 |           0.928571 |                  0.75 |                0.75     |                 0        | False                |

## 6. Extended Chronological OOF Coverage

- first eligible date: 2026-02-12
- dates / rows: 60 / 105

## 7. Expanded Board3 Ranking / Risk Selectivity

- NONLOSS AUC: 0.613778
- pair concordance: 0.766667
- bottom-half loss capture / winner removal / gap: 0.666667 / 0.500000 / 0.166667
- formal signal: PARTIAL

## 8. Early Historical Backcast Replication

- dates / rows: 32 / 50
- NONLOSS AUC: 0.547237

## 9. May-Jun Bridge Performance

- dates / rows: 28 / 55
- NONLOSS AUC: 0.679487

## 10. Recent-Only vs Extended-History Shared-Date Comparison

- recent NONLOSS AUC / gap: 0.692308 / 0.250000
- extended NONLOSS AUC / gap: 0.871795 / 0.000000
- HISTORY_EXTENSION_EFFECT: HARMFUL

## 11. Coefficient Compatibility

- median shared-fold cosine: 0.567304

## 12. Bootstrap / LODO Compatibility

- P(extended selectivity gap > recent): 0.000000

## 13. Temporal Drift Diagnostic

- median-vector cosine: 0.711274
- BOARD3_TEMPORAL_COMPATIBILITY: DRIFTED

## 14. Historical Extension Decision

- BOARD3_HISTORICAL_EXTENSION_CONCLUSION: REJECT_OLD_HISTORY_EXTENSION

## 15. Next Board3 Action

- NEXT_BOARD3_ACTION: REJECT_EXTENDED_HISTORY
