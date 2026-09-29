# v004c Reduced7F Stage1 Temporal Validation v001

## Q1. Reduced7F 是否严格只用了预注册 7F？

YES. The learner receives exactly the ordered preregistered seven rank features; no total_score, D0 timing, interaction, raw absolute, board, or newly engineered input is present.

## Q2. 所有 temporal fold 是否严格遵守 label_available_date < test signal_date？

YES. Every Reduced7F fixed/expanding fit satisfies label_available_date < test signal_date and historical signal_date < test signal_date. Leakage rows are zero. The Current18F comparator is read from archived strict June OOF and the July frozen lock, with zero Current18F fits.

## Q3. May→June 是否优于/至少不差于 Current18F？

YES, modestly. On the 17 strict June dates, Reduced7F improves date-equal Target7 AUC by 1.73pp and Top3 capped excess by 0.25pp; Top3 LOSS excess improves by 1.96pp. The paired Top3 delta is +0.25pp with P(delta>0) 77.84%. Negative-date rate is unchanged, while the worst day is 0.80pp worse.

## Q4. May+June→July 是否优于/至少不差于 Current18F？

YES, modestly. On 21 mature July dates, the once-frozen May+June Reduced7F fit improves date-equal Target7 AUC by 5.24pp and Top3 capped excess by 0.17pp. LOSS excess, negative-date rate, and worst day are unchanged; paired P(delta>0) is 72.45%.

## Q5. Expanding OOF 是否支持稳定 selection alpha？

YES. The 38-date expanding OOF improves date-equal Target7 AUC by 5.71pp and Top3 capped excess by 0.56pp, with 11 positive, 5 negative, and 22 unchanged dates; the date-bootstrap P(delta>0) is 98.54%. Reduced7F adjacent coefficient cosine has median about 0.997 and six of seven features never flip sign; rank_trend_hold_score flips five times.

## Q6. Reduced7F 是否改善 upside ranking 而没有明显恶化 risk？

YES, with a tail caveat. Upside ranking and Top3 excess improve in both fixed chronological folds and expanding OOF. LOSS excess is better or unchanged in the fixed folds and better overall; negative-date rate is not worse. The June/overall worst Top3 day is 0.80pp worse, so this does not establish uniform tail improvement.

## Q7. 是否有资格冻结并进行一次 August final holdout？

YES. The preregistered Reduced7F challenger has consistent chronological direction across June and mature July without systematic LOSS/negative-date deterioration. Freeze the final 485-row/60-date coefficients for one later August final holdout; do not inspect August in this task.


MODEL_STATE = REDUCED7F_TEMPORAL_SIGNAL_SUPPORTED

NEXT_ACTION = FREEZE_REDUCED7F_FOR_AUGUST_HOLDOUT
