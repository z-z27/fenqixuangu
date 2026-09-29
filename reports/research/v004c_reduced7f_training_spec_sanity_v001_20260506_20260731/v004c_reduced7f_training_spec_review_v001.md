# v004c Reduced7F Training Spec Sanity v001

## Q1. Current S0 是否存在明显训练内欠拟合？

YES. S0 exactly reproduces the previously frozen Reduced7F coefficients, but its full-development score standard deviation is only 0.00513, its coefficient norm is 0.06108, and weighted logloss improves by only 0.00118. Target7-vs-LOSS AUC is 0.5343 and the score means order TARGET7 > LOSS > PNT, not the desired TARGET7 > PNT > LOSS.

## Q2. Tail bonus 是否与‘>=7%即强修复’业务定义存在不一致？

YES at the specification level. S0 gives 27 observations in the 10%-12% band an extra 1.5 tail multiplier and 64 observations at >=12% a 2.0 multiplier, although all >=7% observations have the same Target7 business qualification. This is an objective-weighting mismatch, not a label change.

## Q3. 关闭 tail bonus 后实际结果是否改善？

YES, modestly. S1 raises full-development within-date AUC from 0.5808 to 0.5993 and Target7-vs-LOSS AUC from 0.5343 to 0.5420. Against S0, fixed June Top3 capped improves 0.126pp with unchanged Top3 LOSS, while fixed July improves 0.531pp and Top3 LOSS falls 4.762pp.

## Q4. L2=.30 是否过强？

YES. Relative to no-tail S1, S2 increases score standard deviation from 0.00430 to 0.01088, coefficient norm from 0.0591 to 0.1574, weighted logloss improvement from 0.00109 to 0.00273, and Target7-vs-LOSS AUC from 0.5420 to 0.5478. L2=.30 is materially constraining the 7F fit.

## Q5. S2/S3 是否恢复 Target7 vs LOSS separation？

PARTIAL BUT SUFFICIENT FOR THE FIXED SANITY QUESTION. S2 and S3 raise full-development Target7-vs-LOSS AUC to 0.5478 and 0.5595. S2 also improves within-date AUC over S0 in both fixed June (+1.97pp) and fixed July (+2.63pp). S3 has the strongest training separation but loses 0.188pp Top3 capped in fixed June, so S3 is not selected.

## Q6. Rank1/Rank2/Rank3 的问题有没有改善？

S2 improves the lower-rank risk structure without a systematic Top3 risk penalty: fixed June Rank2 LOSS falls 5.26pp and Rank3 LOSS is unchanged; fixed July Rank2 LOSS is unchanged and Rank3 LOSS falls 5.26pp. Fixed Top3 capped improves in both periods. June Rank1 return/LOSS is weaker, so the rankwise repair is meaningful but not uniform.

## Q7. May→June 与 May+June→July 是否支持同一个训练规格？

YES for S2. Fixed May-to-June and fixed May+June-to-July both improve within-date Target7 AUC and Top3 capped versus S0, with Top3 LOSS unchanged or lower. S1 is also directionally favorable, but S2 provides the stronger Target7 separation and clear underfit relief.

## Q8. 当前问题主要属于 TRAINING_SPEC / INFORMATION_LIMIT / MIXED？

TRAINING_SPEC, with a residual INFORMATION_LIMIT caveat. The preregistered tail/L2 repair produces consistent chronological gains, so the diagnosed training contract matters. Absolute forward AUC and oracle-gap recovery remain modest, so this does not establish that the seven features contain all information needed. Freeze S2_NO_TAIL_L2_010 only; do not search further.

TRAINING_SPEC_STATE = TRAINING_SPEC_REPAIR_SUPPORTED

NEXT_ACTION = FREEZE_ONE_REPAIRED_7F_SPEC
