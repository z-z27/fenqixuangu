# v004c Blind 5m Seal-path Information Audit v001

## 简单结论

### Q1. Target7 和 LOSS 在断板前封板路径上到底有没有区别？

同日质量方向 concordance 为 F1=0.447, F2=0.482, F3=0.584。正式判断必须结合头部、错配救援和时间稳定性，不能只看 pooled。

### Q2. 这个区别是 May/June/July 都存在，还是只在某个月？

逐月 May/June/July 分别为 F1: 0.411/0.444/0.500; F2: 0.629/0.436/0.350; F3: 0.595/0.538/0.648。任何 >0.55 与 <0.45 的跨月组合均已标记方向翻转。

### Q3. 当 S2 自己排错时，F1/F2/F3 谁还能把 winner 和 LOSS 分开？

S2 wrong-pair rescue 为 F1=0.531, F2=0.647, F3=0.671；正式 rescue 状态=PARTIAL。

### Q4. Rank2/Rank3 是否真正得到额外信息？

固定 Rank2-6 concordance 为 F1=0.338, F2=0.346, F3=0.461；正式增量状态=NO。Rank1 仅作描述，不主导结论。

### Q5. July 与 May/June 差异有多少可能来自 Sina vs BaoStock？

上一轮已预注册 source shift；本轮盲分布与月份方向联合判断为 SOURCE_SHIFT_IMPACT=LOW。没有按 source 重标化或改 proxy。

### Q6. 这是真正新增信息，还是又一个 development 局部关系？

Family 正式状态=PREBREAK_5M_SEAL_PATH_INFORMATION_PARTIAL。三个 proxy 逐项状态：F1=NO_INFORMATION, F2=TEMPORALLY_UNSTABLE, F3=PARTIAL_INFORMATION。

### Q7. 当前 Stage1 D1 因子研究：继续一次 confirmation 还是正式停止？

STOP_AND_REVIEW

## 审计边界

- Population: 485 rows / 60 dates, 2026-05-06 through 2026-07-29.
- Labels: only rows with label_available_date < 2026-08-01.
- Proxies: exact locked F1/F2/F3; no reconstruction change and no F4.
- Continuous concordance uses 0.5 for exact ties and also exports the strict-better rate; F2 concordance excludes ties.
- S2 score comes from archived coefficients; learner fit count is zero.
- No raw/capped return is loaded into the output contract or tested.
- Bootstrap and permutation both use signal_date as the cluster/permutation boundary.

## Final State

F1_INFORMATION_STATE = NO_INFORMATION

F2_INFORMATION_STATE = TEMPORALLY_UNSTABLE

F3_INFORMATION_STATE = PARTIAL_INFORMATION

PREBREAK_5M_SEAL_PATH_INFORMATION_STATE = PREBREAK_5M_SEAL_PATH_INFORMATION_PARTIAL

SOURCE_SHIFT_IMPACT = LOW

S2_WRONG_PAIR_RESCUE_FOUND = PARTIAL

RANK2_6_INCREMENTAL_INFORMATION = NO

BROADER_D1_INFORMATION_LIMITATION = [待核验]

STAGE1_D1_INFORMATION_RESEARCH_STATE = STOP_AND_REVIEW

MODEL_TRAINED = NO

FEATURE_SEARCH = NO

WINDOW_SEARCH = NO

AUGUST_HOLDOUT_STATUS = CONSUMED

AUGUST_USED_IN_PRIMARY_ANALYSIS = NO

NEXT_ACTION = STOP_AND_REVIEW
