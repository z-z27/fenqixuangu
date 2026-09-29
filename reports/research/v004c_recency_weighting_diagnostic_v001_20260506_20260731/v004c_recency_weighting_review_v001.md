# v004c Recency Weighting Diagnostic v001

S2 source contract: 7F, weighted L2 logistic, L2=0.10, positive_weight=1.50, tail bonus OFF, date_weight=1/candidate_count.

Only HISTORICAL_EQUAL (May=1.0, June=1.0) and RECENT_PRIORITY_2X (May=0.5, June=1.0) were fit.

## Q1. 6月加权是否比5/6月等权更适合7月？

NO。状态为 RECENCY_SIGNAL_NOT_SUPPORTED。相对等权，Recent Priority 的 July Top3 capped 变化 -0.6086%，Target7 率变化 -1.64%，LOSS 率变化 +3.28%。

## Q2. 如果改善，改善发生在 Rank1、Rank2、Rank3，还是整体Top3？

没有整体改善；变化来源为：Rank1: capped -0.2028%, Target7 -4.76%, LOSS +9.52%；Rank2: capped +1.0027%, Target7 +0.00%, LOSS -9.52%；Rank3: capped -2.9022%, Target7 +0.00%, LOSS +10.53%。Top3 Board3 数量由 10 变为 10，并非通过减少 Board3 获益。绝对值最大的 coefficient delta 为 rank_active_money_score -0.04580, rank_log_candidate_base_price +0.03639, rank_d1_close_vwap_pct -0.02169；这里只作诊断，不据此改模。

## Q3. 5月权重降低后，LOSS是否减少？

Top3 LOSS 率变化 +3.28%，severe LOSS 率变化 +3.28%；新增 Top3 中 LOSS 3 只，剔除 Top3 中 LOSS 1 只。

## Q4. 是否只是少数几个July日期造成改善？

逐日 Top3 delta：正 0 日、负 5 日、零 16 日；绝对变化最大的 3 日占比 84.88%。最大三日（按绝对值，保留实际方向）为 2026-07-27 (-4.7078%), 2026-07-16 (-3.9337%), 2026-07-06 (-2.2075%)。本次不是少数日期带来的改善；所有实质变化日期均为负向。

## Q5. 是否值得进行下一阶段的正式 forward 研究？

NO。固定 2x recency 假设未获支持，不值得据此进入正式 forward 模型研究。本任务不选择或冻结模型；July 已消费，不能用于追加权重或生产定版。

JULY_MODEL_SELECTION = NO

AUGUST_SIGNAL_OUTCOME_ACCESSED = NO

RECENCY_WEIGHTING_STATE = RECENCY_SIGNAL_NOT_SUPPORTED

NEXT_ACTION = STOP_AND_REVIEW
