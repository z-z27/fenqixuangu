# v004c S2 Winner vs Loss Failure Attribution v001

本报告是 development failure attribution；S2 分数由已保存系数重建，模型拟合次数为 0。

## Q1. 候选池里是否有足够 Target7 可供排序？

有。485 行、60 个日期的成熟候选池中，date-equal Target7 基准率为 32.85%；漏在 Top3 外的 Target7 与可替换 LOSS 已按日逐项列出。

## Q2. Rank1、Rank2、Rank3分别出了什么问题？

Rank1/2/3 的 Target7 率分别为 45.00%/31.03%/40.00%，LOSS 率分别为 25.00%/31.03%/29.09%；Rank2/3 的 winner 与 LOSS 混排是主要失败结构。

## Q3. Top3中的LOSS有多少其实可以被当天漏掉的Target7替换？

按一只漏选 Target7 最多替换一个 LOSS 的容量约束，25/49 个 LOSS 槽可理论替换（51.02%），发生在 20 个日期。

## Q4. S2 对 Target7 vs LOSS 的同日排序正确率是多少？

共 321 个同日 T7–LOSS pairs，正确率/含 tie 的 concordance 为 55.14%，错误率为 44.86%。

## Q5. 现有7F本身是否包含稳定的 Target7-vs-LOSS 信息？

只有 4/7 个 feature 达到跨月同方向标记；逐月与 pooled 的原始方向、pair 支持均已保留，不能把 pooled 单点当作稳定信息。

## Q6. raw信息比 daily-rank 信息是否明显更强？

没有普遍更强。7 个 pooled raw/rank 对照中，0 个被标为 RAW_INFORMATION_PRESENT_BUT_RANK_TRANSFORM_LOST；其余为 raw 信息不足或 raw/rank 相近。

## Q7. S2为什么会把 LOSS 排得这么高？

固定正系数组合使部分 LOSS 同时获得多项正贡献；pooled 中 LOSS 平均贡献高于 Target7 的 feature 为：rank_d1_close_ma10_pct, rank_d1_low_ma10_pct。错误 pair 的逐 feature 与 contribution gap 已单列，未据此改模。

## Q8. 当前主要瓶颈到底是 MODEL_FORMULATION、INFORMATION_LIMIT 还是 MIXED？

INFORMATION_LIMIT_DOMINANT。证据同时考虑跨月单因子方向、raw/rank 对照、固定 S2 pair concordance、可替换 LOSS 与 7F 几何重叠；本任务不授权后续模型。

WINNER_LOSS_FAILURE_ATTRIBUTION = INFORMATION_LIMIT_DOMINANT

NEXT_ACTION = STOP_AND_REVIEW
