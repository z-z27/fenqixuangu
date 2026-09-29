# v004c Broad Market Context Information Audit v001

本审计只使用固定 M1/M2/M3、冻结 S2 分数与三项预注册 stock signals；模型拟合次数为 0。主结论先在 May-July 冻结，August 仅后验辅助描述。

## Q1. 市场整体强时，v004c candidate 是否更容易修复？

WEAK vs STRONG 的 candidate T7 为 29.43% vs 36.27%，LOSS 为 30.72% vs 24.08%，capped 为 2.45% vs 3.00%。完整 continuous+split environment gate=NOT_SUPPORTED；fixed-split partial evidence=YES。

## Q2. 市场强弱是否能解释 S2 什么时候有效、什么时候失效？

Selection alpha WEAK/STRONG 均值为 -0.11%/0.50%；M1 与 alpha Spearman=0.070。预注册 alpha-dependence gate=NOT_SUPPORTED。

## Q3. Rank2 / Rank3 confusion 是否集中在某种市场环境？

- Rank1: WEAK T7/LOSS/capped=33.33%/26.67%/2.75%; STRONG=56.67%/23.33%/4.06%.
- Rank2: WEAK T7/LOSS/capped=34.48%/34.48%/2.86%; STRONG=27.59%/27.59%/2.37%.
- Rank3: WEAK T7/LOSS/capped=25.93%/40.74%/1.27%; STRONG=53.57%/17.86%/4.06%.

## Q4. S2 Target7-vs-LOSS separation 在强/弱市场中是否不同？

WEAK=0.507 (181 pairs)，STRONG=0.641 (140 pairs)。

## Q5. theme / active_money / close_vwap 的方向不稳定是否被市场状态解释？

- rank_theme_score: WEAK=0.586, STRONG=0.538.
- rank_active_money_score: WEAK=0.529, STRONG=0.574.
- rank_d1_close_vwap_pct: WEAK=0.451, STRONG=0.464.

## Q6. 按 market state 条件化后，May / June / July 是否更一致？

0/5 个预注册 temporal metrics 达到固定的 range-reduction 门；formal flag=NO。逐月/分状态数值见 month-conditioned 表。

## Q7. M2/M3 是否支持 M1 的主要结论？

BROAD_MARKET_STRENGTH_RESULT_ROBUST；固定指标方向匹配率=100.00%。未按结果选择变量。

## Q8. August 辅助结果是否同方向？

MOSTLY_DIFFERENT_DIRECTION；1/4 个固定方向检查一致。该结果不改变 May-July 状态与授权判断。

BROAD_MARKET_CONTEXT_INFORMATION_STATE = BROAD_MARKET_CONTEXT_PARTIALLY_SUPPORTED

PRIMARY_CONTEXT_VARIABLE = MARKET_UP_RATIO

AUGUST_ROLE = POST_HOC_AUXILIARY_ONLY

MODEL_TRAINED = NO

NEW_CONTEXT_VARIABLE_ADDED = NO

NEXT_ACTION = STOP_AND_REVIEW
