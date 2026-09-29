# v004c D1 Coarse Retrieval Capacity Audit v001

## 先说人话

S2 当精确 Top3 排名器不稳定；本轮把它当成只负责 KEEP/DROP 的粗筛器。结果是：K4/K5 相对同日随机有明确优势，但召回太低；扩大到 Top7 后能保住更多 winner，却仍未达到预注册的 80% 总召回，而且 51.67% 的交易日本来就不超过 7 只，实际没有发生筛选，Top7 的选择校正证据也不足。没有同一个 K 同时满足高召回、有效压缩和随机优势。因此只能判为 PARTIAL，不能授权它成为 v004d Stage A。

### Q1. S2 只用来留下值得继续看的股票，有没有用？

有一定用，但容量不够强。Top4/Top5 显示了排序增益，却漏掉太多 Target7；Top7 的跨月 winner-presence 较稳定，却压缩太弱、总体召回仍不足，且相对随机的增益不再可靠。

### Q2. Top4 / Top5 / Top6 / Top7 分别保住多少 Target7？

| K | Target7 row recall | Date-macro recall | Winner presence |
|---:|---:|---:|---:|
| 4 | 53.46% | 64.95% | 88.24% |
| 5 | 61.64% | 73.29% | 94.12% |
| 6 | 68.55% | 78.86% | 94.12% |
| 7 | 75.47% | 83.73% | 96.08% |

### Q3. 需要保留原候选池多少比例？

| K | Retained rows | Compression | No-compression dates |
|---:|---:|---:|---:|
| 4 | 45.98% | 54.02% | 25.00% |
| 5 | 55.26% | 44.74% | 30.00% |
| 6 | 63.92% | 36.08% | 41.67% |
| 7 | 71.13% | 28.87% | 51.67% |

### Q4. 这是 S2 的真实排序能力，还是 K 大以后随机也差不多？

K4/K5 的 row recall 明确高于同日随机，K-selection-adjusted evidence 通过；但它们没有达到高召回门槛。K6/K7 召回更接近粗筛目标时，相对随机的增益缩小且校正证据未通过。具体如下。

| K | S2 minus random recall | Selection-adjusted p | Evidence gate |
|---:|---:|---:|:---:|
| 4 | +7.22 pp | 0.0113 | YES |
| 5 | +5.91 pp | 0.0390 | YES |
| 6 | +3.93 pp | 0.1664 | NO |
| 7 | +3.02 pp | 0.2573 | NO |

### Q5. May / June / July 是否稳定？

Temporal state = SUPPORTED。Top7 的月度 row recall 为 MAY 74.14%, JUNE 72.88%, JULY 80.95%；方向不崩，但这依赖很大的 K。

### Q6. 是否存在候选明显减少、同时保留大部分 Target7 的平衡点？

没有 K 完整通过预注册 gate。Top7 最接近高召回，但 pooled recall 仍低于 80%，且 no-compression date rate 高于 35%。不能降低门槛或追加 K=8。

### Q7. v004c 是否有资格转为 v004d Stage A？

当前没有。它可继续作为冻结参考，但本轮没有授权新的 Stage A operating point。

## Gate details

| K | Passed gates | Full pass |
|---:|---:|:---:|
| 4 | 4/7 | NO |
| 5 | 5/7 | NO |
| 6 | 3/7 | NO |
| 7 | 4/7 | NO |

## Integrity

- Frozen population: 485 rows / 60 dates.
- Frozen score parity max abs error: 1.665e-16.
- Frozen rank mismatch count: 0.
- S2 score/rank read from archived artifacts; fit count is zero.
- K values tested: 3 reference plus exactly 4/5/6/7 retrieval candidates.
- Random baseline: 10000 same-date repetitions, seed 20260906.
- August signal dates used: NO.

## Formal verdict

D1_COARSE_RETRIEVAL_STATE = D1_COARSE_RETRIEVAL_PARTIAL

RECOMMENDED_STAGE_A_K = NONE

TARGET7_RECALL_AT_RECOMMENDED_K = N/A

WINNER_PRESENCE_HIT_AT_RECOMMENDED_K = N/A

RETAINED_ROW_RATIO_AT_RECOMMENDED_K = N/A

RANDOM_BASELINE_UPLIFT = N/A

TEMPORAL_RETRIEVAL_STABILITY = SUPPORTED

V004C_NEW_ROLE = FROZEN_REFERENCE_ONLY

MODEL_TRAINED = NO

FEATURE_SEARCH = NO

PARAMETER_SEARCH = NO

NEW_FACTOR = NO

AUGUST_HOLDOUT_STATUS = CONSUMED

AUGUST_USED = NO

NEXT_ACTION = STOP_AND_REVIEW
