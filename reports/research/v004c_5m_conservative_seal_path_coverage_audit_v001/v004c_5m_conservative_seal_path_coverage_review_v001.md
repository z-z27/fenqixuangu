# v004c 5m Conservative Seal-Path Coverage Audit v001

## 简单结论

Q1. May / June / July 的5分钟K够不够？  F1月度覆盖为 May 94.52%、June 97.11%、July 84.94%；最终状态见下方。

Q2. 涨停价能否可靠确定？  PASS。使用未复权前收盘价 × 1.10、人民币分位 ROUND_HALF_UP，并与日线及5分钟最高/收盘价交叉核对。

Q3. 能否稳定识别最终稳定锁板时间？  可以，达到预注册门槛。

Q4. 能否保守识别明确破板？  可以，达到预注册门槛。 该字段只是5分钟可观察下界，不是真实炸板次数。

Q5. 能否比较D0和前一板谁更晚稳定锁板？  可以，达到预注册门槛。

Q6. 与可靠历史涨停池交叉验证是否合理？  共有 303 个F1可比板日；匹配‘真实final seal之后第一根完整5分钟bar’的比例 84.49%，10分钟内比例 94.39%。F2=1 对原始 open_board_count>0 的精确率 97.56%。

F1通常比逐笔 final seal 晚5–10分钟是定义使然：包含真实封板时刻的那根5分钟K仍可能含有封板前的低价成交；F1要求下一根完整K从头到尾都锁在涨停价。

Q7. 是否存在明显月份 source shift？  是。 主来源：May=baostock_5m，June=baostock_5m，July=minute_5m。

Q8. 是否有资格进入 outcome information audit？  有；本轮仍未读取任何 outcome。

## 审计边界

- 冻结 population：485 rows / 60 signal dates（2026-05-06 至 2026-07-29）。
- 只读取 candidate identity、D0/PREV_BOARD_DAY、本地5分钟缓存、未复权日线和历史涨停池路径字段。
- 未读取 Target7、LOSS、收益、S2 score/rank 或未来标签；未联网、未训练、未搜索窗口/阈值。
- 固定 tolerance：0.011；完整日固定为48根（09:35..11:30，13:05..15:00）。
- F1/F2/F3 是保守5分钟 proxy，不能称为真实 final_limit_up_time/open_board_count。

## Coverage

| field | overall | May | June | July | Board2 | Board3 | ready |
|---|---:|---:|---:|---:|---:|---:|---|
| F1 | 92.16% | 94.52% | 97.11% | 84.94% | 90.89% | 98.73% | YES |
| F2 | 92.99% | 95.21% | 98.27% | 85.54% | 91.87% | 98.73% | YES |
| F3 | 85.57% | 82.88% | 95.38% | 77.71% | 84.48% | 91.14% | YES |

## 最终状态

F1_FINAL_STABLE_LOCK_DATA = READY

F2_CONFIRMED_REOPEN_DATA = READY

F3_LOCK_DETERIORATION_DATA = READY

TEMPORAL_SOURCE_SHIFT = YES

LIMIT_PRICE_LINEAGE = PASS

5M_SEAL_PATH_DATA_STATE = 5M_SEAL_PATH_DATA_READY

OUTCOME_ACCESSED = NO

MODEL_TRAINED = NO

FEATURE_SEARCH = NO

WINDOW_SEARCH = NO

EXTERNAL_DATA_FETCHED = NO

NEXT_ACTION = BLIND_5M_SEAL_PATH_INFORMATION_AUDIT
