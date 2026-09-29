# v004d Auction Data Feasibility Audit v001

## 先说人话

1. **历史上不能完整看到 D2 09:25 集合竞价结果。** 现有缓存可从 D2 日线 open 找到开盘价等价参考，但没有独立 09:25 snapshot，也没有可分离的竞价成交量和成交额。
2. **价格参考覆盖 100.00%，但核心三字段覆盖 0%。** 主窗口共有 492 个 candidate-D2；auction volume/amount 都不能从全日量额或 09:35 bar 伪造。
3. **现有 5 分钟 K 不是集合竞价 K。** Sina/BaoStock 的第一根规范化 bar 是 09:35，已经混入 09:30 后连续竞价。
4. AKShare/Eastmoney 的盘前接口能返回当前/最近交易日 09:15--09:30 状态，但不能回放 May--July，量额究竟是虚拟撮合、累计状态还是最终撮合量仍未被仓库语义证明。
5. 今天是周日，**真实 09:25--09:30 latency 未测试**。实现了 collector、时间戳日志与 N=5/10/15/20/30 dry-run；dry-run P95 最慢约 6.814 ms，但它不含网络 fetch，不能代替 live PASS。
6. 因历史核心数据缺失，当前 auction 路线还没有资格进入 outcome information audit。

## Q1. 历史上到底能不能看到 D2 09:25 结果？

只能看到官方 D2 open 的价格等价参考；看不到独立保存的 09:25 price snapshot、auction volume 和 auction amount，因此答案是 **不能完整看到**。

## Q2. 价格、成交量、成交额是否完整可靠？

- official-open price reference：100.00% coverage；但没有独立 auction feed，无法做独立 exact-match lineage 验证。
- auction volume：0% historical coverage。
- auction amount：0% historical coverage。
- 三字段同时完整：0%，未达到 overall 90% / monthly 85% gate。

## Q3. 这些量是真实集合竞价结果，还是普通 5 分钟 K 的伪解释？

日线 open 只保留作最终开盘价参考；全日 volume/amount 与 09:35 首根 bar 都明确不是 auction-only 数据，未作伪解释。当前盘前接口量额语义仍是 `UNRESOLVED`。

## Q4. 两根 5 分钟竞价数据有没有明确含义？

仓库没有原生 09:15--09:20、09:20--09:25 两根 auction 5m 记录。当前 AKShare 接口提供 1 分钟 supplier state，不是普通成交 OHLCV，也不支持 May--July 回放。因此 `AUCTION_5M_DATA_STATE = NOT_AVAILABLE`；不强迫保留。

## Q5. 实盘最早什么时候稳定可获取？

`PENDING_REAL_TRADING_DAY_VALIDATION`。非交易日 probe 不能回答 09:25 后稳定时点。

## Q6. 5/10/20/30 只需要多久？

本轮只完成 parser/join/basic-calculation dry-run；最大规模 P95 为 6.814 ms。网络 fetch 未计入，详见 scale/timing CSV，不能用该数声明实盘通过。

## Q7. 09:30 前有没有足够时间？

尚不能判断。需要真实交易日 09:25--09:30 记录 source availability、完整率、stale 与 P95 total time；正式 gate 是 P95 <=120s 且核心字段完整率 >=95%。

## Q8. 历史研究与未来实盘能否使用同语义数据？

不能证明。历史只有 official-open price reference，volume/amount 缺失；live prospective source 是 Eastmoney current snapshot，`HISTORICAL_LIVE_SOURCE_SHIFT = UNKNOWN`。

## Q9. 是否有资格进入下一阶段？

没有。问题不是计算速度，而是历史核心数据缺失和量额语义未确认。

## 数据纪律

- Candidate universe：完整 Board2/Board3 first-break，S2 不做 hard filter。
- 输入 CSV 使用显式 identity/date/board 列投影；outcome、future return、model score/rank 未载入。
- 未训练模型、未搜索 feature、未创建 BUY/PASS、entry 仍为 D2 open。
- Sunday probe 只验证接口可达性/解析：盘前接口一次成功（13.804s，返回 2026-09-04 的 16 行），一次复测失败；bid/ask probe 失败。均不属于 live latency evidence。

## Formal result

V004D_CANDIDATE_UNIVERSE = BOARD2_BOARD3_FIRST_BREAK_ALL_CANDIDATES

D1_HARD_FILTER = NO

HISTORICAL_AUCTION_DATA_STATE = NOT_READY

LIVE_AUCTION_DATA_STATE = PENDING_REAL_TRADING_DAY_VALIDATION

AUCTION_FIELD_SEMANTICS_STATE = UNRESOLVED

AUCTION_5M_DATA_STATE = NOT_AVAILABLE

HISTORICAL_LIVE_SOURCE_SHIFT = UNKNOWN

CORE_FIELDS_AVAILABLE = OFFICIAL_D2_OPEN_PRICE_REFERENCE_ONLY

EARLIEST_RELIABLE_LIVE_DATA_TIME = PENDING_REAL_TRADING_DAY_VALIDATION

P95_TOTAL_PIPELINE_TIME = NOT_MEASURED_LIVE

V004D_AUCTION_DATA_FEASIBILITY_STATE = V004D_AUCTION_DATA_NOT_FEASIBLE

OUTCOME_ACCESSED = NO

MODEL_TRAINED = NO

FEATURE_SEARCH = NO

BUY_PASS_RULE_CREATED = NO

ENTRY_PRICE_DEFINITION = D2_OPEN

NEXT_ACTION = STOP_AUCTION_PIPELINE
