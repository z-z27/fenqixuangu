# v004d Direct Auction Source Capability Audit v001

## 先说人话

1. **Sina 有实时行情字段，但没有被证明为独立的集合竞价数据源。** 原始 quote 可批量拿到 open、当前累计量额、五档盘口和时间；可是它没有历史竞价回放，也没有显式的最终撮合量、未匹配量或虚拟匹配字段。只有在真实交易日 09:25 后连续轮询，才能证明当时的累计量额是否恰好等于集合竞价成交量额。
2. **Eastmoney 原始源确实比当前仓库缓存多。** `trends2/get?iscr=1&ndays=1` 返回 09:15--09:30 盘前分钟状态。000001 在 2026-09-04 的 09:25 价格与官方 open 一致；09:26 出现 5,094 手和 6,041,484 元，且 `5094 × 100 × 11.86 = 6,041,484`。这是“最终开盘撮合行”的强线索，但目前只有一个独立 code-date，尚不足以把语义标成 CLEAR。
3. **AKShare 不是第三套独立上游。** `stock_zh_a_hist_pre_min_em` 底层就是 Eastmoney `trends2/get`，固定 `ndays=1`，无日期参数；官方文档也明确为最近一个交易日。`stock_zh_a_hist_min_em(period=1)` 底层固定最近 5 个交易日，但这仍不能回补 May--July。
4. **May--July 历史竞价回补仍然被阻断。** Sina 没有历史竞价接口；Eastmoney/AKShare 已确认的盘前接口只到最近 1 日，最近 5 日接口也远达不到开发窗口。把 `ndays` 从 1 改为 5 的直接试验仍只返回 2026-09-04，并从 09:30 开始，说明这不是简单把 AKShare 参数改大就能解决。
5. **实盘能力有技术可行性迹象，但本轮不能宣称 READY。** Sina 批量 5/10/20/30/50 只读请求均返回；Eastmoney 批量 20/30/50 返回，5/10 遇到暂态传输失败。今天是周日，没有真实 09:15--09:30 时点证据，字段何时稳定、09:26 是否普遍出现、数据是否陈旧仍待交易日验证。
6. **上一轮的问题不只是仓库 wrapper/cache 没保存。** wrapper 的确漏掉了 Eastmoney 盘前接口和大量 quote 字段；但三个审计源也都不能回补 May--July 独立竞价量额。因此 `CURRENT_CACHE_LIMITATION_ONLY = NO`。

## Q1. Sina 到底有没有集合竞价数据？

有可在盘前轮询的实时 quote 候选字段，没有独立历史竞价序列，也没有显式的最终撮合量/未匹配量字段。正式状态：`AUCTION_SOURCE_UNRESOLVED`。

## Q2. Eastmoney / stock 到底有没有？

有。原始 `trends2` 能看到最近交易日的 09:15--09:30 supplier-defined premarket state，且 09:26 行在一个样本中高度符合最终开盘撮合价量额。问题是历史深度太短、样本数太少、真实 09:25 可用时点未验证。正式状态：`AUCTION_SOURCE_PARTIAL`。

## Q3. AKShare 到底有没有？

有 Eastmoney 盘前数据的 wrapper，但不是独立上游。它只暴露最近一个交易日，不能传入历史日期；盘口 wrapper 是单股实时接口。本轮运行还观察到暂态 JSON/连接失败。正式状态：`AUCTION_SOURCE_PARTIAL`。

## Q4. 哪个能拿价格、量、额？

- Eastmoney pre-minute / AKShare pre-minute：最近交易日可见价格、量、额；09:26 行是最有希望的最终撮合行，但语义仍为 provisional。
- Sina / Eastmoney current quote：有 open 和累计量额；在收盘后它们明确是全日累计，不是 auction-only。
- 未匹配买量、未匹配卖量、显式虚拟匹配量：三个源均未确认。

## Q5. 哪个能查历史？

没有一个能回补 May--July。Eastmoney/AKShare 的 pre-minute 是最近 1 个交易日；Eastmoney historical 1m wrapper contract 最多最近 5 个交易日，而且本轮历史域 runtime probe 失败。Sina 的历史 5m 是正常连续交易 bar，不含集合竞价。

## Q6. 哪个能用于实盘？

Eastmoney raw pre-minute 是最合理的潜在 live source；Sina quote 可作为潜在 fallback 观察源。两者都必须在真实交易日 09:15--09:27 轮询，验证字段冻结时点、完整率和语义后才能批准。

## Q7. 之前的问题是 source 没数据，还是 wrapper/cache 没保存？

两者都有：仓库没包 Eastmoney pre-minute，也把 spot quote 字段在 universe normalization 后丢掉；但上游可确认的历史深度同样不足，无法仅靠重新暴露 wrapper 回补 May--July。

## Q8. 现在能不能继续集合竞价路线？

不能直接进入信息审计或 BUY/PASS。可以保留一条数据积累路线：先在真实交易日确认 Eastmoney 09:25/09:26 语义与延迟，再决定是否从当日开始持续归档。由于本轮协议对 PARTIAL 的正式动作是停止评审，本次输出 `STOP_AND_REVIEW`，不自动创建归档任务。

## 能力边界

- 09:15--09:25 Eastmoney 行是 `SUPPLIER_DEFINED_PREMARKET_BAR`，价格变化而量额为零，不能当普通成交 OHLCV。
- 09:26 行只标 `PROVISIONAL_FINAL_AUCTION_MATCH_ROW`；一个样本的精确算术与开盘价一致不足以替代多股票、真实时点验证。
- 非交易日 batch latency 只证明请求形状和解析能力，不是 live latency。
- 无 outcome、无模型、无 BUY/PASS、无阈值或因子搜索。

## Formal result

SINA_AUCTION_SOURCE_STATE = AUCTION_SOURCE_UNRESOLVED

EASTMONEY_AUCTION_SOURCE_STATE = AUCTION_SOURCE_PARTIAL

AKSHARE_AUCTION_SOURCE_STATE = AUCTION_SOURCE_PARTIAL

PRIMARY_AUCTION_SOURCE_CANDIDATE = NONE

POTENTIAL_HISTORICAL_SOURCE = EASTMONEY_AKSHARE_RECENT_1_TO_5_TRADING_DAYS_ONLY

POTENTIAL_LIVE_SOURCE = EASTMONEY_RAW_TRENDS2_PREMARKET

HISTORICAL_AUCTION_BACKFILL_STATE = MAY_JULY_NOT_AVAILABLE_FROM_AUDITED_SOURCES

LIVE_AUCTION_COLLECTION_STATE = PENDING_REAL_TRADING_DAY_SEMANTICS_AND_TIMING_VALIDATION

LIVE_TIMING_VALIDATION = PENDING_TRADING_DAY

CURRENT_CACHE_LIMITATION_ONLY = NO

DIRECT_SOURCE_CAPABILITY_RESOLVED = PARTIAL

HISTORICAL_LIVE_SEMANTIC_SHIFT = UNKNOWN

OUTCOME_ACCESSED = NO

MODEL_TRAINED = NO

FEATURE_SEARCH = NO

BUY_PASS_CREATED = NO

V004D_DIRECT_AUCTION_SOURCE_STATE = DIRECT_AUCTION_SOURCE_PARTIAL

NEXT_ACTION = STOP_AND_REVIEW

## Evidence URLs

- AKShare stock data documentation: https://github.com/akfamily/akshare/blob/main/docs/data/stock/stock.md
- AKShare Eastmoney stock history wrapper source: https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_hist_em.py
- AKShare bid/ask wrapper source: https://github.com/akfamily/akshare/blob/main/akshare/stock/stock_ask_bid_em.py
