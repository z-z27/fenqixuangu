# v004c 日线来源审计 — Forward 数据路径追溯 (v004c_forward_data_lineage_audit)

只读代码与文件事实审计,不猜测。审计日期 2026-08-03,时区 Asia/Shanghai。

## 审计依据的代码事实

- `src/loaders.py::MarketDataService`: `daily_cache = StockFrameCache(config.cache_dir / "daily", "daily")` → **主日线缓存路径 = `data/cache/daily/{code}_daily.pkl`**;另有 `daily_unadjusted_cache = StockFrameCache(config.cache_dir / "daily_unadjusted", "daily")`。
- `get_stock_bars(code, days, end_date)`: 当缓存不满足 `_cache_covers`(≥180 个交易日且末日期 == end_date)时,调用 `provider.fetch_daily_history(...)` 并 **`self.daily_cache.write(normalized, cached_daily)`**(loaders.py:161)。
- `ensure_minute_cache`(loaders.py:357): **只刷新 5min 缓存,不碰日线**。
- `src/backtester.py::prefetch_future_bars_for_signals`(468 行)内部调用 `service.ensure_minute_cache(code, days=eval_days, end_date=end_date)`(486 行)— **未来标签预取只刷新 5min 缓存**。
- `src/v005_daily_selector.py::run_v005_daily_from_market`(297 行): `collect_limit_ups` → `build_signals_for_pool(service, pool, as_of_date=signal_date, ...)` → `apply_daily_research_ranking` → `run_v005_daily_selector`。**信号构建调用 get_stock_bars(end_date=D1),选择器部分只处理信号表,不再加载任何 bar**。
- `src/cli.py::_build_signals`(约 723 行): `bars = service.get_stock_bars(code, days=days, end_date=as_of_date)` — D1 口径,刷新主日线缓存到 D1。
- `src/v005_daily_selector.py` 的 `--signals-file` 模式:**完全不加载 bar,不触碰任何缓存**。

## 12 问回答

### 1. 七月 forward 使用的日线 loader 是什么?
`MarketDataService.get_stock_bars`(`src/loaders.py:126`),内部使用 `StockFrameCache` 读取 `data/cache/daily`。

### 2. 它读取的文件路径是什么?
`data/cache/daily/{code}_daily.pkl`(经 `FQ_CACHE_DIR` 默认 `data/cache`)。

### 3. 是否与 v004c 0.2.1 读取路径完全一致?
**是。** v0.2.1 构建脚本同样读取 `data/cache/daily/{code}_daily.pkl`(同一 `StockFrameCache` 路径)。两者读取的是同一个文件集合。

### 4. Forward 是否只要求 D1 及以前数据?
**是。** `get_stock_bars(end_date=as_of_date)` 中 as_of_date = 当日涨停池最新交易日(D1);信号生成只用 D1 及更早的日线与 5min;`_keep_recent_trade_days` 只保留最近 N 个交易日。forward 流程从不请求 D2/D3 的行情。

### 5. Forward 是否会在 D3 成熟后更新个股日线缓存?
**不会。** 主日线缓存只在"该股被 D1 处理且缓存不覆盖 end_date"时刷新(loaders.py:148-161)。没有任何流程在标签成熟后回写 D2/D3 日线。唯一与"未来"相关的流程 `prefetch_future_bars_for_signals` 只刷新 **5min** 缓存(backtester.py:486),不刷新日线。

### 6. Forward 评分报告是否保存了 D2/D3 日线 OHLC?
**没有。** `v005_daily_scored_candidates_*.csv` 只含分钟口径 `d2open_d3high_return_pct` 等,且**不含 d2_trade_date/d3_trade_date 列**(本次审计用统一日历重建 D2/D3 日期后,2783 条索引命中才可与缺失日期匹配)。日线 OHLC 从未写入任何 forward/holdout 报告。

### 7. 是否存在"评分结果保留,但底层日线缓存后来不完整"的情况?
**是,这是本问题的主因。** 评分(基于 D1 时的信号帧)保存在 daily_v005/holdout 报告中;而 D2/D3 的日线 bar 从未进入 `data/cache/daily`。对 151 条缺失记录: 90 条曾被 v004a 评分(其中 70 条 daily_v005_frozen、11 条 holdout_0626、8 条 grid_v2、1 条 frozen_reconstruction),29 条进入过 v004a Top15,7 只股票出现在 v005 selection/top3 文件(000011、000520、000722、000899、600984、603338、603983),14 只出现在 decision 文件——但它们的 `data/cache/daily/*.pkl` 仍止于各自最后一次 D1 处理日期。

### 8. 是否存在多个缓存目录或多个数据源版本?
**是。** 至少 4 类日线相关本地来源:
| 来源 | 路径 | 数据源 | 用途 |
|---|---|---|---|
| 主日线缓存 | `data/cache/daily/*.pkl` (1576 文件) | tencent_daily(调整口径 none) | loader 正式读取路径 |
| 未复权日线缓存 | `data/cache/daily_unadjusted/*.pkl` (3034 文件) | sina_daily | 派生涨停扫描(`_get_daily_for_limitup_scan`),adjust="none" |
| 原始日线快照 | `data/raw/daily/*.csv` (1560 文件) | 随 get_stock_bars 写出的最近窗口 | 报告/调试 |
| 5min 缓存 | `data/cache/minute_5m/*.pkl` | sina_5m | D1 盘中因子;未来预取唯一刷新对象 |

另: `data/overrides/minute_bar_repairs.csv` 为 **5min** 修复覆盖层(1 条,002857),与日线无关。

### 9. 是否存在缓存被部分覆盖、截断或回退的证据?
**未发现截断或回退。** 证据:
- 主日线缓存末日期分布为"各股最后一次 D1 处理日期"(06-29 344 只、07-03 97 只、07-14 48 只、07-24 108 只、07-31 294 只等),与"池内股票按天增量刷新"一致,是未更新而非截断。
- `daily_unadjusted` 存在 2026-06-29/30 与 07-20/22 两个批量写入波次(派生扫描)。
- **审计窗口内发现外部刷新**: 6 只股票(000037、001331、600396、600644、600936、605488)的 `data/cache/daily/*.pkl` mtime 为 2026-08-03 22:01-22:03,晚于 v0.2.1 构建时间 18:13——本任务期间环境并非静态,缓存正被外部进程(推测为用户日常 run)持续更新。这 6 只股票的 10 条缺失记录在构建时确实缺日期,现在主缓存已含(见缓存审计表 audit_result=CACHE_REFRESHED_EXTERNALLY_AFTER_V021_BUILD)。

### 10. 90 条左右曾被 v004a 评分但当前缺 D2/D3 日线的记录,具体来自什么评分来源?
**90 条 = 全部缺失记录的 59.6%**,来源: daily_v005_frozen 70、holdout_0626_frozen 11、grid_v2_walk_forward 8、frozen_reconstruction 1(另有 61 条无任何 v004a 评分)。这些评分的 D2/D3 标签均为 **5min 分钟口径**(报告内 d2open_d3high_return_pct 来自 `_d2open_d3_metrics`),不包含日线 OHLC。

### 11. 哪些记录进入过 v004a Top15、v5a 候选或最终三票?
- v004a Top15: **29 条**缺失记录(is_v004a_top15=True)。
- v5a(政策组合)候选/选择文件出现: **14 只股票**在 v005 decision 文件、**7 只股票**在 selection/top3 文件(000011、000520、000722、000899、600984、603338、603983)。

### 12. 为什么参与过 forward 不必然代表当前缓存中保存了未来 D2/D3 标签?
因为 forward 的**数据契约是 D1 时点**:
1. 信号构建只在 D1 时点请求 D1 及更早 bar,并把日线缓存刷新到 D1(该股在该日被处理时);
2. 标签(分钟口径)在 D2/D3 成熟后由**事后**的 history-samples/holdout 流程计算,该流程通过 `ensure_minute_cache` 只补 5min;
3. D2/D3 的**日线** OHLC 从不被任何流程写入 `data/cache/daily`;
4. 因此"被 forward 评分过"只证明该股在 D1 时点有缓存,不证明 D2/D3 日线被保存过——两者是不同时点、不同缓存路径的数据。

## 结论(事实层面)

- v004c 0.2.1 与 forward 读取的是**同一个主日线缓存路径**;缺失不是路径不一致。
- 缺失机制 = **主日线缓存只在 D1 处理时增量更新,未来 D2/D3 日线从不回写**;这解释了"参与过 forward 仍缺 D2/D3"。
- 可恢复性: 151 条缺失中 112 条(74.2%)的期望日期存在于其它本地日线来源(`daily_unadjusted` sina_daily 131 个日期条目、`raw/daily` CSV 85 个日期条目、外部刷新后的主缓存 111 个日期条目);39 条仅有 forward/holdout 快照的**分钟**口径证据;0 条完全无本地记录。
