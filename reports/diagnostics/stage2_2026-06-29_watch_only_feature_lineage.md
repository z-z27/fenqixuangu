# 2026-06-29 共同 WATCH_ONLY 特征漂移血缘诊断

## 结论

- 总体分类：`DATE_SPECIFIC_UPSTREAM_EXPANSION_PLUS_NONSCORABLE_FEATURE_DRIFT`。
- A（21→41 模型候选扩张）：最早可观察于 signal pool；新增63只中，D0=2026-06-26 的20只进入 eligible/scorable。旧运行没有 raw-source snapshot，因此不能把扩张具体归因到某个 provider、缓存或旧生成步骤。
- B（12只共同 WATCH_ONLY 漂移）：10只由隔离前的 6月29日盘中 Tencent 日线缓存直接解释；603065、603313 缺少旧 raw，只能保守归为 `UNRESOLVED_NO_OLD_RAW_EVIDENCE`。
- 12只旧版和冻结版均非 eligible/scorable，也未出现在 frozen holdout 的 v004a/v002、Top15、v005 candidate pool 或 final Top3。故该漂移不影响 21→41 的直接模型候选扩张解释，但仍是有效的数据血缘问题。

## 重要日期语义

这12只均为 `d0_date=2026-06-25`、`signal_date=2026-06-29`、`days_since_d0=4`。源码 `build_key_zones()` 取传入日线最后一行，因此样本列名中的 `d1_*` 实际引用 **2026-06-29 signal-date 最新行情**，不是固定的首个 D0 后交易日 2026-06-26。报告同时保留真正 2026-06-26 D1 的 OHLCV/MA10/VWAP，避免混淆。

## 逐股特征差异

| code | name | changed_fields | mismatch_count |
| --- | --- | --- | --- |
| 000766 | 通化金马 | low_absorb_width_pct\|invalid_distance_pct | 2 |
| 002303 | 美盈森 | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price | 5 |
| 002387 | 维信诺 | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price | 5 |
| 002928 | 华夏航空 | d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price | 4 |
| 600152 | 维科技术 | total_score\|support_score\|entry_width_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|low_absorb_width_pct\|invalid_distance_pct\|candidate_base_price | 9 |
| 600500 | 中化国际 | support_score | 1 |
| 600668 | 尖峰集团 | d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price | 4 |
| 601798 | 蓝科高新 | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|low_absorb_width_pct\|invalid_distance_pct\|candidate_base_price | 7 |
| 603065 | 宿迁联盛 | support_score | 1 |
| 603313 | 梦百合 | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price | 5 |
| 603500 | 祥和实业 | total_score\|support_score\|entry_width_score\|d1_low_ma10_pct\|low_absorb_width_pct\|invalid_distance_pct | 6 |
| 603527 | 众源新材 | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price | 5 |

## 逐股根因分类

| code | name | primary_cause | secondary_cause | confidence | changed_model_fields |
| --- | --- | --- | --- | --- | --- |
| 000766 | 通化金马 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | low_absorb_width_pct\|invalid_distance_pct |
| 002303 | 美盈森 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price |
| 002387 | 维信诺 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price |
| 002928 | 华夏航空 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price |
| 600152 | 维科技术 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | total_score\|support_score\|entry_width_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|low_absorb_width_pct\|invalid_distance_pct\|candidate_base_price |
| 600500 | 中化国际 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | support_score |
| 600668 | 尖峰集团 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price |
| 601798 | 蓝科高新 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|low_absorb_width_pct\|invalid_distance_pct\|candidate_base_price |
| 603065 | 宿迁联盛 | UNRESOLVED_NO_OLD_RAW_EVIDENCE | RAW_DAILY_SIDE_CHANGE_INFERRED | LOW | support_score |
| 603313 | 梦百合 | UNRESOLVED_NO_OLD_RAW_EVIDENCE | RAW_DAILY_SIDE_CHANGE_INFERRED | MEDIUM | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price |
| 603500 | 祥和实业 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | total_score\|support_score\|entry_width_score\|d1_low_ma10_pct\|low_absorb_width_pct\|invalid_distance_pct |
| 603527 | 众源新材 | STALE_OR_PARTIAL_DAY_CACHE | RAW_DAILY_INPUT_CHANGED | HIGH | support_score\|d1_low_ma10_pct\|d1_close_ma10_pct\|d1_close_vwap_pct\|candidate_base_price |

## 直接缓存证据

- 既有隔离 manifest：`reports/diagnostics/cache_quarantine_2026-06-29/manifest.csv`。
- 10只隔离文件的 `source=tencent_daily`，mtime 均在 2026-06-29 15:10 Asia/Shanghai 之前，失效原因均为 `stale_intraday_daily_cache`。
- 对这10只，将隔离日线与当前完整的6月29日48根5分钟线按生产 `_merge_minute_amount → enrich_daily_indicators → build_key_zones` 复算，全部精确复现旧样本 key zones、support_score 和 entry_width_score（10/10）。
- 当前12只的6月29日分钟缓存均有48根，最后时间15:00；旧/新样本记录的 `d1_intraday_vwap` 和 `d1_first_5m_low` 全部一致。最早可观察差异位于日线侧，而不是分钟摘要侧。
- 603065、603313 不在隔离 manifest 中；现存 `daily_unadjusted` 旧文件只覆盖到6月26日，无法恢复6月29日旧 raw。两只不得升级为已证明的盘中缓存原因。

## 旧运行时点

- 本任务使用的旧 dedup candidates mtime：`2026-07-01T13:33:37.608938217+08:00`；其上游未去重文件 mtime 为 2026-07-01 12:07:24+08:00，均晚于6月29日收盘。
- 旧生成日志6月29日：signal_rows=133、quality_ok=133、quality_failed=66、future_fetch_ok=133、future_fetch_failed=0。
- 12只旧 data-quality 均记录 `daily_source=cache`、`minute_source=cache`、`from_cache=True`。因此“报告文件在收盘后生成”不代表其日线缓存也在收盘后形成；10只的隔离 mtime 直接证明复用了盘中快照。

## 特征依赖链

| field | source | function | entry | direct_dependencies | daily | minute | ma10 | cross_section | future_data | cache_timing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_base_price | src/backtester.py:1265-1280 | _candidate_base_price | history_samples.evaluate_history_candidate_only | key_zones.d1_close；fallback=signal-date daily close | 是 | 否 | 否 | 否 | 否 | 是（生成 signal 时的日线缓存） |
| d1_low_ma10_pct | src/signal_engine.py:75-78,270-275 | generate_signal/_pct_distance | build_signals_for_pool→generate_signal | key_zones.d1_low, ma10 | 是 | 否 | 是 | 否 | 否 | 是 |
| d1_close_ma10_pct | src/signal_engine.py:75-79,270-275 | generate_signal/_pct_distance | build_signals_for_pool→generate_signal | key_zones.d1_close, ma10 | 是 | 否 | 是 | 否 | 否 | 是 |
| d1_close_vwap_pct | src/signal_engine.py:80-84,270-275 | generate_signal/_pct_distance | build_signals_for_pool→generate_signal | d1_close / (d1_intraday_vwap or d1_vwap) | 是 | 是 | 否 | 否 | 否 | 是 |
| low_absorb_width_pct | src/history_samples.py:1904；src/backtester.py:1360-1364 | evaluate_history_candidate_only/_zone_width_pct | history sample row construction | low_absorb_min, low_absorb_max | 间接 | 间接 | 间接 | 否 | 否 | 是 |
| invalid_distance_pct | src/history_samples.py:1905；src/backtester.py:1366-1369 | evaluate_history_candidate_only/_invalid_distance_pct | history sample row construction | invalid_price, low_absorb_max | 间接 | 间接 | 间接 | 否 | 否 | 是 |
| support_score | src/support_quality.py:6-66 | score_support_quality | generate_signal | latest daily low/close/prev_low/vwap/close_position；latest minute VWAP/above ratio | 是 | 是 | 否 | 否 | 否 | 是 |
| entry_width_score | src/signal_engine.py:75,215-227,253-260 | score_entry_width/_low_absorb_width_pct | generate_signal | zone bounds；base=prev_close or d1_close | 间接 | 间接 | 间接 | 否 | 否 | 是 |
| total_score | src/signal_engine.py:88-98,172-190；src/config.py:51-60 | score_total | generate_signal | trend/graph/active-cooling/entry/theme/support及固定权重 | 间接 | 间接 | 间接 | 仅theme依赖limit-up行业计数；不依赖成功signal集合 | 否 | 是 |

原始链路为：`MarketDataService.get_stock_bars (src/loaders.py:119-205) → _merge_minute_amount (888-897) → enrich_daily_indicators/enrich_5min_indicators (src/indicators.py:6-38) → build_key_zones (41-87) → generate_signal (src/signal_engine.py:52-169) → evaluate_history_candidate_only (src/history_samples.py:1867-1955)`。

## Git 公式版本证据

- 旧样本没有 runtime provenance，无法证明生成时的精确 commit。按文件 mtime，生成前最近已提交版本是 `30d9b2e6338232f26eb6c92e9c752be3b5c33b7d`；这只是版本上界，不排除当时存在未提交工作树。
- `src/indicators.py`、`src/support_quality.py`、`src/signal_engine.py` 在 `30d9b2e6338232f26eb6c92e9c752be3b5c33b7d` 与当前 HEAD 的 Git blob 完全相同：

| path | old_blob | current_blob | same |
| --- | --- | --- | --- |
| src/indicators.py | a23be30a7a9abbe76f728b1c24eb4598b849e807 | a23be30a7a9abbe76f728b1c24eb4598b849e807 | True |
| src/support_quality.py | a58648410d187a095fd971b464f5163deb1d9d8a | a58648410d187a095fd971b464f5163deb1d9d8a | True |
| src/signal_engine.py | 9eeead517399f44407b405846c68e55700c73dbe | 9eeead517399f44407b405846c68e55700c73dbe | True |

- `_candidate_base_price`、`_zone_width_pct`、`_invalid_distance_pct` 的 AST 也完全相同：

| function | old_ast_sha256 | current_ast_sha256 | same |
| --- | --- | --- | --- |
| _candidate_base_price | 99e6003aeb50bcf46001409ff71e9e03ad17e1c03a858d84137d9b2b6a57e1ec | 99e6003aeb50bcf46001409ff71e9e03ad17e1c03a858d84137d9b2b6a57e1ec | True |
| _zone_width_pct | 04c2774057aa122ab5e0efa15ea3a09b9afdb76223344309d5e20f9a04240b43 | 04c2774057aa122ab5e0efa15ea3a09b9afdb76223344309d5e20f9a04240b43 | True |
| _invalid_distance_pct | ff93504cf0aabc1711ca60720b4b28a5a7976f429e24524c189ebc61695542b2 | ff93504cf0aabc1711ca60720b4b28a5a7976f429e24524c189ebc61695542b2 | True |

- `StrategyConfig` 的评分权重未变，且 `support_weight=0.00`。因此 support_score 的变化本身不改变 total_score；600152、603500 的 total_score 变化均来自 entry-width 档位变化。

## 横截面依赖

- candidate_base_price、MA10/VWAP 百分比、support_score、entry_width_score 均为单票特征，不使用 signal pool 排名、分位数或候选总数。
- total_score 的 theme_score 会读取 limit-up pool 的同行业数量，但不读取“成功生成的 signal 集合”；本次12只 theme_score 均未变化。因此 131→194 的 signal 集合扩张不会反向改写这12只字段。
- v004a 的按日 percentile rank 只在 `v004a_scorable_bool=True` 后计算（`src/v004a.py:256-355`）。12只在过滤前即被排除，不参与 v004a/v002 分位数。

## 模型域影响

- 旧21只与当前同21只关键模型输入完全一致，既定语义 hash：`2d0d3521c2af11a4e9a9a33630564d944466f84e0fd4ba8ac8ebdc8d099f65d8`。
- 12只均：旧 eligible=False、当前 eligible=False、当前 scorable=False；frozen holdout 全部后续层均无其 code。
- 结论：`WATCH_ONLY 特征漂移不影响本次 21→41 的直接模型候选扩张解释。`

## 证据限制

- 旧运行没有 raw-source snapshot、完整旧分钟 raw 和 commit provenance。
- 对603065、603313，能证明的是“样本日线侧派生值变化且公式版本证据不支持逻辑漂移”；不能严格证明具体 provider、缓存形成时点或唯一底层字段。
- 旧/新分钟摘要一致不等同于旧/新完整分钟文件逐字节一致。
