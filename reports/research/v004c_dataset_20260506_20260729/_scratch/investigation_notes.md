# v004c 数据结构调查报告 (2026-08-03)

只读调查笔记,供数据集构建与审计使用。

## 1. 数据缓存布局 (data/cache)

| 缓存 | 文件 | 覆盖 | 说明 |
|---|---|---|---|
| daily | `data/cache/daily/{code}_daily.pkl` 1576 文件 | 单票约 2025-03 起,结束日期碎片化(见下) | 列: date, code, market, open, high, low, close, volume, amount, pct_chg, change, amplitude, turnover_rate, source |
| minute_5m | `data/cache/minute_5m/{code}_5min.pkl` 1576 文件 | 约 2026-04-29 起(435 只),结束日期碎片化 | 列: datetime, trade_date, time, code, market, open, high, low, close, volume, amount, pct_chg, change, amplitude, turnover_rate, source, adjust |
| limit_ups | `data/cache/limit_ups/{date}_limitups.pkl` 62 文件 | **2026-05-06 .. 2026-07-31 完整** | 列: trade_date, code, name, market, latest_price, pct_chg, amount, turnover_rate, float_market_cap, total_market_cap, industry, limit_up_time, final_limit_up_time, open_board_count, seal_amount, consecutive_limit_up_count, source |
| universe | `data/cache/universe/eastmoney_main_board_universe.pkl` | 3098 只主板 | code, name, market, industry, float_market_cap, total_market_cap, source |
| suspension_status | 仅 2026-06-26/06-29/06-30/07-07/07-10 五份 | 不完整 | 不能作为断板观察的停牌判定依据 |

### 1.1 关键覆盖限制

- **daily 缓存结束日期分布**: 05-06(2) 05-08(13) 05-11(12) 05-12(18) 05-15(34) 05-18(11) 05-19(14) 05-22(34) 05-25(7) 05-26(10) 05-29(27) 06-01(15) 06-02(10) 06-05(55) 06-08(13) 06-12(1) 06-16(1) 06-18(5) 06-24(1) 06-26(2) 06-29(344) 06-30(17) 07-03(97) 07-06(45) 07-07(53) 07-10(61) 07-13(25) 07-14(48) 07-17(79) 07-20(18) 07-21(12) 07-22(1) 07-24(108) 07-27(67) 07-28(17) 07-29(5) 07-31(294)。
  → 信号日期的 D1 日线特征只能对"日线缓存覆盖到该日期"的股票计算;之后日期属于数据缺口,按 `insufficient_daily_history` 排除。
- **5min 缓存结束日期分布**: 05-06(19) 05-07(14) 05-19(1) 05-22(1) 06-12(26) 06-16(34) 06-18(26) 06-23(33) 06-24(1) 06-26(16) 06-30(27) 07-01(290) 07-03(1) 07-14(352) 07-16(134) 07-20(1) 07-21(80) 07-22(1) 07-24(29) 07-31(490)。
  → D1 分钟特征与 D2/D3 标签受此限制;5min 未覆盖 → `missing_minute_data` / `missing_future_label`。
- **daily 缓存起始**: 1568/1576 文件起始 ≤ 2026-04-01,MA20/20 日回看基本可满足。
- **5min 缓存起始**: 大部分自 2026-04-29 起。
- **amount 缺失**: tencent_daily 来源的日线 amount 为 NaN。项目通过 `_merge_minute_amount`(loaders.py:990)用 5min amount 按日求和回填。本任务沿用该口径:日线 amount 以 5min 求和优先,5min 缺失且日线 NaN 则留空。
- **raw/ 与 processed/**: `data/raw/daily/*.csv` 与 `data/raw/minute_5m/*.csv` 只保存最近窗口(每票 10~480 行),覆盖远差于 cache;`data/processed/recent_limitups.csv` 只是最近涨停池。**本任务以 cache 为准**。

## 2. 涨停判定口径 (沿用项目逻辑)

- 项目明确的涨停判定实现: `src/loaders.py::_is_main_board_limit_up_day`(daily close ≥ limit_price − 0.011 且 pct_chg ≥ 9.7 且 high ≥ limit_price − 0.011;limit_price = prev_close × 1.10 四舍五入到分)。
- 涨停池(akshare zt_pool_em)是"盘中触及并封板"的池子,含 open_board_count/limit_up_time 等信息,是涨停池数据口径。
- **本任务主判定**: 逐票日线收盘封板判定(项目逻辑);涨停池用于交叉核对、名称/行业/换手/封板信息。
- **连板计数**: 日线相邻交易日序列上连续涨停(严格相邻,非池子 gap≤2 天的粗口径)。`signal_engine._count_consecutive_boards`(池子口径,gap≤2)存在周末跨天计数问题(Fri→Mon gap=3 天会被切断),且池子只从 2026-05-06 起,因此本任务采用日线相邻交易日口径,并在审计中记录池子 consecutive_limit_up_count 作交叉核对。此差异在 feature definitions 与 review 中说明。
- ST/退市: 项目日线派生判定要求 pct_chg ≥ 9.7,ST 5% 涨停不会计数——与项目口径一致,ST 股票因此不会形成 2/3 板事件。名称中带 ST/退 的股票保留在审计中说明。

## 3. 目标标签口径 (沿用项目定义)

`src/history_samples.py::_d2open_d3_metrics` + `_finalise_targets`:
- d2/d3 = 该股 5min 交易日序列中 signal_date 之后的第 1、2 个交易日(`_future_trade_dates`),即按个股自己的交易日,天然跳过停牌日。
- d2_open = D2 首根 5min bar 的 open(缺失时回退日线 open)。
- d3_high = D3 全部 5min bar 的 high 最大值;**d3_close = D3 最后一根 5min bar 的 close**。
- **D3 最高价是"D3 当日盘中最高"(分钟级),不是 D2→D3 持有窗口最高价** — 沿用项目口径并在 feature definitions 写明。
- d2open_d3high_return_pct = (d3_high / d2_open − 1) × 100;d2open_d3close_return_pct = (d3_close / d2_open − 1) × 100。
- target7_d2open_d3high = d2open_d3high_return_pct ≥ 7.0。
- 本任务输出字段 d2_high/d2_low/d2_close/d3_low 为补充字段,同样以 5min 优先(首根/最大/最小/末根),5min 缺失回退日线,并在 definitions 注明。
- 标签只允许读到 2026-07-31;signal_date ≤ 07-29 时 D2/D3 ≤ 07-31 可满足;若个股下一交易日晚于 07-31(如停牌跨月),标签不可计算 → LABEL_UNAVAILABLE。

## 4. 现有模型与评分文件 (比较字段来源)

| 数据 | 文件 | 覆盖 signal_date | v004a prob/rank | v002 rank |
|---|---|---|---|---|
| v004a grid_v2 | `reports/v004a/grid_v2_scored/v004a_scored_candidates.csv` | 06-02..06-29 | ✓ (walk-forward 逐日折) | 无 |
| holdout 0626 | `reports/v005_fixed_grid_holdout_2026-06-26_2026-06-30/v005_fixed_grid_holdout_scored_candidates.csv` | 06-26, 06-29, 06-30 | ✓ (冻结 06-26 折) | 无 |
| holdout 0701 | `reports/v005_fixed_grid_holdout_2026-07-01_2026-07-03/v005_fixed_grid_holdout_scored_candidates.csv` | 07-01, 07-02, 07-03 | ✓ (冻结 06-26 折) | 无 |
| daily_v005 | `reports/daily_v005/{date}/v005_daily_scored_candidates_{date}.csv` | 07-02..07-31 | ✓ (冻结 06-26 折) | ✓ daily_rank |
| daily signals | `reports/daily_signals/signals_{date}.csv` | 06-25, 07-01..07-31 | 无 | ✓ daily_rank |
| history candidates | `reports/history_samples/2026-05-06_2026-06-29/history_candidates_*.csv` | 05-06..06-29 | 无(但有完整 v004a 输入字段与标签,可只读重建) | 无 |

- **v004a 覆盖缺口**: 2026-05-06..06-01 无现成评分。计划用冻结系数 `configs/models/v004a_coefficients_2026-06-26.csv` 对 history_candidates 的 scorable 池做只读重建(percentile rank 特征与 predict_logistic 均为纯函数),并在 06-02..06-29 与 grid_v2 的特征列核对特征计算;核对通过才采用,否则留空并在报告说明。
- **v002 覆盖缺口**: 05-06..06-24 及 06-26..06-30 无现成 daily_rank。不重跑 v002 排序(避免与 daily 流程口径漂移),留空并记录。
- v004a 语义说明: grid_v2 为 walk-forward 逐日折评分;holdout/daily_v005 为冻结 06-26 折。比较字段将注明各自语义。

## 5. 冻结模型资产 (existing_model_inventory 用)

- v004a 冻结系数: `configs/models/v004a_coefficients_2026-06-26.csv` (sha256 见 policy json: 78fe05032ca0efd6bc931d0766f6427923ae52168a291c47c5e02145575b2117)
- 冻结策略: `configs/policy_v005_v1.json` (policy_version v005.1-frozen-20260626, policy_id policy_v005_v002_regime_fallback)
- v002 模型: `reports/manual_models/ranking_model_v002_core_momentum_support.json` (sha256 f4a0eaaf60341e4a9a3e729aed879c4a4d84b3dbf107282957493b252f7e07f3)
- v004a 模块: `src/v004a.py`;v004b 模块: `src/v004b.py`;v004b 报告: `reports/v004b/`(smoke/top10/top15 等研究分支)
- v005 链: `src/v005_*.py`, `reports/v005_*`, `reports/daily_v005/`
- **v5a**: 仓库中无独立 "v5a" 名称的代码/配置/文档(全库 grep 无命中)。最接近的是 v005 冻结策略链(policy_v005_v1.json + v005 modules + fixed-grid holdout 报告)。inventory 中如实说明。

## 6. 构建口径决定(将在报告与 feature definitions 中写明)

1. 主样本唯一键: (event_id, signal_date);需查重。
2. 主样本 = 通过身份过滤 + D1 数据完整 + 标签可计算的行;CANDIDATE。其余状态 LABEL_UNAVAILABLE / QUALITY_FAILED / EXCLUDED 只进审计表。
3. days_since_break 按个股日线交易日序列(相邻行)计;停牌日不伪造观察(suspended)。
4. signal_date 本身为涨停日的观察 → EXCLUDED(still_limit_up):断板修复观察只观察非板日。
5. recognition_score 为初步代理: 由 board_day 三个截面排名(amount/turnover/volume)的百分位均值组成,原始组成字段全部保留。
6. 断板类型: break_touched_limit_up(high ≥ 涨停价−容差)、break_opened_from_limit_up(open ≥ 涨停价−容差)按项目容差 0.011 判定。
7. 高位区 = bar typical price ≥ d1_low + 0.7×(d1_high−d1_low),typical price = (high+low+close)/3;尾盘区 = time ≥ 14:00;下跌 bar = close < open。
8. 金额类字段: 5min 求和优先(项目 `_merge_minute_amount` 口径)。
9. v004a 比较字段对 06-02..07-31 用现成文件;05-06..06-01 视只读重建核对结果决定。
10. 所有收益字段统一小数(0.07 = 7%)。

## 7. 数据源完整性清单 (source_files)

- data/cache/limit_ups/2026-05-06..2026-07-31 (62 文件)
- data/cache/daily/*_daily.pkl (1576)
- data/cache/minute_5m/*_5min.pkl (1576)
- data/cache/universe/eastmoney_main_board_universe.pkl
- reports/v004a/grid_v2_scored/v004a_scored_candidates.csv
- reports/v005_fixed_grid_holdout_2026-06-26_2026-06-30/v005_fixed_grid_holdout_scored_candidates.csv
- reports/v005_fixed_grid_holdout_2026-07-01_2026-07-03/v005_fixed_grid_holdout_scored_candidates.csv
- reports/daily_v005/2026-07-02..2026-07-31/v005_daily_scored_candidates_*.csv
- reports/daily_signals/signals_2026-06-25.csv, signals_2026-07-01..2026-07-31.csv
- reports/history_samples/2026-05-06_2026-06-29/history_candidates_2026-05-06_2026-06-29.csv
- configs/models/v004a_coefficients_2026-06-26.csv
- configs/policy_v005_v1.json
