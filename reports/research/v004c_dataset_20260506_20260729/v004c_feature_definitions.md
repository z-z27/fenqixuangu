# v004c 断板修复研究数据集 — 特征定义文档

- 数据集版本: `v004c-dataset-0.1`
- 候选定义版本: `break-repair-0.1`
- 特征定义版本: `v004c-features-0.1`
- 研究标识: `model_id = model_v004c_break_repair_target7`, `model_role = break_repair_single_stock_ranker`
- 生成日期: 2026-08-03
- 时区: Asia/Shanghai (A 股交易日历)

## 通用口径

1. **收益字段统一使用小数**: 0.07 代表 7%。`target7_d2open_d3high = True` 当且仅当 `d2open_to_d3high_return >= 0.07`。
2. **涨停判定**: 复制自 `src/loaders.py::_is_main_board_limit_up_day`(2026-08-03 只读): 收盘价 ≥ 涨停价 − 0.011 且 涨幅 ≥ 9.7% 且 最高价 ≥ 涨停价 − 0.011;涨停价 = 前收 × 1.10 四舍五入到分。ST 股票(5% 板)不会计入,与项目口径一致。
3. **连板/断板**: 在个股日线相邻交易日序列上连续涨停(严格相邻)。断板日 = 连续涨停序列(2 或 3)结束后第一个非涨停日。个股相邻交易日由该股日线缓存行决定,停牌日自然不在序列中。与 `signal_engine._count_consecutive_boards` 的池子粗口径(gap≤2 日历天,存在周末跨天切断问题)不同,本数据集采用日线相邻交易日口径,见 review 文档。
4. **D1 分钟质量**: 信号日 5min bar 数 ≥ 30(全日 48 根)、总量 > 0、日线/5min 收盘差异 ≤ 0.02。这是对项目 `_build_quality_report`(窗口级)的逐日适配,见 review 文档。
5. **金额口径**: 日线 amount 优先用该日 5min 求和(要求该日 5min bar ≥ 30 根),否则日线 amount,否则空。这是对 `src/loaders.py::_merge_minute_amount`(只要有 5min 就替换)的质量收紧: 5min 覆盖不完整时求和会失真(实测出现 795× 伪值)。
6. **除零处理**: 除数为 0、分母为空、high==low 时相应字段置空(NaN),不产生 inf;文档内已注明的除外。
7. **高位区**: bar typical price ≥ `d1_low + 0.7 × (d1_high − d1_low)`;typical price = (high+low+close)/3。
8. **尾盘区**: time ≥ 14:00 的 bar。**下跌 bar**: close < open。**上涨 bar**: close > open。
9. **D3 最高价口径**: 沿用项目定义 — D3 当日 5min bar 最高价的最大值(分钟级盘中最高),**不是** D2→D3 持有窗口最高价(`src/history_samples.py::_d2open_d3_metrics`)。D3 close = D3 最后一根 5min bar 的 close。D2 open = D2 首根 5min bar 的 open,缺失回退日线 open(项目相同)。
10. **D2/D3 日期**: 该股 5min 交易日序列中 signal_date 之后的第 1、2 个交易日(`_future_trade_dates` 口径),自动跳过停牌日。
11. **比较字段语义**:
    - v004a 现有评分: 2026-06-02..06-29 为 grid_v2 walk-forward 逐日折;2026-06-26..06-30 与 2026-07-01..07-03 为 fixed-grid holdout(冻结 2026-06-26 折);2026-07-02..07-31 为 daily_v005(冻结 2026-06-26 折)。缺口 2026-05-07..06-01 用冻结系数只读重建(frozen_reconstruction),重建概率/排名与 holdout 冻结评分完全一致(概率差异 ≤ 2e-16,排名一致率 1.0000,见 `_scratch/v004a_recon_validation.csv`)。2026-05-06 的 scorable 池在现有 history candidates 中为空,无评分。
    - v002: 来自现成 daily signals 的 daily_rank(文件内 trade_date 为准),覆盖 2026-06-30、07-02..07-31 的部分日期;其余日期无现成 v002 排序,留空(不重跑模型)。
12. **每个字段的"可用性"**:
    - `available_at_d1_close = True`: 仅用 ≤ signal_date 的数据,可在 D1 收盘时计算。
    - `available_at_d1_close = False`: 需要未来数据(标签)。
    - `audit only`: 仅为审计/交叉核对保留,不建议进入模型。

---

## A. 身份和日期

| 字段 | 中文含义 | 数学公式/定义 | 数据来源 | D1收盘可得 | 缺失处理 | 极端值处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| event_id | 事件标识 | code + "_" + break_date | 派生 | 是 | 恒有 | - | 否 | 是(身份键) |
| code | 股票代码 | 6 位字符串 | 缓存文件名 | 是 | 恒有 | - | 否 | 是(键) |
| name | 股票名称 | 断板日涨停池名称 → 最新池名称 → universe | 涨停池/universe | 是 | 恒有 | - | 否 | 否(避免命名过拟合) |
| signal_date | 信号日(D1) | 断板日或其后第 1/2 个交易日 | 派生 | 是 | 恒有 | - | 否 | 是(键) |
| break_date | 断板日 | 连续 2/3 板后第一个非涨停日 | 派生 | 是 | 恒有 | - | 否 | 是 |
| board_streak_before_break | 断板前连板数 | 断板日前连续涨停天数 ∈ {2,3} | 派生 | 是 | 恒有 | - | 否 | 是 |
| days_since_break | 距断板日天数 | 0/1/2 = 断板当日/次日/再次日(个股交易日) | 派生 | 是 | 恒有 | - | 否 | 是 |
| days_since_last_limit_up | 距最近涨停日天数 | (signal_date 日线行索引 − 断板前最后涨停行索引) | 派生 | 是 | 恒有 | - | 否 | 是 |
| repair_attempt_count | 修复尝试次数 | 事件内 ≤ 当前 signal_date 的"非涨停观察槽"数(涨停日不算尝试) | 派生 | 是 | 恒有 | - | 否 | 是 |

## B. 连板和辨识度代理

| 字段 | 中文含义 | 数学公式/定义 | 数据来源 | D1收盘可得 | 缺失处理 | 极端值处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| recent_limit_up_count_10d | 近10日涨停次数 | 信号日(含)往前 10 个交易日中日线判定涨停天数 | 日线派生 | 是 | 恒有 | - | 否 | 是 |
| recent_limit_up_count_20d | 近20日涨停次数 | 信号日(含)往前 20 个交易日中日线判定涨停天数 | 日线派生 | 是 | 恒有 | - | 否 | 是 |
| recent_pool_appearance_count_10d | 近10日涨停池出现次数 | 同一窗口内该股出现在涨停池的天数 | 涨停池 | 是 | 2026-05-06 前池无数据,计数只统计有池覆盖的日期(早 5 月样本偏小,已在质量报告说明) | - | 否 | 是(注意覆盖偏倚) |
| recent_pool_appearance_count_20d | 近20日涨停池出现次数 | 同上,20 日窗口 | 涨停池 | 是 | 同上 | - | 否 | 是(注意覆盖偏倚) |
| max_board_streak_20d | 近20日最大连板数 | 20 日窗口内最长的连续涨停天数 | 日线派生 | 是 | 恒有 | - | 否 | 是 |
| board_day_amount_rank | 末板日成交额排名 | 断板前最后涨停日,该股 amount 在该日涨停池成员中的降序名次(1=最高);amount 取池 amount,缺失取日线 | 涨停池/日线 | 是 | 末板日 < 2026-05-06(无池)或值缺失 → 空 | 排名天然有界 | 否 | 是 |
| board_day_turnover_rank | 末板日换手率排名 | 同上,按 turnover_rate 降序 | 涨停池/日线 | 是 | 同上 | - | 否 | 是 |
| board_day_volume_rank | 末板日成交量排名 | 同上,按日线 volume 降序 | 日线 | 是 | 同上 | - | 否 | 是 |
| recognition_score | 辨识度初步代理 | mean(pct_amount, pct_turnover, pct_volume),pct = (N − rank + 1)/N | 派生 | 是 | 任一组成缺失 → 空(不静默代替) | 定义域 [0,1] | 否 | 仅作初步代理,不证明有效 |

## C. 断板日 OHLC 和结构

| 字段 | 中文含义 | 数学公式/定义 | 数据来源 | D1收盘可得 | 缺失处理 | 极端值处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| break_open/high/low/close | 断板日 OHLC | 日线值 | 日线缓存 | 是 | 恒有(候选要求 OHLC>0) | - | 否 | 是 |
| break_prev_close | 断板日昨收 | 末板日收盘 | 日线缓存 | 是 | 恒有 | - | 否 | 是 |
| break_open_return | 断板日开盘收益 | open / prev_close − 1 | 派生 | 是 | prev_close 缺失/≤0 → 空 | - | 否 | 是 |
| break_high_return | 断板日最高收益 | high / prev_close − 1 | 派生 | 是 | 同上 | - | 否 | 是 |
| break_close_return | 断板日收盘收益 | close / prev_close − 1 | 派生 | 是 | 同上 | - | 否 | 是 |
| break_intraday_range | 断板日振幅 | (high − low) / prev_close | 派生 | 是 | 同上 | - | 否 | 是 |
| break_high_to_close_drawdown | 断板日冲高回落 | (high − close) / high | 派生 | 是 | high 缺失 → 空 | 定义域 [0,1] | 否 | 是 |
| break_upper_shadow_ratio | 上影线占比 | (high − max(open,close)) / (high − low) | 派生 | 是 | high==low → 空 | 定义域 [0,1] | 否 | 是 |
| break_lower_shadow_ratio | 下影线占比 | (min(open,close) − low) / (high − low) | 派生 | 是 | high==low → 空 | 定义域 [0,1] | 否 | 是 |
| break_close_location | 收盘位置 | (close − low) / (high − low) | 派生 | 是 | high==low → 空(中性定义: 不猜) | 定义域 [0,1] | 否 | 是 |
| break_volume | 断板日成交量 | 日线 volume(手) | 日线缓存 | 是 | 恒有 | - | 否 | 是 |
| break_amount | 断板日成交额 | 5min 求和(≥30 bar)否则日线 amount(元) | 5min/日线 | 是 | 均缺失 → 空(17 行) | - | 否 | 是 |
| break_volume_ratio_vs_board_days | 断板日量比(对板日) | break_volume / mean(断板前 streak 个板日 volume) | 派生 | 是 | 板日 volume 缺失 → 空 | 一字板小量板日可致大值(实测 ≤ 33) | 否 | 是 |
| break_amount_ratio_vs_board_days | 断板日额比(对板日) | break_amount / mean(板日 amount) | 派生 | 是 | 板日 amount 缺失 → 空(28 行) | 5min 覆盖不足的板日已过滤(修正后 ≤ 36) | 否 | 是 |
| break_turnover_ratio | 断板日换手比(对板日) | break_turnover_rate / mean(板日 turnover_rate) | 派生 | 是 | **全空**: 日线缓存(tencecent_daily)无 turnover_rate,池 turnover 只覆盖涨停日 | - | 否 | 否(字段不可用) |
| break_touched_limit_up | 断板日是否触板 | high ≥ 涨停价(前收×1.10) − 0.011 | 派生 | 是 | 恒有 | - | 否 | 是 |
| break_opened_from_limit_up | 断板日是否开在涨停 | open ≥ 涨停价 − 0.011 | 派生 | 是 | 恒有 | - | 否 | 是 |

## D. D1 日线与均线位置

| 字段 | 中文含义 | 数学公式/定义 | 数据来源 | D1收盘可得 | 缺失处理 | 极端值处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| d1_open/high/low/close | D1(信号日)OHLC | 日线值 | 日线缓存 | 是 | 恒有(候选要求 OHLC>0) | - | 否 | 是 |
| d1_volume | D1 成交量 | 日线 volume(手) | 日线缓存 | 是 | 恒有 | - | 否 | 是 |
| d1_amount | D1 成交额 | 同 break_amount 口径 | 5min/日线 | 是 | 恒有(信号日 5min ≥30 bar 保证) | - | 否 | 是 |
| d1_ma5 / d1_ma10 / d1_ma20 | D1 均线 | close 的 5/10/20 日滚动均值(min_periods=周期) | 日线派生 | 是 | 历史不足 → 候选被排除 | - | 否 | 是 |
| d1_close_to_ma5 | D1 收盘对 MA5 | close / ma5 − 1 | 派生 | 是 | ma 缺失 → 空 | - | 否 | 是 |
| d1_low_to_ma5 | D1 最低对 MA5 | low / ma5 − 1 | 派生 | 是 | 同上 | - | 否 | 是 |
| d1_high_to_ma5 | D1 最高对 MA5 | high / ma5 − 1 | 派生 | 是 | 同上 | - | 否 | 是 |
| d1_close_to_ma10 | D1 收盘对 MA10 | close / ma10 − 1 | 派生 | 是 | 同上 | - | 否 | 是 |
| d1_low_to_ma10 | D1 最低对 MA10 | low / ma10 − 1 | 派生 | 是 | 同上 | - | 否 | 是 |
| d1_close_to_ma20 | D1 收盘对 MA20 | close / ma20 − 1 | 派生 | 是 | 同上 | - | 否 | 是 |
| d1_ma5_slope | MA5 斜率 | ma5_today / ma5_prev − 1(1 日变化率) | 派生 | 是 | 前一日 ma 缺失 → 空 | - | 否 | 是 |
| d1_ma10_slope | MA10 斜率 | ma10_today / ma10_prev − 1 | 派生 | 是 | 同上 | - | 否 | 是 |
| d1_reclaimed_ma5 | D1 收回 MA5 | close ≥ ma5 | 派生 | 是 | 恒有 | - | 否 | 是 |
| d1_reclaimed_ma10 | D1 收回 MA10 | close ≥ ma10 | 派生 | 是 | 恒有 | - | 否 | 是 |
| consecutive_days_below_ma5 | 连续低于 MA5 天数 | 从 signal_date(含)向前数 close < ma5 的连续天数 | 派生 | 是 | 恒有 | - | 否 | 是 |
| consecutive_days_below_ma10 | 连续低于 MA10 天数 | 同上(ma10) | 派生 | 是 | 恒有 | - | 否 | 是 |

## E. D1 修复质量(5min)

| 字段 | 中文含义 | 数学公式/定义 | 数据来源 | D1收盘可得 | 缺失处理 | 极端值处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| d1_open_to_close_return | D1 日内收益 | close / open − 1(5min 首/末 bar) | 5min | 是 | open 缺失 → 空 | - | 否 | 是 |
| d1_low_to_close_recovery | 低点回收 | (close − low) / low(5min 日内低点) | 5min | 是 | low 缺失 → 空 | 定义域 ≥ −1 | 否 | 是 |
| d1_high_to_close_drawdown | 冲高回落 | (high − close) / high | 5min | 是 | high 缺失 → 空 | 定义域 [0,1] | 否 | 是 |
| d1_close_location | 收盘位置 | (close − low) / (high − low) | 5min | 是 | high==low(一字板)→ 空 | [0,1] | 否 | 是 |
| d1_vwap | D1 日内 VWAP | 5min 累计额/累计量(手数自适应,同 `indicators.infer_vwap`)末值 | 5min | 是 | 恒有(候选要求) | - | 否 | 是 |
| d1_close_to_vwap | D1 收盘对 VWAP | close / vwap − 1 | 派生 | 是 | 恒有 | - | 否 | 是 |
| d1_intraday_range | D1 振幅 | (high − low) / prev_close | 5min/日线 | 是 | prev_close 缺失 → 空 | - | 否 | 是 |
| d1_afternoon_return | 午后收益 | close / 13:00 前最后一根 bar 的 close − 1 | 5min | 是 | 无午前 bar → 空 | - | 否 | 是 |
| d1_last_hour_return | 尾盘 1 小时收益 | close / 14:00 bar 的 close − 1 | 5min | 是 | 无 14:00 bar → 空 | - | 否 | 是 |
| d1_up_bar_volume_ratio | 阳线量占比 | Σvolume(close>open) / Σvolume | 5min | 是 | 总量 0 → 空 | [0,1] | 否 | 是 |
| d1_down_bar_volume_ratio | 阴线量占比 | Σvolume(close<open) / Σvolume | 5min | 是 | 同上 | [0,1] | 否 | 是 |

## F. 对手盘和成交压力代理(5min)

| 字段 | 中文含义 | 数学公式/定义 | 数据来源 | D1收盘可得 | 缺失处理 | 极端值处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| volume_above_d1_close_ratio | D1 收盘价上方量占比 | Σvolume(bar close ≥ d1_close) / Σvolume | 5min | 是 | 总量 0 → 空 | [0,1] | 否 | 是 |
| amount_above_d1_close_ratio | D1 收盘价上方额占比 | 同上,按 amount | 5min | 是 | 同上 | [0,1] | 否 | 是 |
| volume_above_break_close_ratio | 断板收盘价上方量占比 | Σvolume(bar close ≥ break_close) / Σvolume | 5min | 是 | 同上 | [0,1] | 否 | 是 |
| high_zone_volume_ratio | 高位区量占比 | Σvolume(typical ≥ d1_low+0.7×(d1_high−d1_low)) / Σvolume | 5min | 是 | 同上 | [0,1] | 否 | 是 |
| high_zone_amount_ratio | 高位区额占比 | 同上,按 amount | 5min | 是 | 同上 | [0,1] | 否 | 是 |
| late_day_sell_volume_ratio | 尾盘下跌量占比 | Σvolume(time≥14:00 且 close<open) / Σvolume | 5min | 是 | 同上 | [0,1] | 否 | 是 |
| late_day_sell_amount_ratio | 尾盘下跌额占比 | 同上,按 amount | 5min | 是 | 同上 | [0,1] | 否 | 是 |
| down_bar_volume_ratio | 全天下跌量占比 | Σvolume(close<open) / Σvolume | 5min | 是 | 同上 | [0,1] | 否 | 是 |
| d1_vwap_to_close_gap | VWAP 对收盘差 | (vwap − close) / close(正 = VWAP 在上 = 卖压) | 派生 | 是 | vwap 缺失 → 空 | 与 d1_close_to_vwap 互补 | 否 | 是 |

## G. 与现有模型的比较字段

| 字段 | 中文含义 | 数学公式/定义 | 数据来源 | D1收盘可得 | 缺失处理 | 极端值处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| is_v004a_scorable | 是否在 v004a 可评分池 | 该日 scorable 池是否包含该 code(现成文件行 / 只读重建行) | 现有评分文件/重建 | 是 | 恒有(False 或 True) | - | 否 | 是(对照) |
| v004a_probability | v004a 概率 | 现成评分(来源见通用口径 11)或冻结 06-26 折只读重建 | 现有评分文件/重建 | 是 | 非 scorable → 空(43.7%) | 定义域 (0,1) | 否 | 是(对照) |
| v004a_rank | v004a 当日排名 | 按概率降序的当日排名(现成口径) | 同上 | 是 | 同上 | - | 否 | 是(对照) |
| is_v004a_top3/10/15 | 是否进入 v004a 前 3/10/15 | v004a_rank ≤ 3/10/15 | 派生 | 是 | 恒有(非 scorable 为 False) | - | 否 | 是(对照) |
| is_v002_scorable | 是否在 v002 排序中 | 现成 daily signals 该日是否含该 code 且有 daily_rank | 现成 signals CSV | 是 | 恒有 | - | 否 | 是(对照) |
| v002_rank | v002 当日排名 | 现成 daily signals daily_rank(文件内 trade_date 为准) | 现成 signals CSV | 是 | 无现成评分日期 → 空(80.4%) | - | 否 | 是(对照) |
| is_v002_top3/10/15 | 是否 v002 前 3/10/15 | v002_rank ≤ 3/10/15 | 派生 | 是 | 恒有 | - | 否 | 是(对照) |

## H. 标签字段

| 字段 | 中文含义 | 数学公式/定义 | 数据来源 | D1收盘可得 | 缺失处理 | 极端值处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| d2_trade_date / d3_trade_date | D2/D3 交易日 | 该股 5min 序列中 signal_date 后第 1/2 个交易日 | 5min | 否 | 无 → 标签不可用 | - | 否 | 否(标签) |
| d2_open | D2 开盘价 | D2 首根 5min bar open,缺失回退日线 open | 5min/日线 | 否 | 均缺失 → 标签不可用 | - | 否 | 否(标签) |
| d2_high / d2_low / d2_close | D2 高/低/收 | 5min 最大/最小/末根,缺失回退日线 | 5min/日线 | 否 | 同上 | - | 否 | 否(标签) |
| d3_high | D3 最高价 | **D3 当日 5min high 最大值(项目口径)** | 5min | 否 | 缺失 → 标签不可用 | - | 否 | 否(标签) |
| d3_low | D3 最低价 | 5min 最小值,缺失回退日线(补充字段) | 5min/日线 | 否 | 同上 | - | 否 | 否(标签) |
| d3_close | D3 收盘价 | D3 末根 5min close(项目口径) | 5min | 否 | 缺失 → 标签不可用 | - | 否 | 否(标签) |
| d2open_to_d3high_return | D2 开→D3 高收益 | d3_high / d2_open − 1 | 派生 | 否 | 恒有(候选要求) | 实测 [−0.32, 0.28] | 否 | 否(标签) |
| d2open_to_d3close_return | D2 开→D3 收收益 | d3_close / d2_open − 1 | 派生 | 否 | 恒有 | 实测 [−0.36, 0.28] | 否 | 否(标签) |
| target7_d2open_d3high | 正式目标 | d2open_to_d3high_return ≥ 0.07 | 派生 | 否 | 恒有 | - | 否 | 否(标签) |
| tail_loss_5pct | 尾部亏损 5% | d2open_to_d3close_return ≤ −0.05 | 派生 | 否 | 恒有 | - | 否 | 否(标签) |
| future_data_status | 未来数据状态 | 主样本恒为 "complete" | 派生 | 否 | 恒有 | - | 否 | 否 |

## 审计专用字段(不建议进入模型)

| 字段 | 含义 | 说明 |
|---|---|---|
| last_board_date | 末板日 | 断板前最后涨停日,与池交叉核对用 |
| break_day_in_pool | 断板日是否在涨停池 | 断板日理论上不在池中(收盘未封),用于交叉核对 |
| last_board_day_in_pool | 末板日是否在涨停池 | 末板日应在池中 |
| pool_consecutive_count_last_board | 池 consecutive_limit_up_count | akshare 池口径连板数,与日线相邻口径交叉核对(见 review 争议说明) |
| v004a_score_source | v004a 评分来源 | grid_v2_walk_forward / holdout_0626_frozen / holdout_0701_frozen / daily_v005_frozen / frozen_reconstruction / 空 |

## 缺失率最高的字段(质量报告摘要)

| 字段 | 缺失率 | 原因 |
|---|---|---|
| break_turnover_ratio | 100% | 日线缓存(tencent_daily)无 turnover_rate |
| v002_rank | 80.4% | 现成 v002 排序仅覆盖 06-30、07-02 起的部分日期 |
| v004a_probability / rank | 42.7% | 该日该 code 不在 v004a scorable 池(含 2026-05-06 无任何池) |
| board_day_amount/turnover_rank + recognition_score | 24.0% | 末板日早于 2026-05-06(池无数据)或值缺失 |
| pool_consecutive_count_last_board | 3.8% | 末板日不在池中 |
| board_day_volume_rank | 3.8% | 末板日不在池中(排名池按池成员) |
