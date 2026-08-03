# v004c 断板修复研究数据集 0.2 — 特征定义文档

- 数据集版本: `v004c-dataset-0.2`
- 候选定义版本: `break-repair-0.1`(与 0.1 相同,候选身份规则未变)
- 特征定义版本: `v004c-features-0.2`
- 研究标识: `model_id = model_v004c_break_repair_target7`, `model_role = break_repair_single_stock_ranker`
- 目标: `target7_d2open_d3high`;部署状态: `research_only`
- 生成日期: 2026-08-03;时区: Asia/Shanghai

## 与 v0.1 的关系

- v0.2 全量表保留 v0.1 全部 1020 个候选观察行,标签值不变(v0.1↔v0.2 标签变化: **0** 条)。
- v0.2 新增: D1/D2/D3 标签质量审计字段、sample_role 划分、阶段表达修正、MA5/MA10 语义修正、分桶研究表达、训练视图与去冗余。
- v0.1 中未变化的字段定义(断板判定、涨停判定、D1 修复原始连续值、对手盘原始值、标签口径)沿用 v0.1 定义文档,本文只记录 v0.2 变更与新增,并给出训练视图字段的完整定义。

## 通用口径(v0.2)

1. **收益字段统一小数**: 0.07 代表 7%。`target7_d2open_d3high = d2open_to_d3high_return >= 0.07`;`tail_loss_5pct = d2open_to_d3close_return <= -0.05`。
2. **涨停/连板/断板判定**: 同 v0.1(复制自 `src/loaders.py::_is_main_board_limit_up_day`;日线相邻交易日连续)。
3. **D2 开盘(标签分母)**: 沿用项目正式实现(`src/history_samples.py::_d2open_d3_metrics`)— D2 首根 5min bar 的 open,缺失回退日线 open。**不擅自切换为日线**。v0.2 审计同时检查: D2 首根 bar 是否为预期首根(09:35),并记录 `d2_daily_minute_open_diff = |日线D2开盘 − 5min首bar开盘|`。
4. **D3 最高/收盘(标签)**: 沿用项目正式实现 — D3 当日 5min bar 最高价最大值 / D3 末根 5min bar 收盘。v0.2 审计记录 `d3_daily_minute_high_diff` 与 `d3_daily_minute_close_diff`。
5. **预期分钟网格(本数据集规则,写入审计)**: 基于当前 5min 缓存实测(2026-04-01..07-31 全量 11624 个完整交易日,其中 11554 日为 48 根 bar,首 09:35,末 15:00):
   - `expected_bar_count = 48`(09:35..11:30 共 24 根 + 13:05..15:00 共 24 根)
   - `expected_first_bar_time = 09:35`, `expected_last_bar_time = 15:00`
   - `minute_complete = (bar_count == 48) AND (first == 09:35) AND (last == 15:00)`
   - 缺失早盘/尾盘、bar 数不足均视为不完整。不使用"bar 数 ≥ 30"作为完整标准。
6. **D2/D3 日期与统一交易日历**: 统一交易日历 = 涨停池缓存日期 ∪ 全部日线缓存日期(2026-05-06..07-31 窗口内 62 个交易日,经交叉验证无缺口)。
   - `expected_d2_date` = 统一日历中 signal_date 的下一交易日;`expected_d3_date` = expected_d2 的下一交易日。
   - `actual_d2_date` / `actual_d3_date` = 实际用于标签的个股交易日(个股 5min 序列,项目口径)。
   - 若 actual ≠ expected,只有存在**严格停牌证明**(`data/cache/suspension_status/` 缓存,`suspension_duration ∈ {连续停牌, 停牌一天}` 且区间覆盖期望日)才允许偏移(`d2_date_shift_reason = suspended_proven`);无证明 → `shift_without_proof` 且 `label_quality_ok = false`。**不得自动跳到下一个缓存日期。**
7. **label_quality_ok**: D1、D2、D3 三个交易日的分钟网格全部完整,且任何日期偏移都有严格停牌证明,且 expected_d2/expected_d3 存在(在日历范围内)。任一不满足 → `label_quality_ok = false`,原因记录在 `label_quality_reason`。**不完整 bar 的标签不得进入主训练表。**
8. **缺失值处理**: 不做均值/中位数/零值填充。除明确说明的字段外,缺失即留空。
9. **MA5/MA10 语义修正(v0.2 核心变更)**:
   - 旧 `d1_reclaimed_ma5/ma10`(仅 `close >= ma`)废弃,重命名为 `deprecated_d1_reclaimed_ma5_v01` / `deprecated_d1_reclaimed_ma10_v01` 仅审计保留。
   - 新增 `d1_close_above_ma5 = d1_close >= d1_ma5`(与旧字段公式相同,但语义命名正确)。
   - 新增 `d1_true_reclaim_ma5 = (d1_low <= d1_ma5) AND (d1_close >= d1_ma5)`(真正的"下探后收回 MA5")。
10. **阶段表达(v0.2 核心变更)**:
    - `stage_group`: days_since_break==0 → `d0`;∈{1,2} → `post`。
    - `post_day`: d0 样本为空;days_since_break==1 → 1;==2 → 2。
    - `training_stage_feature = "stage_group"`: 未来模型应使用 stage_group(阶段是离散分组,post_day 是组内细分;训练视图只保留 stage_group + post_day)。
    - `days_since_break` / `days_since_last_limit_up` / `repair_attempt_count` 为重复阶段表达,`feature_role = deprecated_redundant`,仅在全量表审计保留,不进训练视图。
11. **feature_role 取值**: `identity`(身份/分组)/ `training`(v0.2 训练视图候选因子)/ `audit_only`(仅审计,不进训练)/ `label`(标签)/ `label_quality`(标签质量审计)/ `weight_helper`(权重辅助)/ `deprecated_redundant`(冗余或废弃,全量表保留)/ `research_hold`(研究保留,未进入 v0.2 训练视图)。
12. **sample_role 取值**: `TRAIN_PRIMARY`(2026-06-01..07-29,标签质量合格)/ `SENSITIVITY_MAY_COMPLETE`(2026-05-06..05-31,标签质量合格;不进入首版主训练,原因 `minute_data_coverage_selection_bias`)/ `AUDIT_ONLY_MAY_INCOMPLETE`(五月候选事件但分钟/标签/特征不完整,只进审计)/ `EXCLUDED_QUALITY`(质量不合格或规则排除)。

---

## 训练视图字段(v004c_training_primary_20260601_20260729.csv)

### 身份和分组

| 字段 | 中文含义 | 公式/定义 | 来源 | D1收盘可得 | 缺失处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|
| event_id | 事件标识 | code + "_" + break_date | 派生 | 是 | 恒有 | 否 | 是(分组键) |
| code | 股票代码 | 6 位字符串 | 缓存 | 是 | 恒有 | 否 | 是(键) |
| name | 股票名称 | 断板日池名称→最新池名称→universe | 池/universe | 是 | 恒有 | 否 | 否(命名过拟合) |
| signal_date | 信号日(D1) | 断板日或其后第 1/2 交易日 | 派生 | 是 | 恒有 | 否 | 是(分组键) |
| break_date | 断板日 | 连续 2/3 板后首个非涨停日 | 派生 | 是 | 恒有 | 否 | 是 |
| stage_group | 阶段组 | days_since_break==0→d0;∈{1,2}→post | 派生 | 是 | 恒有 | 否 | 是(未来模型主阶段字段) |
| post_day | 阶段内天数 | d0 为空;1 或 2 | 派生 | 是 | d0 恒为空(定义) | 否 | 是 |
| board_streak_before_break | 断板前连板数 | ∈{2,3} | 派生 | 是 | 恒有 | 否 | 是 |

### MA 位置

| 字段 | 中文含义 | 公式/定义 | 来源 | D1收盘可得 | 缺失处理 | 极端值 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| d1_close_to_ma5_raw | 收盘对 MA5 距离 | d1_close / d1_ma5 − 1 | 日线派生 | 是 | ma 缺失→空 | 小数(非百分数) | 否 | 是 |
| d1_close_to_ma5_bucket | MA5 距离分桶(研究) | <−5% / −5~−2% / −2~+2% / +2~+5% / +5~+10% / ≥+10% | 派生 | 是 | 同 raw | 分桶有界 | 否 | 是(研究表达,非硬门槛) |
| d1_close_above_ma5 | 收盘在 MA5 上方 | d1_close >= d1_ma5 | 派生 | 是 | 恒有 | - | 否 | 是 |
| d1_true_reclaim_ma5 | 真正收回 MA5 | (d1_low <= d1_ma5) AND (d1_close >= d1_ma5) | 派生 | 是 | 恒有 | - | 否 | 是 |
| d1_close_to_ma10_raw | 收盘对 MA10 距离 | d1_close / d1_ma10 − 1 | 日线派生 | 是 | 同上 | 小数 | 否 | 是 |
| d1_close_above_ma10 | 收盘在 MA10 上方 | d1_close >= d1_ma10 | 派生 | 是 | 恒有 | - | 否 | 是 |
| d1_true_reclaim_ma10 | 真正收回 MA10 | (d1_low <= d1_ma10) AND (d1_close >= d1_ma10) | 派生 | 是 | 恒有 | - | 否 | 是 |

### D1 修复

| 字段 | 中文含义 | 公式/定义 | 来源 | D1收盘可得 | 缺失处理 | 极端值 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| d1_open_to_close_return_raw | 日内收益 | d1_close / d1_open − 1(5min 首/末 bar) | 5min | 是 | open 缺失→空 | 实测 [−0.17, 0.18] | 否 | 是 |
| d1_open_to_close_bucket | 日内收益分桶(研究) | <−5% / −5~−2% / −2~0% / 0~+2% / +2~+5% / ≥+5% | 派生 | 是 | 同 raw | 分桶有界 | 否 | 是(研究表达) |
| d1_high_to_close_drawdown_raw | 冲高回落 | (d1_high − d1_close) / d1_high | 5min | 是 | high 缺失→空 | 非负,≤1;异常进质量报告 | 否 | 是 |
| d1_close_location | 收盘位置 | (d1_close − d1_low) / (d1_high − d1_low) | 5min | 是 | high==low(一字板)→空 | [0,1] | 否 | 是 |
| d1_close_to_vwap_raw | 收盘对 VWAP | d1_close / d1_vwap − 1 | 5min | 是 | vwap 缺失→空 | - | 否 | 是 |

### 成交和抛压(每类只保留一个代表)

| 字段 | 中文含义 | 公式/定义 | 来源 | D1收盘可得 | 缺失处理 | 仅审计 | 可用于模型 | 被移除的孪生字段 |
|---|---|---|---|---|---|---|---|---|
| down_bar_volume_ratio | 下跌 bar 量占比 | Σvol(close<open) / Σvol | 5min | 是 | 总量 0→空 | 否 | 是 | d1_down_bar_volume_ratio(完全相同) |
| late_day_sell_volume_ratio | 尾盘抛压量占比 | Σvol(time≥14:00 且 close<open) / Σvol | 5min | 是 | 同上 | 否 | 是 | late_day_sell_amount_ratio |
| high_zone_volume_ratio | 高位区量占比 | Σvol(typical≥d1_low+0.7×(d1_high−d1_low)) / Σvol | 5min | 是 | 同上 | 否 | 是 | high_zone_amount_ratio |
| volume_above_d1_close_ratio | 收盘价上方量占比 | Σvol(bar close≥d1_close) / Σvol | 5min | 是 | 同上 | 否 | 是 | amount_above_d1_close_ratio |

### 断板结构

| 字段 | 中文含义 | 公式/定义 | 来源 | D1收盘可得 | 缺失处理 | 极端值 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|---|
| break_high_to_close_drawdown | 断板日冲高回落 | (break_high − break_close) / break_high | 日线派生 | 是 | high 缺失→空 | [0,1] | 否 | 是 |
| break_close_location | 断板日收盘位置 | (break_close − break_low) / (break_high − break_low) | 日线派生 | 是 | high==low→空 | [0,1] | 否 | 是 |
| break_volume_ratio_vs_board_days | 断板量比 | break_volume / mean(板日 volume) | 日线派生 | 是 | 板日缺失→空 | 一字板小量板日可大(≤33) | 否 | 是 |
| break_touched_limit_up | 断板日触板 | break_high ≥ 涨停价 − 0.011 | 派生 | 是 | 恒有 | - | 否 | 是 |
| break_opened_from_limit_up | 断板日开在涨停 | break_open ≥ 涨停价 − 0.011 | 派生 | 是 | 恒有 | - | 否 | 是 |

### 标签

| 字段 | 中文含义 | 公式/定义 | 来源 | D1收盘可得 | 缺失处理 | 仅审计 | 可用于模型 |
|---|---|---|---|---|---|---|---|
| d2_open | D2 开盘(标签分母) | D2 首根 5min bar open,缺失回退日线 open(项目口径) | 5min/日线 | 否 | 缺失→标签不可用 | 否 | 否(标签) |
| d3_high | D3 最高价 | D3 当日 5min high 最大值(项目口径) | 5min | 否 | 缺失→标签不可用 | 否 | 否(标签) |
| d3_close | D3 收盘价 | D3 末根 5min close(项目口径) | 5min | 否 | 同上 | 否 | 否(标签) |
| d2open_to_d3high_return | D2 开→D3 高收益 | d3_high / d2_open − 1 | 派生 | 否 | 恒有 | 否 | 否(标签) |
| d2open_to_d3close_return | D2 开→D3 收收益 | d3_close / d2_open − 1 | 派生 | 否 | 恒有 | 否 | 否(标签) |
| target7_d2open_d3high | 正式目标 | d2open_to_d3high_return ≥ 0.07 | 派生 | 否 | 恒有 | 否 | 否(标签) |
| tail_loss_5pct | 尾部亏损目标 | d2open_to_d3close_return ≤ −0.05 | 派生 | 否 | 恒有 | 否 | 否(标签) |

### 权重辅助与质量门

| 字段 | 中文含义 | 公式/定义 | 说明 |
|---|---|---|---|
| event_observation_count | 事件观察数 | 该 event_id 在 TRAIN_PRIMARY 中的行数 | 同一事件多次观察按事件加权 |
| signal_date_candidate_count | 当日候选数 | 该 signal_date 在 TRAIN_PRIMARY 中的行数 | 横截面日数加权 |
| proposed_training_weight | 建议训练权重 | 1 / (signal_date_candidate_count × event_observation_count),TRAIN_PRIMARY 内均值归一化为 1 | 只计算输出,不实际训练;均值=1.0,实测 [0.40, 5.80] |
| label_quality_ok / target_quality_ok / tail_label_quality_ok | 标签质量门 | 见通用口径 7;target/tail 各自的可计算性 | 训练表内恒为 true |
| training_stage_feature | 阶段字段声明 | "stage_group" | 未来模型使用 stage_group |

## 全量表保留但不进训练视图的字段

| 分组 | 字段 | feature_role | 说明 |
|---|---|---|---|
| 阶段冗余 | days_since_break, days_since_last_limit_up, repair_attempt_count | deprecated_redundant | 训练视图只用 stage_group/post_day;审计保留 |
| 辨识度(审计) | recognition_score, board_day_amount_rank, board_day_turnover_rank, board_day_volume_rank | audit_only | recognition 缺失率高(24%);单因子排序能力不足;不作为训练输入 |
| 现有模型对照(审计) | v004a_probability, v004a_rank, v002_rank, is_v004a_scorable/top3/10/15, is_v002_scorable/top3/10/15, v004a_score_source | audit_only | 现有模型覆盖不完整;避免 v004c 成为 v004a 下游重排模型 |
| 涨停频率(研究保留) | recent_limit_up_count_10d/20d, recent_pool_appearance_count_10d/20d, max_board_streak_20d | research_hold | 未进入 v0.2 训练视图,保留供后续研究 |
| 冗余成交额 | amount_above_d1_close_ratio, high_zone_amount_ratio, late_day_sell_amount_ratio, break_amount_ratio_vs_board_days, d1_down_bar_volume_ratio | deprecated_redundant | 与保留 volume 字段 |spearman| ≥ 0.99(见冗余报告);全量表保留 |
| VWAP 反向 | d1_vwap_to_close_gap | deprecated_redundant | 与 d1_close_to_vwap_raw 完全反向(spearman −1.0),只保留一种方向 |
| 废弃 reclaimed | deprecated_d1_reclaimed_ma5_v01, deprecated_d1_reclaimed_ma10_v01 | deprecated_redundant | 与 d1_close_above_* 完全相同(pearson 1.0),命名错误已废弃 |
| 其余原始字段 | break_* 原始 OHLC、d1_* 原始值、d1_ma5/10/20、d1_ma5_slope/ma10_slope、consecutive_days_below_*、d1_low_to_close_recovery、d1_intraday_range、d1_afternoon_return、d1_last_hour_return、d1_up_bar_volume_ratio、volume_above_break_close_ratio、d1_amount 等 | research_hold / identity | 保留研究;d1_ma5/10/20 等原始锚点可作后续因子来源 |

## 标签质量审计字段(全量表)

| 字段 | 含义 | 规则 |
|---|---|---|
| d1_bar_count / d2_bar_count / d3_bar_count | 各日 5min bar 数 | 实测: D1 全部 48;D2/D3 各 1/2 行不足 48 |
| d1/2/3_first_bar_time, d1/2/3_last_bar_time | 各日首末 bar 时间 | 完整日 = 09:35 / 15:00 |
| d1/2/3_expected_bar_count | 预期 bar 数 | 48(实测网格规则,见通用口径 5) |
| d1/2/3_minute_complete | 各日分钟完整性 | bar=48 且首 09:35 且末 15:00 |
| d2_daily_minute_open_diff | D2 日线/分钟开盘差异 | 记录,不设硬阈值;大差异提示数据源不一致 |
| d3_daily_minute_high_diff | D3 日线/分钟最高差异 | 同上 |
| d3_daily_minute_close_diff | D3 日线/分钟收盘差异 | 同上 |
| expected_d2_date / actual_d2_date / expected_d3_date / actual_d3_date | D2/D3 期望与实际日期 | 期望日期 = 统一日历下一交易日 |
| d2_date_shift_reason / d3_date_shift_reason | 偏移原因 | none / suspended_proven / shift_without_proof |
| suspension_proof_status | 停牌证明状态 | not_applicable / proven / not_proven |
| label_quality_ok | 标签质量总开关 | 见通用口径 7 |
| label_quality_reason | 失败原因 | 竖线分隔的具体原因 |
| target_quality_ok / tail_label_quality_ok | 两个目标各自的质量门 | label_quality_ok 且相应收益可计算 |

## 备注

1. 本任务未训练任何模型;`proposed_training_weight` 仅为建议权重。
2. MA5 分桶与日内收益分桶是研究表达,不是交易硬门槛,未据此删除任何样本。
3. v0.2 标签值与 v0.1 完全一致(v0.1↔v0.2 Target7/tail_loss 变化 = 0);v0.2 的变化在于标签质量审计、因子语义、命名与训练视图划分。
4. 数据修订风险与 v0.1 相同: 本地缓存可能随时间修订,训练时应以本数据包哈希为准。
