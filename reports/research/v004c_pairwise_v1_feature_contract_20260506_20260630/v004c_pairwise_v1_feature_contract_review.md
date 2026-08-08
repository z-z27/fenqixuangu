# v004c Pairwise v1 Feature Contract Review (May+June 冻结)

- 输出: v004c_pairwise_v1_feature_contract_20260506_20260630 (两轮构建同路径, 保证 review 字节一致)
- 窗口: 2026-05-06 .. 2026-06-30 (May 146 行 / 18 信号日, June 173 行 / 21 信号日)
- universe: 319 行 / 39 信号日
- 与 v002 身份逐行一致 (event_id/code/signal_date/break_date/board_streak_before_break/source_window): PASS
- FEATURE 数: 53

## 1. 信息边界 (available by D1 close)

所有 FEATURE `available_as_of <= D1_CLOSE`。允许 D1 当日信息、最近 7 日详细路径、MA5/10/20、10/20 日历史状态、连板历史、涨停池历史、历史成交状态。标签 (target7/tail_loss) 与 D2/D3 价格被物理隔离在 X 之后, 仅在 Pairwise 训练时附加。
- 违反 D1 边界的 FEATURE: 无

## 2. 特征清单

- 候选清单元数据: v004c_pairwise_v1_feature_inventory.csv (21 列)
- FEATURE 契约: v004c_pairwise_v1_feature_contract.csv (53 行, feature_order 1..53)
- 标签契约: v004c_pairwise_v1_label_contract.csv (LABEL_ONLY)

### 53 个 FEATURE 按 semantic_group

- **BOARD_HISTORY** (6): board_streak_is_3, max_board_streak_20d, recent_limit_up_count_10d, recent_limit_up_count_20d, recent_pool_appearance_count_10d, recent_pool_appearance_count_20d
- **D0_CROSS_SECTION** (1): board_day_volume_rank
- **D1_PRICE_ACTION** (14): break_open_return, break_high_return, break_close_return, break_touched_limit_up, break_opened_from_limit_up, break_upper_shadow_ratio, break_lower_shadow_ratio, d1_high_to_close_drawdown_raw, d1_low_to_close_recovery, d1_open_to_close_return_raw, d1_close_location, d1_intraday_range, d1_afternoon_return, d1_last_hour_return
- **D1_VOLUME_ACTIVITY** (3): break_volume_ratio_vs_board_days, d1_up_bar_volume_ratio, down_bar_volume_ratio
- **D1_MA_POSITION** (15): d1_close_to_ma5_raw, d1_low_to_ma5_raw, d1_high_to_ma5_raw, d1_close_to_ma10_raw, d1_low_to_ma10_raw, d1_close_to_ma20, d1_ma5_slope, d1_ma10_slope, d1_ma20_slope, d1_close_above_ma5, d1_close_above_ma10, d1_true_reclaim_ma5, d1_true_reclaim_ma10, consecutive_days_below_ma5, consecutive_days_below_ma10
- **D1_CHIP_DISTRIBUTION** (4): volume_above_d1_close_ratio, amount_above_d1_close_ratio, high_zone_volume_ratio, high_zone_amount_ratio
- **D1_LATE_DAY_PRESSURE** (2): late_day_sell_volume_ratio, late_day_sell_amount_ratio
- **D1_VWAP_POSITION** (1): d1_close_to_vwap_raw
- **POOL_MEMBERSHIP** (3): break_day_in_pool, last_board_day_in_pool, pool_consecutive_count_last_board
- **RECENT_7D_PATH** (4): recent_7d_cumulative_return, recent_7d_max_drawdown, recent_7d_close_position, recent_7d_limit_up_count

### 按 information_class

- D1_STRUCTURE: 24
- KNOWN_STATE: 25
- RECENT_PATH: 4

### 重建口径 (19 个缺失特征)

以下特征不在 v002 冻结列中, 由本工具从日线缓存 + 涨停池缓存确定性重建, 公式与 _scratch/build_v004c_dataset.py 官方定义逐条对齐:
- 重建特征 (19): board_streak_is_3, max_board_streak_20d, recent_limit_up_count_10d, recent_limit_up_count_20d, recent_pool_appearance_count_10d, recent_pool_appearance_count_20d, board_day_volume_rank, break_high_return, break_close_return, break_touched_limit_up, break_opened_from_limit_up, break_upper_shadow_ratio, break_lower_shadow_ratio, break_volume_ratio_vs_board_days, break_day_in_pool, last_board_day_in_pool, pool_consecutive_count_last_board, d1_ma20_slope, recent_7d_limit_up_count

### 旧 CONTEXT_ONLY 审查

v004c_model_table.py 中 CONTEXT_ONLY_COLUMNS (recent_limit_up_count_10d/20d, recent_pool_appearance_count_10d/20d, max_board_streak_20d) 定义正确、D1_CLOSE 可得、不依赖模型输出, 现 PROMOTE 为 FEATURE (信息边界允许的 10/20 日历史状态)。

### 旧 SENSITIVITY 审查

SENSITIVITY_FEATURE_COLUMNS 中 board_day_amount_rank / board_day_turnover_rank 因 May 池 amount/turnover 全 NaN 且 daily amount 全 NaN 而 EXCLUDE_SOURCE_UNAVAILABLE; break_amount_ratio_vs_board_days 分母 (board 日 amount) 在 May 不可重建, 同样排除。其余 SENSITIVITY 特征全部进入 FEATURE (定义正确且可得)。

### 手动综合因子排除

recognition_score (旧模型综合评分), profit_pressure, overrepair, break_volume_abnormality, ma5_overheat_10, d1_close_to_ma5_bucket, d1_open_to_close_bucket, d1_vwap_to_close_gap, Repair-State composites (DIVERGENCE/RECLAIM/DAMAGE/SUPPLY/DIVERGENCE_SQ/DIVERGENCE_X_*), v004a/v004b handcrafted nonlinear features 全部不进契约 (EXCLUDE_MANUAL_COMPOSITE / EXCLUDE_REDUNDANT_DETERMINISTIC_TRANSFORM / EXCLUDE_MODEL_OUTPUT)。

### 精确重复排除

ED 组 break_open/d1_open, break_high/d1_high, break_low/d1_low, break_close/d1_close, break_intraday_range/d1_intraday_range, break_high_to_close_drawdown/d1_high_to_close_drawdown_raw, break_close_location/d1_close_location, break_volume/d1_volume, break_amount/d1_amount, d1_down_bar_volume_ratio/down_bar_volume_ratio, volume_above_break_close_ratio/volume_above_d1_close_ratio, deprecated_d1_reclaimed_ma5_v01/d1_close_above_ma5, deprecated_d1_reclaimed_ma10_v01/d1_close_above_ma10 只保留 canonical 一份。v002 中除 canonical 名外的别名一律不进入输入表。

### 特征缺失

- structural_missing_count 合计: 35 (定义允许的数学退化 / 池窗口外 / 池源覆盖缺口)
- unexpected_missing_count 合计: 0 (必须为 0)
- A) D0=2026-04-30 池窗口外事件 (6): board_day_volume_rank, last_board_day_in_pool, pool_consecutive_count_last_board 各 6 个结构缺失; recent_pool_appearance_count 按池覆盖子集计算 (下界)。
- B) 池源覆盖缺口事件 (7): D0 日线涨停但涨停池未收录该股, board_day_volume_rank 与 pool_consecutive_count_last_board 各 7 个结构缺失; last_board_day_in_pool=False 为合法值 (池成员记录缺)。
- C) 一字板 603065 事件 (1): break_upper_shadow_ratio / break_lower_shadow_ratio / d1_close_location 各 1 个结构缺失 (high==low 分母退化)。
  - board_streak_is_3: May 1.0000 | June 1.0000 | Combined 1.0000
  - max_board_streak_20d: May 1.0000 | June 1.0000 | Combined 1.0000
  - recent_limit_up_count_10d: May 1.0000 | June 1.0000 | Combined 1.0000
  - recent_limit_up_count_20d: May 1.0000 | June 1.0000 | Combined 1.0000
  - recent_pool_appearance_count_10d: May 1.0000 | June 1.0000 | Combined 1.0000
  - recent_pool_appearance_count_20d: May 1.0000 | June 1.0000 | Combined 1.0000
  - board_day_volume_rank: May 0.9521 | June 0.9653 | Combined 0.9592
  - break_open_return: May 1.0000 | June 1.0000 | Combined 1.0000
  - break_high_return: May 1.0000 | June 1.0000 | Combined 1.0000
  - break_close_return: May 1.0000 | June 1.0000 | Combined 1.0000
  - break_touched_limit_up: May 1.0000 | June 1.0000 | Combined 1.0000
  - break_opened_from_limit_up: May 1.0000 | June 1.0000 | Combined 1.0000
  - break_upper_shadow_ratio: May 1.0000 | June 0.9942 | Combined 0.9969
  - break_lower_shadow_ratio: May 1.0000 | June 0.9942 | Combined 0.9969
  - d1_high_to_close_drawdown_raw: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_low_to_close_recovery: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_open_to_close_return_raw: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_close_location: May 1.0000 | June 0.9942 | Combined 0.9969
  - d1_intraday_range: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_afternoon_return: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_last_hour_return: May 1.0000 | June 1.0000 | Combined 1.0000
  - break_volume_ratio_vs_board_days: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_up_bar_volume_ratio: May 1.0000 | June 1.0000 | Combined 1.0000
  - down_bar_volume_ratio: May 1.0000 | June 1.0000 | Combined 1.0000
  - volume_above_d1_close_ratio: May 1.0000 | June 1.0000 | Combined 1.0000
  - amount_above_d1_close_ratio: May 1.0000 | June 1.0000 | Combined 1.0000
  - high_zone_volume_ratio: May 1.0000 | June 1.0000 | Combined 1.0000
  - high_zone_amount_ratio: May 1.0000 | June 1.0000 | Combined 1.0000
  - late_day_sell_volume_ratio: May 1.0000 | June 1.0000 | Combined 1.0000
  - late_day_sell_amount_ratio: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_close_to_vwap_raw: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_close_to_ma5_raw: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_low_to_ma5_raw: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_high_to_ma5_raw: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_close_to_ma10_raw: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_low_to_ma10_raw: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_close_to_ma20: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_ma5_slope: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_ma10_slope: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_ma20_slope: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_close_above_ma5: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_close_above_ma10: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_true_reclaim_ma5: May 1.0000 | June 1.0000 | Combined 1.0000
  - d1_true_reclaim_ma10: May 1.0000 | June 1.0000 | Combined 1.0000
  - consecutive_days_below_ma5: May 1.0000 | June 1.0000 | Combined 1.0000
  - consecutive_days_below_ma10: May 1.0000 | June 1.0000 | Combined 1.0000
  - break_day_in_pool: May 1.0000 | June 1.0000 | Combined 1.0000
  - last_board_day_in_pool: May 0.9589 | June 1.0000 | Combined 0.9812
  - pool_consecutive_count_last_board: May 0.9521 | June 0.9653 | Combined 0.9592
  - recent_7d_cumulative_return: May 1.0000 | June 1.0000 | Combined 1.0000
  - recent_7d_max_drawdown: May 1.0000 | June 1.0000 | Combined 1.0000
  - recent_7d_close_position: May 1.0000 | June 1.0000 | Combined 1.0000
  - recent_7d_limit_up_count: May 1.0000 | June 1.0000 | Combined 1.0000

### 泄漏审计

- FEATURE future leakage: 0 (必须 0)
- FEATURE model-output leakage: 0 (必须 0)
- FEATURE label-lineage leakage: 0 (必须 0)

## 3. 输入表

- v004c_pairwise_v1_input_table.csv: 319 行 × 67 列
- 列序: 身份(5) + break_date(审计) + 53 FEATURE(冻结序) + 标签(2) + dev 资格(3) + 缺失统计(2) + provenance
- 2 板 / 3 板: 261 / 58
- dev_label_complete: 319/319
- dev_feature_source_complete: 319/319
- dev_training_eligible: 319/319

## 4. Deterministic Rebuild

- 两轮构建核心资产字节一致: PASS
- 标签附加方式: 特征生成时不读标签; 标签列由 v002 冻结值原样携带至输入表末尾, 未来 Pairwise 训练时以事件 id 对齐。

## 5. 严格禁止确认

- 未训练 Pairwise Ridge / Logistic / Tree/GBDT
- 未搜索 lambda
- 未做 walk-forward 模型
- 未计算模型 probability / 排名 / Top1/Top3
- 未根据 Target7 / tail-loss / 单因子 AUC / IC / 月份表现选择特征
- 未创建新的人工综合评分 / Recognition Score 类 composite
- 未使用 v004a/v004b/Repair-State 预测输出或旧模型系数
- 未创建 feature_x_board3 交互列

## 6. 结论

- Feature Contract 状态: READY_FOR_PAIRWISE_V1

