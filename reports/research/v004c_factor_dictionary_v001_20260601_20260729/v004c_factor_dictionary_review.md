# v004c 阶段2.1 — 因子字典、去重与准入报告 (自动生成)

- 字典版本: v004c-factor-dictionary-0.1
- 来源: v004c-d1-dataset-0.1 (tag v004c-d1-dataset-0.1 → a65661f4738b849a06efb4859d4100842871e971)
- 生成器提交: 7f4db91e06513a08de382831d401c5cc4d5ea3c4
- 分支: research-sample-analysis / HEAD: aa62b315dd1f792dd7657b190e41c1e992f93112

## 版本链验证

- 来源 tag = v004c-d1-dataset-0.1 对应目标提交 = a65661f4738b849a06efb4859d4100842871e971 (通过)
- 数据提交是当前 HEAD 祖先: True
- 输入 SHA 与阶段1 manifest 一致: 全部通过 (v004c_d1_column_lineage.csv, v004c_d1_snapshot_v001.csv, v004c_training_d1_v001.csv)
- source-ref 字节锚定: 4 个阶段1文件与 tag 逐字节一致 (bytes_equal=true, LF 规范化比较, autocrlf 环境), 阶段1 manifest 审计参考 ae2a2b48b0b97390e08e81c42cadaa84d5ff76b95e847a80a66e480936ccd618 校验通过
- current_git_root = F:\fenqixuangu / stage1_recorded_git_root = F:\fenqixuangu (后者仅供审计)

## 冻结门 (十九)

| 指标 | 值 |
|---|---|
| 输入行数 | 333 |
| 信号日数 | 42 |
| lineage 行数 | 197 |
| stage1 allowed 原始候选 | 79 |
| 精确重复组 | 13 |
| forbidden_unknown 机制组 | 0 |
| break_turnover_ratio 缺失 | 333 |
| target-blind | True |
| hard_failures | 0 |

## 197 列字典统计

- 字典行数: 197
- 候选 / 非候选: 79 / 118

## 79 原始候选分区

| 状态 | 数量 |
|---|---|
| PRIMARY_RAW | 32 |
| SENSITIVITY_RAW | 21 |
| DERIVE_ONLY | 12 |
| EXCLUDE_ALL_MISSING | 1 |
| EXCLUDE_CONSTANT | 0 |
| EXCLUDE_DEPRECATED | 2 |
| EXCLUDE_EXACT_DUPLICATE_ALIAS | 11 |

### PRIMARY_RAW (32 个)

break_close_return, break_day_in_pool, break_high_return, break_open_return, break_touched_limit_up, break_volume_ratio_vs_board_days, d1_afternoon_return, d1_close_to_ma10_raw, d1_close_to_ma20, d1_close_to_ma5_raw, d1_close_to_vwap_raw, d1_high_to_close_drawdown_raw, d1_high_to_ma5_raw, d1_intraday_range, d1_last_hour_return, d1_low_to_close_recovery, d1_low_to_ma10_raw, d1_low_to_ma5_raw, d1_ma10_slope, d1_ma5_slope, d1_open_to_close_return_raw, d1_true_reclaim_ma5, d1_up_bar_volume_ratio, down_bar_volume_ratio, high_zone_volume_ratio, late_day_sell_volume_ratio, max_board_streak_20d, recent_limit_up_count_10d, recent_limit_up_count_20d, recent_pool_appearance_count_10d, recent_pool_appearance_count_20d, volume_above_d1_close_ratio

### SENSITIVITY_RAW (21 个)

amount_above_d1_close_ratio, board_day_amount_rank, board_day_turnover_rank, board_day_volume_rank, break_amount_ratio_vs_board_days, break_lower_shadow_ratio, break_opened_from_limit_up, break_upper_shadow_ratio, consecutive_days_below_ma10, consecutive_days_below_ma5, d1_close_above_ma10, d1_close_above_ma5, d1_close_location, d1_close_to_ma5_bucket, d1_open_to_close_bucket, d1_true_reclaim_ma10, d1_vwap_to_close_gap, high_zone_amount_ratio, last_board_day_in_pool, late_day_sell_amount_ratio, pool_consecutive_count_last_board

### DERIVE_ONLY (12 个, 绝对尺度)

board_streak_before_break, break_prev_close, d1_amount, d1_close, d1_high, d1_low, d1_ma10, d1_ma20, d1_ma5, d1_open, d1_volume, d1_vwap

## 精确重复组 (13)

| group_id | canonical | alias | alias_status |
|---|---|---|---|
| ED01 | d1_open | break_open | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED02 | d1_high | break_high | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED03 | d1_low | break_low | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED04 | d1_close | break_close | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED05 | d1_intraday_range | break_intraday_range | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED06 | d1_high_to_close_drawdown_raw | break_high_to_close_drawdown | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED07 | d1_close_location | break_close_location | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED08 | d1_volume | break_volume | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED09 | d1_amount | break_amount | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED10 | d1_close_above_ma5 | deprecated_d1_reclaimed_ma5_v01 | EXCLUDE_DEPRECATED |
| ED11 | d1_close_above_ma10 | deprecated_d1_reclaimed_ma10_v01 | EXCLUDE_DEPRECATED |
| ED12 | down_bar_volume_ratio | d1_down_bar_volume_ratio | EXCLUDE_EXACT_DUPLICATE_ALIAS |
| ED13 | volume_above_d1_close_ratio | volume_above_break_close_ratio | EXCLUDE_EXACT_DUPLICATE_ALIAS |

## 近重复 / 互斥组

- 自动检测命中 5 对
- 预声明互斥组 7 组 (ME01-ME07), 全部登记于近重复对表

| pair_id | left | right | overlap | pearson | spearman | source | me_group |
|---|---|---|---|---|---|---|---|
| NP01 | amount_above_d1_close_ratio | volume_above_d1_close_ratio | 333 | 0.999927 | 0.999926 | BOTH | ME02_ABOVE_CLOSE_CHIP |
| NP02 | break_amount_ratio_vs_board_days | break_volume_ratio_vs_board_days | 331 |  |  | PREDECLARED_SEMANTIC | ME01_BREAK_ACTIVITY_RATIO |
| NP03 | consecutive_days_below_ma10 | d1_close_above_ma10 | 333 |  |  | PREDECLARED_SEMANTIC | ME07_MA10_POSITION |
| NP04 | consecutive_days_below_ma5 | d1_close_above_ma5 | 333 | -1.0 | -1.0 | BOTH | ME06_MA5_POSITION |
| NP05 | d1_close_to_vwap_raw | d1_vwap_to_close_gap | 333 | -0.999254 | -1.0 | BOTH | ME05_VWAP_GAP |
| NP06 | high_zone_amount_ratio | high_zone_volume_ratio | 333 | 0.999834 | 0.99984 | BOTH | ME03_HIGH_ZONE |
| NP07 | late_day_sell_amount_ratio | late_day_sell_volume_ratio | 333 | 0.998865 | 0.998642 | BOTH | ME04_LATE_SELL |

## 派生因子规格与域审计 (6 个)

| feature | formula | 域规则 | 非空 | 缺失 | 有限 | 域违规 | 唯一值 | min | max | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| board_streak_is_3 | 1[board_streak_before_break == 3] | values in {0, 1} | 333 | 0 | 333 | 0 | 2 | 0.0 | 1.0 | OK |
| ma5_overheat_10 | 1[d1_close_to_ma5_raw >= 0.10] | values in {0, 1} | 333 | 0 | 333 | 0 | 2 | 0.0 | 1.0 | OK |
| overrepair | max(d1_open_to_close_return_raw - 0.03, 0) | >= 0 | 333 | 0 | 333 | 0 | 72 | 0.0 | 0.0780391322841343 | OK |
| break_volume_abnormality | abs(log(max(break_volume_ratio_vs_board_days, 1e-6))) | >= 0 | 333 | 0 | 333 | 0 | 333 | 0.001393621415825322 | 3.4947310447090336 | OK |
| profit_chip_ratio | 1 - volume_above_d1_close_ratio | [0, 1] | 333 | 0 | 333 | 0 | 283 | 0.0 | 0.9821376216580825 | OK |
| profit_pressure | profit_chip_ratio * d1_low_to_close_recovery * log1p(break_volume_ratio_vs_board_days) | finite; no sign constraint (components may be negative) | 333 | 0 | 333 | 0 | 283 | 0.0 | 0.29759486845410066 | OK |

## 已有字段公式重算

| 字段 | 公式 | mismatch | 状态 |
|---|---|---|---|
| d1_true_reclaim_ma5 | 1[d1_low <= d1_ma5 and d1_close >= d1_ma5] | 0 | RECOMPUTE_MATCH |
| d1_close_above_ma5 | 1[d1_close >= d1_ma5] | 0 | RECOMPUTE_MATCH |
| d1_close_above_ma10 | 1[d1_close >= d1_ma10] | 0 | RECOMPUTE_MATCH |

## 声明

阶段2.1 只冻结因子字典、去重和准入; 没有使用 Target7 选择因子;
没有训练模型; 没有运行 Walk-forward; 没有修改阶段1; 没有自动 commit 或 push;
没有读取或刷新 data/cache; 没有使用 recognition_score / v004a / v002 / v005 字段。
