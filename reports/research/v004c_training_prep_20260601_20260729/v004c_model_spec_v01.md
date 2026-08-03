# v004c 模型数学规格 (v004c-model-spec-0.1)

- 数据集版本: `v004c-dataset-0.2.1`;特征定义版本: `v004c-features-0.3`
- 目标标签源: `daily_ohlc`(日线 OHLC 正式标签)
- 部署状态: `research_only`;本规格**未训练任何模型**,仅供 v004c.0-research 训练前冻结评审。

## 1. 四个模型头

| 头 | stage_group | 目标 | 训练人口 (v0.2.1 TRAIN_PRIMARY) |
|---|---|---|---|
| v004c_d0_target | d0 (days_since_break == 0) | target7_daily_d2open_d3high | 280 行 / 280 事件 / 93 正样本 (33.2%) |
| v004c_d0_tail | d0 | tail_loss_daily_5pct | 280 行 / 76 尾亏 (27.1%) |
| v004c_post_target | post (days_since_break ∈ {1,2}) | target7_daily_d2open_d3high | 448 行 / 149 正样本 (33.3%) |
| v004c_post_tail | post | tail_loss_daily_5pct | 448 行 / 153 尾亏 (34.2%) |

## 2. 每个模型的目标

- d0/post × Target7: 估计 P(target7_daily_d2open_d3high = 1)。
- d0/post × 尾亏: 估计 P(tail_loss_daily_5pct = 1)。
- 两个目标共用同一特征集,独立建模;模型输出概率,不作交易名单。

## 3. 日线标签公式(正式)

```
expected_d2_date = 统一交易日历中 signal_date 的下一交易日
expected_d3_date = 统一交易日历中 expected_d2 的下一交易日
actual_daily_d2_date = 个股日线序列中 ≥ expected_d2 的首个有日线 bar 的交易日
                      (无 bar 时必须具备严格停牌证明, 否则标签无效)
actual_daily_d3_date = 个股日线序列中 > actual_daily_d2 的首个有日线 bar 的交易日 (规则同上)

d2_open_daily        = actual_daily_d2_date 的日线 open
d3_high_daily        = actual_daily_d3_date 的日线 high
d3_close_daily       = actual_daily_d3_date 的日线 close
daily_d2open_to_d3high_return  = d3_high_daily / d2_open_daily - 1
daily_d2open_to_d3close_return = d3_close_daily / d2_open_daily - 1
target7_daily_d2open_d3high = 1 if daily_d2open_to_d3high_return >= 0.07 else 0
tail_loss_daily_5pct        = 1 if daily_d2open_to_d3close_return <= -0.05 else 0
```

- **D2 买入价使用日线 open(含正式集合竞价结果), 不使用 5min 首根 K 线 open。**
- 5min 数据只用于 D1 盘中因子,不用于 D2/D3 正式标签价格。
- 旧 5min 标签保留为 `audit_minute_*` 审计对照(见 v0.2.1 数据)。

## 4. 候选池定义

沿用 `break-repair-0.1`(与 v0.1/v0.2 一致):

- 主板(涨停池∪日线缓存);连续 2/3 板(日线相邻交易日,`_is_main_board_limit_up_day` 判定)后第一个非涨停日为断板日;
- 观察日 = 断板日及其后第 1/2 个该股交易日(days_since_break ∈ {0,1,2});signal_date 本身为涨停日的观察排除;
- 候选阶段无 MA/量能/收回 VWAP/recognition/v004a 等硬过滤。
- **v0.2.1 训练视图额外要求**: `daily_label_quality_ok = true` 且 `d1_minute_complete = true`(48 bar / 09:35 / 15:00)。

## 5. d0/post 定义

```
stage_group = d0   当 days_since_break == 0
stage_group = post 当 days_since_break ∈ {1, 2}
post_day ∈ {1, 2}  仅 post 头使用
training_stage_feature = "stage_group"
```

## 6. 每个模型推荐因子(基于正式单因子分析,规则合成,非拟合)

| 头 | 推荐主要因子 (ADMIT_PRIMARY) | 有效自由度 | 备用 (ADMIT_ALTERNATIVE) |
|---|---|---|---|
| d0_target | down_bar_volume_ratio; board_streak_is_3; d1_close_to_vwap_raw*; d1_true_reclaim_ma5* | 4 | break_touched_limit_up |
| d0_tail | late_day_sell_volume_ratio; break_volume_distance_from_one; volume_above_d1_close_ratio; break_touched_limit_up | 4 | break_volume_ratio_vs_board_days; break_volume_log_ratio; d1_high_to_close_drawdown_raw; break_high_to_close_drawdown; board_streak_is_3 |
| post_target | d1_high_to_close_drawdown_raw; volume_above_d1_close_ratio; down_bar_volume_ratio; ma5_below_minus_2_hinge | 4 | break_volume_distance_from_one; break_volume_ratio_vs_board_days; break_volume_log_ratio |
| post_tail | d1_close_to_vwap_raw; break_touched_limit_up; ma5_above_5_hinge; late_day_sell_volume_ratio | 4 | ma5_above_10_hinge; break_volume_distance_from_one; break_volume_ratio_vs_board_days; break_volume_log_ratio; d1_oc_return_clipped; d1_open_to_close_return_raw; d1_close_location; d1_repair_above_2_hinge; d1_open_to_close_bucket |

`*` = 补充准入(0.51 ≤ AUC < 0.53,为达到最低 4 因子表达;证据弱,必须在 walk-forward 中复核)。所有单因子 AUC 0.51–0.62,信号弱但方向在多数子区间一致(详见 `v004c_factor_admission_decision.csv`)。

## 7. 每个因子的数学表达

```
down_bar_volume_ratio            = Σvol(close<open) / Σvol           (D1 全天)
late_day_sell_volume_ratio       = Σvol(time≥14:00 & close<open) / Σvol
volume_above_d1_close_ratio      = Σvol(bar close ≥ d1_close) / Σvol
d1_high_to_close_drawdown_raw    = (d1_high - d1_close) / d1_high
d1_close_to_vwap_raw             = d1_close / d1_vwap - 1
d1_true_reclaim_ma5              = 1 if (d1_low ≤ d1_ma5) & (d1_close ≥ d1_ma5) else 0
ma5_below_minus_2_hinge          = max(-0.02 - (d1_close/d1_ma5 - 1), 0)
ma5_above_5_hinge                = max((d1_close/d1_ma5 - 1) - 0.05, 0)
break_volume_ratio_vs_board_days = break_volume / mean(板日 volume)
break_volume_log_ratio           = log(max(break_volume_ratio_vs_board_days, 1e-9))
break_volume_distance_from_one   = |break_volume_log_ratio|
break_high_to_close_drawdown     = (break_high - break_close) / break_high
break_touched_limit_up           = 1 if break_high ≥ 涨停价 - 0.011 else 0
board_streak_is_3                = 1 if board_streak_before_break == 3 else 0
post_day                         = 1 or 2 (仅 post 头)
```

## 8. 每个模型有效自由度

- 主推荐: 每头 4 个连续/哑变量表达,df = 4(远低于 8 上限)。
- 若加入分桶表达(d1_close_to_ma5_bucket / d1_open_to_close_bucket),每个分桶按 5 df 计,须相应削减连续表达;v0.2.1 首版**不建议**同时使用分桶与连续表达。

## 9. 缺失值处理

- **不允许均值/中位数/零值静默填充**(任务约束)。
- 缺失策略: (a) 该样本从相应模型头训练中排除(记录缺失原因);(b) 对少数缺失字段(如 d1_close_location 一字板 10 行、break_close_location 5 行)可单独加"缺失指示"哑变量,但首版不推荐(样本小)。
- D1 分钟因子缺失(d1_minute_complete=false)→ 该行不可训练。
- 分桶与 hinge 的缺失与原始值一致传递。

## 10. 截尾和标准化规则

- 因子截尾(仅研究性,已随因子定义给出): ma5_gap_clipped = clip(d1_close_to_ma5_raw, −0.15, 0.20); d1_oc_return_clipped = clip(d1_open_to_close_return_raw, −0.10, 0.10)。
- 首版建议: 对连续因子做**秩标准化或分位标准化**(v004a 同款 percentile rank 口径,按 signal_date 横截面 rank(pct=True)),而非原始值回归——与项目 v004a 特征口径一致。
- 不使用 z-score 原始值标准化(量纲与右偏敏感)。

## 11. Fold 内权重公式

```
compute_fold_weights(train_df):           # 输入 = 当前训练 fold 数据
  event_count_i  = fold 内 event_id i 的观察次数
  base_weight_i  = 1 / event_count_i
  weight_i       = base_weight_i / Σ_{j ∈ 同 signal_date} base_weight_j   # 日期内归一
  weight_i       = weight_i / mean(weight)                                 # fold 均值归一 = 1
```

- 只读取 fold 内 event_id 与 signal_date;不读取验证集/全数据集次数/未来候选数/CSV 中 proposed_training_weight。
- 9 项测试全部通过(`v004c_fold_weight_test_report.txt`),包括: 有限/正/均值1/日期权重和相等(≤1e-10)/事件重复减半/验证集无关/行序不变/输入不变。
- 旧字段 `proposed_training_weight`(v0.2 定义: 1/(日数×事件数))保留在全量数据中,标记 `audit_only_do_not_use_for_training`;正式训练使用本工具按 fold 动态计算。

## 12. Walk-forward 规则

- 基于 TRAIN_PRIMARY 的 42 个唯一 signal_date,按时间顺序扩展式 fold(`v004c_walk_forward_fold_spec.csv`):
  - 初始训练 ≥ 14 个信号日(实际 14);
  - 验证窗 6-8 日(实际 6,6,6,2;末窗因样本末尾压缩,已记录);
  - 训练/验证边界 purge = 2 个交易日;
  - 每个 fold 验证 event_id 无跨训练/验证(实测 0 冲突);
  - 每个 fold 对四个头分别输出训练行数/事件数/正样本数。
- 正样本充足性: fold1 的 d0_target(34)与 d0_tail(24)训练正样本低于 40 建议下限,已在 fold spec 中记录;正式训练时对该 fold 的 d0 头评估需降权或调整边界(记录在案,不在本任务中执行)。
- 本任务不运行任何 fold 训练。

## 13. 未来允许使用的模型类型

1. **低复杂度 Logistic 回归**(主选): 4-6 个线性项,weighted logistic(L2 可选,正则参数在 walk-forward 网格内选择——该选择属于"训练阶段",本任务未执行);
2. **Elastic Net Logistic**(受限备选): 仅当因子共线性在 walk-forward 中造成不稳定时,以轻 α 使用。

## 14. 暂不允许

- GBDT / 树集成;
- 神经网络;
- 大量自动交互项(仅允许规格内列出的 hinge/分桶表达);
- 任何形式的自动特征搜索后接入策略。

## 15. 正式训练开始前尚需确认的问题

1. **d0 头样本量**: d0_target 280 行 / 93 正样本、d0_tail 280 行 / 76 尾亏,低于"正样本 ≥100"的宽松门槛,是否接受 d0 头以 4 df 训练并仅作探索性输出?(建议: 接受训练,但 d0 头结论标注 low-confidence)
2. **fold1 正样本不足**: 是否对 d0 头 fold1 验证结果降权/剔除,或把初始训练延长至 16 个信号日(将只剩 3 个验证 fold)后重新评估?
3. **d1_close_to_vwap_raw 与 d1_true_reclaim_ma5 的补充准入**: 证据弱(AUC 0.52/0.52),是否保留在主要集合中还是降为备用?
4. **C 组选择**: volume_above_d1_close_ratio(d0_tail/post_target)与 d1_close_to_vwap_raw(d0_target/post_tail)在不同头选择了不同成员——是否允许跨头不同(建议允许,记录理由)?
5. **尾亏目标的分桶**: 尾亏头是否需要对 tail_loss 做样本加权(尾亏 27-34% 与 Target7 相当,无需特别重加权,但需确认);
6. **缺失指示**: 10 行一字板 d1_close_location 缺失与 5 行 break_close_location 缺失的处理(排除 vs 缺失哑变量);
7. **五月敏感性数据的角色**: 仅做系数稳健性检查,不进入训练;确认评审口径;
8. **截尾参数**: ma5 clip(−0.15, 0.20) 与 oc clip(−0.10, 0.10) 是否作为冻结规格(建议冻结,避免调参污染);
9. **多空方向**: 因子方向由 walk-forward 训练数据内学习(不预先固定方向),但需在训练后人工复核方向与单因子分析一致;
10. **v004a/v002 对照**: 是否在正式 walk-forward 中同时输出 v004a/v002 对照列(建议输出,仅对照)。

---

附: 本规格的因子推荐来自 `v004c_factor_univariate_formal.csv`、`v004c_factor_stability_formal.csv`、`v004c_factor_redundancy_formal.csv`、`v004c_factor_admission_decision.csv` 的规则合成;**未拟合任何多因子模型**,所有 AUC 均为单因子秩法统计量。
