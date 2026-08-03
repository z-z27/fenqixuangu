# v004c 训练准备研究评审报告 (v004c_training_prep_review)

- 数据集版本: `v004c-dataset-0.2.1`;特征定义: `v004c-features-0.3`;模型规格: `v004c-model-spec-0.1`
- 标签源: `daily_ohlc`;部署状态: `research_only`
- 生成日期 2026-08-03,时区 Asia/Shanghai。
- 本报告**不声称任何模型已建成或训练完成**;本任务未训练任何模型。

## 1. 当前是在数据阶段、因子阶段还是训练阶段?

**处于"数据集构建与标签质量修正 + 正式因子筛选 + 模型规格冻结前准备"阶段。** 已完成: 0.2.1 日线标签数据集、四头×27 因子正式单因子分析、模型数学规格 v0.1、walk-forward fold 规格、fold 内动态权重工具与测试。**尚未进入训练阶段**(未拟合逻辑回归/Elastic Net/GBDT,未选超参,未做自动特征选择,未输出交易名单)。

## 2. 是否已经训练模型?

**没有。** 所有 AUC 均为单因子秩法统计量(Mann-Whitney U),未拟合任何多因子模型;准入结论为规则合成,不是拟合结果。

## 3. 日线标签与分钟标签有多少差异?

- 标签**判定分歧: Target7 0 条、尾亏 0 条**(在日线标签质量合格的 866 行上)。
- 价格差异极小: d2_open 差异最大 0.00 元(p50=0);d3_high 差异最大 0.01 元;d3_close 差异最大 2.93 元(单条数据修订离群,即 603580 复牌日);收益差异 p99 ≈ 0.00。
- 原因: 日线与 5min 价格同源,分歧几乎只来自极少数数据修订。

## 4. 日线口径是否能够稳定生成全部标签?

**不能覆盖全部,原因与分钟口径相同——本地缓存覆盖。** 1020 行中日线标签合格 866 行(84.9%),154 行因"预期 D2/D3 交易日无日线 bar 且无严格停牌证明"被判无效(这些个股的日线缓存早于 5min 缓存结束,如 000090 日线止于 06-02)。被排除行全部记录在 `v004c_daily_label_quality_report.csv`(原因: d2/d3_no_bar_without_proof)。**未联网补数据、未跳日、未复制前日 K 线**。日线口径在"有日线缓存覆盖"的样本上稳定生成,覆盖缺口是本地数据限制,不是定义问题。

## 5. d0 和 post 分别有多少样本和正样本?

| 头 | 行数 | 事件数 | Target7 正样本 | 尾亏样本 |
|---|---|---|---|---|
| d0 (v0.2.1 训练视图) | 280 | 280 | 93 (33.2%) | 76 (27.1%) |
| post (v0.2.1 训练视图) | 448 | 327 | 149 (33.3%) | 153 (34.2%) |
| 合计 | 728 | 343 | 242 (33.2%) | 229 (31.5%) |

## 6. 哪些因子正式建议进入四个模型头?

| 头 | ADMIT_PRIMARY | df | ADMIT_ALTERNATIVE |
|---|---|---|---|
| d0_target | down_bar_volume_ratio; board_streak_is_3; d1_close_to_vwap_raw*; d1_true_reclaim_ma5* | 4 | break_touched_limit_up |
| d0_tail | late_day_sell_volume_ratio; break_volume_distance_from_one; volume_above_d1_close_ratio; break_touched_limit_up | 4 | break_volume_ratio_vs_board_days; break_volume_log_ratio; d1_high_to_close_drawdown_raw; break_high_to_close_drawdown; board_streak_is_3 |
| post_target | d1_high_to_close_drawdown_raw; volume_above_d1_close_ratio; down_bar_volume_ratio; ma5_below_minus_2_hinge | 4 | break_volume_distance_from_one; break_volume_ratio_vs_board_days; break_volume_log_ratio |
| post_tail | d1_close_to_vwap_raw; break_touched_limit_up; ma5_above_5_hinge; late_day_sell_volume_ratio | 4 | ma5_above_10_hinge; break_volume_distance_from_one; break_volume_ratio_vs_board_days; break_volume_log_ratio; d1_oc_return_clipped; d1_open_to_close_return_raw; d1_close_location; d1_repair_above_2_hinge; d1_open_to_close_bucket |

`*` = 补充准入(0.51 ≤ AUC < 0.53;证据弱,须 walk-forward 复核)。每组表达多样性受限: A/B/C/D 组内名额与相关性门槛在 `v004c_factor_admission_decision.csv` 逐行记录。单因子 AUC 0.51–0.62,最强为 d0_tail 的 late_day_sell_volume_ratio(0.616,方向一致率 1.0)。

## 7. 哪些因子被拒绝,原因是什么?

- **REJECT_UNSTABLE (11)**: 如 d0_tail 的 d1_close_to_ma5_raw/ma5_gap_clipped(方向一致率 0.33)、d0_target 的 d1_high_to_close_drawdown_raw/break_high_to_close_drawdown(一致率 0.33)、d1_repair_above_5_flag(一致率 0.00)等——时间子区间方向翻转。
- **REJECT_REDUNDANT (0)**: 本轮无因子与已准入因子 |spearman| ≥ 0.98(等价表达通过组内名额/备用机制处理,如 ma5_gap_clipped 与 raw AUC 相同被判 AUDIT_ONLY)。
- **REJECT_LOW_COVERAGE (0)**: 训练视图中所有候选因子覆盖率 ≥ 98.7%(d1_close_to_ma5_bucket/d1_open_to_close_bucket 覆盖率 86.8% 以上,未到 0.85 阈值以下)。
- **AUDIT_ONLY (61)**: 单因子 AUC < 0.51 或证据过弱,包括大部分 MA5 原始距离、D1 修复幅度、断板结构因子。
- 另: 识别度/模型对照字段(recognition_score、board_day_*_rank、v004a_probability/rank、v002_rank、month、指数/市场字段、集合竞价字段)按规格标记 `audit_only`,不进入任何模型头;**月份不作预测因子**(仅用于稳定性分析)。

## 8. 总有效自由度是否符合限制?

**符合。** 每个头主推荐 4 个表达,df = 4 ≤ 8;若引入分桶表达(每桶 5 df)须相应削减连续表达(规格第 8 节写明)。四头各自独立 ≤ 8。

## 9. fold 规格是否满足时间隔离?

**满足。**
- 42 个唯一 signal_date,4 个扩展式 fold: 初始训练 14 日、purge 2 交易日、验证窗 6/6/6/2(末窗因样本末尾压缩,已记录);
- **每个 fold 的 train/valid 事件交集 = 0**(2 日 purge 保证断板事件不会跨越边界,实测确认);
- 训练覆盖 [0,14),[0,22),[0,30),[0,38) 个信号日,验证分别 16-21, 24-29, 32-37, 40-41。
- 正样本充足性: fold1 的 d0_target(34)与 d0_tail(24)训练正样本 < 40 建议下限——已记录在 fold spec 的 boundary_note;正式训练时对该 fold 的 d0 头评估需降权或调整边界(不在此任务执行)。

## 10. fold 权重测试是否全部通过?

**9/9 全部通过**(`v004c_fold_weight_test_report.txt`):
有限、正、均值=1(1.000000000000)、每日期权重和相等(max−min = 7.11e-15)、事件重复基础影响减半(E3 双次 base=0.5 = E1 单次 1.0 的一半)、验证集增删无关、确定性、行序不变(728 行全匹配)、输入 DataFrame 未被修改。

## 11. 是否已经具备进入 v004c.0 正式 walk-forward 训练的条件?

**数据与流程侧基本具备,模型侧有保留条件。** 已具备: 日线标签数据集(728 训练行)、标签质量门、四头因子准入、fold 规格与事件隔离、fold 内权重工具(测试全过)、模型规格 v0.1。**保留条件**: d0 头正样本 93/76 低于 100 建议门槛;fold1 d0 头训练正样本 24-34 不足;补充准入的 2 个因子证据弱;五月敏感性数据待作为稳健性检查。**结论: 可以进入 v004c.0-research 的"训练准备"最后一步(冻结规格评审),但在正式训练前必须解决第 12 节列出的确认问题。**

## 12. 如果尚未具备,剩余具体问题是什么?

1. d0 头(93 正/76 尾)是否接受低样本训练并标注 low-confidence(建议接受,限制 df ≤ 4);
2. fold1 d0 头正样本不足的处理(降权该 fold 评估 vs 延长初始训练至 16 日,后者只剩 3 个验证 fold——需选择并记录);
3. 补充准入因子(d1_close_to_vwap_raw、d1_true_reclaim_ma5)保留在主要集还是降为备用;
4. C 组跨头选择不一致的确认(建议允许,记录理由);
5. 缺失行处理(一字板 10 行 d1_close_location、5 行 break_close_location: 排除 vs 缺失哑变量);
6. 截尾参数与秩标准化口径冻结确认(ma5 clip(−0.15,0.20)、oc clip(−0.10,0.10)、per-date percentile rank);
7. 尾亏头是否需要额外样本加权(现状尾亏率 27-34%,建议不加重,需确认);
8. v004a/v002 对照列是否加入正式 walk-forward 输出(建议加入,仅对照)。

## 附: 数据与流程事实

- v0.2 全量 1020 行 → v0.2.1 全量 1020 行(全部保留);TRAIN_PRIMARY 877 → 728(154 行因日线标签质量排除,其中 2 行为 v0.2 排除但 v0.2.1 日线标签合格重新进入: 603580、605028);SENSITIVITY_MAY_COMPLETE 141 → 138(3 行五月因日线缓存缺口降级 AUDIT_ONLY_MAY_INCOMPLETE)。
- 日线标签: Target7 284 正样本 / 尾亏 264;与分钟标签判定分歧 0/0。
- 未修改任何现有文件;未联网;未刷新缓存;未训练模型;五月数据未删除。
