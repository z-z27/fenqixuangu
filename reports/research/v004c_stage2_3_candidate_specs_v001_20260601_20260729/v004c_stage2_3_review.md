# v004c 阶段2.3 — 人工候选模型规格冻结报告 (自动生成)

- 规格版本: v004c-stage2-3-candidate-specs-0.1
- 来源: 阶段1 v004c-d1-dataset-0.1 → a65661f4738b849a06efb4859d4100842871e971 / 阶段2.1 v004c-factor-dictionary-0.1 → 11f455dc9a6643a67250321cce68868be904cf06 / 阶段2.2 v004c-stage2-2-univariate-0.1 → fba302ffc58deb7b15812e4dd2ada113ae1f3a34
- 分支: research-sample-analysis / HEAD: 274019f4cbae900c0548ddd82bb6fa5fcd5d3000

## 版本链与输入验证

- 阶段2.2 tag 目标 = fba302ffc58deb7b15812e4dd2ada113ae1f3a34 (True)
- 阶段2.1 tag 目标 = 11f455dc9a6643a67250321cce68868be904cf06 (通过)
- 阶段1 tag 目标 = a65661f4738b849a06efb4859d4100842871e971 (通过)
- 阶段2.2数据提交是当前 HEAD 祖先: True
- 阶段2.1 git_head.txt = aa62b315dd1f792dd7657b190e41c1e992f93112 (通过)
- 阶段2.2 git_head.txt = ee0306fc4a5f3dadd3f0a780721754a014c58c58 (通过)
- 输入 SHA 与对应 manifest 一致: v004c_canonical_feature_map_v001.csv=OK; v004c_d1_column_lineage.csv=OK; v004c_d1_data_manifest.json=OK; v004c_d1_snapshot_v001.csv=OK; v004c_exact_duplicate_groups_v001.csv=OK; v004c_factor_dictionary_manifest.json=OK; v004c_factor_dictionary_v001.csv=OK; v004c_feature_allowlist_primary_v001.csv=OK; v004c_feature_allowlist_sensitivity_v001.csv=OK; v004c_near_duplicate_pairs_v001.csv=OK; v004c_predeclared_derived_factor_audit_v001.csv=OK; v004c_predeclared_derived_factor_spec_v001.csv=OK; v004c_stage2_2_board_subgroup_dev_v001.csv=OK; v004c_stage2_2_cluster_bootstrap_dev_v001.csv=OK; v004c_stage2_2_distribution_shift_unlabeled_v001.csv=OK; v004c_stage2_2_feature_structure_v001.csv=OK; v004c_stage2_2_manifest.json=OK; v004c_stage2_2_review.md=OK; v004c_stage2_2_univariate_summary_dev_v001.csv=OK
- 阶段1 manifest 审计参考校验: True
- 结构门: 333 行 / 42 信号日
- snapshot 仅读取允许特征列: break_open_return, break_volume_ratio_vs_board_days, d1_close_to_ma5_raw, d1_close_to_vwap_raw, d1_open_to_close_return_raw, d1_vwap_to_close_gap, down_bar_volume_ratio, high_zone_amount_ratio, high_zone_volume_ratio, late_day_sell_amount_ratio, late_day_sell_volume_ratio, signal_date, volume_above_d1_close_ratio
- 时间切片: 六月 173 行 / 七月锁定 160 行 (仅无标签使用)

## 阶段2.3定位: 人工设计 + 规格先冻结

- candidate_selection_mode = human_fixed_after_june_development_review
- M1、M2和M3由研究人员在阅读六月阶段2.2单因子结构和稳定性证据后人工确定。
- 六月是开发与候选设计数据 (development_and_manual_candidate_design)。
- 候选规格在任何七月标签揭示前冻结 (candidate_specs_frozen_before_july_label_access=true)。
- 阶段2.3运行时未读取 Target 列, 未根据 AUC、bootstrap、flags 或其它统计 自动搜索、增加、删除或替换候选成员。
- 运行时成员不变性 (runtime membership invariance): 模型目录/特征规格/预处理规格/候选 JSON/评价协议 对阶段2.2 证据字段扰动与 Target 列扰动逐字节不变

## 固定候选模型 (M0 + 3 个实质候选, 共 4 个)

- **M0 INTERCEPT_BASELINE** (role=BASELINE, feature_count=0, family=intercept_prevalence_baseline, penalty=none)
- **M1 PRIMARY_MECHANISM_CORE** (role=PRIMARY_CANDIDATE, feature_count=5, family=binary_logistic_regression, penalty=L2)
- **M2 SENSITIVITY_EXPRESSION_ALTERNATIVES** (role=SENSITIVITY_CANDIDATE, feature_count=5, family=binary_logistic_regression, penalty=L2)
- **M3 PREDECLARED_DERIVED_MECHANISMS** (role=DERIVED_SENSITIVITY_CANDIDATE, feature_count=5, family=binary_logistic_regression, penalty=L2)

### M1 固定特征顺序与预期方向 (direction_is_constraint=false)

| 位置 | 特征 | 预期方向 | 机制组 | 预处理 | 互斥组 |
|---|---|---|---|---|---|
| 1 | break_open_return | + | D1_PRICE_ACTION | FOLD_CLIP_Z |  |
| 2 | down_bar_volume_ratio | + | D1_VOLUME_ACTIVITY | FOLD_CLIP_Z |  |
| 3 | high_zone_volume_ratio | - | D1_CHIP_DISTRIBUTION | FOLD_CLIP_Z | ME03_HIGH_ZONE |
| 4 | late_day_sell_volume_ratio | - | D1_LATE_DAY_PRESSURE | FOLD_CLIP_Z | ME04_LATE_SELL |
| 5 | d1_close_to_vwap_raw | - | D1_VWAP_POSITION | FOLD_CLIP_Z | ME05_VWAP_GAP |

设计目的: primary 跨机制核心模型: 5 个不同机制组的 primary 表达; 无缺失处理需求; 不使用稀疏二元字段; 不使用日期不稳定或单日主导字段; 作为主要紧凑候选。

### M2 固定特征顺序与预期方向 (direction_is_constraint=false)

| 位置 | 特征 | 预期方向 | 机制组 | 预处理 | 互斥组 |
|---|---|---|---|---|---|
| 1 | break_open_return | + | D1_PRICE_ACTION | FOLD_CLIP_Z |  |
| 2 | down_bar_volume_ratio | + | D1_VOLUME_ACTIVITY | FOLD_CLIP_Z |  |
| 3 | high_zone_amount_ratio | - | D1_CHIP_DISTRIBUTION | FOLD_CLIP_Z | ME03_HIGH_ZONE |
| 4 | late_day_sell_amount_ratio | - | D1_LATE_DAY_PRESSURE | FOLD_CLIP_Z | ME04_LATE_SELL |
| 5 | d1_vwap_to_close_gap | + | D1_VWAP_POSITION | FOLD_CLIP_Z | ME05_VWAP_GAP |

设计目的: 表达替代敏感性模型: 保持与 M1 相同的主要机制结构, 保留 break_open_return 与 down_bar_volume_ratio, 替换 high_zone_volume_ratio→high_zone_amount_ratio、late_day_sell_volume_ratio→late_day_sell_amount_ratio、d1_close_to_vwap_raw→d1_vwap_to_close_gap; 用于判断信号是否来自机制本身, 而不是只来自 volume/amount 或正向/反向表达。M1 与 M2 的互斥表达不得放进同一模型。

### M3 固定特征顺序与预期方向 (direction_is_constraint=false)

| 位置 | 特征 | 预期方向 | 机制组 | 预处理 | 互斥组 |
|---|---|---|---|---|---|
| 1 | overrepair | - | PREDECLARED_DERIVED | FOLD_CLIP_Z |  |
| 2 | ma5_overheat_10 | - | PREDECLARED_DERIVED | RAW_BINARY |  |
| 3 | break_volume_abnormality | - | PREDECLARED_DERIVED | FOLD_CLIP_Z |  |
| 4 | profit_chip_ratio | + | PREDECLARED_DERIVED | FOLD_CLIP_Z |  |
| 5 | late_day_sell_volume_ratio | - | D1_LATE_DAY_PRESSURE | FOLD_CLIP_Z | ME04_LATE_SELL |

设计目的: 预声明派生机制模型: 检验阶段2.1 预声明派生表达是否能形成 结构更明确的跨机制组合; 不是根据阶段2.2 表现自动挑选派生因子。

### M1 与 M2 的表达替换关系

| 机制组 | M1 | M2 |
|---|---|---|
| D1_PRICE_ACTION | break_open_return | break_open_return |
| D1_VOLUME_ACTIVITY | down_bar_volume_ratio | down_bar_volume_ratio |
| D1_CHIP_DISTRIBUTION | high_zone_volume_ratio | high_zone_amount_ratio |
| D1_LATE_DAY_PRESSURE | late_day_sell_volume_ratio | late_day_sell_amount_ratio |
| D1_VWAP_POSITION | d1_close_to_vwap_raw | d1_vwap_to_close_gap |

M1 与 M2 的互斥表达不得放进同一模型; M2 用于判断信号是否来自机制本身, 而不是只来自 volume/amount 或正向/反向表达。

### M3 派生与底层机制关系

| 位置 | 派生特征 | 冻结公式 | 直接源 | 底层机制 |
|---|---|---|---|---|
| 1 | overrepair | max(d1_open_to_close_return_raw - 0.03, 0) | d1_open_to_close_return_raw | D1_PRICE_ACTION |
| 2 | ma5_overheat_10 | 1[d1_close_to_ma5_raw >= 0.10] | d1_close_to_ma5_raw | D1_MA_POSITION |
| 3 | break_volume_abnormality | abs(log(max(break_volume_ratio_vs_board_days, 1e-6))) | break_volume_ratio_vs_board_days | D1_VOLUME_ACTIVITY |
| 4 | profit_chip_ratio | 1 - volume_above_d1_close_ratio | volume_above_d1_close_ratio | D1_CHIP_DISTRIBUTION |
| 5 | late_day_sell_volume_ratio | (源字段) | - | D1_LATE_DAY_PRESSURE |

M3 不包含任何派生因子的直接源变量; 检验阶段2.1 预声明派生表达能否形成 结构更明确的跨机制组合, 不是根据阶段2.2 表现自动挑选派生因子。

## 12 个唯一入选特征 (首次出现顺序)

1. break_open_return, down_bar_volume_ratio, high_zone_volume_ratio, late_day_sell_volume_ratio, d1_close_to_vwap_raw, high_zone_amount_ratio, late_day_sell_amount_ratio, d1_vwap_to_close_gap, overrepair, ma5_overheat_10, break_volume_abnormality, profit_chip_ratio

## 候选边界 (59 行)

- 59 个阶段2.2 因子逐一登记; SELECTED_FIXED = 12, NOT_SELECTED_NOT_REJECTED = 47
- 未进入 M1/M2/M3 的字段只表示不属于本次预声明候选设计, 不代表永久淘汰; 禁止按 AUC 或 flags 自动填写淘汰理由

## 无标签冗余审计 (30 行)

- 数据切片: 六月 2026-06-01..2026-06-30 / 七月锁定 2026-07-01..2026-07-29 (只读特征值, 不读标签)
- 每模型 10 对, 共 30 行; 指标: dev/locked Spearman rho、pair 计数、最大绝对相关、六月七月相关变化
- 冗余 flag 数量: ABS_RHO_LT_0_70=30
- 最高相关特征对: M3 overrepair vs ma5_overheat_10 (|rho|=0.581433670594)
- 六月/七月相关性变化最大对: overrepair vs break_volume_abnormality (shift=0.321675541883)
- 冗余指标只作描述, 不得自动修改模型成员; 高相关也保留固定候选并在此披露

## VWAP 符号重参数化披露

- 核查字段: d1_close_to_vwap_raw (M1) 与 d1_vwap_to_close_gap (M2); 浮点容差 atol=rtol=1e-12
- 六月核查: pair_non_null=173, max_abs_sum=0.008057137084739804, Pearson=-0.9991696599734359, Spearman=-1.0, exact_negative_equivalence=False
- 七月无标签核查: pair_non_null=160, max_abs_sum=0.0066904950411876, Pearson=-0.9993953570493527, Spearman=-1.0, exact_negative_equivalence=False
- relationship_type = NOT_EXACT_NEGATIVE (由数据判定, 未硬编码)
- 该两字段在所选切片中不构成严格负向关系, 如实记录, 不作符号重参数化声明。

## 固定预处理

- FOLD_CLIP_Z: 每个训练 fold 独立, 只用训练窗口非缺失值计算 p01/p99 (0.01/0.99), 训练值与测试值裁剪到训练 p01/p99, 再用裁剪后训练均值/标准差 z-score (ddof=0); 测试窗口不参与任何参数计算; 训练标准差 <= 1e-12 时 fail closed
- RAW_BINARY: 保持 0/1, 不裁剪不标准化, 严格验证值域 {0, 1} (M3 的 ma5_overheat_10)
- 派生因子逐行计算, 不用标签, 不用跨行统计; 派生完成后执行冻结预处理政策
- 缺失策略 = FAIL_CLOSED: 任何窗口出现缺失该 fold 失败并记录, 禁止均值/中位数/众数填补、缺失 indicator、删除含缺失样本
- 禁止: 全样本先裁剪、全样本先标准化、七月参与预处理、监督式分箱、重新创建 bucket、PCA、特征选择

## 固定模型族与超参数

- M1/M2/M3: binary_logistic_regression, penalty=L2, C=1.0, solver=lbfgs, fit_intercept=True, class_weight=None, max_iter=5000, tol=1e-08, random_seed=20260806
- M0: intercept_prevalence_baseline, penalty=none
- 禁止后续搜索 C / penalty / solver / class_weight / tol / max_iter; 阶段2.4 和 2.5 不得根据七月或 Walk-forward 表现调参
- 收敛状态定义: CONVERGED / MAX_ITER_REACHED / NUMERICAL_FAILURE / NON_FINITE_COEFFICIENT / NON_FINITE_PREDICTION / INPUT_VALIDATION_FAILURE; solver 报告未收敛或达到 max_iter 仍未满足停止条件 → convergence_status=FAILED, 即使预测有限也不得标为收敛 (本阶段只冻结定义, 不运行拟合)

## 七月一次性评价协议 (只写协议, 不执行)

- 训练 2026-06-01..2026-06-30 / 测试 2026-07-01..2026-07-29
- 概率指标: ROC-AUC / PR-AUC / Brier score / log loss / calibration intercept / calibration slope (校准只评估, 不重新校准)
- 日期级排序指标: date Top1/Top3 Target7 hit rate / mean Target7 count in Top3 / eligible signal_date count; 不创建交易阈值; 排序 = predicted_probability 降序 + stock_code 升序
- 模型规格提交并打标签后才允许揭示七月结果

## 日期级 Walk-forward 协议 (只写协议, 不运行)

- 切分: 按 signal_date 升序, 训练 = 测试日期之前的全部日期, 测试 = 下一完整 signal_date, 同日全部样本同一 fold
- 最低训练门: train_signal_dates >= 10 / train_rows >= 80 / train_positive >= 20 / train_negative >= 40; 首个同时满足全部门的日期 作为首个正式测试日期
- 每 fold 独立物化派生/检查缺失/裁剪/标准化/拟合; 测试日期不参与训练参数

## 最终候选资格门 (阶段2.6 不得临时创造规则)

- 1. 七月正式拟合收敛
- 2. 所有正式 Walk-forward fold 均收敛
- 3. 所有预测均有限且在 [0,1]
- 4. 七月 log loss < M0 七月 log loss
- 5. 完整 Walk-forward OOS log loss < M0 OOS log loss
- 6. 完整 Walk-forward OOS ROC-AUC >= 0.50
- 无候选满足全部条件 → 最终状态 = REJECT_NO_STABLE_MODEL, 不得降低门槛

## 最终候选比较算法 (global_sequential_tolerance_filter)

- 模式: global_sequential_tolerance_filter; 资格集合优先 (eligibility_first=True); 基于全体合格候选的顺序集合过滤, 不进行任何两模型逐对比较; 全部差值为绝对数值差
- Step 1 OOS log loss: retain_within_absolute_difference_of_global_minimum, 绝对容差 = 0.005
- Step 2 OOS Brier: retain_within_absolute_difference_of_stage_minimum, 绝对容差 = 0.002
- Step 3 July log loss: minimum, 容差 = 0.0 (无额外容差; 数值完全相同才进入下一步)
- Step 4 feature_count: minimum
- Step 5 固定模型优先序: M1 > M2 > M3
- 禁止: 逐对比较模型 / 相对差值 / 百分比差 / 七月 AUC 打破并列 / Top1 或 Top3 打破并列 / 临时调整容差 / 看到结果后修改层级; Top1/Top3 只作为描述性业务排序指标

## 阶段2.4 指标精确定义 (只冻结口径, 不执行)

- ROC-AUC: implementation = sklearn.metrics.roc_auc_score, positive_label = 1; 评估标签必须同时包含 0 和 1, 否则 否则 metric_status = METRIC_UNDEFINED, metric_value = null (不得填 0.5)
- PR-AUC: implementation = sklearn.metrics.average_precision_score (metric_name = average_precision); 不是 precision-recall 曲线梯形积分; 无正样本 → METRIC_UNDEFINED
- Brier score: 公式 = mean((y - p)^2); 预测必须有限且位于 [0,1]; 否则模型评估失败, 不得裁剪后掩盖非法预测
- Log loss: epsilon = 1e-15; 该裁剪只用于计算 log loss, 不得修改保存的原始预测概率
- Calibration: logit(P(Y=1)) = alpha + beta * z (implementation = statsmodels GLM Binomial, maxiter = 100, tol = 1e-10); calibration_intercept = alpha, calibration_slope = beta
- M0 校准: calibration_slope = NOT_APPLICABLE; 校准截距 = calibration-in-the-large: logit(clip(observed_rate, 1e-15, 1 - 1e-15)) - logit(clip(predicted_rate, 1e-15, 1 - 1e-15)); observed_rate = mean(y), predicted_rate = constant_prediction
- 校准指标 calibration metrics are descriptive only; 不属于六条最终资格门, 不参与候选比较算法
- 指标状态字段: OK / METRIC_UNDEFINED / NOT_APPLICABLE / INPUT_VALIDATION_FAILURE / NUMERICAL_FAILURE; 不得使用 0、0.5 或空字符串伪装不可计算指标

## 系数稳定性只作诊断 (阶段2.5 报告, 不作硬资格门)

- 报告每个特征在有效 fold 中的系数、符号、expected direction 一致比例、均值/标准差/最小/最大
- expected direction 一致比例不作为阶段2.6 硬资格门; 不得因系数方向不符而重拟合或删除变量

## Target-blind 锁定测试

- 七月锁定审计: 39/39 项 PASS
- 审计明细: v004c_stage2_3_holdout_lock_audit_v001.csv

## 声明

阶段2.3 只冻结候选模型规格和后续评价协议;
M1、M2和M3由研究人员在阅读六月阶段2.2单因子结构和稳定性证据后人工确定, 候选规格在任何七月标签揭示前冻结;
阶段2.3运行时未读取 Target 列, 未根据 AUC、bootstrap、flags 或其它统计 自动搜索、增加、删除或替换候选成员;
没有训练模型; 没有生成预测; 没有读取七月标签; 没有运行 Walk-forward;
没有搜索超参数; 没有搜索交易阈值;
阶段1、阶段2.1 和阶段2.2 资产未被修改;
没有自动 commit 或 push。
