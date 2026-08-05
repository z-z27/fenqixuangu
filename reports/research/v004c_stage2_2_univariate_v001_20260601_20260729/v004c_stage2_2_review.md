# v004c 阶段2.2 — 单因子结构、六月开发集方向与日期稳定性审计报告 (自动生成)

- 分析版本: v004c-stage2-2-univariate-0.1
- 来源: 阶段1 v004c-d1-dataset-0.1 → a65661f4738b849a06efb4859d4100842871e971 / 阶段2.1 v004c-factor-dictionary-0.1 → 11f455dc9a6643a67250321cce68868be904cf06
- 分支: research-sample-analysis / HEAD: ee0306fc4a5f3dadd3f0a780721754a014c58c58

## 版本链与输入验证

- 阶段2.1 tag 目标 = 11f455dc9a6643a67250321cce68868be904cf06 (通过)
- 阶段1 tag 目标 = a65661f4738b849a06efb4859d4100842871e971 (通过)
- 阶段2.1数据提交是当前 HEAD 祖先: True
- 阶段2.1 git_head.txt = aa62b315dd1f792dd7657b190e41c1e992f93112 (通过)
- 输入 SHA 与对应 manifest 一致: v004c_d1_column_lineage.csv=OK, v004c_d1_data_manifest.json=OK, v004c_d1_snapshot_v001.csv=OK, v004c_factor_dictionary_manifest.json=OK, v004c_factor_dictionary_v001.csv=OK, v004c_feature_allowlist_primary_v001.csv=OK, v004c_feature_allowlist_sensitivity_v001.csv=OK, v004c_near_duplicate_pairs_v001.csv=OK, v004c_predeclared_derived_factor_audit_v001.csv=OK, v004c_predeclared_derived_factor_spec_v001.csv=OK, v004c_training_d1_v001.csv=OK
- 阶段1 manifest 审计参考校验: True
- 结构门: 333 行 / 197 列 / 42 信号日

## 时间切片 (七月锁定)

- 六月开发集: 2026-06-01 .. 2026-06-30 / 173 行 / Target7 正样本 59
- 七月锁定切片: 2026-07-01 .. 2026-07-29 / 160 行 (仅无标签使用)
- 七月标签列已删除/mask: 57 列 (locked_target_masked=True)
- locked_target_access = False

## 59 因子完整覆盖

- primary 分析项: 38 (source 32 + derived 6)
- sensitivity: 21
- 总分析因子: 59 (唯一)
- 分析类型: CONTINUOUS 36 / ORDINAL 8 / RANK 3 / BINARY 10 / BUCKET 2
- 派生因子重算与阶段2.1审计一致: True

## 各机制组因子数

| 机制组 | 因子数 | primary | sensitivity | derived |
|---|---|---|---|---|
| BOARD_HISTORY | 5 | 5 | 0 | 0 |
| D0_CROSS_SECTION | 3 | 0 | 3 | 0 |
| D1_CHIP_DISTRIBUTION | 4 | 2 | 2 | 0 |
| D1_LATE_DAY_PRESSURE | 2 | 1 | 1 | 0 |
| D1_MA_POSITION | 15 | 9 | 6 | 0 |
| D1_PRICE_ACTION | 15 | 10 | 5 | 0 |
| D1_VOLUME_ACTIVITY | 4 | 3 | 1 | 0 |
| D1_VWAP_POSITION | 2 | 1 | 1 | 0 |
| POOL_MEMBERSHIP | 3 | 1 | 2 | 0 |
| PREDECLARED_DERIVED | 6 | 6 | 0 | 6 |

## 描述性 flags 数量

| flag | 数量 |
|---|---|
| BOARD_DIRECTION_DISAGREEMENT | 18 |
| BOARD_SUPPORT_INSUFFICIENT | 0 |
| BOOTSTRAP_CI_CROSSES_ZERO | 53 |
| BOOTSTRAP_LOW_VALID_REPLICATES | 4 |
| CLIP_DIRECTION_FLIP | 0 |
| DATE_SIGN_UNSTABLE | 5 |
| DISTRIBUTION_SHIFT | 4 |
| HIGH_DOMINANCE | 8 |
| LOW_BINARY_SUPPORT | 7 |
| LOW_COVERAGE | 0 |
| MISSINGNESS_RATE_DIFFERENCE | 2 |
| MUTUAL_EXCLUSION_ALTERNATIVE | 7 |
| QUANTILE_COLLAPSE | 10 |
| SENSITIVITY_ONLY | 21 |
| SINGLE_DATE_INFLUENTIAL | 4 |

## bootstrap 设置

- 日期 cluster bootstrap: 1000 replicates, seed = 20260805
- 每个 replicate 从 21 个日期 cluster 中有放回抽取 21 次 (bootstrap_unit=signal_date_cluster);
- 抽中日期时带入该日期全部样本行, 同一日期抽中多次时其全部行重复进入样本 (不去重, 不拆分 cluster);
- 59 个因子共用同一抽样矩阵 (shared_draw_matrix_across_features=true);
- CI 只使用有效 replicate; 无有效 replicate 时均值/中位数/CI 留空 (不填 0);
- 有效 replicate 低于 800 的因子数: 4 (有效 = 1000 的因子数: 52)
- bucket 区间效应 (max-min 天然 >= 0) 不解释为相对 0 的方向性证据;

## LODO 日期稳定性

- 六月 signal_date 数: 21
- 每个因子逐日期删除整日样本后重算主效应, 记录符号一致性与最大绝对变化
- 因子结果完整保存在 v004c_stage2_2_lodo_date_stability_v001.csv

## 二板/三板子组支持

- 支持门使用因子 complete-case 样本 (non_null>=20, positive>=5, negative>=5):
- board=2: VERIFIED 59 / 59
- board=3: VERIFIED 59 / 59
- 方向统计: 可判定 42 (一致 24 / 相反 18) / 不可判定 17
- 反向因子中近零 (|效应| < 0.01) 数量: 3 (仅人工解释, 不改变冲突定义)

## 缺失结构

- 六月存在缺失的因子数: 8
- 缺失结构明细: v004c_stage2_2_missingness_dev_v001.csv

## 无标签分布漂移 (六月 vs 七月)

- 触发 DISTRIBUTION_SHIFT 的因子数: 4 (数值 4 / 类别 0)
- 漂移指标明细: v004c_stage2_2_distribution_shift_unlabeled_v001.csv

## Target-blind 锁定测试

- 七月锁定审计: 20/20 项 PASS
- 审计明细: v004c_stage2_2_holdout_lock_audit_v001.csv

## 声明

阶段2.2 只完成单因子结构和六月开发集稳定性审计;
七月标签未用于任何因子分析、因子选择或关联计算;
没有新增因子; 没有自动筛选因子; 没有训练模型; 没有运行完整 Walk-forward;
没有修改阶段1或阶段2.1资产; 没有自动 commit 或 push。
