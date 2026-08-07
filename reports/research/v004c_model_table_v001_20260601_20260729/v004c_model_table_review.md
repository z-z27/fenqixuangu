# v004c 统一建模表 v001 (v004c_model_table_v001)

- 阶段: 正式多变量模型开发前的最后一个数据工程阶段 (构建模型输入表, 不训练模型)
- 构建日期: 2026-08-07
- 分支: research-sample-analysis (模型表构建 commit ab2b067; 特征覆盖 commit 91bd2009)

## 0. 输入版本与来源

1. **Stage 1 输入版本**: `v004c_d1_dataset_v001_20260601_20260729` (ref `v004c-d1-dataset-0.1`)
   - `v004c_training_d1_v001.csv` SHA256: `cd0108bb80ed4f3bfa1a2573fe89094562ee43578decd28ed1529a849ca1f214`
   - `v004c_d1_column_lineage.csv` SHA256: `914f0a34a34a2d9e9f5b8a0420e89666c8b8d3a3ba8d28e46a8dd54c1791ba6c`
2. **Stage 2.1 输入版本**: `v004c_factor_dictionary_v001_20260601_20260729` (ref `v004c-factor-dictionary-0.1`)
   - `v004c_feature_allowlist_primary_v001.csv` SHA256: `03e8416bc8428dc2f1fd2c054e66e5a744e270fb27a19bbd9aa79f030241e9c9`
   - `v004c_feature_allowlist_sensitivity_v001.csv` SHA256: `10e41546c409e40fef66471b300a3b69dec09877f321e59c0fd7d9224804ad1b`
   - `v004c_feature_exclusions_v001.csv` SHA256: `5e9b2bd9c540b4cef27e27e22737ac061b0ef3e8a861d8e998ac8624e4877979`
   - `v004c_predeclared_derived_factor_spec_v001.csv` SHA256: `d495112272ef15427894aec630b09c805ca6fecccd2d75f3bc416f958e678896`
3. **特征覆盖结论**: `v004c_feature_coverage_v001.csv` SHA256: `835fb7541b9845cc149b7c737789d325ba3a2c3eed3f7c3311a4cc9f864ccdfa` (commit 91bd2009)
4. **日线缓存**: `data\cache\daily` (offline, 未联网, 未修改 180/120 窗口)

## 1. 表结构总览

- 总行数: 333 (一行一个 D1 事件)
- signal dates: 42
- model table 列数: 65
- schema rows: 65 (schema 只描述实际输出表列, 与 table 列一一对应)
- feature 数量: 55 (primary 29 + sensitivity 26)
- identifier 数量: 4
- label 数量: 1
- audit 数量: 5
- event_id 唯一: True

## 2. 列角色

- **IDENTIFIER**: event_id, code, signal_date, board_streak_before_break
- **FEATURE (PRIMARY, 29)**: break_open_return, break_high_return, break_close_return, break_touched_limit_up, volume_above_d1_close_ratio, late_day_sell_volume_ratio, d1_close_to_ma5_raw, d1_high_to_close_drawdown_raw, d1_low_to_close_recovery, d1_open_to_close_return_raw, break_volume_ratio_vs_board_days, d1_close_to_vwap_raw, break_day_in_pool, high_zone_volume_ratio, d1_close_to_ma10_raw, d1_close_to_ma20, d1_ma10_slope, d1_low_to_ma5_raw, d1_ma5_slope, d1_true_reclaim_ma5, d1_afternoon_return, d1_intraday_range, d1_last_hour_return, d1_up_bar_volume_ratio, down_bar_volume_ratio, board_streak_is_3, recent_7d_cumulative_return, recent_7d_max_drawdown, recent_7d_close_position
- **FEATURE (SENSITIVITY, 26)**: board_day_amount_rank, board_day_turnover_rank, board_day_volume_rank, break_upper_shadow_ratio, break_lower_shadow_ratio, break_amount_ratio_vs_board_days, break_opened_from_limit_up, consecutive_days_below_ma5, consecutive_days_below_ma10, d1_close_location, amount_above_d1_close_ratio, high_zone_amount_ratio, late_day_sell_amount_ratio, last_board_day_in_pool, pool_consecutive_count_last_board, d1_close_above_ma5, d1_close_above_ma10, d1_true_reclaim_ma10, d1_close_to_ma5_bucket, d1_open_to_close_bucket, d1_high_to_ma5_raw, d1_low_to_ma10_raw, ma5_overheat_10, overrepair, break_volume_abnormality, profit_pressure
- **LABEL**: target7_daily_d2open_d3high
- **AUDIT_ONLY**: break_date, daily_label_quality_ok, target_training_eligible, sample_role_v022, d1_factor_quality_ok

## 3. 3 个 recent-7d 字段 (RECENT_7D_PATH 机制)

窗口: T-6..D1 共 7 个个股实际有效日线相邻行 (不使用自然日; 不足 7 行失败当前事件, 不换窗口)。

| field | 定义 | 覆盖率 | min | median | max |
|---|---|---|---|---|---|
| recent_7d_cumulative_return | close(D1) / close(T-6) - 1, T-6 = D1 前第 6 个交易日 (窗口 7 行) | 333/333 (0 缺失) | -0.206580 | 0.190361 | 0.639281 |
| recent_7d_max_drawdown | max_{t∈[T-5..D1]} max(0, (prior_peak_t - low_t) / prior_peak_t); prior_peak_t = max(high_s), s < t (严格较早交易日, 不使用同日 high/low 顺序) | 333/333 (0 缺失) | 0.025271 | 0.100567 | 0.403666 |
| recent_7d_close_position | (close(D1) - 7d_low) / (7d_high - 7d_low); 7d_high==7d_low 置空 | 333/333 (0 缺失) | 0.356787 | 0.779079 | 1.000000 |

严格日级最大回撤: `prior_peak_t = max(high_s), s < t` (只用严格较早交易日 high, 不使用同日 high/low 先后顺序; 当前日 high 只成为后续 prior peak; 所有后续 low 高于 prior peak 时回撤 = 0)。

recent-7d 三字段的授权来源是 feature coverage (coverage_status=NEW_REQUIRED / recommended_role=M2_PRIMARY_CANDIDATE / available_as_of=D1_CLOSE / mechanism=RECENT_7D_PATH), 不要求存在于旧 Stage 2.1 allowlist。

## 4. 六月/七月数据完整性描述 (只允许行数与标签计数, 禁止模型表现)

| 月份 | 行数 | signal date 数 | Target7 阳性计数 |
|---|---|---|---|
| 2026-06 | 173 | 21 | 59 |
| 2026-07 | 160 | 21 | 41 |

## 5. Target 处理

- `target7_daily_d2open_d3high`: binary (True 100 / False 233), 严格二元验证通过, 未标准化/未裁剪/未做任何预处理。
- 只属于 LABEL 角色 (1 列), 绝不在 FEATURE_COLUMNS 中; feature contract 来自 Stage 2.1 规则 + 91bd2009 覆盖结论, 与标签表现无关。

## 6. 数据质量与泄漏检查

- 未来字段 (D2/D3/POST_D3/future/outcome/tail_loss 等 token 扫描): 无
- 旧模型输出 (v002/v004a/v004b/v005/recognition/policy): 无
- duplicate alias (ED01-13 别名列): 无 (仅 canonical 列进入 universe)
- primary 缺失: 0 (任何缺失 => FAIL MODEL TABLE BUILD)
- primary ±inf: 0
- binary 非法值: 0 (严格二元验证: board_streak_is_3, break_day_in_pool, break_opened_from_limit_up, break_touched_limit_up, d1_close_above_ma10, d1_close_above_ma5, d1_true_reclaim_ma10, d1_true_reclaim_ma5, last_board_day_in_pool, ma5_overheat_10)
- 全部数值 feature ±inf: 0
- event_id 唯一: True
- code 合法 (6 位数字): True
- signal_date 有效 (YYYY-MM-DD 且存在于个股日线缓存): True
- signal_date == break_date 逐行一致: True

## 7. Sensitivity 字段结构性缺失 (允许, 明确记录)

以下 sensitivity 字段在冻结 Stage 1 中既有结构性缺失, 模型表保留原值 (不填充):

| feature | missing |
|---|---|
| board_day_amount_rank | 34 |
| board_day_turnover_rank | 34 |
| board_day_volume_rank | 9 |
| break_amount_ratio_vs_board_days | 2 |
| break_lower_shadow_ratio | 2 |
| break_upper_shadow_ratio | 2 |
| d1_close_location | 2 |
| pool_consecutive_count_last_board | 9 |

未来第一版 M1/M2 预计只使用完整 primary 字段。

## 8. 明确排除的字段 (排除清单与原因由 feature coverage 资产负责)

以下字段不得进入 PRIMARY/SENSITIVITY universe; 逐字段排除原因见 `v004c_feature_coverage_v001.csv` (coverage_status / recommended_role / reason 列), model-table schema 不重复维护排除清单:

- **CONTEXT_ONLY (REUSE_AS_CONTEXT, 不升级为 primary)**: recent_limit_up_count_10d, recent_limit_up_count_20d, recent_pool_appearance_count_10d, recent_pool_appearance_count_20d, max_board_streak_20d
- **DEFER_NOT_REQUIRED_FOR_V001**: d1_ma20_slope, recent_7d_limit_up_count, board_stage_volume_price_decomposition
- **REDUNDANT (可由已有字段精确确定)**: profit_chip_ratio, d1_vwap_to_close_gap
- **FORBIDDEN (未来/标签/旧模型输出)**: d2_open_daily, d3_high_daily, recognition_score, v002_rank, v004a_probability
- 其余 REDUNDANT / NOT_NEEDED / FORBIDDEN / AUDIT_ONLY / DERIVE_ONLY 候选字段同样不进入模型表; 全部由 `v004c_feature_coverage_v001.csv` 统一记录。
- 本 schema (`v004c_model_table_schema_v001.csv`) 只描述实际输出表列 (65 行 = model table 列数), 不含任何不在表中的字段。

## 9. M1/M2 槽位 (本阶段不选模型)

- **M1 D1 STRUCTURE BASELINE**: exact features = **NOT_FROZEN**
- **M2 INTEGRATED PRIMARY**: exact features = **NOT_FROZEN**
- 不输出 m1_features.json / m2_features.json / candidate_model_specs

## 10. 预处理声明

- 本阶段未做任何全样本预处理 (无 z-score / winsorize / P01-P99 clip / 标准化)。
- 模型表保存 raw legal feature values; 预处理参数必须由下一阶段每个 walk-forward fold 只用训练 fold 计算。
- recent-7d 三字段在 schema 中的 preprocess_policy 标记为 FOLD_CLIP_Z (临时约定, 与 Stage 2.1 连续变量约定一致, 不构成模型冻结); 实际 clip 阈值、均值、标准差等参数只能在下一阶段每个 training fold 内部拟合, 本阶段未生成任何全样本预处理参数。

## 11. 原子输出

- 构建到 sibling tmp 目录 → 写 3 文件 → 磁盘读回全验证 → rename → 正式目录
- 失败时正式目录不存在; 若正式目录已存在则 FAIL (不覆盖)
- 正式输出目录: `reports\research\v004c_model_table_v001_20260601_20260729`

## 12. 最终状态

- 完成: `v004c_model_table_v001.csv` 确定性生成 (333 行 / 42 signal dates / 55 features) ✓
- 训练: 无 (本阶段禁止 Logistic Regression / M0 / M1 / M2)
- 预测: 无 (无概率 / rank / Top3 输出)
- M1/M2 冻结: 无
