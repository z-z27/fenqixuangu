# v004c Pairwise v1 Feature Contract v002 Review (Pool Correctness Fix)

- 输出: v004c_pairwise_v1_feature_contract_v002_20260506_20260630
- universe: {len(input_df)} 行 / {input_df['signal_date'].nunique()} 信号日
- 与 v002 身份逐行一致: PASS
- FEATURE 数 (after): 53 (before: 53)

## 1. Pool 历史恢复情况

- Required pool range:
  - start: 2026-04-03
  - end: 2026-06-30
  - required trading dates: 58
- Before:
  - pool dates complete: 39
  - pool dates missing: 19 (2026-04-03..2026-04-30)
- Recovery:
  - attempted: 19 (recover_v004c_pool_history_v002.py, canonical daily_limitup_derived)
  - recovered: 19
  - still missing: 0
- After:
  - complete dates: 58
  - incomplete dates: 0
- pool source: daily_limitup_derived (canonical pipeline)
- 完整性核对 (recover_v004c_pool_completeness_v002.py): 04-03..04-30 恢复池零缺口; 06 月原有池文件 12 个日期共 23 只缺口已补日线并重新 derive (002217 型缓存缺口根因)

## 2. Pool-derived FEATURE 决策

- recent_pool_appearance_count_10d: KEEP_FEATURE
- recent_pool_appearance_count_20d: KEEP_FEATURE
- board_day_volume_rank: KEEP_FEATURE
- break_day_in_pool: KEEP_FEATURE
- last_board_day_in_pool: KEEP_FEATURE
- pool_consecutive_count_last_board: KEEP_FEATURE

## 3. board_day_volume_rank

- denominator definition: FULL_D0_LIMIT_UP_POOL (当日完整涨停池成员, 非 development candidate subset)
- D0 dates audited: 39
- complete rank denominator dates: 39
- incomplete rank denominator dates: 0
- total pool members audited: 25405
- daily volume available: 25405
- daily volume missing: 0
- old subset-ranking bug eliminated: YES

## 4. Missing 分类

- STRUCTURAL_MISSING: 3 (数学退化, 如 603065 high==low)
- SOURCE_MISSING: 0 (pool 数据源缺口)
- UNEXPECTED_MISSING: 0 (必须 0)

## 5. High-zone contract 公式修正

- high_zone_volume_ratio: Σvolume[typical_price >= d1_low + 0.7*(d1_high-d1_low)] / Σvolume; typical_price = (high + low + close) / 3
- high_zone_amount_ratio: Σamount[typical_price >= d1_low + 0.7*(d1_high-d1_low)] / Σamount; typical_price = (high + low + close) / 3
- 实现核对 (v004c_minute_features.py): typical_price=(high+low+close)/3, 分子 Σ[typical_price >= thr], 一致

## 6. Pairwise 层 eligibility

- pairwise_v1_feature_source_complete: 319/319
- pairwise_v1_training_eligible: 319/319
- dev_* 列保持 AUDIT_ONLY (v002 foundation 层)

## 7. 泄漏审计

- FEATURE future leakage: 0 (必须 0)
- FEATURE model-output leakage: 0 (必须 0)
- FEATURE label-lineage leakage: 0 (必须 0)
- D1 availability violations: 无

## 8. Deterministic Rebuild

- 两轮构建 9 个核心资产字节一致: PASS

## 9. 严格禁止确认

- 未训练 Pairwise Ridge / Logistic / Tree/GBDT
- 未搜索 lambda / 未做 walk-forward
- 未做 Target7 / tail-loss / AUC / IC 分析
- 未新增 FEATURE / 未恢复 amount/turnover rank
- 未创建 feature_x_board3 交互列

## 10. 结论

- Feature Contract 状态: READY_FOR_PAIRWISE_V1

