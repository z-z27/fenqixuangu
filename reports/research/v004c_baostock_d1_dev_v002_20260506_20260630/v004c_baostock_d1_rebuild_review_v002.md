# v004c BaoStock Dev Foundation Correctness Fix v002 — Review

任务性质: **定点正确性修复** (不训练 / 不筛选 / 不调参 / 不推翻 v001 已验证结论)。修复项: P1 May label recovery 执行顺序 (recovery -> patch universe -> build rows, 恢复结果真正写入 dev CSV) + UNKNOWN != FALSE (不完整 label 的 target 显式 NA) + daily recovery 只追加缺失日期 (existing canonical rows immutable) + development eligibility 语义 (dev_label_complete / dev_feature_source_complete / dev_training_eligible, May 全量 live re-audit, June 冻结链, 字段语义一致) + structural missing 分类 (603065) + BaoStock provider strict datetime clipping / fail-closed + provenance 从 cache meta 读取 (含 SHA256 / 包版本) + DETERMINISTIC_REBUILD 正式写入本 review。

- 窗口: 2026-05-06..2026-06-30; 输出: v004c_baostock_d1_dev_v002_20260506_20260630/; v001 报告保留审计; fetch timestamp 仅存于 provenance JSON

- July 不进入开发数据; 冻结资产 (repair_state / logistic_walkforward / d1_dataset) 未修改; v001 的 319/319 D1 5min 覆盖结论不推翻


## 1. Universe (v001 -> v002 候选池必须精确一致)

| window | rows | signal_dates | codes |
|---|---|---|---|
| may | 146 | 18 | 137 |
| june | 173 | 21 | 158 |

- v001 rows: 319; v002 rows: 319; event_id/code/signal_date/board_streak_before_break exact match: True (319/319)

## 2. BaoStock 5min 获取与覆盖 (source=baostock_5m, adjustment=none, interval=5m)

| window | bao_fetched | 48-bar complete | minute_features_complete(legacy) | daily_features_complete | final_x_complete(legacy audit) |
|---|---|---|---|---|---|
| may | 146/146 | 146/146 (100.0%) | 146/146 | 146/146 | 146/146 |
| june | 173/173 | 173/173 (100.0%) | 172/173 | 173/173 | 172/173 |

- minute_feature_complete / final_x_complete 为 legacy literal audit (§22/§45), 不直接用于模型资格; 资格见第 8 节 dev_feature_source_complete。

## 3. D1 48-bar completeness (正式规则: 48 bar / 09:35 首 / 15:00 末)

逐 event 明细见 v004c_baostock_d1_minute_quality_v002.csv; 缺失原因分类 (missing_reason) 复用 v004c_may_d1_coverage 正式分类。

- may: 146/146 完整 (100.0%)
- june: 173/173 完整 (100.0%)

## 4. May label recovery (§7-§10) 与正式 dev CSV 更新

| event_id | code | signal_date | label_d2_date | label_d3_date | reason_before | label_complete_before | label_complete_after | label_recovered | target7_after | tail_loss_after |
|---|---|---|---|---|---|---|---|---|---|---|
| 000402_2026-05-07 | 000402 | 2026-05-07 | 2026-05-08 | 2026-05-11 | d3_high_invalid(2026-05-11)|d3_close_invalid(2026-05-11) | False | True | True | False | False |
| 000553_2026-05-07 | 000553 | 2026-05-07 | 2026-05-08 | 2026-05-11 | d3_high_invalid(2026-05-11)|d3_close_invalid(2026-05-11) | False | True | True | False | False |
| 002149_2026-05-07 | 002149 | 2026-05-07 | 2026-05-08 | 2026-05-11 | d3_high_invalid(2026-05-11)|d3_close_invalid(2026-05-11) | False | True | True | False | False |
| 603158_2026-05-08 | 603158 | 2026-05-08 | 2026-05-11 | 2026-05-12 | d3_high_invalid(2026-05-12)|d3_close_invalid(2026-05-12) | False | True | True | False | False |

- recovered: 4/4 (before 状态取冻结 candidates CSV, 跨运行确定; after 取当前 canonical daily cache 的正式 audit_daily_label)

- **formal dev CSV updated: YES** — recovery 先于 build_event_row 完成, universe label state 统一由 audit_may_labels_live 重新审计后进入行构建; 4 条 recovered 事件在 v002 dev CSV 中 dev_label_complete=True 且 target 为实际审计结果 (程序内 assert, 非人工 review)

- **UNKNOWN != FALSE**: dev_label_complete=False 的事件 target7/…/tail_loss 在 dev CSV 中为显式 NA (非 False), 三态 (True/False/NA) 可区分 (§5/§37)

### Daily cache recovery safety (§53C)

| metric | value |
|---|---|
| overlapping existing dates overwritten | 0 (append-only 过程保证) |
| new dates appended | 0 |
| overlap dates observed | 0 |
| overlap value diff count | 0 |

## 5. June 重建 vs 冻结表交叉核对

| column | match_rate | max_abs_diff |
|---|---|---|
| d1_open | 1.0000 | 0.000e+00 |
| d1_high | 1.0000 | 0.000e+00 |
| d1_low | 1.0000 | 0.000e+00 |
| d1_close | 1.0000 | 0.000e+00 |
| d1_ma5 | 1.0000 | 2.842e-14 |
| d1_ma10 | 1.0000 | 5.684e-14 |
| d1_ma20 | 1.0000 | 5.684e-14 |
| d1_close_to_ma5_raw | 1.0000 | 2.359e-16 |
| d1_low_to_ma5_raw | 1.0000 | 2.429e-16 |
| d1_high_to_ma5_raw | 1.0000 | 4.718e-16 |
| d1_close_to_ma10_raw | 1.0000 | 4.996e-16 |
| d1_low_to_ma10_raw | 1.0000 | 2.776e-16 |
| d1_close_to_ma20 | 1.0000 | 4.996e-16 |
| d1_ma5_slope | 1.0000 | 3.018e-16 |
| d1_ma10_slope | 1.0000 | 4.233e-16 |
| d1_close_above_ma5 | 1.0000 | 0.000e+00 |
| d1_close_above_ma10 | 1.0000 | 0.000e+00 |
| d1_true_reclaim_ma5 | 1.0000 | 0.000e+00 |
| d1_true_reclaim_ma10 | 1.0000 | 0.000e+00 |
| consecutive_days_below_ma5 | 1.0000 | 0.000e+00 |
| consecutive_days_below_ma10 | 1.0000 | 0.000e+00 |
| break_open_return | 1.0000 | 9.910e-17 |

- 口径: 重建日线/MA 状态与冻结 v004c_training_d1_v001.csv 同列对比, epsilon=1e-09; d1_volume/d1_amount 为 5min 派生, 不在列内。

## 6. Sina vs BaoStock regression (normalized contract)

- 比对事件数: 165 (June 两源均有 48-bar 的事件)
- normalized columns same: False (schema 差异列: ['interval'])
- timestamps same (09:35..15:00): True
- 48-bar semantics same: True (sina=165 bao=165)
- volume units same (median bao/sina per-bar ratio): 1.0 (p10=0.9952972074822931, p90=1.0077948541804271)
- amount units same (median bao/sina): 1.0000000045823965
- 不要求 OHLC 相同 (源间 bar 级取整差异由 48-bar 语义审计覆盖)

## 7. minute_data_coverage_selection_bias

- **ELIMINATED**: 全部 319 个 event 的 D1 5min 数据均由 BaoStock 获取且 48-bar 完整 (319/319), 覆盖率不再由旧 Sina cache 存在与否决定。
  证据: may 146/146 (100%), june 173/173 (100%)

- 该结论独立于 603065 的 d1_close_location structural NaN (第 9 节): structural missing 是公式数学退化, 不是 minute data coverage 缺失 (§41)。

## 8. Development eligibility (§54)

| window | rows | dev_label_complete | dev_feature_source_complete | dev_training_eligible | board2 | board3 |
|---|---|---|---|---|---|---|
| may | 146 | 146/146 | 146/146 | 146/146 | 116 | 30 |
| june | 173 | 173/173 | 173/173 | 173/173 | 145 | 28 |
| combined | 319 | 319/319 | 319/319 | 319/319 | 261 | 58 |

- signal dates: May 18 + June 21 = 39; minute_source unique values: ['baostock_5m']

- dev_training_eligible 是**数据资格** (数据层有资格进入后续模型开发), 不是模型选择 / Target 驱动筛选 / sample weight; 下一阶段模型仍可能因 fold-only preprocessing / 特征可用性 / 训练起始窗口等原因变化 (§24)。

- legacy target_training_eligible (v0.2.2 role) 仅作 AUDIT_ONLY lineage, 不进入本表统计; May 无 legacy chain, 该列为 NA (§6/§32)。

## 9. Structural missing (§55)

| event_id | feature | reason |
|---|---|---|
| 603065_2026-06-11 | d1_close_location | intraday high == low (D1 日内 5min high==low, d1_close_location 官方公式分母 (high-low) 为 0 -> None; STRUCTURAL_MISSING, 非 vendor / data coverage failure) |

- structural missing 是**公式数学退化** (明确定义允许), 不是 vendor / data coverage failure; 不使 dev_feature_source_complete 变 False (§20/§21)。
- structural missing rows: 1

## 10. Unexpected missing (§55)

- 无
- unexpected missing rows: 0

## 11. 603065_2026-06-11 专项确认 (§52)

- D1 open=high=low=close=12.36 (一字板, 日内 5min high==low); d1_close_location 官方公式 (close-low)/(high-low) 分母为 0 -> 返回 None, 数学定义未修改 (禁止填 0/0.5/1, §19)
- 分类: STRUCTURAL_MISSING (structural_missing_count=1: d1_close_location); unexpected_missing_count=0; dev_feature_source_complete=True; dev_training_eligible=True (label complete) — 样本保留在新开发集 (程序内 assert)

## 12. Provenance (§56)

- source = baostock_5m; adjust = none; interval = 5m
- cache dir (repo-relative): data/cache/baostock_5m
- cached codes: 264; date range: 2026-05-06 .. 2026-06-30; total rows: 493344
- cache SHA256 coverage: 264/264 codes
- baostock package version: 0.9.3
- 每 code 明细 (含 fetch_timestamp) 见 v004c_baostock_d1_fetch_provenance_v002.json; fetch timestamp 独立于确定性资产

## 13. DETERMINISTIC_REBUILD (§29/§30/§57)

| asset | build #1 sha256 | cache-only 重建 #2 sha256 | identical |
|---|---|---|---|
| dev CSV | 1bf17299b78a58dfa2773722e5c3f48540351cbfa3e9cac3b0f4c2c8154e7254 | 1bf17299b78a58dfa2773722e5c3f48540351cbfa3e9cac3b0f4c2c8154e7254 | True |
| quality CSV | 04ea279febd3c2528796baedfb65c561f5d1e60872ba96d9e21c0ed8aec1c30e | 04ea279febd3c2528796baedfb65c561f5d1e60872ba96d9e21c0ed8aec1c30e | True |
| lineage CSV | 8953214aced5c2839886e537e36cb5301f01acba584c9b6166b77b4ffb38250a | 8953214aced5c2839886e537e36cb5301f01acba584c9b6166b77b4ffb38250a | True |
| review | (两轮渲染字节比较, 不内嵌自身 hash §30) | | True |

- **DETERMINISTIC_REBUILD = PASS** (build #1 vs cache-only 重建 #2: dev / quality / lineage byte-identical, review 两轮渲染字节一致)

## 14. Provider contract (§53A)

- strict datetime clipping: **PASS** — baostock API 只支持按日参数, normalize 后再次执行 start<=datetime<=end, 与 sina_5m 对外 contract 一致; partial-window (10:00..11:00) 请求返回无 09:35/15:00 等区间外 bar (单元测试)
- raw duplicate fail-closed: **PASS** — normalize_baostock_5m_frame 对重复 (code, datetime) raise RuntimeError, 无 silent drop (单元测试)
- invalid-row fail-closed: **PASS** — unparseable datetime / NaN OHLC / NaN volume/amount / 负价格 / high<open / low>open / 负 volume/amount 一律 fail closed (normalize + validate)
- suspension placeholder: **PASS** — OHLC 全 0 停牌占位 bar 显式分类为 BAOSTOCK_SUSPENSION_PLACEHOLDER 后移除, 与 INVALID_MARKET_BAR 严格区分
- silent fallback: **保持禁止** — baostock 失败不尝试 sina
- Sina compatibility: **保留** — normalize_5min_frame (Sina 路径) 行为未改

## 15. 限制

- 日线 / MA / recent-7d 状态来自现有 canonical daily cache; 若缓存含历史回补, June 交叉核对 (第 5 节) 会显示差异
- d1_volume/d1_amount 属 22 个 minute features (官方列表), 由 BaoStock 5min 求和; legacy 官方构建为 Sina 5min 求和, 两者数值等价但非同一 raw lineage (见 lineage CSV lineage_note, §25)
- 5min 源替换只影响 minute-dependent features; 冻结模型 / 阈值 / lambda / 特征选择均未重做 (本任务禁止); Feature Contract 扩展 (break_high_return 等) 属下一阶段, 本任务未添加
- 本任务实际运行状态为已恢复 (v001 已写回 daily cache): 两轮构建均无网络 / 无 daily cache 写入 (rows_appended=0); 确定性证据基于该状态 (§9)
