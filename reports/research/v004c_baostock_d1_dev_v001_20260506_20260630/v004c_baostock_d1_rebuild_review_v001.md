# v004c BaoStock 5m Integration + May/June Historical Rebuild v001 — Review

任务性质: **HISTORICAL DATA FOUNDATION REBUILD** (不训练 / 不筛选 / 不调参)。BaoStock 5m 为 May+June D1 历史 5min 主数据源, Sina 行为完全保留; 22 个 minute-dependent features 用官方公式 (build_v004c_dataset.py E/F 段 1:1) 从 BaoStock 重建, 日线 / MA / recent-7d 状态用现有 canonical daily cache 公式重建, Target 标签全部来自现有审计链, 不重算。

- 窗口: 2026-05-06..2026-06-30; 输出: v004c_baostock_d1_dev_v001_20260506_20260630/; fetch timestamp 仅存于 v004c_baostock_d1_fetch_provenance.json

- July 不进入开发数据; 冻结资产 (repair_state / logistic_walkforward / d1_dataset) 未修改


## 1. Universe

| window | rows | signal_dates | codes |
|---|---|---|---|
| may | 146 | 18 | 137 |
| june | 173 | 21 | 158 |

## 2. BaoStock 5min 获取与覆盖 (source=baostock_5m, adjustment=none, interval=5m)

| window | bao_fetched | 48-bar complete | minute_features_complete | daily_features_complete | final_x_complete |
|---|---|---|---|---|---|
| may | 146/146 | 146/146 (100.0%) | 146/146 | 146/146 | 146/146 |
| june | 173/173 | 173/173 (100.0%) | 172/173 | 173/173 | 172/173 |

## 3. D1 48-bar completeness (正式规则: 48 bar / 09:35 首 / 15:00 末)

逐 event 明细见 v004c_baostock_d1_minute_quality_v001.csv; 缺失原因分类 (missing_reason) 复用 v004c_may_d1_coverage 正式分类: cache_missing / d1_date_missing / bar_grid_incomplete / grid_violation。

- may: 146/146 完整 (100.0%)
- june: 173/173 完整 (100.0%)

## 4. May label recovery (§15)

| event_id | code | signal_date | label_d2_date | label_d3_date | reason_before | label_complete_before | label_complete_after | label_recovered | target7_after | tail_loss_after |
|---|---|---|---|---|---|---|---|---|---|---|
| 000402_2026-05-07 | 000402 | 2026-05-07 | 2026-05-08 | 2026-05-11 | d3_high_invalid(2026-05-11)|d3_close_invalid(2026-05-11) | False | True | True | False | False |
| 000553_2026-05-07 | 000553 | 2026-05-07 | 2026-05-08 | 2026-05-11 | d3_high_invalid(2026-05-11)|d3_close_invalid(2026-05-11) | False | True | True | False | False |
| 002149_2026-05-07 | 002149 | 2026-05-07 | 2026-05-08 | 2026-05-11 | d3_high_invalid(2026-05-11)|d3_close_invalid(2026-05-11) | False | True | True | False | False |
| 603158_2026-05-08 | 603158 | 2026-05-08 | 2026-05-11 | 2026-05-12 | d3_high_invalid(2026-05-12)|d3_close_invalid(2026-05-12) | False | True | True | False | False |

- recovered: 4/4; 未恢复事件保持 label 不完整, 不造假标签。

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

## 8. 确定性构建 (byte-identical)

- 单次构建 (未做第二次 cache-only 重建)

## 9. 限制

- 日线 / MA / recent-7d 状态来自现有 canonical daily cache; 若缓存含历史回补, June 交叉核对 (第 5 节) 会显示差异
- 5min 源替换只影响 minute-dependent features; 冻结模型 / 阈值 / lambda / 特征选择均未重做 (本任务禁止)
- d1_volume/d1_amount 属 22 个 minute features (官方列表), 由 BaoStock 5min 求和
