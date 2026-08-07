# v004c Repair-State Model v002 — Frozen July Retrospective OOT Review

OOT 考试: 冻结 June 模型 (Repair-State v002) 在 July (2026-07-01 ~ 2026-07-29)
上的回顾式正式考试。本报告只评价冻结模型, 不训练 / 不重拟合 / 不调参 /
不根据 July 结果修改模型。

## 1. Method (方法)

- 冻结模型唯一来源: `F:\fenqixuangu\reports\research\v004c_repair_state_model_freeze_v001_202606\v004c_repair_state_frozen_model_v001.json` (frozen model JSON, 全精度系数)
- transform / factor / 概率全部复用冻结 JSON 与 spec v002 的 apply 逻辑
  (src/v004c_repair_state_model_v002.py + src/v004c_repair_state_spec_v002.py);
  不复制第三套公式; 不调用任何模型拟合
- 概率重建: logit = X @ beta + intercept; p = 1 / (1 + exp(-logit))
  (numpy/scipy sigmoid; 无 refit / recalibrate)
- R0 = 冻结 June prevalence = 59/173 = 0.341040
- 判定门 (§26-29) 在打开 July Target 前预声明; bootstrap (replicates=2000,
  seed=20260808, unit=signal_date) 只做不确定性描述, 不改变 pass/fail
- 所有输出确定性: 无时间戳 / 固定 seed / CSV float_format %.6f

## 2. Frozen model contract (冻结模型身份)

| field | value |
|---|---|
| model_version | v004c_repair_state_model_v001 |
| factor_spec_version | v004c_repair_state_spec_v002 |
| training_period | 2026-06-01 ~ 2026-06-30 |
| training_rows | 173 |
| training_dates | 21 |
| post_june_hypothesis | true |
| june_is_validation | false |
| july_target_seen | false |
| R1 factor_order | OPEN, DIVERGENCE, CLOSE_DAMAGE, SUPPLY, RECLAIM (5) |
| R2 factor_order | R1 + DIVERGENCE_SQ, DIVERGENCE_X_RECLAIM, DIVERGENCE_X_DAMAGE (8) |
| ranking | probability descending, tie -> event_id ascending |

身份校验通过 (无 FROZEN_MODEL_CONTRACT_MISMATCH)。

## 3. July slice (July 切片)

| field | value |
|---|---|
| window_rows | 333 |
| out_of_window_rows | 0 |
| july rows | 160 |
| july dates | 21 |
| july date range | 2026-07-01 ~ 2026-07-29 |

防御性验证通过: rows = 160 (预期 160), dates = 21 (预期 21)。

## 4. Two-phase audit (两阶段打开审计)

| field | value |
|---|---|
| PHASE_A_COMPLETE | YES |
| PRE_TARGET_PREDICTION_SHA256 | 8fe5e0745495e948149a29c833e6cce938ec997f78e93f4bab2ad6bec1409eea |
| TARGET_OPENED | YES |
| POST_TARGET_PREDICTION_SHA256 | 8fe5e0745495e948149a29c833e6cce938ec997f78e93f4bab2ad6bec1409eea |
| IDENTICAL | YES |

Phase A (target-blind) 完成后才允许打开 July Target; 打开后 POST hash 与
PRE hash 完全一致, 证明打开 Target 没有改变任何模型预测 / 排名。July Target
一旦打开即 permanently consumed, 不得再称 holdout / unseen / blind。

## 5. R0 frozen probability (R0 冻结概率)

R0 = 冻结 June prevalence = 59/173 = 0.341040 (全行常数)。来源为冻结
metadata (FROZEN_JUNE_POSITIVE=59 / FROZEN_JUNE_TOTAL=173), 并由 June
development metrics CSV 的 R1 pooled_candidate_rate 交叉验证
(_verify_june_prevalence, 1e-12 内一致)。R0 不使用 July prevalence;
R0 ranking = N/A (无 ranking 规则)。

## 6. Probability metrics (LogLoss / Brier / AUC / AP)

| model | logloss | brier | auc | average_precision | mean_probability | observed_rate | calibration_gap |
|---|---|---|---|---|---|---|---|
| R0 | 0.585875 | 0.197775 | 0.5000 | 0.2562 | 0.341040 | 0.256250 | 0.084790
| R1 | 0.601038 | 0.205352 | 0.4978 | 0.2594 | 0.339811 | 0.256250 | 0.083561
| R2 | 0.610552 | 0.208952 | 0.4636 | 0.2693 | 0.340187 | 0.256250 | 0.083937

- 自然候选池 pooled Target7 rate = 0.2562 (mean daily = 0.2469)
- improvement = baseline - candidate (positive = better):
  R1 vs R0: LogLoss -0.015163, Brier -0.007577; R2 vs R0: LogLoss -0.024676,
  Brier -0.011177; R2 vs R1: LogLoss -0.009514, Brier -0.003600

## 7. Within-date ranking (同日 ranking 主指标)

pair-weighted within-date AUC (Mann-Whitney U 的 within-date 版本; 只计同日
(positive, negative) 对, 跨日对不计):

| model | valid_auc_dates | mean_daily_auc | median_daily_auc | pair_weighted_auc |
|---|---|---|---|---|
| R1 | 14 | 0.4997 | 0.5000 | 0.5240 |
| R2 | 14 | 0.4817 | 0.4688 | 0.4847 |

## 8. Natural candidate baselines (自然候选基线)

- Top1Baseline = mean daily candidate rate = 0.2469
- Top3Baseline = sum(k_t * CandidateRate_t) / sum(k_t) = 0.2592
  (k_t = min(3, candidates))
- Top1 / Top3 不是硬 gate (21 个 picks 方差大, 只作 secondary)

## 9. R1-R2 ranking (Top1 / Top3 vs matched baseline)

| model | top1 hits | top1 rate | top1 baseline | top1 lift | top3 rate | top3 lift |
|---|---|---|---|---|---|---|
| R1 | 6 | 0.2857 | 0.2469 | 0.0388 | 0.2667 | 0.0074
| R2 | 8 | 0.3810 | 0.2469 | 0.1341 | 0.2667 | 0.0074

Top3 picks: R1 = 60, R2 = 60; Top3 命中日期: R1 = 13,
R2 = 13; zero-hit 日期: R1 = 8, R2 = 8。

## 10. OOT gates (预声明判定门, §26-29)

| gate | value |
|---|---|
| R1_PROBABILITY_PASS | NO |
| R1_RANKING_PASS | YES |
| R1_OOT_PASS | NO |
| R2_PROBABILITY_PASS | NO |
| R2_RANKING_PASS | NO |
| R2_OOT_PASS | NO |
| R2_INCREMENTAL_PASS | NO |
| R2_ONLY_OOT_PASS | NO |
| FINAL_DECISION | REJECT_REPAIR_STATE_V002_OOT |

- R1_PROBABILITY_PASS: R1 LogLoss < R0 LogLoss AND R1 Brier < R0 Brier
- R1_RANKING_PASS: R1 pair-weighted within-date AUC > 0.50 AND R1 Top3 lift > 0
- R2_INCREMENTAL_PASS (§28): R2 LogLoss < R1 LogLoss AND R2 Brier < R1 Brier
  AND R2 pair-weighted AUC >= R1 AND R2 Top3 rate >= R1
- 最终晋级 (§29): Case A 双 NO -> REJECT; Case B R1 YES R2 NO -> PROMOTE R1;
  Case C 双 YES + incremental NO -> PROMOTE R1 (简约); Case D 双 YES +
  incremental YES -> PROMOTE R2; Case E R1 NO R2 YES -> PROMOTE R2 (R2_ONLY=YES)
- Top1 不参与 hard gate (只作 secondary)

## 11. Bootstrap uncertainty (信号日 cluster bootstrap, 95% percentile CI)

unit = signal_date (有放回抽 21 日期, 重复日期允许重复), replicates=2000,
seed=20260808。CI 只描述证据精度, 不改变 §26-29 的 pass/fail。

| statistic | lower | upper | n_finite |
|---|---|---|---|
| r1_ll_improvement_vs_r0 | -0.048669 | 0.009287 | 2000 |
| r1_brier_improvement_vs_r0 | -0.023178 | 0.003563 | 2000 |
| r2_ll_improvement_vs_r0 | -0.058838 | 0.000239 | 2000 |
| r2_brier_improvement_vs_r0 | -0.026150 | 0.000109 | 2000 |
| r2_ll_improvement_vs_r1 | -0.017273 | -0.000716 | 2000 |
| r2_brier_improvement_vs_r1 | -0.006878 | 0.000323 | 2000 |
| r1_pair_weighted_within_date_auc | 0.443168 | 0.602276 | 2000 |
| r2_pair_weighted_within_date_auc | 0.388880 | 0.566265 | 2000 |
| r1_top3_lift | -0.042931 | 0.056901 | 2000 |
| r2_top3_lift | -0.056588 | 0.065962 | 2000 |

## 12. Confidence (§32)

| field | value |
|---|---|
| Confidence | N/A |

CONFIDENCE_STRONG 要求被晋级模型的 LogLoss improvement CI 下界 > 0 AND
Brier improvement CI 下界 > 0 AND Top3 lift CI 下界 > 0; 否则
CONFIDENCE_MIXED (MIXED 不改变晋级结果, 只提醒 July 只有 21 个日期,
后续 August+ forward shadow 更重要)。

## 13. Calibration (校准描述, 禁止 recalibration)

mean p / observed / gap 见 §6; 概率 tercile (按概率三等分):

| model | tercile | count | mean_predicted | observed_rate |
|---|---|---|---|---|
| R1 | LOW | 54 | 0.2595 | 0.2593 |
| R1 | MID | 53 | 0.3240 | 0.2642 |
| R1 | HIGH | 53 | 0.4375 | 0.2453 |
| R2 | LOW | 54 | 0.2472 | 0.2963 |
| R2 | MID | 53 | 0.3341 | 0.2453 |
| R2 | HIGH | 53 | 0.4410 | 0.2264 |

只描述, 不做 Platt / isotonic / 任何校准。

## 14. Concentration (ranking 集中度, §34)

| model | bucket | count | target7_rate | spread |
|---|---|---|---|---|
| R1 | TOP20 | 32 | 0.2188 | |
| R1 | BOTTOM20 | 32 | 0.2500 | |
| R1 | SPREAD | | | -0.0312 |
| R2 | TOP20 | 32 | 0.2500 | |
| R2 | BOTTOM20 | 32 | 0.3125 | |
| R2 | SPREAD | | | -0.0625 |

Top20% / Bottom20% 按 probability 降序 (tie -> event_id 升序) 分桶;
只描述, 不是硬 gate。

## 15. R1-R2 disagreement (分歧审计, §35)

| field | value |
|---|---|
| same_top1_dates | 18 |
| different_top1_dates | 3 |
| mean_top3_overlap | 2.5714 |

逐日详情见 daily ranking CSV (same_top1 / top3_overlap 列)。

## 16. Transform audit (冻结 transform 审计, §37)

| primitive | below_q01 | above_q99 | total_clipped | clip_rate |
|---|---|---|---|---|
| break_open_return | 9 | 1 | 10 | 0.0625 |
| d1_intraday_range | 3 | 0 | 3 | 0.0187 |
| d1_open_to_close_return_raw | 1 | 2 | 3 | 0.0187 |
| d1_high_to_close_drawdown_raw | 2 | 1 | 3 | 0.0187 |
| d1_close_to_vwap_raw | 6 | 0 | 6 | 0.0375 |
| late_day_sell_volume_ratio | 0 | 5 | 5 | 0.0312 |
| d1_low_to_close_recovery | 0 | 0 | 0 | 0.0000 |
| d1_afternoon_return | 3 | 0 | 3 | 0.0187 |

frozen q01/q99 未被修改; 只描述越界, 禁止删除越界行 / 重新 clip。

## 17. Interaction audit (R2 contribution 审计, §36)

- max |contribution| = coefficient * factor (July 全样本):
| OPEN | coef=0.119904 | max_abs_contrib=0.3092 | event_id=002412_2026-07-24 | date=2026-07-24 |
| DIVERGENCE | coef=-0.156602 | max_abs_contrib=0.4040 | event_id=002969_2026-07-29 | date=2026-07-29 |
| CLOSE_DAMAGE | coef=0.539700 | max_abs_contrib=1.3693 | event_id=002767_2026-07-07 | date=2026-07-07 |
| SUPPLY | coef=-0.162977 | max_abs_contrib=0.3798 | event_id=002167_2026-07-02 | date=2026-07-02 |
| RECLAIM | coef=0.414053 | max_abs_contrib=1.0411 | event_id=002969_2026-07-29 | date=2026-07-29 |
| DIVERGENCE_SQ | coef=0.278701 | max_abs_contrib=1.0530 | event_id=002969_2026-07-29 | date=2026-07-29 |
| DIVERGENCE_X_RECLAIM | coef=-0.286307 | max_abs_contrib=1.3494 | event_id=002969_2026-07-29 | date=2026-07-29 |
| DIVERGENCE_X_DAMAGE | coef=-0.364459 | max_abs_contrib=1.2639 | event_id=000892_2026-07-13 | date=2026-07-13 |
- 非有限 contribution / |contribution| >= 10 记录: 0 条
  (无)
- 只记录; 禁止 clip / drop row / 调整 coefficient

## 18. Q1 — R1 是否真正超过冻结 June base-rate R0?

根据 §6 (LogLoss / Brier) 与 §7/§9 (within-date ranking / Top3 lift) 的
预声明 gate 判定: R1_OOT_PASS = NO。
结论必须依赖 pre-registered gate, 而非任何 post-hoc 解释。

## 19. Q2 — R2 的三个条件项是否在 July 提供 R1 之外的真实增量价值?

R2_INCREMENTAL_PASS = NO (§28 四条件全部成立才为 YES)。逐项:
- R2 LogLoss < R1 LogLoss: NO
- R2 Brier < R1 Brier: NO
- R2 pair-weighted AUC >= R1: NO
- R2 Top3 rate >= R1: YES
任何一项失败都意味着条件项 (DIVERGENCE_SQ / DIVERGENCE_X_RECLAIM /
DIVERGENCE_X_DAMAGE) 在 July 上的增量价值未获得证据支持 (简约原则)。

## 20. Q3 — 模型是否把 July 候选池天然 Target7 成功率浓缩成更高 Top1/Top3 强修复成功率?

- 自然候选池 pooled rate = 0.2562; Top3 rate: R1 = 0.2667,
  R2 = 0.2667; Top3 lift: R1 = 0.0074, R2 = 0.0074
- pair-weighted within-date AUC: R1 = 0.5240, R2 = 0.4847
- Top1 (secondary): R1 = 0.2857 vs baseline 0.2469;
  R2 = 0.3810
浓缩效果以 Top3 lift > 0 与 within-date AUC > 0.50 为证据; 21 个日期下
Top1 的方差过大, 不作判定依据。

## 21. Important interpretation (重要解释)

- July 是 OOT 考试: 冻结模型未被 refit / recalibrate / 调参; 冻结 JSON
  未被修改; July 结果未回写训练阶段
- July Target 已打开, permanently consumed: 不得再称 holdout / unseen /
  blind; 下一阶段只能 August+ forward shadow (或 REJECT 后先做失败归因:
  calibration failure / within-date ranking failure / temporal instability /
  factor-state failure / nonlinearity failure)
- 本次判定: REJECT_REPAIR_STATE_V002_OOT; forward_shadow_candidate = NONE;
  Confidence = N/A
- 历史冻结资产 (v001/v002 freeze, walkforward, model table, src spec/model)
  全部未修改 (July 只能新增, 不回写)

## 22. Next step (下一阶段)

无论 PASS / FAIL / MIXED, 本任务到此停止。若晋级, 下一阶段只允许
August+ forward shadow (冻结模型只读, 输出预测, 不修改模型); 若 REJECT,
先做失败归因再决定是否修订模型假设 (修订需要新的 pre-registered 流程)。
