# v004c Repair-Strength Pairwise Ridge v001 — A/B vs Binary Target7 Ridge

**输入**: rows = 319 | signal_dates = 39 | features = 53 | May = 146 (18 dates) | June = 173 (21 dates) | board2 = 261 | board3 = 58
**OOF 评价集**: 233 rows | 29 signal dates (warm-up 10 dates) | max signal_date = 2026-06-30 | July accessed = NO

**Objective**: RAW_REPAIR_STRENGTH_PAIRWISE_ORDERING
- training outcome: `d3_high_daily / d2_open_daily - 1` (raw, 训练不封顶 7%)
- training capped at 7%: NO
- Target7 used for pair construction: NO (EVALUATION_ONLY)
- pair type multipliers: NONE
- magnitude weighting: NONE (pair 等权, 收益差大小不放大)
- per-date total pair weight: 1 (weight = 1/N_pair_date)

## Repair Pair Counts & Gap Diagnostics (§70-§71, §97)

- CROSS_THRESHOLD: count 656 | share 43.88% | mean per-date share 46.55%
- WITHIN_NON_TARGET: count 612 | share 40.94% | mean per-date share 40.49%
- WITHIN_TARGET7: count 227 | share 15.18% | mean per-date share 12.96%
- ALL: count 1495 | share 100.00% | mean per-date share 100.00%
- gap percentiles (ALL): p25 +2.73pp | median +6.05pp | p75 +10.10pp | p90 +14.60pp | max +39.94pp
- small-gap share (ALL): <0.1pp 0.87% | <0.5pp 4.35% | <1.0pp 8.49%
- gap 只诊断, 不据此重训 (§20, §71)

## 1. Did Repair-Strength Training Improve Actual Top3 Selection?

| Metric | Binary Ridge | Repair Ridge | Universe | Oracle |
|---|---|---|---|---|
| Rank1 capped return | 2.82% | 1.63% | 3.00% | 6.98% |
| Rank2 capped return | 3.45% | 3.45% | 3.00% | 5.82% |
| Rank3 capped return | 1.67% | 1.62% | 3.00% | 5.39% |
| Top3 capped return | 2.55% | 2.11% | 3.00% | 6.10% |
| Top1 Target7 hit | 41.38% | 34.48% | 37.95% | 93.10% |
| Top3 Target7 precision | 38.51% | 39.66% | 37.95% | 71.84% |
| Available winner capture | 54.32% | 55.56% | N/A | 100.00% |

Daily universe baseline (Combined): 3.00%

> Scope note: practical TopK 按 v001 口径只统计候选数 >= K 的日期 (29 个 OOF 日期中 2 个候选不足 3 只: 2026-05-22 x1, 2026-06-02 x2); regret / winner capture / oracle 沿用 Top3 Failure Decomposition 口径 K=min(3,n), 含全部 29 天。两个口径数字均正确 (§48 vs §63)。

## 2. Can It Catch Real +7% Winners?

All positive-winner dates (n=27):
- Binary: mean capture 54.32% | median 66.67% | full-capture date rate 25.93%
- Repair: mean capture 55.56% | median 66.67% | full-capture date rate 25.93%
Group A >=3 Target7 (n=15 dates):
- Binary: 3/3 1 | 2/3 8 | 1/3 4 | 0/3 2 | capture 51.11% | Top3 4.00%
- Repair: 3/3 1 | 2/3 8 | 1/3 5 | 0/3 1 | capture 53.33% | Top3 3.41%
- Oracle Top3 (Group A): 7.00%

## 3. Did It Learn Continuous Repair Strength?

| Concordance (date-weighted) | Binary Ridge | Repair Ridge |
|---|---|---|
| all repair | 49.95% | 47.93% |
| cross-threshold | 50.50% | 52.60% |
| within non-target | 45.40% | 40.89% |
| within Target7 | 59.75% | 55.93% |

## 4. What Happened Below the 7% Success Line?

within-non-target concordance: Binary 45.40% | Repair 40.89% (chance = 50%)
Group B 1-2 Target7 (n=12 dates):
- Binary: winner capture 58.33% | full-capture dates 6 | selected fill 0.28% | oracle fill 2.66% | fill gap +2.38pp | best fill capture 45.45%
- Repair: winner capture 58.33% | full-capture dates 6 | selected fill 0.28% | oracle fill 2.66% | fill gap +2.38pp | best fill capture 45.45%
Group C 0 Target7 (n=2 dates, baseline 0.52%):
- Binary: Top3 2.11% | Oracle 2.34% | gap +0.23pp
- Repair: Top3 0.56% | Oracle 2.34% | gap +1.78pp

## 5. Did Winner-Capture Regret Fall?

| Regret (mean over dates) | Binary Ridge | Repair Ridge |
|---|---|---|
| total top3 regret | +3.30pp | +3.71pp |
| winner-capture regret | +2.63pp | +2.94pp |
| repair-ordering regret | +0.67pp | +0.78pp |
| winner regret share | 79.64% | 79.02% |
| repair-ordering regret share | 20.36% | 20.98% |
> 注: regret 分解是结果分解, 不是因果份额 (§69)。

## 6. May vs June

May: Top3 Binary 3.03% | Repair 2.21% | winner capture Binary 58.33% | Repair 58.33% | all-repair conc Binary 52.48% | Repair 53.89% | within-non-target conc Binary 38.29% | Repair 39.07%
June: Top3 Binary 2.39% | Repair 2.08% | winner capture Binary 52.63% | Repair 54.39% | all-repair conc Binary 49.11% | Repair 45.94% | within-non-target conc Binary 47.53% | Repair 41.43%
- 不根据月份调模型; 若存在月份差异只记录为 temporal instability evidence (§77)

## 7. Lambda Selection

> lambda 只由 date-weighted OOF raw-repair pairwise logloss 选择 (§37); 下列收益/一致性列 EVALUATION_ONLY / NOT_USED_FOR_SELECTION (§38, §74)
- lambda 0.1: repair OOF pairwise logloss 0.804833 | evaluable dates 28 | all-repair concordance 48.07% (EVAL_ONLY) | Top3 return 2.92% (EVAL_ONLY) | winner capture 56.79% (EVAL_ONLY)
- lambda 0.3: repair OOF pairwise logloss 0.762672 | evaluable dates 28 | all-repair concordance 45.46% (EVAL_ONLY) | Top3 return 2.65% (EVAL_ONLY) | winner capture 51.23% (EVAL_ONLY)
- lambda 1.0: repair OOF pairwise logloss 0.729344 | evaluable dates 28 | all-repair concordance 45.61% (EVAL_ONLY) | Top3 return 2.61% (EVAL_ONLY) | winner capture 52.47% (EVAL_ONLY)
- lambda 3.0: repair OOF pairwise logloss 0.709412 | evaluable dates 28 | all-repair concordance 47.89% (EVAL_ONLY) | Top3 return 2.63% (EVAL_ONLY) | winner capture 53.70% (EVAL_ONLY)
- lambda 10.0: repair OOF pairwise logloss 0.698291 | evaluable dates 28 | all-repair concordance 47.93% (EVAL_ONLY) | Top3 return 2.11% (EVAL_ONLY) | winner capture 55.56% (EVAL_ONLY) <== selected
- selected lambda: 10.0

## 8. What Changed vs Binary Ridge?

- Universe: SAME (319 rows / 39 dates / 146+173 / 261+58)
- Features: SAME (53, pairwise v1 v002 frozen)
- Architecture: SAME (DATE_CONDITIONAL_PAIRWISE_RIDGE structure, board3 interaction)
- Preprocessing: SAME (fold-only q01/q99 -> median -> z-score)
- Walk-forward: SAME (chronological, warmup 10, 29 OOF dates)
- Lambda grid: SAME (0.1/0.3/1/3/10)
- Evaluation: SAME (capped 7%, Top1-3, winner capture, regret, concordance)
- Changed: TRAINING PAIR CONSTRUCTION ONLY — definition (binary Target7 -> raw repair ordering) AND scope (v001 implementation trained on whole-train cross-date pairs; this task's repair pairs are strictly same-date per spec §16/§26, weight 1/N_pair_date)
> 审计注 (外部复核项, 不修改历史): v001 的 chronological_walkforward 把整个 train 集合传给 build_same_date_pairs (fold 0: train_pairs 1749 = 33x53 跨日全组合, 同日和仅 195), 与其 docstring 声称的 'same-date only' 不符; 历史资产冻结, control 直接用冻结 score 评价。因此 A/B 的 pair 变化包含 定义+scope 两个维度。

## Model Behavior Overlap (诊断, 非控制变量)

- Repair OOF score vs Binary OOF score: Spearman 0.891 | Top3 成员集合一致 20/29 日期 (含全部 12 个 Group B 日期) —— 两个 objective 在这批 D1 feature 上几乎学到同一方向, 因此 Group B / May 的捕获与填充指标高度雷同; 这本身说明 pair 定义变化对排序行为的改变有限。

## Diagnosis

Q1 Did raw repair-strength ordering improve Top3 capped return?
- NO | Binary Top3 2.55% -> Repair 2.11% | baseline 3.00%
Q2 Did it improve available winner capture?
- YES | Binary 54.32% -> Repair 55.56%
Q3 Did it improve within-non-target repair ordering?
- NO | Binary 45.40% -> Repair 40.89% (chance 50%)
Q4 Did it improve Group A >=3-winner selection?
- YES | Binary 51.11% -> Repair 53.33% | 3/3 dates Binary 1 / Repair 1 (of 15)
Q5 Does the evidence support that Binary Target7 supervision was materially misaligned with the repair-strength objective?
- INCONCLUSIVE | within-non-target: Binary 45.40% -> Repair 40.89%; 若 within-non-target 改善但 Top3/赢家捕获不改善, 结论不能叫成功 (§82)

## Development Gate (§80)

1. Combined Top1 > baseline: FAIL (1.63% vs 3.00%)
2. Combined Top3 > baseline: FAIL (2.11% vs 3.00%)
3. Combined Top3 > Binary control: FAIL (2.11% vs 2.55%)
4. Combined winner capture > Binary control: PASS (55.56% vs 54.32%)
5. Combined all-repair concordance > 0.50: FAIL (47.93%)
REPAIR_OBJECTIVE_DEV_SIGNAL = ABSENT
REPAIR_READY_FOR_JULY_OOT = NO
JULY_OOT = NOT RUN

## Training Fit Sanity (§72)

- in-sample (all 319 rows) date-weighted all-repair concordance: 54.87% | pairwise logloss 0.689563 — optimization/capacity sanity ONLY, 不证明泛化 (§73)

## Board2 / Board3 Minimal Diagnostics (§78)

- board2: Top3 2.50% | winner capture 62.00% | all-repair concordance 49.19% | rows 193 (诊断 only)
- board3: Top3 2.47% | winner capture 89.74% | all-repair concordance 42.99% | rows 40 (诊断 only)

## Coefficient Diagnostics (§98)

Largest shared |beta|:
- d1_true_reclaim_ma5: 0.0052 (gamma -0.0001)
- d1_ma10_slope: 0.0045 (gamma -0.0008)
- d1_last_hour_return: 0.0044 (gamma +0.0007)
- d1_close_to_ma20: 0.0042 (gamma +0.0004)
- d1_intraday_range: 0.0035 (gamma -0.0001)
delta_board3: 0.0007
Fold stability (n_folds=29): beta sign agreement median 0.862 | gamma sign agreement median 0.793 | delta min -0.0007 / max 0.0015
- 系数仅诊断, 不据此删 feature (§98)

## Leakage Audit

- repair_strength_raw_return / capped_opportunity_return_7 / d2_open_daily / d3_high_daily / Target7 / source_window / signal_date / event_id / code in X: NO
- train/test chronology: strict chronological walk-forward (warmup 10), max(train) < test asserted per fold
- July accessed: NO (max signal_date 2026-06-30)

## Model Status

- selected lambda: 10.0
- model hash: de31476dded984c64fe8adb667979f11ae70c9df4f901e1c5bad20dd88090a30
- model status: DEV_FROZEN (July OOT not run)
- feature selection: NO
- new features: NO
- pair type weighting: NONE
- magnitude weighting: NONE
- Tail: NOT RUN
- July: NOT ACCESSED
- historical assets modified: NO

## Determinism

- full pipeline executed twice; all 9 outputs byte-identical: PASS

## 本实验最重要的原则 (§134)

7% 是业务成功线, 不是训练信息截断线。
新模型学习完整的同日修复强弱顺序 (raw repair ordering), 最终仍用 Top3 7% 封顶收益 + Target7 赢家捕获判断交易价值。

