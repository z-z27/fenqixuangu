# v004c Pairwise Ridge v001 — Development Model Review

**模型**: DATE_CONDITIONAL_PAIRWISE_RIDGE (开发期)
**输入**: rows = 319 | signal_dates = 39 | features = 53 | May = 146 | June = 173 | board2 = 261 | board3 = 58
**OOF 评价集**: 233 rows | 29 test signal dates (warm-up 10 dates 无 OOF)
**合约**: pairwise_v1_v002 (v002 corrected)

> 注意 (tail label 语义): v002 label contract 文本写 `1[(d3_high / d2_open - 1) <= -0.05]`, 但冻结标签值 (May and June 源表, 0 mismatch) 按实现 `1[(d3_close / d2_open - 1) <= -0.05]` 计算 (src/v004c_may_d1_coverage.py:485)。本任务按冻结值训练/评价, 文本/实现差异标记为外部复核项, 不修改任何冻结资产。

## Practical Upside OOF Results

Combined daily universe baseline: 3.00%

Rank1:
- mean capped return: 2.82%
- median: 4.74%
- excess vs universe: -0.18pp
- positive rate: 65.52%
- beat universe rate: 55.17%
Rank2:
- mean capped return: 3.45%
- median: 4.87%
- excess vs universe: +0.59pp
- positive rate: 82.14%
- beat universe rate: 67.86%
Rank3:
- mean capped return: 1.67%
- median: 2.94%
- excess vs universe: -1.12pp
- positive rate: 62.96%
- beat universe rate: 51.85%
Rank4:
- mean capped return: 2.01%
- median: 3.01%
- excess vs universe: -0.82pp
- positive rate: 70.83%
- beat universe rate: 45.83%
Rank5:
- mean capped return: 5.02%
- median: 6.99%
- excess vs universe: +2.22pp
- positive rate: 90.91%
- beat universe rate: 77.27%
Top1:
- mean capped return: 2.82%
- median: 4.74%
- excess: -0.18pp
- positive day rate: 65.52%
- beat universe rate: 55.17%
Top2:
- mean capped return: 3.06%
- median: 3.32%
- excess: +0.21pp
- positive day rate: 82.14%
- beat universe rate: 50.00%
Top3:
- mean capped return: 2.55%
- median: 2.96%
- excess: -0.23pp
- positive day rate: 85.19%
- beat universe rate: 48.15%
Rank1-5 return profile:
- 2.82% / 3.45% / 1.67% / 2.01% / 5.02%
- 梯度结论: 无一致排序梯度 — 模型排名与实际上涨机会之间没有稳定正相关, Top1 平均 2.82% 低于当日随机基准 3.00% (§38 明确标注)

## May Practical OOF
daily universe baseline: 3.63%
Rank1:
- mean capped return: 3.76%
- median: 7.00%
- excess vs universe: +0.13pp
- positive rate: 75.00%
- beat universe rate: 62.50%
Rank2:
- mean capped return: 5.86%
- median: 7.00%
- excess vs universe: +2.71pp
- positive rate: 100.00%
- beat universe rate: 85.71%
Rank3:
- mean capped return: -0.07%
- median: -0.41%
- excess vs universe: -3.21pp
- positive rate: 42.86%
- beat universe rate: 28.57%
Top1:
- mean capped return: 3.76%
- median: 7.00%
- excess: +0.13pp
- positive day rate: 75.00%
- beat universe rate: 62.50%
Top2:
- mean capped return: 4.58%
- median: 6.37%
- excess: +1.43pp
- positive day rate: 85.71%
- beat universe rate: 85.71%
Top3:
- mean capped return: 3.03%
- median: 2.41%
- excess: -0.12pp
- positive day rate: 100.00%
- beat universe rate: 42.86%

## June Practical OOF
daily universe baseline: 2.76%
Rank1:
- mean capped return: 2.47%
- median: 4.55%
- excess vs universe: -0.29pp
- positive rate: 61.90%
- beat universe rate: 52.38%
Rank2:
- mean capped return: 2.65%
- median: 4.39%
- excess vs universe: -0.11pp
- positive rate: 76.19%
- beat universe rate: 61.90%
Rank3:
- mean capped return: 2.27%
- median: 6.54%
- excess vs universe: -0.39pp
- positive rate: 70.00%
- beat universe rate: 60.00%
Top1:
- mean capped return: 2.47%
- median: 4.55%
- excess: -0.29pp
- positive day rate: 61.90%
- beat universe rate: 52.38%
Top2:
- mean capped return: 2.56%
- median: 2.44%
- excess: -0.20pp
- positive day rate: 80.95%
- beat universe rate: 38.10%
Top3:
- mean capped return: 2.39%
- median: 3.06%
- excess: -0.27pp
- positive day rate: 80.00%
- beat universe rate: 50.00%

## Statistical Ranking Diagnostics

Upside:
- pairwise logloss: 0.695107
- pairwise AUC: 0.5050
- pair accuracy: 0.5050
- Top1 Target7 hit: 41.38%
- Top3 Target7 precision: 38.51%
- daily random Target7 baseline: 37.95%

Tail:
- pairwise logloss: 0.689387
- pairwise AUC: 0.5614
- pair accuracy: 0.5614

## Lambda Selection

Upside (唯一标准 = date-weighted OOF pairwise logloss; 收益列 EVALUATION_ONLY / NOT_USED_FOR_SELECTION):
- lambda 0.1: date-weighted OOF pairwise logloss 0.828174 | AUC 0.5002 | accuracy 0.5002 | Rank1 return 3.47% (EVALUATION_ONLY)
- lambda 0.3: date-weighted OOF pairwise logloss 0.772134 | AUC 0.5006 | accuracy 0.5006 | Rank1 return 3.02% (EVALUATION_ONLY)
- lambda 1.0: date-weighted OOF pairwise logloss 0.731639 | AUC 0.4973 | accuracy 0.4973 | Rank1 return 2.51% (EVALUATION_ONLY)
- lambda 3.0: date-weighted OOF pairwise logloss 0.707943 | AUC 0.5158 | accuracy 0.5158 | Rank1 return 2.74% (EVALUATION_ONLY)
- lambda 10.0: date-weighted OOF pairwise logloss 0.695107 | AUC 0.5050 | accuracy 0.5050 | Rank1 return 2.82% (EVALUATION_ONLY) <== selected
- selected_lambda: 10.0

Tail:
- lambda 0.1: date-weighted OOF pairwise logloss 0.865019 | AUC 0.4831 | accuracy 0.4831 | Rank1 return 2.97% (EVALUATION_ONLY)
- lambda 0.3: date-weighted OOF pairwise logloss 0.771519 | AUC 0.4948 | accuracy 0.4948 | Rank1 return 3.22% (EVALUATION_ONLY)
- lambda 1.0: date-weighted OOF pairwise logloss 0.717166 | AUC 0.5536 | accuracy 0.5536 | Rank1 return 3.10% (EVALUATION_ONLY)
- lambda 3.0: date-weighted OOF pairwise logloss 0.695290 | AUC 0.5751 | accuracy 0.5751 | Rank1 return 2.82% (EVALUATION_ONLY)
- lambda 10.0: date-weighted OOF pairwise logloss 0.689387 | AUC 0.5614 | accuracy 0.5614 | Rank1 return 2.77% (EVALUATION_ONLY) <== selected
- selected_lambda: 10.0

## Tail-Risk Practical Evaluation

Daily base tail incidence: 25.06%

Highest-Risk Rank1: tail incidence 24.14% | capped return 2.77%
Highest-Risk Top3: tail incidence 32.14% | capped return 2.61%
Safest Rank1: tail incidence 17.24% | capped return 3.49%
Safest Top3: tail incidence 17.86% | capped return 3.39%

## Board2 / Board3 Diagnostics (仅诊断)

- board2 upside: AUC 0.5177 | Rank1 3.99% | Top3 2.68% | Target7 rate 34.72%
- board2 tail: AUC 0.6150 | Rank1 2.51% | Top3 2.98% | Target7 rate 34.72%
- board3 upside: AUC 0.4143 | Rank1 2.94% | Top3 3.21% | Target7 rate 42.50%
- board3 tail: AUC 0.5542 | Rank1 2.81% | Top3 3.33% | Target7 rate 42.50%

## Coefficient Diagnostics

Largest upside shared |beta|:
- d1_ma10_slope: 0.0092 (gamma -0.0008)
- d1_close_to_ma20: 0.0090 (gamma +0.0022)
- d1_high_to_ma5_raw: 0.0081 (gamma +0.0009)
- d1_intraday_range: 0.0080 (gamma +0.0029)
- d1_true_reclaim_ma5: 0.0067 (gamma -0.0002)
Largest upside board3 deviations |gamma|:
- recent_7d_limit_up_count: 0.0050
- recent_limit_up_count_10d: 0.0046
- recent_pool_appearance_count_10d: 0.0046
- board_day_volume_rank: -0.0042
- recent_limit_up_count_20d: 0.0041
delta_board3: upside 0.0011 | tail 0.0020
Fold stability (n_folds=29): upside beta sign agreement median 0.931 | tail beta sign agreement median 0.966

## Final Frozen Dev Models

Upside: lambda 10.0 | model hash 0bc50da15fed3c55e8d00957cc7f0b2265ddd376c8d45184367f4e8faf8abb99 | train 2026-05-06..2026-06-30 (319 rows, 39 dates) | status DEV_FROZEN
Tail: lambda 10.0 | model hash 31c9d51c637ac284ee1c2b54bb17d1d3d64c31a4b2e39c9505cdb8463e373a4f | train 2026-05-06..2026-06-30 (319 rows, 39 dates) | status DEV_FROZEN

## Independent Temporal Units

- rows = 319 | signal_dates = 39
- pairs are training comparisons, not independent market observations.
- upside pairable dates (OOF evaluable): 26 | train pairs across folds: 267914
- tail pairable dates (OOF evaluable): 24 | train pairs across folds: 206323

## Leakage Audit

- future in X: NO
- labels in X: NO
- outcome return in X: NO
- source_window in X: NO
- signal_date numeric in X: NO
- train/test chronology: strict chronological walk-forward (warmup 10 signal dates), max(train) < test asserted per fold
- July accessed: NO (max signal_date 2026-06-30)

## Model Status

- model status: DEV_FROZEN (July OOT not run)
- UPSIDE_DEV_GO: NO
- UPSIDE_READY_FOR_JULY_OOT: NO
- TAIL_DEV_SIGNAL: PRESENT
- TAIL_READY_FOR_JULY_OOT: YES
- feature selection: NO
- model zoo: NO
- July OOT: NOT RUN

## Determinism

- full pipeline executed twice; all 9 outputs byte-identical: PASS

