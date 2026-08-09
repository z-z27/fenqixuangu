# v004c Corrected Same-Date Binary vs Same-Date Repair — Objective A/B

**输入**: rows = 319 | signal_dates = 39 | features = 53 | OOF = 233 rows / 29 dates (warm-up 10) | max signal_date = 2026-06-30 | July accessed = NO

## Historical Bug Audit (§85)

HISTORICAL_BINARY_V001_PAIR_SCOPE_AUDIT: **MISMATCH_CONFIRMED**
- 历史文档声称 same-date, 实际实现 pooled cross-date (chronological_walkforward 把整个 train 集合一次性传入 build_same_date_pairs, 该函数无日期参数)
- 示例 fold 0: train positives 33 | train negatives 53 | pooled cross-date pair count 1749 (= N_pos_total x N_neg_total) | correct same-date pair count 195
- Historical Binary v001 保留用于追溯 (HISTORICAL_BINARY_CROSS_DATE), 不再作为严格同日 Binary control (§36); 历史资产未修改

## Corrected Binary Pair Construction (§56-§58)

- total same-date binary pairs (all 39 dates): 656
- pair-evaluable dates: 35 | single-class dates: 4 (2026-05-18 x1, 2026-05-22 x1, 2026-06-04 x6, 2026-06-08 x3)
- per-evaluable-date positive count: 1..12 | negative count: 1..13
- 恒等式: 同日 binary pairs (N_pos_t x N_neg_t) == Repair 的 CROSS_THRESHOLD pairs (656 == 656, 数学必然: 同日候选里 t7=1 vs t7=0 的组合与 raw>=7% vs raw<7% 的组合完全一致); Repair 额外包含 839 个 within-class pairs (within-non-target 612 + within-target7 227) 的监督
- fold audit: min evaluable train dates 9 | max 34 | min total weight 9.0 | max 34.0 | cross-date pairs (all folds): 0 (MUST BE 0)

## Lambda Selection (§71)

> lambda 只由 date-weighted OOF same-date binary pairwise logloss 选择 (§24); STRICT Top3 / winner capture 列 EVALUATION_ONLY (§26)
- lambda 0.1: OOF binary pairwise logloss 0.860825 | AUC 0.4968 | evaluable dates 26 | STRICT Top3 2.83% (EVAL_ONLY) | winner capture 53.70% (EVAL_ONLY)
- lambda 0.3: OOF binary pairwise logloss 0.792266 | AUC 0.5017 | evaluable dates 26 | STRICT Top3 2.96% (EVAL_ONLY) | winner capture 56.17% (EVAL_ONLY)
- lambda 1.0: OOF binary pairwise logloss 0.743198 | AUC 0.5018 | evaluable dates 26 | STRICT Top3 2.86% (EVAL_ONLY) | winner capture 53.70% (EVAL_ONLY)
- lambda 3.0: OOF binary pairwise logloss 0.713961 | AUC 0.4923 | evaluable dates 26 | STRICT Top3 2.19% (EVAL_ONLY) | winner capture 51.23% (EVAL_ONLY)
- lambda 10.0: OOF binary pairwise logloss 0.697844 | AUC 0.4993 | evaluable dates 26 | STRICT Top3 2.61% (EVAL_ONLY) | winner capture 52.47% (EVAL_ONLY) <== selected
- selected lambda: 10.0

## 1. Corrected Binary vs Repair: Which Ranks Better Stocks?

| Metric | Corrected Binary | Repair | Universe | Oracle |
|---|---|---|---|---|
| Rank1 capped return | 1.76% | 1.63% | 3.00% | — |
| Rank2 capped return | 2.98% | 3.45% | 2.86% | — |
| Rank3 capped return | 3.45% | 1.62% | 2.78% | — |
| STRICT Top1 | 1.76% | 1.63% | 3.00% | — |
| STRICT Top2 | 2.27% | 2.44% | 2.86% | — |
| STRICT Top3 | 2.61% | 2.11% | 2.78% | — |
| Up-To-3 | 2.83% | 2.37% | 3.00% | — |
| Top1 Target7 hit | 34.48% | 34.48% | 37.95% | 93.10% |
| STRICT Top3 Target7 precision | 33.33% | 37.04% | 37.95% | 71.84% |
| Available winner capture | 52.47% | 55.56% | N/A | 100.00% |

> 口径: STRICT TopK = 只统计候选数 >= K 的日期 (§38); Up-To-3 = 全部 29 天 K_t=min(3,n) (§39); oracle 列 Up-To-3 口径。

## 2. Did the Old Cross-Date Pair Bug Matter? (§92)

| Metric | Historical Binary | Corrected Binary |
|---|---|---|
| Rank1 | 2.82% | 1.76% |
| STRICT Top3 | 2.55% | 2.61% |
| Up-To-3 | 2.78% | 2.83% |
| winner capture | 54.32% | 52.47% |
| binary pairwise AUC | 0.5050 | 0.4993 |
| all-repair concordance | 49.95% | 48.80% |
| score Spearman | 0.938 | 1.0000 |
| Top3 exact match dates | 15/29 | — |
| mean Top3 Jaccard | 0.721 | — |
> 若 Historical 更好: 跨日配对可能引入了不同的 pooled 历史分类效应, 但这不能证明跨日 pairwise 排序符合每日选择目标 (§55, §79)。

## 3. Winner Capture (§93)

All positive-winner dates (n=27):
- Corrected: capture 52.47% | full-capture rate 29.63%
- Repair: capture 55.56% | full-capture rate 25.93%
Group A >=3 Target7 (n=15):
- Corrected: 3/3 2 | 2/3 5 | 1/3 4 | 0/3 4 | capture 44.44% | STRICT Top3 3.26%
- Repair: 3/3 1 | 2/3 8 | 1/3 5 | 0/3 1 | capture 53.33% | STRICT Top3 3.41%
- Oracle (Group A): 7.00%
Group B 1-2 Target7 (n=12):
- Corrected: capture 62.50% | full-capture 6 | fill 0.28% | oracle fill 2.66% | gap +2.38pp | best fill capture 50.00%
- Repair: capture 58.33% | full-capture 6 | fill 0.28% | oracle fill 2.66% | gap +2.38pp | best fill capture 45.45%
Group C 0 Target7 (n=2, baseline 0.52%):
- Corrected: Up-To-3 0.56% | Oracle 2.34% | gap +1.78pp
- Repair: Up-To-3 0.56% | Oracle 2.34% | gap +1.78pp

## 4. Continuous Repair Ordering (§94)

| Concordance (raw, date-weighted) | Corrected | Repair |
|---|---|---|
| all repair | 48.80% | 47.93% |
| cross-threshold | 49.93% | 52.60% |
| within non-target | 44.03% | 40.89% |
| within Target7 | 57.02% | 55.93% |

## 5. Binary Target7 Discrimination (§95)

| Metric | Corrected | Repair |
|---|---|---|
| same-date binary pairwise logloss | 0.6978 | 0.6937 |
| same-date binary pairwise AUC/concordance | 0.4993 | 0.5260 |
| evaluable dates | 26 | 26 |

## 6. Top3 Regret (§96)

| Regret (mean over 29 dates, K=min(3,n)) | Corrected | Repair |
|---|---|---|
| total | +3.25pp | +3.71pp |
| winner capture | +2.62pp | +2.94pp |
| repair ordering | +0.64pp | +0.78pp |
| winner share | 80.38% | 79.02% |

## 7. May vs June (§97)

May: STRICT Top3 Corrected 2.16% | Repair 2.21% | Up-To-3 Corrected 2.76% | Repair 2.81% | capture Corrected 54.17% | Repair 58.33% | binary AUC Corrected 0.4953 | Repair 0.5996 | all-repair conc Corrected 48.62% | Repair 53.89% | within-nt Corrected 39.30% | Repair 39.07%
June: STRICT Top3 Corrected 2.76% | Repair 2.08% | Up-To-3 Corrected 2.86% | Repair 2.21% | capture Corrected 51.75% | Repair 54.39% | binary AUC Corrected 0.5007 | Repair 0.4989 | all-repair conc Corrected 48.86% | Repair 45.94% | within-nt Corrected 45.45% | Repair 41.43%
- DIAGNOSTIC_ONLY, 不根据月份调模型 (§70)

## 8. Controlled A/B Audit (§98)

- Universe: SAME (319/39)
- Features: SAME (53, exact order)
- Architecture: SAME (DATE_CONDITIONAL structure, board3)
- Preprocessing: SAME (fold-only)
- Walk-forward: SAME (chronological, warmup 10, 29 OOF dates)
- Lambda grid: SAME (0.1/0.3/1/3/10)
- Pair scope: SAME-DATE (both models)
- Cross-date pairs: 0 (corrected binary, fold-audited)
- Date weighting principle: SAME (per-date total weight = 1)
- Evaluation: SAME (unified evaluator on all models)
- Primary changed variable: OBJECTIVE / PAIR DEFINITION ONLY

## Diagnosis (§99-§105)

Q1 Did correcting Binary pair scope materially change the Binary model?
- YES | score Spearman (HIST vs CORR) 0.938 | Top3 exact match 15/29 | mean Jaccard 0.721 | STRICT Top3 HIST 2.55% vs CORR 2.61% | winner capture HIST 54.32% vs CORR 52.47%
Q2 Same-date objective comparison: BOTH_WEAK
- evidence: Rank1 CORR 1.76% vs REPAIR 1.63% | STRICT Top3 CORR 2.61% vs REPAIR 2.11% | capture CORR 52.47% vs REPAIR 55.56% | all-repair conc CORR 48.80% vs REPAIR 47.93%
Q3 Corrected Binary practical signal: Rank1 > baseline NO | STRICT Top3 > baseline NO
Q4 Repair vs Binary continuous repair ordering: NO
- evidence: all-repair CORR 48.80% vs REPAIR 47.93% | within-non-target CORR 44.03% vs REPAIR 40.89%
Q5 Repair vs Binary winner capture: REPAIR | Corrected 52.47% vs Repair 55.56%
Q6 Binary Target7 objective misalignment: INCONCLUSIVE
- evidence (fair same-date A/B): within-non-target CORR 44.03% vs REPAIR 40.89% | STRICT Top3 CORR 2.61% vs REPAIR 2.11% | capture CORR 52.47% vs REPAIR 55.56%
Q7 CURRENT_MODELING_BOTTLENECK: CURRENT_D1_FEATURE_SET_OR_TEMPORAL_GENERALIZATION

## Development Gate (§72)

1. STRICT Top1 > matching baseline: FAIL (1.76% vs 3.00%)
2. STRICT Top3 > matching baseline: FAIL (2.61% vs 2.78%)
3. available winner capture > 0.50: PASS (52.47%)
4. same-date binary pairwise AUC > 0.50: FAIL (0.4993)
CORRECTED_BINARY_DEV_SIGNAL = ABSENT
READY_FOR_JULY_OOT = NO
JULY_OOT = NOT RUN

## Model Status

- selected lambda: 10.0
- model hash: d79f01930286f3c033c444ba79ce6579ca0d8052ce1b2750dce3f4570e5aa84d
- model status: DEV_FROZEN
- Tail: NOT RUN | July: NOT ACCESSED
- new features: NO | feature selection: NO
- historical assets modified: NO
- fold beta sign agreement median 0.931 | gamma 0.897 (n_folds=29)

## Determinism

- full pipeline executed twice; all 8 outputs byte-identical: PASS

## 本任务最重要的原则 (§109)

历史 Binary v001 保留用于追溯, 但因为训练 Pair 跨交易日, 它不能再作为严格同日 Binary control。真正有效的目标函数比较, 只能是「Corrected Same-Date Binary」对「Same-Date Raw Repair」。

