# v004c Upside Predictability Diagnostic v001

**输入**: rows = 319 | signal dates = 39 | features = 53 | OOF test dates = 29 | max signal_date = 2026-06-30 (July NOT accessed)

## 1. Can Three +7% Stocks Actually Exist Each Day?

All 39 dates (39 dates):
- 0 Target7: 3 天 (7.7%)
- 1 Target7: 10 天 (25.6%)
- 2 Target7: 6 天 (15.4%)
- >=3 Target7: 20 天 (51.3%)

OOF 29 dates (29 dates):
- 0 Target7: 2 天 (6.9%)
- 1 Target7: 9 天 (31.0%)
- 2 Target7: 3 天 (10.3%)
- >=3 Target7: 15 天 (51.7%)

May 18 dates (18 dates):
- 0 Target7: 1 天 (5.6%)
- 1 Target7: 4 天 (22.2%)
- 2 Target7: 3 天 (16.7%)
- >=3 Target7: 10 天 (55.6%)

June 21 dates (21 dates):
- 0 Target7: 2 天 (9.5%)
- 1 Target7: 6 天 (28.6%)
- 2 Target7: 3 天 (14.3%)
- >=3 Target7: 10 天 (47.6%)

Oracle Top1 hit rate (39 dates): 92.31%
Oracle Top3 Target7 precision (39 dates): 72.22%
Oracle Top1/2/3 capped return (39 dates): 6.93% / 6.10% / 6.22%
Oracle Top1 hit rate (OOF 29 dates): 93.10%
Oracle Top3 Target7 precision (OOF 29 dates): 71.84%

> 解释: 只有 20/39 (51.3%) 的交易日理论上存在 至少 3 只 Target7; 因此「每天 3 只全部 +7%」的不可实现部分是 数据层面决定的。随机选 3 只的期望命中 36.94%。

## 2. Can Linear Pairwise Ridge Fit the Historical Data?

Ridge λ0.1 TRAIN (IN_SAMPLE, NOT PREDICTIVE EVIDENCE):
- AUC: 0.7620 | Rank1: 4.95% | Top3: 3.86% | baseline: 3.21%
Ridge λ10 TRAIN (IN_SAMPLE, NOT PREDICTIVE EVIDENCE):
- AUC: 0.6140 | Rank1: 3.42% | Top3: 3.15% | baseline: 3.21%
Ridge λ10 OOF (v001 正式资产, 程序读取验证):
- AUC: 0.5050 | Rank1: 2.82% | Top3: 2.55% | baseline: 3.00%

## 3. Does Fixed Nonlinear GBDT Recover Predictive Signal?

Configuration: GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=2, min_samples_leaf=10, subsample=1.0, random_state=20260809)
GBDT TRAIN (IN_SAMPLE, CAPACITY_DIAGNOSTIC_ONLY):
- AUC: 0.8819 | Rank1: 5.93% | Top3: 4.42% | baseline: 3.21%
GBDT OOF Combined:
- within-date pairwise AUC: 0.5352 | pair accuracy: 0.5352 | date-balanced logloss: 0.6931
- baseline capped return: 3.00%
- Rank1 capped return: 3.68% (excess +0.68pp)
- Top3 capped return: 2.69% (excess -0.09pp)
- Top1 beat-universe rate: 62.07%
- Top1 Target7 hit: 51.72% | Top3 Target7 precision: 44.25%

GBDT OOF May:
- AUC: 0.4221 | baseline: 3.63% | Rank1: 4.14% | Top3: 2.55% | Top1 hit: 50.00% | Top3 precision: 50.00%
GBDT OOF June:
- AUC: 0.5768 | baseline: 2.76% | Rank1: 3.50% | Top3: 2.74% | Top1 hit: 52.38% | Top3 precision: 42.06%

## 4. Direct Comparison

| Mode | AUC | Rank1 | Top3 | Baseline | Top1 Hit | Top3 Prec | Top1 Beat |
|---|---|---|---|---|---|---|---|
| Random | 0.5000 | 3.21% | 3.21% | 3.21% | 36.94% | 36.94% | 50.00% |
| Ridge_lam10_OOF | 0.5050 | 2.82% | 2.55% | 3.00% | 41.38% | 38.51% | 55.17% |
| GBDT_OOF | 0.5352 | 3.68% | 2.69% | 3.00% | 51.72% | 44.25% | 62.07% |
| Oracle | 1.0000 | 6.93% | 6.22% | 3.21% | 100.00% | 100.00% | 100.00% |
| Ridge_lam0.1_TRAIN | 0.7620 | 4.95% | 3.86% | 3.21% | 69.23% | 51.71% | 74.36% |
| Ridge_lam10_TRAIN | 0.6140 | 3.42% | 3.15% | 3.21% | 46.15% | 44.87% | 61.54% |
| GBDT_TRAIN | 0.8819 | 5.93% | 4.42% | 3.21% | 79.49% | 61.97% | 84.62% |

## 5. Train vs OOF Gap

Ridge: TRAIN λ10 AUC 0.6140 / Rank1 3.42% -> OOF AUC 0.5050 / Rank1 2.82%
GBDT: TRAIN AUC 0.8819 / Rank1 5.93% -> OOF AUC 0.5352 / Rank1 3.68%

## Board2 / Board3 Minimal Diagnostics (仅诊断, §50)

- board2 ridge: AUC 0.5177 | Top3 2.68%
- board2 gbdt: AUC 0.5538 | Top3 2.91%
- board3 ridge: AUC 0.4143 | Top3 3.21%
- board3 gbdt: AUC 0.7000 | Top3 4.87%

## Diagnosis

NONLINEAR_OOF_SIGNAL_PRESENT: NO

Q1 Pairwise Ridge expression limitation:
- UNCLEAR
- evidence: Ridge TRAIN gate=PASS (λ0.1 AUC 0.7620, λ10 AUC 0.6140), Ridge OOF gate=FAIL (AUC 0.5050), GBDT OOF gate=FAIL (AUC 0.5352)

Q2 Same 53 features nonlinear OOF signal:
- NO
- evidence: GBDT OOF AUC 0.5352, Rank1 3.68% vs baseline 3.00%, Top1 beat 62.07%

Q3 Stable D1 Target7 information:
- YES
- evidence: Ridge TRAIN gate=PASS, GBDT TRAIN gate=PASS, GBDT OOF gate=FAIL

PRIMARY_DIAGNOSIS: TEMPORAL_INSTABILITY

TASK_CEILING: MODERATE (oracle top3 precision 72.2%)

Plain-language conclusion:
- 历史数据可以拟合 (Ridge λ0.1/λ10 TRAIN 与 GBDT TRAIN 训练端均通过 gate), 但两种模型 walk-forward 都不能稳定预测下一交易日; 主要失败在时间稳定性。
- 补充: GBDT OOF 的 Top1 侧有部分信号 (Rank1 3.68% > baseline 3.00%, beat 62.07%, Top1 hit 51.72% vs 随机 36.94%), 但 Top3 2.69% 低于 baseline 且 AUC 仅 0.5352, 不足以通过 gate; May OOF AUC 0.4221 vs June 0.5768 显示月份间结构不稳定 — 与 TEMPORAL_INSTABILITY 一致。

This result DOES prove:
- 每日 Target7 数量分布与 Oracle 上限 (理论 Top3 精度 72.22%, 随机基线 36.94%);
- Ridge 在 May+June 上的训练端拟合能力 (λ10 TRAIN AUC 0.6140);
- 固定 GBDT 与 Ridge 的 walk-forward OOF 对比 (AUC 0.5352 vs 0.5050)。
This result DOES NOT prove:
- 任何模型在 July 或未来的预测能力 (July NOT accessed);
- 非线性模型已可推广 (若 OOF 通过 gate, 仍需独立 OOT 验证);
- 特征信息不足是永久结论 (只对当前 53 特征有效)。

Recommended next step (external review 后决定):
- 保留现有 53 特征与双模型结论; 评估新 D1 信息源前不扩大 模型; 或接受 D1 信息限制。

## Determinism

- full pipeline executed twice; all 6 outputs byte-identical: PASS

