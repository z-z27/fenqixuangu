# v004c Top3 Failure Decomposition v001

**输入**: Ridge OOF rows = 233 | GBDT OOF rows = 233 | OOF signal dates = 29 | exact event identity = PASS | max signal_date = 2026-06-30 | July NOT accessed | model training performed = NO

## 1. Why Is Top3 Failing?

Ridge Top3:
- model return: 2.78%
- oracle return: 6.09%
- total regret: +3.30pp
- winner-capture regret: +2.63pp
- repair-ordering regret: +0.67pp

GBDT Top3:
- model return: 2.92%
- oracle return: 6.09%
- total regret: +3.17pp
- winner-capture regret: +2.33pp
- repair-ordering regret: +0.84pp

Ridge 更大的损失来自: WINNER_CAPTURE (winner share 79.64%, repair share 20.36%)
GBDT 更大的损失来自: WINNER_CAPTURE (winner share 73.47%, repair share 26.53%)

## 2. When >=3 Real +7% Winners Exist

Ridge (dates=15):
- 3/3 dates: 1 | 2/3: 8 | 1/3: 4 | 0/3: 2
- mean capture: 51.11%
- Top3 return: 4.00% | Oracle Top3: 7.00%
GBDT (dates=15):
- 3/3 dates: 3 | 2/3: 7 | 1/3: 4 | 0/3: 1
- mean capture: 60.00%
- Top3 return: 4.17% | Oracle Top3: 7.00%

## 3. When Only 1-2 +7% Winners Exist

Ridge Group B (dates=12):
- winner capture: 58.33%
- full-winner-capture dates: 6
- selected remaining non-target return: 0.28%
- oracle remaining non-target return: 2.66%
- fill return gap: +2.38pp
- best non-target fill capture: 45.45%
- winner regret: +2.61pp | repair-ordering regret: +1.59pp
GBDT Group B (dates=12):
- winner capture: 62.50%
- full-winner-capture dates: 6
- selected remaining non-target return: -0.03%
- oracle remaining non-target return: 2.89%
- fill return gap: +2.92pp
- best non-target fill capture: 45.45%
- winner regret: +2.09pp | repair-ordering regret: +1.63pp

## 4. When No Stock Reaches +7%

Ridge Group C (dates=2):
- daily baseline: 0.52% | Top3: 2.11% | Oracle Top3: 2.34% | gap: +0.23pp
- repair concordance: 40.00%
GBDT Group C (dates=2):
- daily baseline: 0.52% | Top3: -0.07% | Oracle Top3: 2.34% | gap: +2.42pp
- repair concordance: 56.67%

## 5. Can the Models Rank Repair Strength Below 7%?

| Metric | Ridge | GBDT |
|---|---|---|
| Combined cross-threshold | 50.50% | 53.52% |
| Combined within non-target | 45.40% | 46.86% |
| Combined all repair pairs | 49.34% | 50.05% |
| May cross-threshold | 58.40% | 42.21% |
| May within non-target | 38.29% | 49.62% |
| May all repair pairs | 52.16% | 42.68% |
| June cross-threshold | 47.59% | 57.68% |
| June within non-target | 47.53% | 46.03% |
| June all repair pairs | 48.41% | 52.50% |

## 6. Rank1 vs Rank2 vs Rank3

Ridge:
- Rank1: return 2.82% | Target7 hit 41.38% | excess -0.18pp
- Rank2: return 3.45% | Target7 hit 39.29% | excess +0.59pp
- Rank3: return 1.67% | Target7 hit 29.63% | excess -1.12pp
GBDT:
- Rank1: return 3.68% | Target7 hit 51.72% | excess +0.68pp
- Rank2: return 3.13% | Target7 hit 32.14% | excess +0.27pp
- Rank3: return 1.50% | Target7 hit 44.44% | excess -1.28pp

## 7. Does Binary Target7 Lose Repair-Strength Information?

Ridge: within-non-target concordance 45.40% vs cross-threshold 50.50%
GBDT: within-non-target concordance 46.86% vs cross-threshold 53.52%

## Oracle Display Audit (§48)

PREVIOUS_ORACLE_DISPLAY_BUG: CONFIRMED (旧 comparison 表硬编码 Oracle Top1 Hit = 100% / Top3 Prec = 100%; 正确口径见下)
Correct All-39: Oracle Top1 hit 92.31% | Top3 precision 72.22%
Correct OOF-29: Oracle Top1 hit 93.10% | Top3 precision 71.84% | Top3 capped return 6.09%

## Diagnosis

Q1 When >=3 winners exist, can Ridge catch them?
- NO | evidence: Group A mean capture 51.11% (3/3 只有 1/15 天)
Q1b When >=3 winners exist, can GBDT catch them?
- NO | evidence: Group A mean capture 60.00% (3/3 3/15 天)
Q2 When only 1-2 winners exist, can Ridge capture available winners?
- PARTIAL | evidence: Group B capture 58.33%
Q2b GBDT:
- PARTIAL | evidence: Group B capture 62.50%
Q3 Can Ridge rank repair strength among non-Target7 stocks?
- NO | evidence: within-non-target concordance 45.40% (chance = 50%), cross-threshold 50.50%
Q3b GBDT:
- NO | evidence: within-non-target concordance 46.86%, cross-threshold 53.52%
Q4 What causes more Top3 regret?
- Ridge: winner capture 79.64% | repair ordering 20.36% | larger = WINNER_CAPTURE
- GBDT: winner capture 73.47% | repair ordering 26.53% | larger = WINNER_CAPTURE
Q5 Binary Target7 objective misalignment:
- SUPPORTED
- Evidence A: within-non-target 45.40%/46.86% vs cross-threshold 50.50%/53.52% (相对差 -5.10pp/-6.66pp)
- Evidence B: Group B full-winner-capture 后 fill gap +2.38pp/+2.92pp
- Evidence C: Group C Ridge Top3 2.11% vs baseline 0.52% (gap +0.23pp); GBDT Top3 -0.07% (gap +2.42pp)
- Evidence D: Rank1->Rank3 hit 衰减 Ridge 41.38% -> 29.63%; GBDT 51.72% -> 44.44%
- 附注 (§66): Group A winner capture 本身仅 51.11%/60.00%, cross-threshold 仅 50.50%/53.52% —— 问题不只是 binary 目标错位: 当前 D1 score 连真正 >=7% 的赢家都无法稳定识别。

TOP3_FAILURE_PICTURE: WINNER_CAPTURE

Plain-language conclusion:
- Top3 失败约 4/5 来自赢家没抓进 Top3 (Ridge 79.6% / GBDT 73.5% of regret): 即使当天存在 >=3 只 +7% 股票, 模型平均也只抓到 51.11%/60.00%;
- 剩余约 1/5 来自 7% 以下股票之间的修复强弱排序 (Ridge 20.4% / GBDT 26.5%): 全赢家捕获日期的非赢家填充收益 0.28%/-0.03% vs oracle 2.66%/2.89%;
- within-non-target 修复一致性 45.40%/46.86% (≈/低于 50% 随机), 提示 binary Target7 目标在 7% 以下丢失修复强度信息, 但该错位是次要成分; 主因仍是赢家识别失败。

Does current evidence support changing the next model objective from binary Target7 to repair-strength ranking?
- YES (部分支持) | reason: within-non-target 排序 ≤ 随机且 fill gap 实际损失存在; 但 79.6%/73.5% regret 来自 winner capture, 且 cross-threshold 也仅勉强 ≥ 随机 —— 换目标前必须先解决赢家识别本身 (D1 信息/时间稳定性), 否则 repair-strength 目标同样无法泛化 (diagnostic §TEMPORAL_INSTABILITY 结论)。

Recommended next step (external review 后决定):
- BOTH: (1) 修复赢家识别 (D1 信息/时间稳定性问题, 见 predictability diagnostic); (2) 若赢家识别改善后仍存在 fill gap, 再冻结 repair-strength ranking objective 作为下一阶段实验 (本任务不实现)。

## Scope Note

- 本诊断 Top3 按 §9 K = min(3, candidate_count), 29 个 OOF 日期全部参与 (v001 review 的 Top3 2.55% 只统计 >=3 候选日期, 因此本报告 Combined model return 2.78%/2.92% 与 v001 2.55% 口径不同, 数字均正确)。

## Determinism

- full pipeline executed twice; all 5 outputs byte-identical: PASS

