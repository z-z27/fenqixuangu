# v004c Frozen-v4a Architecture Transfer Benchmark v001

CAPABILITY BENCHMARK ONLY — ONE FROZEN ARCHITECTURE — NO JULY

## 1. Did Frozen v004a Architecture Work Inside v004c?

| Model | Rank1 | Baseline | Excess | Top3 | Baseline | Excess | Rank1 Beat | Top3 Beat | Top1 Hit | Top3 Precision |
|---|---|---|---|---|---|---|---|---|---|---|
| V4A_ARCH_TRANSFER_V4C | 2.8393% | 2.7596% | 0.0797% | 1.8539% | 2.6575% | -0.8036% | 57.1429% | 40.0000% | 28.5714% | 28.3333% |
| CORRECTED_BINARY_SAME_DATE | 1.0715% | 2.7596% | -1.6881% | 2.7619% | 2.6575% | 0.1044% | 42.8571% | 55.0000% | 28.5714% | 31.6667% |
| REPAIR_SAME_DATE | 0.7708% | 2.7596% | -1.9889% | 2.0773% | 2.6575% | -0.5802% | 42.8571% | 45.0000% | 28.5714% | 35.0000% |
| FOUNDATION_RAW_M0 | 2.4520% | 2.7596% | -0.3077% | 2.6335% | 2.6575% | -0.0240% | 52.3810% | 50.0000% | 28.5714% | 33.3333% |

- Architecture transfer signal: **PARTIAL**
- Exact score tie rows: **0**

## 2. High-Opportunity vs Low-Opportunity Days

| Bucket | Dates | Universe | Rank1 | Rank1 Excess | Top3 | Top3 Excess | Negative Rate | Oracle Rank1 | Oracle Top3 |
|---|---|---|---|---|---|---|---|---|---|
| LOW | 7 | 0.7922% | -0.3685% | -1.1607% | -0.9211% | -1.7133% | 71.4286% | 6.9162% | 5.2087% |
| MID | 7 | 2.7096% | 3.2762% | 0.5666% | 2.2631% | -0.4464% | 14.2857% | 7.0000% | 6.6285% |
| HIGH | 7 | 4.7770% | 5.6103% | 0.8332% | 4.6139% | -0.1589% | 0.0000% | 7.0000% | 6.5823% |

High-opportunity upside capture: **NO**.
Low-opportunity capital preservation under the capped-opportunity evaluator: **NO**.

## 3. Comparison With Existing v004c Models

- Practical comparison vs Corrected Binary and Repair: **MIXED**.
- Every row above uses the same 21 June dates, same event universe, same capped-opportunity evaluator, and deterministic event_id tie-break.

## 4. Exact v004a Architecture Parity

- 18 feature exact: **YES**
- l2: **0.3**
- positive weight: **1.5**
- tail weighting: **MATCH**
- date weighting: **MATCH**
- interactions: **MATCH**
- rank transform: **MATCH**
- extra features: **NONE**
- new model logic: **NONE**
- feature gate: **PASS**

## 5. What Changed From v004a?

- Candidate universe: broad v004a → v004c first-break universe
- Daily percentile reference: v004c candidates
- Training history: v004c dates
- Training architecture: **UNCHANGED**

## 6. What Did NOT Change?

- Target: Target7 binary
- Model family: weighted L2 Logistic
- L2: 0.30
- Positive weight: 1.50
- Tail weight: >=10% 1.5; >=12% 2.0
- Date weight: 1/n before class and tail multiplication
- Feature structure: frozen 18 dimensions
- Interaction formulas: frozen six

## 7. Feature Provenance and Leakage Boundary

- Frozen historical raw rows: **304**
- Canonical reconstructed raw rows: **15**
- Reconstructed rows use the canonical generate_signal chain with daily data clipped at D1, BaoStock 5m clipped at D1, and the original five-calendar-day historical limit-up pool.
- Missing raw ranks retain v004a's average-percentile then 0.5 fallback semantics.
- Test outcomes never enter feature transformation or scoring; train outcomes enter only Target7 and frozen class/tail sample weights.
- July was not accessed by the formal pipeline.

## 8. Downside / Upside Distribution

- Top3 mean: **1.8539%**
- Top3 p25: **-0.2577%**
- Top3 median: **2.5742%**
- Top3 worst: **-6.7688%**
- Top3 negative-date rate: **30.0000%**

### Best 3 dates

| Date | Universe | Top3 | Excess | Selected codes |
|---|---|---|---|---|
| 2026-06-23 | 5.0686% | 5.4643% | 0.3957% | 603663|002297|003031 |
| 2026-06-10 | 4.3895% | 5.2019% | 0.8124% | 603929|603616|603186 |
| 2026-06-22 | 4.6541% | 5.1373% | 0.4832% | 001339|002989|002491 |

### Worst 3 dates

| Date | Universe | Top3 | Excess | Selected codes |
|---|---|---|---|---|
| 2026-06-03 | -0.4759% | -6.7688% | -6.2928% | 600280|002579|001210 |
| 2026-06-30 | 1.8285% | -1.4689% | -3.2973% | 601133|603650|603163 |
| 2026-06-26 | 1.6014% | -0.8163% | -2.4177% | 603989|002990|002080 |

## 9. Oracle

- Oracle Rank1: **6.9721%**
- Oracle STRICT Top3: **6.1177%**

## 10. Final Answers

- Q1 exact D1-safe 18-feature reconstruction: **YES**
- Q2 beat June universe Rank1: **YES**
- Q3 beat June universe STRICT Top3: **NO**
- Q4 beat Corrected Binary + Repair: **MIXED**
- Q5 HIGH-opportunity upside capture: **NO**
- Q6 LOW-opportunity capital preservation: **NO**
- Q7 previous v004c modeling-route problem: **INCONCLUSIVE**

ARCHITECTURE_TRANSFER_SIGNAL: **PARTIAL**

PREVIOUS_V4C_MODELING_ROUTE_MISDESIGNED: **INCONCLUSIVE**

READY_FOR_JULY_OOT: **NO**
