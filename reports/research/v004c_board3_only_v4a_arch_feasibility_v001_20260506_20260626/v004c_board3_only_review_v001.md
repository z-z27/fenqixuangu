# v004c Board3-Only Frozen-V4A-Architecture Feasibility Benchmark v001

## 1. Experimental Contract

- Development architecture feasibility: **YES**; pristine OOT: **NO**.
- Board3 internal ranking only. The only changed dimension is the training population.
- No feature, target, learner, weighting, hyperparameter, model-zoo, cross-board merge, or July-result access.

## 2. Matured Board3 Population

- Matured pre-July: **307 rows / 37 dates**; Board3: **55 rows / 28 dates**.
- Board3 Target7: **23**; LOSS: **13**.

## 3. Chronological OOF Availability

- Board3-only eligible dates: **2026-06-12, 2026-06-16, 2026-06-17, 2026-06-18, 2026-06-22, 2026-06-23, 2026-06-24, 2026-06-25**.
- Common evaluation: **8 dates / 19 rows**.
- Unfittable single-class dates: **0**.

## 4. Common Evaluation Population

- Target7 rows: **5**; LOSS rows: **6**.
- Board3 candidates/date: min **1**, median **2.00**, max **8**.

## 5. Frozen 18-Feature Parity

- Model: `V4A_ARCH_TRANSFER_V4C` / `WEIGHTED_L2_LOGISTIC`; 18 exact features; L2 `0.30`; positive weight `1.50`.
- Percentile-rank features were constructed on the complete daily V4C universe before Board3 filtering.

## 6. Training / Leakage Audit

- Self leakage: **0**; current-test leakage: **0**; Board2 training rows: **0**; July rows: **0**.
- Prediction lock SHA256: `2f85e79738dbbedf19382db0da5f1136de02e84c69557ea5c85341bcf62fbd6b`; outcome perturbation: **PASS**.

## 7. Coefficient Stability

- Instability warning: **NO**; extreme fold: **NO**; prediction collapse: **NO**.
- Median / max beta L2 norm: **0.317945 / 0.330675**.

## 8. Unified vs Board3-Only Coefficient Changes

| Feature | Unified median | Board3-only median | Delta | Same sign |
|---|---:|---:|---:|:---:|
| rank_d1_close_ma10_pct | 0.038963 | -0.057921 | -0.096884 | NO |
| rank_d1_low_ma10_pct | 0.043766 | -0.096840 | -0.140606 | NO |
| rank_trend_hold_score | 0.015172 | 0.073469 | 0.058297 | YES |
| rank_total_score | -0.001608 | 0.090753 | 0.092361 | NO |
| rank_theme_score | 0.044081 | 0.056953 | 0.012872 | YES |
| rank_days_since_d0 | 0.018261 | 0.020139 | 0.001877 | YES |
| rank_log_candidate_base_price | 0.057126 | 0.087361 | 0.030235 | YES |
| rank_active_money_score | 0.057076 | -0.048740 | -0.105815 | NO |
| rank_d1_close_vwap_pct | 0.026790 | 0.159235 | 0.132445 | YES |
| inter_close_low | 0.048711 | -0.095520 | -0.144231 | NO |
| inter_close_trend | 0.029335 | 0.017754 | -0.011581 | YES |
| inter_total_trend | 0.014699 | 0.105733 | 0.091035 | YES |
| inter_total_active | 0.043344 | 0.008224 | -0.035120 | YES |
| inter_low_active | 0.061016 | -0.093265 | -0.154280 | NO |
| spread_close_low | -0.007066 | 0.039670 | 0.046736 | NO |
| days_since_d0_le1 | 0.106826 | 0.052148 | -0.054678 | YES |
| days_since_d0_eq2 | 0.000000 | 0.000000 | 0.000000 | YES |
| days_since_d0_ge3 | -0.108429 | -0.052148 | 0.056281 | YES |

## 9. Target7 Ranking

- Unified AUC: **0.4714**; Board3-only AUC: **0.4571**; delta: **-0.0143**.

## 10. NONLOSS Ranking

- Unified AUC: **0.2308**; Board3-only AUC: **0.6923**; delta: **0.4615**.

## 11. Same-Date Winner / Loss Ordering

- TARGET7_VS_NONTARGET: 8 pairs; unified **0.5000**, Board3-only **0.6250**, delta **0.1250**.
- NONLOSS_VS_LOSS: 14 pairs; unified **0.2143**, Board3-only **0.9286**, delta **0.7143**.
- TARGET7_VS_LOSS: 2 pairs; unified **0.0000**, Board3-only **1.0000**, delta **1.0000**.

## 12. Rank1 / Top2 Practical Performance

| Policy | Target7 | LOSS | Mean capped | Median capped | Worst |
|---|---:|---:|---:|---:|---:|
| BOARD3_UNIVERSE | 32.8125% | 40.6250% | 2.3351% | 1.4965% | -2.1505% |
| UNIFIED_BOARD3_RANK1 | 25.0000% | 37.5000% | 2.1404% | 3.2674% | -3.6816% |
| BOARD3_ONLY_RANK1 | 25.0000% | 37.5000% | 2.3574% | 2.0594% | -2.1505% |
| UNIFIED_BOARD3_TOP2 | 31.2500% | 43.7500% | 2.3014% | 1.4324% | -2.1505% |
| BOARD3_ONLY_TOP2 | 31.2500% | 37.5000% | 2.3579% | 1.5878% | -2.1505% |
| UNIFIED_BOARD3_TOP3 | 31.2500% | 45.8333% | 2.0639% | 1.4324% | -2.1505% |
| BOARD3_ONLY_TOP3 | 35.4167% | 37.5000% | 2.5785% | 2.4702% | -2.1505% |

## 13. Winner and Loss Rank Movement

- Unified / Board3-only rank selectivity: **-0.3571 / 0.1429**.
- Rank1 changed dates: **4**; net Target7: **0**; net losses removed: **0**.

## 14. Bootstrap / LODO

- Rank1 daily delta: mean **0.2169%**, positive/negative/zero **1/2/5**.
- Bootstrap P(Rank1 capped delta > 0): **50.7100%**.
- Bootstrap P(NONLOSS AUC delta > 0): **93.1172%**.

## 15. Feasibility Decision

| Gate | Pass |
|---|:---:|
| A | YES |
| B | NO |
| C | YES |
| D | YES |
| E | YES |
| F | NO |
| G | NO |
| H | YES |
| I | YES |
| J | NO |
| K | NO |
| L | YES |

- `BOARD3_ONLY_SIGNAL = PARTIAL`
- `NEXT_BOARD3_ACTION = NO_NEW_MODEL_YET`
- This benchmark evaluates Board3-internal mapping feasibility only. It does not calibrate or merge Board2 and Board3 scores.
- No July result was accessed, and this small May/June development sample is not production evidence.
- Recommended action: preserve the result and do not start another model or tune this specification yet.
