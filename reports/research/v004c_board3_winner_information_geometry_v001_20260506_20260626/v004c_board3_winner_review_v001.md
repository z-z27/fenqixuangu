# v004c Board3 Winner Information Geometry Audit v001

## 1. Experimental Contract

- Board3-only May/June information audit; no model, new feature, feature engineering, target search, or threshold search.
- `ORIENTATION_SOURCE = FULL_DEVELOPMENT_T7_VS_PNT`; orientation is post-hoc diagnostic only and is fixed for all endpoints.
- `JULY_RESULT_ROWS_ACCESSED = 0`.

## 2. Board3 Matured Population

- Rows/dates: **55 / 28**.
- LOSS / POSITIVE_NON_TARGET / TARGET7: **13 / 19 / 23**.

## 3. LOSS / POSITIVE_NON_TARGET / TARGET7 Distribution

| State | n | Mean | Median | p25 | p75 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOSS | 13 | -2.8204% | -2.1505% | -4.1667% | -1.4531% | -5.9423% | -0.7092% |
| POSITIVE_NON_TARGET | 19 | 3.0921% | 3.2295% | 1.6182% | 4.5592% | 0.2513% | 6.4714% |
| TARGET7 | 23 | 11.9732% | 9.9733% | 8.7226% | 14.5137% | 7.1353% | 22.2997% |

- PNT 0–3% / 3–5% / 5–7% counts: **9 / 7 / 3**.
- T7 7–10% / 10–12% / >=12% counts: **13 / 4 / 6**.
- Rows within +/-1pp of 7%: **6**; median absolute distance for PNT/T7: **3.7705% / 2.9733%**.

## 4. Feature Families

- Low-level family: **53**; frozen Stage1 representation: **18**.
- Benjamini-Hochberg correction was performed separately over exactly 53 and exactly 18 primary permutation p-values.

| Family | Group | Features | Winner pass | Ordinal pass | Best T7/PNT AUC | Best q | Best NONLOSS rho |
|---|---|---:|---:|---:|---:|---:|---:|
| FROZEN_STAGE1_18 | DAY_BUCKET | 3 | 0 | 0 | 0.5137 | 1.000000 | 0.1403 |
| FROZEN_STAGE1_18 | INTERACTION_OR_SPREAD | 6 | 0 | 0 | 0.6751 | 0.882856 | 0.2620 |
| FROZEN_STAGE1_18 | RANK | 9 | 0 | 0 | 0.7414 | 0.882856 | 0.2574 |
| LOW_LEVEL_53 | BOARD_HISTORY | 6 | 0 | 0 | 0.6350 | 0.632435 | 0.2389 |
| LOW_LEVEL_53 | D0_CROSS_SECTION | 1 | 0 | 0 | 0.5595 | 1.000000 | 0.0835 |
| LOW_LEVEL_53 | D1_CHIP_DISTRIBUTION | 4 | 0 | 0 | 0.5355 | 1.000000 | 0.1251 |
| LOW_LEVEL_53 | D1_LATE_DAY_PRESSURE | 2 | 0 | 0 | 0.5812 | 1.000000 | 0.1845 |
| LOW_LEVEL_53 | D1_MA_POSITION | 15 | 0 | 0 | 0.6201 | 0.750442 | 0.3189 |
| LOW_LEVEL_53 | D1_PRICE_ACTION | 14 | 0 | 0 | 0.6247 | 0.750442 | 0.2279 |
| LOW_LEVEL_53 | D1_VOLUME_ACTIVITY | 3 | 0 | 0 | 0.5423 | 1.000000 | 0.0824 |
| LOW_LEVEL_53 | D1_VWAP_POSITION | 1 | 0 | 0 | 0.5011 | 1.000000 | -0.1237 |
| LOW_LEVEL_53 | POOL_MEMBERSHIP | 3 | 0 | 0 | 0.5000 | 1.000000 | NA |
| LOW_LEVEL_53 | RECENT_7D_PATH | 4 | 0 | 0 | 0.6362 | 0.632435 | 0.2778 |

## 5. Low-Level 53 Winner Screen

- WINNER_PASS / EXCEPTIONAL / ORDINAL: **0 / 0 / 0**; q<=.10: **0**; lowest q: **0.632435**.

| Feature | Group | T7/PNT AUC | Cliff | PNT/LOSS | T7/LOSS | Spearman | Date consistency | q | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| recent_limit_up_count_20d | BOARD_HISTORY | 0.6350 | 0.2700 | 0.3644 | 0.5100 | 0.2325 | 100.0000% | 0.632435 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|FDR |
| recent_pool_appearance_count_20d | BOARD_HISTORY | 0.6350 | 0.2700 | 0.3644 | 0.5100 | 0.2325 | 100.0000% | 0.632435 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|FDR |
| recent_7d_limit_up_count | RECENT_7D_PATH | 0.6053 | 0.2105 | 0.4879 | 0.5936 | 0.2778 | 100.0000% | 0.632435 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|FDR |
| d1_intraday_range | D1_PRICE_ACTION | 0.6247 | 0.2494 | 0.4291 | 0.5987 | 0.2194 | 62.5000% | 0.750442 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|DATE_CONSISTENCY|BOOTSTRAP|FDR|PAIR_CONCORDANCE |
| d1_ma20_slope | D1_MA_POSITION | 0.6201 | 0.2403 | 0.3522 | 0.4749 | 0.2631 | 75.0000% | 0.750442 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|BOOTSTRAP|FDR |
| d1_high_to_ma5_raw | D1_MA_POSITION | 0.6156 | 0.2311 | 0.2591 | 0.4114 | 0.1783 | 62.5000% | 0.750442 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|DATE_CONSISTENCY|BOOTSTRAP|FDR |
| d1_close_to_ma20 | D1_MA_POSITION | 0.6110 | 0.2220 | 0.3077 | 0.4114 | 0.3189 | 62.5000% | 0.750442 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|DATE_CONSISTENCY|BOOTSTRAP|FDR |
| recent_limit_up_count_10d | BOARD_HISTORY | 0.6098 | 0.2197 | 0.4413 | 0.5686 | 0.2389 | 100.0000% | 0.750442 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|FDR |
| recent_pool_appearance_count_10d | BOARD_HISTORY | 0.6098 | 0.2197 | 0.4413 | 0.5686 | 0.2389 | 100.0000% | 0.750442 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|FDR |
| d1_last_hour_return | D1_PRICE_ACTION | 0.5973 | 0.1945 | 0.5182 | 0.5619 | 0.1865 | 75.0000% | 0.750442 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|CONTINUOUS_INFO|BOOTSTRAP|FDR |

## 6. Frozen Stage1 18 Winner Screen

- WINNER_PASS / EXCEPTIONAL / ORDINAL: **0 / 0 / 0**; q<=.10: **0**; lowest q: **0.882856**.

| Feature | Group | T7/PNT AUC | Cliff | PNT/LOSS | T7/LOSS | Spearman | Date consistency | q | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| inter_close_trend | INTERACTION_OR_SPREAD | 0.6751 | 0.3501 | 0.2794 | 0.4749 | 0.2620 | 62.5000% | 0.882856 | FAIL:T7_LOSS_SAFETY|PNT_LOSS_SAFETY|DATE_CONSISTENCY|FDR |
| rank_trend_hold_score | RANK | 0.6728 | 0.3455 | 0.4858 | 0.6689 | 0.1875 | 100.0000% | 0.882856 | FAIL:PNT_LOSS_SAFETY|FDR |
| inter_total_trend | INTERACTION_OR_SPREAD | 0.6236 | 0.2471 | 0.6296 | 0.7559 | 0.0720 | 57.1429% | 0.882856 | FAIL:PRIMARY_AUC|EFFECT_SIZE|CONTINUOUS_INFO|DATE_CONSISTENCY|FDR|PAIR_CONCORDANCE |
| rank_total_score | RANK | 0.6098 | 0.2197 | 0.6640 | 0.7575 | 0.0429 | 57.1429% | 0.882856 | FAIL:PRIMARY_AUC|EFFECT_SIZE|CONTINUOUS_INFO|DATE_CONSISTENCY|BOOTSTRAP|FDR|PAIR_CONCORDANCE |
| rank_days_since_d0 | RANK | 0.7414 | 0.4828 | 0.4049 | 0.6037 | 0.2574 | NA | 1.000000 | FAIL:PNT_LOSS_SAFETY|DATE_CONSISTENCY|FDR|PAIR_CONCORDANCE |
| rank_theme_score | RANK | 0.6030 | 0.2059 | 0.4838 | 0.5769 | 0.1164 | 100.0000% | 1.000000 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|CONTINUOUS_INFO|FDR|PAIR_CONCORDANCE |
| rank_d1_close_vwap_pct | RANK | 0.5755 | 0.1510 | 0.5425 | 0.5903 | 0.2212 | 57.1429% | 1.000000 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|DATE_CONSISTENCY|BOOTSTRAP|FDR |
| rank_active_money_score | RANK | 0.5664 | 0.1327 | 0.5506 | 0.6371 | -0.0268 | 57.1429% | 1.000000 | FAIL:PRIMARY_AUC|EFFECT_SIZE|CONTINUOUS_INFO|DATE_CONSISTENCY|BOOTSTRAP|FDR|PAIR_CONCORDANCE |
| rank_log_candidate_base_price | RANK | 0.5378 | 0.0755 | 0.5243 | 0.5401 | 0.0948 | 62.5000% | 1.000000 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|DATE_CONSISTENCY|BOOTSTRAP|FDR|PAIR_CONCORDANCE |
| spread_close_low | INTERACTION_OR_SPREAD | 0.5355 | 0.0709 | 0.5000 | 0.5502 | 0.1194 | 57.1429% | 1.000000 | FAIL:PRIMARY_AUC|EFFECT_SIZE|T7_LOSS_SAFETY|PNT_LOSS_SAFETY|CONTINUOUS_INFO|DATE_CONSISTENCY|BOOTSTRAP|FDR|PAIR_CONCORDANCE |

## 7. TARGET7 vs POSITIVE_NON_TARGET

- Strongest low-level oriented AUC: **0.6362**.
- Strongest Stage1-representation oriented AUC: **0.7414**.

## 8. POSITIVE_NON_TARGET vs LOSS Safety

- Winner-pass features safe at AUC>=.52: **0/0**.

## 9. TARGET7 vs LOSS Extreme Separation

- Best extreme AUC: **0.7926**.

## 10. Continuous NONLOSS Raw-Repair Ordering

- Features with Spearman>=.20: **14**; raw pair concordance>=.57: **30**.

## 11. Same-Date Pair Geometry

- T7/PNT pair support range: **17–17 pairs**, informative dates **8–8**.

## 12. Bootstrap / LODO / Permutation / FDR

- Low-level / Stage1 q<=.10: **0 / 0**.
- Bootstrap resamples dates; LODO fixes the full-development orientation; permutation shuffles only T7/PNT labels within each date among NONLOSS rows.

## 13. Redundancy

- Passing features: **0**; near-redundant pass pairs: **0**.

## 14. Winner Information Foundation

- `BOARD3_WINNER_INFORMATION_FOUNDATION = PARTIAL`
- `FOUNDATION_SOURCE = PARTIAL_ONLY`

## 15. Ordinal Repair Information

- `BOARD3_ORDINAL_REPAIR_INFORMATION = PARTIAL`

## 16. Next Board3 Decision

- `NEXT_BOARD3_ACTION = NO_NEW_MODEL_YET`
- No model is trained in this task; no July result is accessed; this is not production evidence.
- Preserve this partial audit and do not train another Board3 model yet.
