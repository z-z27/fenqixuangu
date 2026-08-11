# v004c Board2 vs Board3 Stage1 Feature-Response Audit v001
## 1. Experimental Contract
- Frozen V4C_STAGE1 reconstruction only; no new model, feature, policy, threshold, or July result access.
- `DIAGNOSTIC_CONTRIBUTION_ONLY = YES`; `NEW_MODEL_FEATURE = NO`.
## 2. Strict Population / Stage1 Parity
- Population: **155 rows / 17 dates; Board2 130, Board3 25**.
- Board2 Target7/LOSS: **44 / 33**; Board3: **9 / 8**.
- Stage1 Rank1/Rank2/Rank3/Top2/Top3: **2.9829% / 3.6804% / 0.2655% / 3.3316% / 2.3096%**.
## 3. Frozen 18-Feature Contract
- Model: `V4A_ARCH_TRANSFER_V4C` / `WEIGHTED_L2_LOGISTIC`; L2 **0.30**; positive weight **1.50**; features **18**.
- `rank_d1_close_ma10_pct`, `rank_d1_low_ma10_pct`, `rank_trend_hold_score`, `rank_total_score`, `rank_theme_score`, `rank_days_since_d0`, `rank_log_candidate_base_price`, `rank_active_money_score`, `rank_d1_close_vwap_pct`, `inter_close_low`, `inter_close_trend`, `inter_total_trend`, `inter_total_active`, `inter_low_active`, `spread_close_low`, `days_since_d0_le1`, `days_since_d0_eq2`, `days_since_d0_ge3`.
## 4. Coefficient Stability
| Feature | Mean beta | Median | Min | Max | + folds | - folds | Stable |
|---|---:|---:|---:|---:|---:|---:|---|
| rank_d1_close_ma10_pct | 0.044247 | 0.046106 | 0.030256 | 0.058645 | 100.0000% | 0.0000% | YES |
| rank_d1_low_ma10_pct | 0.050568 | 0.054803 | 0.035496 | 0.061652 | 100.0000% | 0.0000% | YES |
| rank_trend_hold_score | 0.018002 | 0.017240 | 0.006587 | 0.026108 | 100.0000% | 0.0000% | YES |
| rank_total_score | -0.012081 | -0.015190 | -0.031886 | 0.009944 | 29.4118% | 70.5882% | NO |
| rank_theme_score | 0.045301 | 0.045099 | 0.037608 | 0.053874 | 100.0000% | 0.0000% | YES |
| rank_days_since_d0 | 0.017893 | 0.018350 | 0.014237 | 0.020290 | 100.0000% | 0.0000% | YES |
| rank_log_candidate_base_price | 0.040367 | 0.037728 | 0.001200 | 0.066623 | 100.0000% | 0.0000% | YES |
| rank_active_money_score | 0.070406 | 0.078680 | 0.041552 | 0.091405 | 100.0000% | 0.0000% | YES |
| rank_d1_close_vwap_pct | 0.028925 | 0.027653 | 0.010271 | 0.048967 | 100.0000% | 0.0000% | YES |
| inter_close_low | 0.058551 | 0.064459 | 0.038613 | 0.074644 | 100.0000% | 0.0000% | YES |
| inter_close_trend | 0.038227 | 0.044497 | 0.016503 | 0.052081 | 100.0000% | 0.0000% | YES |
| inter_total_trend | 0.009009 | 0.011699 | -0.006210 | 0.016483 | 76.4706% | 23.5294% | NO |
| inter_total_active | 0.043380 | 0.043469 | 0.035654 | 0.048808 | 100.0000% | 0.0000% | YES |
| inter_low_active | 0.073049 | 0.076900 | 0.045404 | 0.096421 | 100.0000% | 0.0000% | YES |
| spread_close_low | -0.006322 | -0.006433 | -0.012211 | 0.000633 | 5.8824% | 94.1176% | YES |
| days_since_d0_le1 | 0.112026 | 0.112647 | 0.081660 | 0.134867 | 100.0000% | 0.0000% | YES |
| days_since_d0_eq2 | 0.000956 | 0.000000 | 0.000000 | 0.006267 | 23.5294% | 0.0000% | NO |
| days_since_d0_ge3 | -0.112982 | -0.112647 | -0.134867 | -0.084768 | 0.0000% | 100.0000% | YES |
## 5. Board Feature Exposure
| Feature | B2 median | B3 median | B3-vs-B2 Cliff | Same-date positive gap |
|---|---:|---:|---:|---:|
| rank_d1_close_ma10_pct | 0.500000 | 0.866667 | 0.498154 | 85.7143% |
| inter_close_low | 0.263889 | 0.694444 | 0.452923 | 78.5714% |
| inter_low_active | 0.239636 | 0.506667 | 0.447385 | 85.7143% |
| inter_close_trend | 0.285714 | 0.551020 | 0.440923 | 92.8571% |
| rank_d1_low_ma10_pct | 0.514706 | 0.833333 | 0.400000 | 85.7143% |
| rank_active_money_score | 0.516667 | 0.666667 | 0.342462 | 85.7143% |
| rank_d1_close_vwap_pct | 0.514706 | 0.777778 | 0.302462 | 71.4286% |
| inter_total_active | 0.232743 | 0.337778 | 0.285538 | 78.5714% |
| rank_trend_hold_score | 0.583333 | 0.714286 | 0.251692 | 85.7143% |
| inter_total_trend | 0.232743 | 0.437500 | 0.162462 | 64.2857% |
## 6. Board Contribution Exposure
- Board3-Board2 mean/median daily total logit gap: **0.105937 / 0.129375**; positive dates **13**.
| Driver | Mean daily gap | Median | Beta median | B2 NONLOSS AUC | B3 NONLOSS AUC | Category | Pass |
|---|---:|---:|---:|---:|---:|---|---|
| inter_low_active | 0.020606 | 0.020915 | 0.076900 | 0.516870 | 0.147059 | OTHER | NO |
| inter_close_low | 0.018948 | 0.023466 | 0.064459 | 0.545923 | 0.191176 | OTHER | NO |
| rank_active_money_score | 0.015069 | 0.014725 | 0.078680 | 0.475789 | 0.257353 | OTHER | NO |
| rank_d1_close_ma10_pct | 0.013899 | 0.015379 | 0.046106 | 0.551390 | 0.227941 | BOARD3_INVERTED | YES |
| rank_d1_low_ma10_pct | 0.011834 | 0.014982 | 0.054803 | 0.562949 | 0.139706 | BOARD3_INVERTED | YES |
| inter_close_trend | 0.010086 | 0.009671 | 0.044497 | 0.515776 | 0.264706 | OTHER | NO |
| rank_d1_close_vwap_pct | 0.006319 | 0.008400 | 0.027653 | 0.546079 | 0.411765 | OTHER | NO |
| inter_total_active | 0.003998 | 0.004772 | 0.043469 | 0.585442 | 0.470588 | BOARD3_DEGRADED | NO |
| rank_trend_hold_score | 0.003058 | 0.002415 | 0.017240 | 0.558419 | 0.393382 | BOARD3_INVERTED | NO |
| rank_log_candidate_base_price | 0.002256 | 0.002232 | 0.037728 | 0.507966 | 0.691176 | OTHER | NO |
## 7. Board2 vs Board3 Target7 Response
| Feature | B2 AUC | B3 AUC | Delta | B2 pair | B3 pair | Winner safety |
|---|---:|---:|---:|---:|---:|---|
| rank_d1_close_ma10_pct | 0.522463 | 0.402778 | -0.119685 | 0.493023 | 0.500000 | WINNER_HARMFUL |
| rank_d1_low_ma10_pct | 0.534884 | 0.416667 | -0.118217 | 0.520930 | 0.625000 | WINNER_HARMFUL |
| days_since_d0_le1 | 0.471855 | 0.319444 | -0.152411 | 0.490698 | 0.500000 | WINNER_HARMFUL |
| rank_trend_hold_score | 0.546512 | 0.576389 | 0.029877 | 0.488372 | 0.750000 | WINNER_ALIGNED |
| inter_total_active | 0.531712 | 0.614583 | 0.082871 | 0.511628 | 0.937500 | WINNER_ALIGNED |
## 8. Board2 vs Board3 NONLOSS Response
| Feature | B2 AUC | B3 AUC | Delta | B2 pair | B3 pair | Gap | Bootstrap P(delta<0) | q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rank_d1_close_ma10_pct | 0.551390 | 0.227941 | -0.323449 | 0.497006 | 0.142857 | 0.013899 | 98.8750% | 0.082440 |
| rank_d1_low_ma10_pct | 0.562949 | 0.139706 | -0.423243 | 0.514970 | 0.142857 | 0.011834 | 99.8300% | 0.016200 |
| days_since_d0_le1 | 0.571853 | 0.235294 | -0.336558 | 0.535928 | 0.500000 | 0.002100 | 99.4150% | 0.079650 |
| rank_trend_hold_score | 0.558419 | 0.393382 | -0.165037 | 0.458084 | 0.678571 | 0.003058 | 87.3950% | 0.223800 |
| inter_total_active | 0.585442 | 0.470588 | -0.114854 | 0.574850 | 0.642857 | 0.003998 | 86.1650% | 0.168525 |
## 9. Same-Date Pair Concordance
- Aggregate score Board2 Target7/NONLOSS pairs: **215 / 167**; Board3: **8 / 14**.
- Feature/board/endpoint pair counts and weak-support flags are preserved in the pair artifact.
## 10. Bootstrap / LODO / Permutation
- Primary NONLOSS interaction tests: **18**; q<=0.10 **5**; q<=0.05 **3**; minimum q **0.016200**.
## 11. Specific Response Inversions
- Formal passes: **3**; mechanism candidates: **5**.
- `rank_d1_close_ma10_pct`: B2/B3 NONLOSS AUC **0.551390 / 0.227941**, gap **0.013899**, q **0.082440**, WINNER_HARMFUL.
- `rank_d1_low_ma10_pct`: B2/B3 NONLOSS AUC **0.562949 / 0.139706**, gap **0.011834**, q **0.016200**, WINNER_HARMFUL.
- `days_since_d0_le1`: B2/B3 NONLOSS AUC **0.571853 / 0.235294**, gap **0.002100**, q **0.079650**, WINNER_HARMFUL.
## 12. Contribution Concentration
- Total positive gap **0.109101**; Top1/Top3/Top5 shares **18.8871% / 50.0659% / 73.6523%**.
- Harmful inversion positive gap/share: **0.027833 / 25.5115%**.
## 13. Aggregate Score Control
- Board2 Target7/NONLOSS AUC: **0.403277 / 0.472977**; pair concordance **0.483721 / 0.556886**.
- Board3 Target7/NONLOSS AUC: **0.541667 / 0.264706**; pair concordance **0.500000 / 0.214286**.
## 14. Board-Aware Formulation Decision
- Q1 Frozen features reverse good/bad response from Board2 to Board3: **YES**.
- Q2 Such features give Board3 positive Stage1 contribution: **YES**.
- Q3 Any inversion survives bootstrap, LODO, and BH: **YES**.
- Q4 Established inversions non-redundant: **YES**.
- Q5 Overpromotion mainly attributable to a small feature set: **PARTIAL**.
- Q6 Broad Stage1 score misalignment inside Board3: **PARTIAL**.
- Q7 Minimal board×feature formulation supported now: **NO**.
- `BOARD3_STAGE1_RESPONSE_MECHANISM = PARTIAL_FEATURE_MISALIGNMENT`
- `BOARD_AWARE_FORMULATION_DIRECTION = NO_BOARD_AWARE_MODEL_YET`
- Why Stage1 over-selects Board3: Board3 receives a date-equal logit lift of **0.105937**; positive price/active-money interaction contributions dominate the lift, while three formally inverted responses explain only **25.5115%** of all positive contribution gap.
- Largest positive Board3 contribution sources: **inter_low_active, inter_close_low, rank_active_money_score, rank_d1_close_ma10_pct, rank_d1_low_ma10_pct**.
- Formal harmful/inverted contributions: **rank_d1_close_ma10_pct, rank_d1_low_ma10_pct, days_since_d0_le1**.
- Board2 uses each formal inversion feature in the intended NONLOSS direction (AUC >=0.55), but their Board3 winner-safety is **WINNER_HARMFUL**.
- Aggregate Board3 retains Target7 ordering above chance (**0.541667**) while failing NONLOSS ordering (**0.264706**); however same-date Board3 pair support is weak and the formal harmful share is below 30%.
- The failure is partially localized but not sufficiently independent/concentrated for a minimal interaction experiment; the evidence also does not satisfy the formal broad-separation state.
- Recommended next technical action: preserve this audit and do not train a board-aware model yet; require independent predeclared evidence before choosing calibration, minimal interactions, or broader separation.
- No board interaction, calibration, board-specific model, penalty, exclusion, or policy was trained/tested.
