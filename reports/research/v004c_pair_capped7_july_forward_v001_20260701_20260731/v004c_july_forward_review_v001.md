# v004c PAIR_CAPPED7 July Chronological Forward Stress v001

## 1. Frozen Specification

- Frozen development commit: `c6af1289f66f8c2cad610b9564573607ce57ac14`
- Forward class: `SEMI_OOT_CHRONOLOGICAL`
- Stage1: frozen 18-feature weighted L2 Logistic; L2=0.30; positive weight=1.50; original tail/date weights.
- Stage2: frozen same-date pairwise weighted Ridge; CAPPED7; 3 predictors; L2=0.30; intercept=0; Top10 only.
- July model selected before outcome evaluation: **YES**; tuning/new features/July refit: **NO**.

## 2. Temporal / Label Availability Audit

- Label available date: D3 date; training rule: `label_available_date < 2026-07-01`.
- June immature label rows across original folds: **337**; affected folds: **21**.
- June temporal caveat: **YES**.

## 3. Pre-July Training Snapshot

- Eligible rows/dates: **307 / 37**.
- Latest signal / label date: **2026-06-26 / 2026-06-30**.
- Stage1 fitted once; Stage2 fitted once; meta rows/pairs: **268 / 994**.
- Self-label leakage / July training / August training: **0 / 0 / 0**.

## 4. Prediction Lock and Outcome Independence

- Prediction rows/dates: **178 / 23**.
- SHA256: `725ffa3e386034233bb6f2cca1fdeff13a31a286b5c3621043c85c5fd5333a08`.
- Outcome columns present: **NO**; perturbation test: **PASS**; outcome influence: **0**.

## 5. July Universe and Data Coverage

- Rows/dates: **178 / 23**; Board2/Board3: **155 / 23**.
- Candidate count min/median/max: **2 / 6.0 / 19**.
- Outcome complete/missing: **178 / 0**; August only for July outcome completion: **YES**.

## 6. CONTROL vs PAIR_CAPPED7

| Model | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Excess | Top1 Hit | Top3 Precision | Beat Universe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNIVERSE | 1.9920% | 1.9920% | 2.4492% | 1.9920% | 2.4492% | 0.0000% | 25.7992% | 25.7992% | NA |
| CONTROL_V4A | 1.7203% | 1.4595% | 1.8782% | 1.5899% | 1.9653% | -0.4840% | 26.0870% | 28.5714% | 38.0952% |
| PAIR_CAPPED7 | 1.6677% | 1.8681% | 2.2792% | 1.7679% | 2.2289% | -0.2203% | 17.3913% | 26.9841% | 47.6190% |
| TOP10_ORACLE | 6.1712% | 4.3247% | 3.6087% | 5.2479% | 5.2131% | 2.7638% | 69.5652% | 49.2063% | 85.7143% |
| FULL_ORACLE | 6.1989% | 4.5700% | 4.0236% | 5.3845% | 5.4511% | 3.0018% | 73.9130% | 57.1429% | 85.7143% |
- CAPPED - CONTROL: **0.2636%**; CAPPED - Universe: **-0.2203%**.

## 7. Downside Risk

| Model | Mean | Median | P25 | Worst | Positive Dates | Negative Dates | <=-3% | <=-5% | <=-10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CONTROL | 1.9653% | 2.8958% | -0.1494% | -6.3540% | 71.4286% | 28.5714% | 2 | 1 | 0 |
| PAIR_CAPPED7 | 2.2289% | 2.6745% | 0.7257% | -4.4428% | 76.1905% | 23.8095% | 1 | 0 | 0 |
- Raw selected-stock mean/median/worst: CONTROL **3.2885% / 2.9017% / -13.0864%**; CAPPED **3.6718% / 2.9017% / -13.0864%**.

## 8. LOW / MID / HIGH Opportunity

- LOW Top3: Universe **-0.0760%**; CONTROL **-2.0842%**; CAPPED **-0.7612%**; delta **1.3229%**; negative dates **66.6667% → 50.0000%**.
- MID Top3: CONTROL **1.7028%**; CAPPED **1.8615%**.
- HIGH Top3: CONTROL **5.7362%**; CAPPED **5.2118%**; delta **-0.5244%**; precision **57.1429% → 47.6190%**.

## 9. Stage1 Retrieval and Oracle Gap

- Target7 recall@3/@5/@7/@10: **39.1304% / 56.5217% / 78.2609% / 89.1304%**.
- Date-equal winner capture@3/@5/@10: **49.0196% / 57.8431% / 87.2549%**.
- Top10 oracle / full oracle / retrieval oracle ratio: **5.2131% / 5.4511% / 92.0717%**.
- Available rerank gap / recovered gap / recovery ratio: **3.2478% / 0.2636% / 8.1177%**.

## 10. Membership Changes

- Changed dates/slots: **17 / 27**.
- Demoted Target7/non-target: **9 / 18**; promoted Target7/non-target/loss/severe: **8 / 19 / 7 / 2**.
- Net Target7 slots gained: **-1**.

## 11. Daily Robustness

- Daily delta mean/median; positive/negative/zero: **0.2407% / 0.0000%; 6 / 9 / 8**.
- Bootstrap p2.5/p50/p97.5; P(delta>0): **-0.3700% / 0.2241% / 0.9310%; 75.5100%**.
- LODO positive; min/median/max: **100.0000%; 0.0540% / 0.2517% / 0.3408%**.

## 12. Forward-Stress Decision

- Gates: top3_control=PASS, top3_universe=FAIL, delta=FAIL, precision_control=FAIL, precision_universe=PASS, beat_universe=FAIL, daily_breadth=FAIL, negative_rate=PASS, worst=PASS, days_minus5=PASS, low_return=PASS, low_negative=PASS, high_upside=FAIL, bootstrap=PASS, lodo=PASS.
- `FORWARD_STRESS_SIGNAL = PARTIAL`
- `PRIMARY_FAILURE_ATTRIBUTION = STAGE2_RERANK_FAILURE`
- June development reference: CONTROL 1.8539%; CAPPED 3.0443%; delta +1.1904pp.
- July: CONTROL 1.9653%; CAPPED 2.2289%; delta 0.2636%.
- No June/July retuning, no alternative target, no production model, no deployment action.
