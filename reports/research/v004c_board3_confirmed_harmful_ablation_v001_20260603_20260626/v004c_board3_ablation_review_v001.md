# v004c Board3 Confirmed-Harmful Contribution Ablation v001
## 1. Experimental Contract
- Post-hoc June mechanism probe; frozen V4C_STAGE1 only; exactly one predeclared three-feature contribution ablation; no model, feature, subset, scale, policy, or July-result access.
## 2. Strict Stage1 Parity
- Population: **155 rows / 17 dates; Board2 130, Board3 25**. Stage1 Top3 **2.3096%**, precision **31.3725%**.
## 3. Predeclared Three-Feature Ablation
- Board3 only: `rank_d1_close_ma10_pct`, `rank_d1_low_ma10_pct`, `days_since_d0_le1`; each frozen contribution is set exactly to zero, including negative contributions. Board2 is unchanged.
## 4. Prediction Lock / Leakage Audit
- SHA256: `662228eb8f511c2d16714215c28af7021b3530161761af5f55656e1577766867`; outcome perturbation PASS; self/current/July rows **0 / 0 / 0**.
## 5. Board3 Score Movement
- Mean original/ablated logit **0.317140 / 0.148190**.
- Removed contribution mean/median/p25/p75/min/max: **0.168951 / 0.175839 / 0.140895 / 0.214851 / 0.048086 / 0.227274**.
| Feature | Mean removed | Median | Positive rows | Negative rows |
|---|---:|---:|---:|---:|
| rank_d1_close_ma10_pct | 0.033227 | 0.033989 | 100.0000% | 0.0000% |
| rank_d1_low_ma10_pct | 0.034819 | 0.037479 | 100.0000% | 0.0000% |
| days_since_d0_le1 | 0.100905 | 0.107255 | 92.0000% | 0.0000% |
## 6. Funnel Composition Change
- Candidate Board3 share: **16.1290%**.
| Ranking | Level | B2 | B3 | B3 share | B3 composition lift | B3 selection rate |
|---|---|---:|---:|---:|---:|---:|
| ORIGINAL | Top10 | 108 | 24 | 18.1818% | 1.127273 | 96.0000% |
| ORIGINAL | Top5 | 62 | 19 | 23.4568% | 1.454321 | 76.0000% |
| ORIGINAL | Top3 | 36 | 15 | 29.4118% | 1.823529 | 60.0000% |
| ABLATED | Top10 | 114 | 18 | 13.6364% | 0.845455 | 72.0000% |
| ABLATED | Top5 | 74 | 7 | 8.6420% | 0.535802 | 28.0000% |
| ABLATED | Top3 | 50 | 1 | 1.9608% | 0.121569 | 4.0000% |
- Board3 Top3 slot reduction: **14 / 93.3333%**; suppression high **YES**; effectively excluded **NO**.
## 7. Board3 Rank Movement by Outcome
| State | Mean rank change | Median | p25 | p75 |
|---|---:|---:|---:|---:|
| LOSS | 4.750000 | 3.500000 | 3.000000 | 6.750000 |
| TARGET7 | 3.333333 | 3.000000 | 2.000000 | 5.000000 |
| POSITIVE_NON_TARGET | 4.625000 | 4.000000 | 3.000000 | 6.000000 |
- Median LOSS-minus-Target7 rank-change gap: **0.500000**.
- Same-date Board3 LOSS-vs-Target7: dates **1**, pairs **2**, improved/worsened/ties **0/1/1**, improvement rate **0.0000%**.
## 8. Top3 Membership Transitions
- Changed dates/slots **12 / 14**.
- Demoted Board3/Board2 **14/0**; promoted Board3/Board2 **0/14**.
- Demoted Target7/LOSS **4/7**; promoted Target7/LOSS **4/5**; net Target7 **0**, net LOSS removed **2**.
## 9. Board3 Loss Removal vs Winner Removal
- Board3 loss removal **7/8 (87.5000%)**; winner removal **4/4 (100.0000%)**; selectivity gap **-12.5000%**.
## 10. Overall Practical Performance
| Policy | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Precision | Capture | Negative dates | Worst | <=-3 | <=-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STRICT_STAGE1 | 2.9829% | 3.6804% | 0.2655% | 3.3316% | 2.3096% | 31.3725% | 47.7778% | 23.5294% | -6.7688% | 1 | 1 |
| BOARD3_ABLATION | 4.2898% | -0.0602% | 2.3618% | 2.1148% | 2.1971% | 31.3725% | 42.2222% | 17.6471% | -5.3533% | 2 | 1 |
- Stage1/Ablation Top3 **2.3096% / 2.1971%**; delta **-0.1125%**; precision **31.3725% / 31.3725%**.
- Board3 Top3 loss **53.3333% → 100.0000%**; mean capped **1.4147% → -2.0413%**; Target7 count **4 → 0**.
- Board2 Top3 slots/Target7/LOSS/mean capped: **36/12/8/2.6824% → 50/16/13/2.2819%**.
- Raw downside Stage1/Ablation selected-stock mean **3.3236%/3.3241%**, median **4.5483%/4.5675%**, worst stock **-24.4558%/-24.4558%**, worst Top3 date **-6.7688%/-5.3533%**.
## 11. LOW / MID / HIGH
- LOW: Stage1/Ablation Top3 **-0.1668% / -0.4160%** (delta **-0.2492%**); Board3 slots/Target7/LOSS/mean capped **6/1/4/0.5165% → 1/0/1/-2.0413%**; SMALL_SAMPLE.
- MID: Stage1/Ablation Top3 **2.8944% / 1.9910%** (delta **-0.9035%**); Board3 slots/Target7/LOSS/mean capped **6/3/2/3.3764% → 0/0/0/NA**; SMALL_SAMPLE.
- HIGH: Stage1/Ablation Top3 **4.2986% / 4.9821%** (delta **0.6835%**); Board3 slots/Target7/LOSS/mean capped **3/0/2/-0.7121% → 0/0/0/NA**; SMALL_SAMPLE.
## 12. Daily Robustness / Bootstrap / LODO
- Daily delta mean/median **-0.1125%/0.0000%**; positive/negative/zero **6/5/6**.
- Bootstrap overall delta p2.5/p50/p97.5/P>0 **-1.2788%/-0.0897%/0.9469%/43.7500%**.
- Bootstrap Board3 LOSS delta valid/p2.5/p50/p97.5/P<0 **12822/25.0000%/45.0000%/64.7059%/0.0000%**.
- Bootstrap Board3 capped delta valid/p2.5/p50/p97.5/P>0 **12822/-5.0609%/-3.2156%/-1.5606%/0.0000%**.
- LODO overall valid/favorable/min/median/max **17/29.4118%/-0.3138%/-0.1195%/0.2820%**.
- LODO Board3 LOSS valid/favorable/min/median/max **16/0.0000%/42.8571%/46.6667%/50.0000%**; capped **16/0.0000%/-3.8201%/-3.4560%/-3.0571%**.
## 13. Return Attribution
- Attributed/observed Top3 delta **-0.1125% / -0.1125%**; closure PASS.
| Demoted category | Count | Contribution |
|---|---:|---:|
| DEMOTED_BOARD3_LOSS | 7 | 0.8024% |
| DEMOTED_BOARD3_TARGET7 | 4 | -0.6544% |
| DEMOTED_BOARD3_POSITIVE_NON_TARGET | 3 | -0.2605% |
| DEMOTED_BOARD2_LOSS | 0 | 0.0000% |
| DEMOTED_BOARD2_TARGET7 | 0 | 0.0000% |
| DEMOTED_BOARD2_POSITIVE_NON_TARGET | 0 | 0.0000% |
## 14. Mechanism Decision
| Gate | Pass |
|---|---|
| A | NO |
| B | YES |
| C | NO |
| D | NO |
| E | NO |
| F | NO |
| G | NO |
| H | NO |
| I | YES |
| J | YES |
| K | NO |
| L | NO |
- `ABLATION_MECHANISM_SIGNAL = ABSENT`
- `NEXT_BOARD_ARCHITECTURE_DIRECTION = BROADER_BOARD_SEPARATION`
- The ablation did not selectively reduce Board3 loss slots; it removed **7/8 losses and 4/4 winners**, leaving one Board3 Top3 slot, which was a loss. Board3 slot suppression was **93.3333%**.
- Overall Top3 fell **-0.1125%**. The three contributions explain broad Board3 promotion pressure, but not a safe loss-selective mechanism; their removal suppresses winners at least as aggressively as losses.
- Recommended next technical action: preserve this negative probe and, in a separate task only, consider broader Board2/Board3 formulation separation. Do not test another ablation, subset, scale, or minimal interaction now.
- This is same-June mechanism plausibility only, not validation or forward evidence. No second ablation or board-aware model is authorized here.
