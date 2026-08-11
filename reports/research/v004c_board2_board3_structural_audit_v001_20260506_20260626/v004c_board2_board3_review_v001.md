# v004c Board2 vs Board3 Structural Audit v001
## 1. Experimental Contract
- May/June only; `JULY_RESULT_ROWS_ACCESSED = 0`; no model, feature, threshold, board rule, or policy was developed.
- Current first stage: `V4C_STAGE1` (`V4A_ARCH_TRANSFER_V4C`); historical original v004a is not the same model instance.
- Outcomes: `raw=D3 high/D2 open-1`; `Target7=raw>=7%`; `LOSS=raw<0`; capped return is upper-cap-only at 7%.
## 2. Matured Pre-July Universe
- Rows/dates: **307 / 37**; Board2/Board3: **252 / 55**.
| Board | Rows | Dates | Target7 | LOSS | Severe <=-5% | Raw mean | Raw median | Capped mean | Capped median | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Board2 | 252 | 37 | 35.7143% | 23.4127% | 4.7619% | 4.7539% | 4.6170% | 3.1387% | 4.6170% | -24.4558% |
| Board3 | 55 | 28 | 41.8182% | 23.6364% | 1.8182% | 5.4085% | 4.5701% | 3.3288% | 4.5701% | -5.9423% |
- Board3-Board2: Target7 **6.1039%**; LOSS **0.2237%**; severe **-2.9437%**; capped mean **0.1901%**.
## 3. Board2 vs Board3 Baseline Outcomes
- Bootstrap LOSS gap p2.5/p50/p97.5; P(Board3>Board2): **-11.8562% / 0.2403% / 13.1098%; 51.4550%**.
- Bootstrap Target7 gap p2.5/p50/p97.5; P(Board3<Board2): **-9.0262% / 6.4459% / 23.3940%; 20.8950%**.
- Bootstrap capped gap p2.5/p50/p97.5; P(Board3<Board2): **-1.1234% / 0.2102% / 1.5830%; 37.6550%**.
- Stratified permutation p: LOSS **1.000000**; Target7 **0.406380**; capped **0.761712**.
## 4. Same-Date Board Comparison
- Mixed dates: **28**. LOSS Board3 worse/better/tie: **11 / 12 / 5**; direction consistency **39.2857%**; mean/median gap **5.1727% / 0.0000%**.
- Target7 Board3 better/worse/tie: **13 / 14 / 1**; mean/median gap **13.1253% / -0.8929%**.
- Capped return Board3 better/worse/tie: **13 / 15 / 0**; mean/median gap **0.5185% / -0.1314%**.
## 5. Strict V4C_STAGE1 Funnel Composition
- Strict parity: 17 dates; Rank1/2/3 **2.9829% / 3.6804% / 0.2655%**; Top3 **2.3096%**; precision **31.3725%**; winner capture **47.7778%**; negative dates **23.5294%**; worst **-6.7688%**.
| Funnel | Total | Board2 | Board3 | Board3 share | Composition lift | Selection ratio B3/B2 |
|---|---:|---:|---:|---:|---:|---:|
| Universe | 155 | 130 | 25 | 16.1290% | 1.0000 | 1.0000 |
| TOP10 | 132 | 108 | 24 | 18.1818% | 1.1273 | 1.1556 |
| TOP5 | 81 | 62 | 19 | 23.4568% | 1.4543 | 1.5935 |
| TOP3 | 51 | 36 | 15 | 29.4118% | 1.8235 | 2.1667 |
## 6. Board-Specific Funnel Outcomes
| Board | Funnel | Rows | Target7 | LOSS | Severe | Mean capped | Median | P25 | Worst |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BOARD2 | UNIVERSE | 130 | 33.8462% | 25.3846% | 5.3846% | 2.8877% | 4.2725% | -0.0904% | -24.4558% |
| BOARD2 | TOP10 | 108 | 31.4815% | 25.9259% | 5.5556% | 2.7062% | 3.8345% | -0.1231% | -24.4558% |
| BOARD2 | TOP5 | 62 | 30.6452% | 27.4194% | 9.6774% | 2.2066% | 4.2029% | -1.1224% | -24.4558% |
| BOARD2 | TOP3 | 36 | 33.3333% | 22.2222% | 5.5556% | 2.6824% | 5.4192% | 0.5882% | -24.4558% |
- BOARD2 Top3-Universe: Target7 **-0.5128%**; loss-lift (Universe-Top3) **3.1624%**; return lift **-0.2053%**.
| BOARD3 | UNIVERSE | 25 | 36.0000% | 32.0000% | 0.0000% | 2.7926% | 3.2295% | -1.2531% | -4.4457% |
| BOARD3 | TOP10 | 24 | 37.5000% | 33.3333% | 0.0000% | 2.8889% | 3.8112% | -1.3031% | -4.4457% |
| BOARD3 | TOP5 | 19 | 36.8421% | 42.1053% | 0.0000% | 2.3262% | 2.1419% | -1.7472% | -4.4457% |
| BOARD3 | TOP3 | 15 | 26.6667% | 53.3333% | 0.0000% | 1.4147% | -0.7092% | -2.0845% | -4.4457% |
- BOARD3 Top3-Universe: Target7 **-9.3333%**; loss-lift (Universe-Top3) **-21.3333%**; return lift **-1.3779%**.
## 7. Stage1 Ranking Quality by Board
| Board | Target7 AUC | NONLOSS AUC | Recall@3 | @5 | @10 | LOSS@3 | @5 | @10 | Winner median pct rank | Loss median pct rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BOARD2 | 0.4033 | 0.4730 | 27.2727% | 43.1818% | 77.2727% | 24.2424% | 51.5152% | 84.8485% | 57.1429% | 64.2857% |
| BOARD3 | 0.5417 | 0.2647 | 44.4444% | 77.7778% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 21.4286% | 7.1429% |
## 8. Rank1 / Rank2 / Rank3 Breakdown
| Rank | Board | Rows | Target7 | LOSS | Mean capped |
|---:|---|---:|---:|---:|---:|
| 1 | BOARD2 | 11 | 18.1818% | 18.1818% | 3.5325% |
| 1 | BOARD3 | 6 | 16.6667% | 50.0000% | 1.9752% |
| 2 | BOARD2 | 10 | 40.0000% | 10.0000% | 4.7298% |
| 2 | BOARD3 | 7 | 42.8571% | 42.8571% | 2.1812% |
| 3 | BOARD2 | 15 | 40.0000% | 33.3333% | 0.6941% |
| 3 | BOARD3 | 2 | 0.0000% | 100.0000% | -2.9494% |
- Overall Rank3 weakness mainly Board3-related: **INCONCLUSIVE**.
- Board3 Top3 selection-conditioned dates/rows are listed below; this is descriptive, not a stock-specific rule.
| Date | Code | Rank | Score | Target7 | LOSS | Raw | Capped |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-06-03 | 600280 | 1 | 0.661570 | 0 | 1 | -1.2531% | -1.2531% |
| 2026-06-04 | 600121 | 1 | 0.652929 | 0 | 1 | -0.7092% | -0.7092% |
| 2026-06-05 | 600255 | 2 | 0.643192 | 1 | 0 | 8.9655% | 7.0000% |
| 2026-06-09 | 603500 | 2 | 0.611438 | 1 | 0 | 9.3627% | 7.0000% |
| 2026-06-11 | 002636 | 1 | 0.600536 | 1 | 0 | 8.9010% | 7.0000% |
| 2026-06-12 | 000048 | 2 | 0.570389 | 0 | 1 | -2.1505% | -2.1505% |
| 2026-06-16 | 601958 | 1 | 0.592320 | 0 | 0 | 4.3929% | 4.3929% |
| 2026-06-16 | 000510 | 3 | 0.572216 | 0 | 1 | -1.4531% | -1.4531% |
| 2026-06-17 | 605198 | 1 | 0.595348 | 0 | 1 | -2.1277% | -2.1277% |
| 2026-06-18 | 603186 | 1 | 0.585704 | 0 | 0 | 4.5483% | 4.5483% |
| 2026-06-18 | 002741 | 2 | 0.585158 | 0 | 1 | -2.0413% | -2.0413% |
| 2026-06-18 | 600110 | 3 | 0.580453 | 0 | 1 | -4.4457% | -4.4457% |
| 2026-06-22 | 002989 | 2 | 0.515856 | 0 | 0 | 2.1419% | 2.1419% |
| 2026-06-24 | 600851 | 2 | 0.586702 | 0 | 1 | -3.6816% | -3.6816% |
| 2026-06-25 | 605069 | 2 | 0.585334 | 1 | 0 | 10.8333% | 7.0000% |
## 9. LOW / MID / HIGH Opportunity Buckets
| Bucket | Board | Candidate / Top3 | Target7 U→T3 | LOSS U→T3 | Capped U→T3 | Warning |
|---|---|---:|---:|---:|---:|---|
| LOW | BOARD2 | 41 / 12 | 12.1951% → 8.3333% | 39.0244% → 41.6667% | 0.6156% → -0.5084% |  |
| LOW | BOARD3 | 11 / 6 | 18.1818% → 16.6667% | 36.3636% → 66.6667% | 1.5657% → 0.5165% |  |
| MID | BOARD2 | 39 / 9 | 25.6410% → 33.3333% | 35.8974% → 33.3333% | 2.2299% → 2.5731% |  |
| MID | BOARD3 | 8 / 6 | 50.0000% → 50.0000% | 25.0000% → 33.3333% | 4.2162% → 3.3764% |  |
| HIGH | BOARD2 | 50 / 15 | 58.0000% → 53.3333% | 6.0000% → 0.0000% | 5.2640% → 5.3007% |  |
| HIGH | BOARD3 | 6 / 3 | 50.0000% → 0.0000% | 33.3333% → 66.6667% | 3.1440% → -0.7121% | SMALL_SAMPLE |
## 10. Bootstrap / Permutation / LODO
- Strict Top3 LOSS gap bootstrap p2.5/p50/p97.5; P(B3>B2); valid: **2.6786% / 31.3131% / 57.7273%; 98.3550%; 20000**.
- Strict Top3 Target7 gap bootstrap p2.5/p50/p97.5; P(B3<B2): **-34.8485% / -6.5637% / 27.0455%; 65.6200%**.
- Strict Top3 capped gap bootstrap p2.5/p50/p97.5; P(B3<B2): **-4.0164% / -1.3019% / 1.8625%; 79.4050%**.
- LODO LOSS gap worse direction/min/median/max: **100.0000%; 26.4706% / 29.4118% / 36.5546%**.
- LODO Target7 gap worse direction/min/median/max: **94.1176%; -13.8655% / -6.6667% / 0.0000%**.
- LODO capped gap worse direction/min/median/max: **100.0000%; -2.0541% / -1.1528% / -0.7525%**.
## 11. Structural Attribution
- Inherent gates: A1=FAIL, A2=FAIL, A3=FAIL, A4=FAIL.
- Overpromotion gates: B1=PASS, B2=PASS, B3=PASS, B4=PASS, B5=PASS.
- `INHERENT_BOARD3_RISK_SUPPORT = NO`
- `STAGE1_BOARD3_OVERPROMOTION_SUPPORT = YES`
- Q1 Board3 materially riskier before Stage1: **NO**.
- Q2 Stage1 preferentially pushes Board3 into Top3: **YES**.
- Q3 Board3 quality deteriorates as funnel narrows: **YES**.
- Q4 Board2 shows equivalent deterioration: **NO**.
- Q5 Target7 ranking materially weaker inside Board3: **NO**.
- Q6 Loss contamination materially worse inside Board3: **YES**.
- Q7 Rank3 weakness mainly Board3-related: **INCONCLUSIVE**.
- Q8 Same-date evidence supports structural difference: **PARTIAL** (Population A does not; strict selected funnel does).
## 12. Board-Separation Decision
- `BOARD_STRUCTURE_MODE = STAGE1_BOARD3_OVERPROMOTION`
- `BOARD2_BOARD3_SEPARATION_EXPERIMENT_WARRANTED = YES`
- No Board3 exclusion, penalty, board-specific threshold, TopK, or model was tested.
- Recommended action follows the formal state only; this audit establishes structure, not a tradable board rule.
