# v004c Frozen Current Stage1 Top2 July Chronological Forward Stress v001

## 1. Frozen Contract

- Evaluation: CHRONOLOGICAL_FORWARD_STRESS / SEMI-OOT; pristine OOT: NO.
- Current V4C_STAGE1_V4A_ARCH; 18 features; weighted L2 Logistic; L2=0.30; positive weight=1.50; TARGET7.
- Top2 was predeclared; Top3 is a fixed diversification comparator. No model, feature, TopK, threshold, Stage2, or Board3-policy search.

## 2. Prediction Provenance / Leakage Audit

- Reused artifact: `reports/research/v004c_pair_capped7_july_forward_v001_20260701_20260731/v004c_july_forward_prediction_lock_v001.csv`.
- Source/purpose lock SHA256: `725ffa3e386034233bb6f2cca1fdeff13a31a286b5c3621043c85c5fd5333a08` / `2db6c036995d2db758482d20aa390de4e5bdc874850fa2f251dd5ee60b4a795d`.
- Equivalent frozen identity: development commit `c6af1289f66f8c2cad610b9564573607ce57ac14` + source lock SHA256; feature manifest is the imported exact 18-column `FROZEN_FEATURE_COLUMNS`.
- Training rows/dates: 307/37; latest signal/label: 2026-06-26/2026-06-30.
- July training/refits: 0/0; fit count in this audit: 0; outcome perturbation: PASS.

## 3. July Population

- Dates/rows: 23/178; Board2/Board3: 155/23.
- Target7/PNT/LOSS: 46/75/57.
- Universe Target7/LOSS/raw/capped: 25.7992%/33.1426%/3.7339%/1.9920%.

## 4. Rank-wise Forward Value

| Rank | Rows | Target7 | LOSS | Severe | Raw | Capped | Value |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 23 | 26.0870% | 30.4348% | 13.0435% | 3.8596% | 1.7203% | PARTIAL |
| 2 | 23 | 26.0870% | 39.1304% | 13.0435% | 2.8447% | 1.4595% | PARTIAL |
| 3 | 21 | 28.5714% | 38.0952% | 0.0000% | 3.1492% | 1.8782% | PARTIAL |

## 5. Top1 / Top2 / Top3

| Portfolio | Dates | Precision | Capture | LOSS | Raw | Capped | Excess | Negative | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top1 | 23 | 26.0870% | 13.0435% | 30.4348% | 3.8596% | 1.7203% | -0.2717% | 30.4348% | -10.0097% |
| Top2 | 23 | 26.0870% | 26.0870% | 34.7826% | 3.3521% | 1.5899% | -0.4021% | 39.1304% | -8.7537% |
| Top3 | 21 | 28.5714% | 39.1304% | 33.3333% | 3.6757% | 1.9653% | -0.4840% | 28.5714% | -6.3540% |

## 6. Rank3 Marginal / Diversification

- Common Top2/Top3 dates: 21; paired capped: 2.0088% vs 1.9653%.
- Rank3 marginal mean/median: -0.0435%/0.0000%; positive/negative/zero: 10/9/2.
- Bootstrap P(Top2>Top3): 54.5250%.

## 7. Bootstrap / LODO

- Top2-Top3 capped interval: [-0.5804%, 0.6957%].
- Top2-Universe capped estimate/P>0: -0.4021%/24.7650%.

## 8. Temporal Replication

- May-Jun and July are reported separately and never pooled.
- May-Jun Top2-Top3: 1.0220%; July paired: 0.0435%.

## 9. Forward Decision

| Gate | Pass |
|---|:---:|
| A | YES |
| B | NO |
| C | NO |
| D | YES |
| E | YES |
| F | YES |
| G | YES |
| H | YES |
| I | NO |
| J | YES |

- `TOP2_CAPACITY_FORWARD_SIGNAL = PARTIAL`
- `PRIMARY_FORWARD_FAILURE = STAGE1_SELECTION_ALPHA_FAILURE`
- `V4C_CORE_ARCHITECTURE_STATE = REOPEN_STAGE1_QUESTION`
- AUTO_OPTIMIZE_AFTER_FAILURE = NO.
