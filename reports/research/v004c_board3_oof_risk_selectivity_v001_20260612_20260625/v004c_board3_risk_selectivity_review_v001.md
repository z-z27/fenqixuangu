# v004c Board3-Only OOF Risk Selectivity Diagnostic v001

## 1. Experimental Contract

- Post-hoc OOF risk-selectivity diagnostic: **YES**.
- Existing locked score only; no fit, new model, feature, score, threshold search, veto, backfill, Board2 modification, cross-board combination, or July result access.

## 2. Existing OOF Prediction Lock

- Expected / observed SHA256: `2f85e79738dbbedf19382db0da5f1136de02e84c69557ea5c85341bcf62fbd6b` / `2f85e79738dbbedf19382db0da5f1136de02e84c69557ea5c85341bcf62fbd6b`.
- Prediction-lock parity and outcome perturbation: **PASS**.

## 3. Population / Support

- Population A: **8 dates / 19 rows**; Target7/PNT/LOSS: **5/8/6**.
- Population B: **5 dates / 16 rows**; Target7/PNT/LOSS: **4/8/4**.

## 4. Known Risk-Ordering Controls

- Unified / Board3-only NONLOSS AUC: **0.2308 / 0.6923**.
- Same-date NONLOSS-vs-LOSS: **14 pairs**; unified / Board3-only concordance **0.2143 / 0.9286**.

- Locked Board3-only score by outcome (mean / median / P25 / P75):
  - LOSS: **0.6266 / 0.6267 / 0.6192 / 0.6336**.
  - PNT: **0.6597 / 0.6616 / 0.6466 / 0.6795**.
  - Target7: **0.6437 / 0.6473 / 0.6185 / 0.6574**.

## 5. LOSS / PNT / TARGET7 Risk-Rank Geometry

| Outcome | Mean risk pct | Median | P25 | P75 |
|---|---:|---:|---:|---:|
| LOSS | 71.4286% | 92.8571% | 64.2857% | 100.0000% |
| PNT | 35.7143% | 28.5714% | 0.0000% | 60.7143% |
| TARGET7 | 57.1429% | 64.2857% | 21.4286% | 100.0000% |

- `RISK_MEDIAN_ORDER = LOSS_RISKIEST`.
- Mixed-date favorable/unfavorable/tie: **2/1/0**.

## 6. Bottom-Half Tail Selectivity

- Flagged **8**: LOSS/PNT/Target7 **3/3/2**.
- Unflagged **8**: LOSS/PNT/Target7 **1/5/2**.
- LOSS enrichment **12.5000%**; loss capture **75.0000%**; winner removal **50.0000%**; selectivity gap **25.0000%**.

## 7. Worst-One Corroboration

- Flagged **5**: LOSS/PNT/Target7 **2/1/2**.
- Loss capture **50.0000%**; winner removal **50.0000%**; selectivity gap **0.0000%**.

## 8. Cumulative Loss-Capture vs Winner-Removal Curve

- Steps **16**; mean selectivity gap **15.6250%**; dominance **62.5000%**.
- Descriptive pattern: **MIXED**. No cutoff was selected.

## 9. Same-Date Pair Selectivity

- NONLOSS-vs-LOSS: **3 dates / 14 pairs / 0.9286**.
- PNT-vs-LOSS: **3 dates / 12 pairs / 0.9167**.
- Target7-vs-LOSS: **1 dates / 2 pairs / 1.0000**.

- `WINNER_LOSS_PAIR_SUPPORT_VERY_WEAK = YES`.

## 10. Unified Stage1 Top3 Decision Surface

- Rows/dates: **10/7**; Target7/PNT/LOSS **1/3/6**.
- NONLOSS AUC **0.6667**; Target7-vs-LOSS AUC **0.3333**; support **ADEQUATE**; corroboration **YES**.

## 11. Bootstrap / LODO

- Bootstrap P(bottom-half gap > 0): **58.5099%**; 95% interval **[-66.6667%, 70.0000%]**.
- LODO bottom-half positive gap: **87.5000%**.

## 12. Risk-Selectivity Decision

| Gate | Pass |
|---|:---:|
| A | YES |
| B | YES |
| C | NO |
| D | YES |
| E | NO |
| F | YES |
| G | NO |
| H | NO |
| I | NO |
| J | YES |
| K | YES |

- Decision-surface gate: **YES**.
- `BOARD3_RISK_SELECTIVITY_SIGNAL = PARTIAL`

## 13. Next Architecture Decision

- `NEXT_BOARD3_RISK_ACTION = NO_BOARD3_RISK_POLICY_YET`
- No threshold was selected, no veto/backfill or combined Top3 was simulated, and Board2 was not modified.
- Recommended action: preserve this diagnostic; do not design a Board3 risk policy or select a threshold yet.
