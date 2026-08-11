# v004c Stage1 × Risk-Protection Architecture Diagnostic v001

## 1. Experimental Contract

- Scope: June-only architecture diagnostic; no new model, feature, target, threshold, or tuning.
- July result rows accessed: 0. JULY_USED_FOR_ARCHITECTURE_DESIGN: NO.
- Oracle policies are non-tradable counterfactual upper bounds.

## 2. Naming and Frozen Specifications

- Current Stage1: V4C_STAGE1 (architecture source V4A_ARCH_TRANSFER_V4C).
- Historical original v004a is not the same fitted model instance.
- Stage2: PAIR_CAPPED7; same-date weighted Ridge, L2 0.30, zero intercept, three frozen predictors.

## 3. Strict June Temporal Reconstruction

- Raw June dates: 21; common strict dates: 17.
- Unavailable dates: 2026-06-01|2026-06-02|2026-06-29|2026-06-30.
- Training rule: label_available_date < test_date; minimum 18 matured dates.
- SELF_LABEL_LEAKAGE_ROWS: 0; CURRENT_TEST_LEAKAGE_ROWS: 0.
- Legacy June values retain temporal caveat: 337 immature-label rows across 21 folds.

## 4. STRICT_STAGE1 vs STRICT_CAPPED

| Policy | Rank1 | Rank2 | Rank3 | Top2 | Top3 | Top3 precision | Negative dates | Worst |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNIVERSE | 2.6786% | 2.6786% | 2.6786% | 2.6786% | 2.6786% | 31.9160% | NA | NA |
| STRICT_STAGE1 | 2.9829% | 3.6804% | 0.2655% | 3.3316% | 2.3096% | 31.3725% | 23.5294% | -6.7688% |
| STRICT_CAPPED | 2.2733% | 3.4467% | -0.0977% | 2.8600% | 1.8741% | 29.4118% | 17.6471% | -6.8535% |

## 5. Swap Attribution

- Changed dates / slots: 12 / 19.
- Demotion loss precision: 26.3158%; Target7 error rate: 47.3684%.
- Contribution from demoting Target7: -0.8438%.
- Contribution from demoting positive non-target: -0.0367%.
- Contribution from demoting loss: 0.4450%.
- Attributed total / observed delta: -0.4355% / -0.4355%; closure PASS.

## 6. Winner-Protection Oracle

- Stage1 winners protected from CAPPED demotion: 9.
- Winner-protected Top3: 3.1943%; delta vs Stage1 0.8847%; delta vs CAPPED 1.3202%.
- Precision 45.0980%; winner capture 64.4444%; negative dates 11.7647%; worst -6.7688%; days <= -5% 1.

## 7. Loss-Only Intervention Oracle

- Original demotions: 19; approved actual-loss demotions: 5; blocked non-loss: 14.
- Loss-permission Top3: 3.1324%; delta vs Stage1 0.8229%; precision 37.2549%; winner capture 55.5556%.
- Negative dates 17.6471%; worst -6.7688%; days <= -5% 1.

## 8. Perfect Loss Detection + Stage1 Backfill

- Stage1 Top3 actual losses: 16; actually removed with available backfill: 14.
- Simple backfill Top3: 3.8258%; delta 1.5162%; precision 39.2157%; winner capture 58.8889%.
- Negative dates 0.0000%; worst 0.2546%.
- Simple backfills: 14; Target7 4; positive non-target 4; loss 6.

## 9. Architecture Upper Bounds

| Policy | Top3 | Precision | Winner capture | Negative dates | Worst |
| --- | --- | --- | --- | --- | --- |
| STAGE1_WITH_ORACLE_LOSS_VETO_STAGE1_BACKFILL | 3.8258% | 39.2157% | 58.8889% | 0.0000% | 0.2546% |
| STAGE1_WITH_ORACLE_RISK_AND_ORACLE_BACKFILL | 5.4440% | 50.9804% | 76.6667% | 0.0000% | 1.3323% |
| CAPPED_WITH_ORACLE_WINNER_PROTECTION | 3.1943% | 45.0980% | 64.4444% | 11.7647% | -6.7688% |
| TOP10_ORACLE | 6.1364% | 64.7059% | 92.2222% | 0.0000% | 1.3323% |

Raw downside audit:

| Policy | Raw mean | Raw median | Worst stock | Worst Top3 date |
| --- | --- | --- | --- | --- |
| STRICT_STAGE1 | 3.3236% | 4.5483% | -24.4558% | -6.7688% |
| STRICT_CAPPED | 2.8129% | 3.2295% | -24.4558% | -6.8535% |
| CAPPED_WITH_ORACLE_WINNER_PROTECTION | 4.5282% | 6.5937% | -24.4558% | -6.7688% |
| CAPPED_WITH_ORACLE_LOSS_PERMISSION | 4.3850% | 5.4025% | -24.4558% | -6.7688% |
| STAGE1_WITH_ORACLE_LOSS_VETO_STAGE1_BACKFILL | 5.1268% | 5.6447% | -12.2694% | 0.2546% |

## 10. LOW / MID / HIGH

| Policy | LOW Top3 | MID Top3 | HIGH Top3 | HIGH precision |
| --- | --- | --- | --- | --- |
| UNIVERSE | 0.6195% | 2.6364% | 4.7728% | 53.0345% |
| STRICT_STAGE1 | -0.1668% | 2.8944% | 4.2986% | 44.4444% |
| STRICT_CAPPED | -1.1737% | 2.3103% | 4.5584% | 44.4444% |
| CAPPED_WITH_ORACLE_WINNER_PROTECTION | -0.0314% | 4.6629% | 5.1962% | 61.1111% |
| CAPPED_WITH_ORACLE_LOSS_PERMISSION | 0.6044% | 4.7667% | 4.2986% | 44.4444% |
| STAGE1_WITH_ORACLE_LOSS_VETO_STAGE1_BACKFILL | 2.6225% | 3.4839% | 5.3140% | 55.5556% |
| STAGE1_WITH_ORACLE_RISK_AND_ORACLE_BACKFILL | 5.0312% | 6.0955% | 5.3140% | 55.5556% |

## 11. Architecture Decision

- ARCHITECTURE_COMPLEMENTARITY_SIGNAL: **STRONG**.
- RISK_PROTECTOR_TRAINING_WARRANTED: **YES**.
- Preferred theoretical direction: **STAGE1_OWNS_SELECTION_PLUS_RISK_PROTECTION**.
- Replacement selection major bottleneck: **NO**.
- Formal gates: A_simple_top3_plus_50bp=PASS; B_simple_precision_preserved=PASS; C_simple_risk_improved=PASS; D_oracle_headroom_100bp=PASS; E_winner_protection_repairs_capped=PASS.
- This diagnostic establishes architecture headroom only; it does not prove a learnable or production-ready risk model.

## 12. v5 Structural-Repetition Risk

- V5_REPETITION_RISK_FOR_FULL_RERANKER: **HIGH**.
- Another full Top10 reranker is not justified by this task.
