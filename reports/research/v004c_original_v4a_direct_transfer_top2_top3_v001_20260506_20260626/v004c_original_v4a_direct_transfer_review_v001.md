# v004c Original-v4a Direct-Transfer & Top2-vs-Top3 Benchmark v001

## 1. Experimental Contract

- Original-v4a must score its complete original universe before V4C intersection.
- Archived scores have priority; no replacement refit is allowed without archived parity.
- Strict point-in-time chronology requires every training label to be available before the test date.
- No model, feature, hyperparameter, Stage2, or Board3 repair was used.

## 2. Provenance Result

- ORIGINAL_V4A_PROVENANCE: **BLOCKED**
- Archived score rows/dates: **2355 / 18**
- Frozen configuration: **logistic_v004a_weighted; L2=0.3; positive_weight=1.5; features=18; initial_train_days=18**
- Full-universe score-lock SHA256: **8828fa23574a9656d022113cd180dd928c1739e131ecfd607e9d336b392f4e47**

## 3. Strict Temporal Audit

- Audited folds: **17**
- Folds with current/future D3 labels in training: **17**
- Current/future-label rows per fold: **84..380**
- Maximum training label-available date: **2026-06-30**
- The archived walk-forward split is chronological by signal date, not by label availability.
- A strict reconstruction that removes those rows would not be score/rank-parity with archived original v4a and would therefore be a different model instance.

## 4. V4C Coverage (QA Only; No Outcomes Attached)

- Strict V4C rows/dates: **155 / 17**
- Archived original-v4a covered rows/dates: **132 / 16**
- Coverage: **85.1613%**
- Missing rows: **23**
- Zero-coverage dates: **2026-06-22**

## 5. Stop Decision

- Direct-transfer practical benchmark: **NOT RUN**
- Top2-vs-Top3 outcome benchmark: **NOT RUN**
- Bootstrap / LODO / membership attribution: **NOT RUN**
- ORIGINAL_V4A_DIRECT_TRANSFER_SIGNAL: **INVALID**
- FINAL_TRADING_CAPACITY_SIGNAL: **INVALID**
- NEXT_V4C_ACTION: **INVALID**
- JULY_RESULT_ROWS_ACCESSED: **>=1 (exact count not queried; provenance probe breach)**

The benchmark is blocked because no archived original-v4a score artifact satisfies the required strict point-in-time label-maturity contract. Re-fitting canonical features with a new maturity filter would be a new reconstructed instance and cannot be presented as archived original-v4a direct transfer without a separate, explicitly authorized experiment.

JULY_RESULT_ROWS_ACCESSED >= 1; JULY_USED_FOR_MODEL_OR_OUTCOME_CONCLUSION = NO
