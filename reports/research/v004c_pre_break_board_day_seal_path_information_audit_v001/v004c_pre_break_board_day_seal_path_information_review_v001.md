# v004c Pre-break Board-day Seal-path Information Audit v001

## 简单版

### Q1. 现有历史数据能否可靠看到以前两板怎么封？

不能覆盖完整开发期。真实 historical pool 的封板时间/开板次数主要出现在部分 June 和 July；May 使用 daily-derived fallback，这些字段按规则全部缺失。

### Q2. Target7 与 LOSS 在封板路径上是否有区别？

未测试。Phase A 只有 0/5 字段通过门槛，family gate 失败，因此没有读取 outcome。

### Q3. May / June / July 是否稳定？

未测试；不能用缺失结构替代时间稳定性结论。

### Q4. S2 排错时 seal path 能否救回？

未测试；S2 score/rank 在 Phase A 未读取。

### Q5. 这是新信息还是另一个不稳定关系？

目前只能确认它是语义上新的 family，不能确认其信息效果。

### Q6. 这个 family 值不值得继续？

现有数据不足以直接继续信息判断；先停止并评审数据可用性，不能补零或临时用5分钟重建。

## Phase A — Blind Coverage / Lineage

Frozen population: 485 rows / 60 dates, 2026-05-06 through 2026-07-29.

Phase A used only event identity/date/board columns and historical limit-up pool fields. It did not load target, return, S2 score, or S2 rank.

- F1_D0_FINAL_SEAL_MINUTE: overall 35.46%; May 0.00%; June 22.54%; July 80.12%; ready=NO.
- F2_D0_OPEN_BOARD_COUNT: overall 35.46%; May 0.00%; June 22.54%; July 80.12%; ready=NO.
- F3_D0_RESEAL_DELAY_MINUTE: overall 35.46%; May 0.00%; June 22.54%; July 80.12%; ready=NO.
- F4_FINAL_SEAL_DETERIORATION: overall 25.36%; May 0.00%; June 4.62%; July 69.28%; ready=NO.
- F5_OPEN_BOARD_DETERIORATION: overall 25.36%; May 0.00%; June 4.62%; July 69.28%; ready=NO.

Fields ready: 0/5; eligible=NONE.

PHASE_A_LOCK_SHA256 = 2eaed9c472b85a8d6ef7213ff4276c5e0118fb966a0eb52c3b1c93f5b5dd6c78

`daily_limitup_derived` rows retain MISSING time/open-board semantics; no zero-fill, 15:00 substitution, OHLC inference, 5-minute repair, or cross-date fill was used.

`seal_amount` and `amount` are coverage notes only and never enter the five-field audit.

## Phase B — Fixed Information Audit

NOT RUN: Phase A family coverage gate failed before outcome access.

## Final State

PREBREAK_SEAL_PATH_DATA_STATE = NOT_READY

PREBREAK_SEAL_PATH_INFORMATION_STATE = NOT_TESTED_DUE_TO_DATA

BROADER_D1_INFORMATION_LIMITATION = [待核验]

STAGE1_D1_INFORMATION_RESEARCH_STATE = STOP_AND_REVIEW

MODEL_TRAINED = NO

FEATURE_SEARCH = NO

WINDOW_SEARCH = NO

AUGUST_HOLDOUT_STATUS = CONSUMED

AUGUST_USED_AS_FRESH_OOT = NO

NEXT_ACTION = STOP_AND_REVIEW
