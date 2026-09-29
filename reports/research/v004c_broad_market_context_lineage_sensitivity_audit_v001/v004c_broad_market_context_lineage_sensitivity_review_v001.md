# v004c Broad Market Context Lineage Sensitivity Audit v001

This is a blind lineage-materiality audit. It used only current universe metadata, local historical daily bars, and the prior blind U0 coverage output.

## Q1. 上一轮 U0 能否精确复现？

YES. Mismatch dates=0; max value error=2.22e-16; max count error=0.

## Q2. U0 / U1 / U2 每天到底差多少只股票？

Across the 62 primary May-July dates, U1 minus U0 has median 12, p90 12, max 12 codes. U0 minus U1 has median 0, max 0. U0_EQUALS_U2=YES across all audited full-quality dates.

## Q3. Membership difference 对 MARKET_UP_RATIO 影响多大？

Median abs difference 0.0594pp; p90 0.1317pp; Spearman 0.999950; state agreement 100.00%; LINEAGE_ROBUST.

## Q4. 对 MARKET_MEDIAN_RETURN 影响多大？

Median abs difference 0.0024pp; p90 0.0083pp; Spearman 0.999899; state agreement 100.00%; LINEAGE_ROBUST.

## Q5. 对 MARKET_TAIL_BALANCE 影响多大？

Median abs difference 0.0627pp; p90 0.1866pp; Spearman 0.999597; state agreement 100.00%; LINEAGE_ROBUST.

## Q6. 不同 universe definition 下，每天市场强弱排序是否基本一致？

YES. All three variables exceed the pre-registered date-order threshold and have 100% primary-window state agreement; no outcome information was used.

## Q7. May / June / July 的相对市场状态排序是否改变？

- market_up_ratio: U0=2026-07>2026-05>2026-06; U1=2026-07>2026-05>2026-06; U2=2026-07>2026-05>2026-06; stable=YES.
- market_median_return: U0=2026-07>2026-05>2026-06; U1=2026-07>2026-05>2026-06; U2=2026-07>2026-05>2026-06; stable=YES.
- market_tail_balance: U0=2026-05>2026-07>2026-06; U1=2026-05>2026-07>2026-06; U2=2026-05>2026-07>2026-06; stable=YES.

## Q8. 把所有 disputed stocks 全部排除后，结论是否仍稳定？

YES. The stress test uses the larger U0/U1 proxy as reference and removes the entire symmetric-difference set, leaving only the daily intersection; all three variables still pass the same robustness interpretation.

## Q9. historical universe lineage 到底是否 materially matters？

The lineage is formally imperfect, but the pre-registered aggregate M1/M2/M3 descriptions are not materially changed by the tested proxy definitions.

BROAD_MARKET_CONTEXT_LINEAGE_STATE = BROAD_CONTEXT_LINEAGE_ROBUST

LABEL_OR_OUTCOME_ACCESSED = NO

MODEL_TRAINED = NO

EXTERNAL_DATA_FETCHED = NO

NEXT_ACTION = BROAD_MARKET_CONTEXT_INFORMATION_AUDIT
