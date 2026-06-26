# 2026-06-25 Data Acceptance Report

## Summary

- total codes: **300**
- accepted: **293**
- failed: **7**
- MA recalc check passed: **300**
- daily/5m close check passed: **295**
- dataaccept daily comparisons: **1**
- dataaccept 5m comparisons: **2**

## Failed Codes

| code | failures | warnings |
|---|---|---|
| 001206 | daily/5m close mismatch max_diff=0.03999999999999915 | dataaccept daily reference missing for 001206; dataaccept 5m reference missing for 001206 |
| 001296 | daily/5m close mismatch max_diff=0.05000000000000071 | dataaccept daily reference missing for 001296; dataaccept 5m reference missing for 001296 |
| 002258 | daily/5m close mismatch max_diff=0.02999999999999936 | dataaccept daily reference missing for 002258; dataaccept 5m reference missing for 002258 |
| 002976 | daily/5m close mismatch max_diff=0.21000000000000085 | dataaccept daily reference missing for 002976; dataaccept 5m reference missing for 002976 |
| 003043 | daily/5m close mismatch max_diff=0.030000000000001137 | dataaccept daily reference missing for 003043; dataaccept 5m reference missing for 003043 |
| 603407 | daily trade days 33 < required 180; 5m trade days 33 < required 40 | dataaccept daily reference missing for 603407; dataaccept 5m reference missing for 603407 |
| 603459 | daily trade days 53 < required 180 | dataaccept daily reference missing for 603459; dataaccept 5m reference missing for 603459 |

## Dataaccept Cross Checks

| code | daily dates | daily max close diff | 5m bars | 5m max close diff | volume ratio |
|---|---:|---:|---:|---:|---:|
| 000636 | 264 | 0 | 1778 | 0 | 100 |
| 605589 |  |  | 384 | 0 |  |

## Rules

- daily MA5/MA10/MA20/MA30 are recalculated from full daily history and compared with project indicators.
- at least 120 trading days of warmup are required; current default requires 180 daily rows.
- 5m data must cover the requested trading-day window.
- daily close must match the last 5m close for overlapping dates within 0.02.
- overlapping `F:\dataaccept` caches are compared when readable; missing references are warnings, not failures.