# 2026-06-25 Failed Data Review

## Summary

- failed codes reviewed: **7**
- category `daily_5m_close_mismatch`: **5**
- category `insufficient_daily_history`: **2**
- repair `still_failed`: **7**

## Details

| code | name | category | action | max close diff | mismatch date | repair |
|---|---|---|---|---:|---|---|
| 001206 | 依依股份 | daily_5m_close_mismatch | force_refetch_and_compare_sources_before_release | 0.04 | 2026-04-29 | still_failed |
| 001296 | 长江材料 | daily_5m_close_mismatch | force_refetch_and_compare_sources_before_release | 0.05 | 2026-05-27 | still_failed |
| 002258 | 利尔化学 | daily_5m_close_mismatch | force_refetch_and_compare_sources_before_release | 0.03 | 2026-04-27 | still_failed |
| 002976 | 瑞玛精密 | daily_5m_close_mismatch | force_refetch_and_compare_sources_before_release | 0.21 | 2026-06-04 | still_failed |
| 003043 | 华亚智能 | daily_5m_close_mismatch | force_refetch_and_compare_sources_before_release | 0.03 | 2026-04-30 | still_failed |
| 603407 | 长裕集团 | insufficient_daily_history | exclude_until_180_daily_rows_or_validated_alternate_source | 0 | 2026-05-11 | still_failed |
| 603459 | 红板科技 | insufficient_daily_history | exclude_until_180_daily_rows_or_validated_alternate_source | 0 | 2026-04-27 | still_failed |

## Policy

- Failed rows remain excluded from signal generation unless a forced refetch returns a full `ok` quality result.
- Insufficient daily history is not manually padded; wait until enough listed trading days exist or use a validated alternate source.
- Daily/5m close mismatches are not loosened here; the source pair must pass the same threshold before the code is tradable.