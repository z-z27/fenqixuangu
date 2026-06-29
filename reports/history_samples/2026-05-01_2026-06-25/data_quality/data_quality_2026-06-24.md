# 2026-06-24 数据质量报告

## 概览

- 总标的: **250**
- 数据成功: **244**
- 数据失败: **6**
- 最新日线 MA 覆盖正常: **250**
- 日线/分钟收盘价校验通过: **245**

## 数据源

| 类型 | 来源 | 数量 |
|---|---|---:|
| 日线 | cache | 249 |
| 日线 | tencent_daily | 1 |
| 5分钟 | sina_5m | 249 |
| 5分钟 | cache | 1 |

## 失败标的

| 代码 | 名称 | D0日期 | 错误 |
|---|---|---|---|
| 001206 | 依依股份 | 2026-06-23 | 001206 data quality check failed: daily/minute close cross-check failed: matched=39, max_diff=0.03999999999999915 |
| 001296 | 长江材料 | 2026-06-22 | 001296 data quality check failed: daily/minute close cross-check failed: matched=39, max_diff=0.05000000000000071 |
| 002258 | 利尔化学 | 2026-06-22 | 002258 data quality check failed: daily/minute close cross-check failed: matched=39, max_diff=0.02999999999999936 |
| 002976 | 瑞玛精密 | 2026-06-23 | 002976 data quality check failed: daily/minute close cross-check failed: matched=39, max_diff=0.21000000000000085 |
| 003043 | 华亚智能 | 2026-06-24 | 003043 data quality check failed: daily/minute close cross-check failed: matched=39, max_diff=0.030000000000001137 |
| 603407 | 长裕集团 | 2026-06-24 | 603407 data quality check failed: daily history rows 32 < required 180 |

## 质量提示

| 代码 | 名称 | 日线源 | 5分钟源 | 提示 |
|---|---|---|---|---|
| 001206 | 依依股份 | cache | sina_5m | daily/minute close cross-check failed: matched=39, max_diff=0.03999999999999915 |
| 001296 | 长江材料 | cache | sina_5m | daily/minute close cross-check failed: matched=39, max_diff=0.05000000000000071 |
| 002258 | 利尔化学 | cache | sina_5m | daily/minute close cross-check failed: matched=39, max_diff=0.02999999999999936 |
| 002976 | 瑞玛精密 | cache | sina_5m | daily/minute close cross-check failed: matched=39, max_diff=0.21000000000000085 |
| 003043 | 华亚智能 | cache | sina_5m | daily/minute close cross-check failed: matched=39, max_diff=0.030000000000001137 |
| 603407 | 长裕集团 | tencent_daily | sina_5m | daily history rows 32 < required 180 |

## 字段说明

| 字段 | 含义 |
|---|---|
| daily_history_rows | 参与日线 MA 计算的日线历史交易日数量 |
| daily_required_days | 当前策略要求的最少日线历史交易日数量 |
| daily_ma_coverage_ok | 最新行 MA5/MA10/MA20/MA30 是否都存在 |
| daily_minute_close_check_ok | 日线收盘与当日最后一根 5 分钟收盘是否基本一致 |
| missing_*_count | 对应字段缺失数量 |