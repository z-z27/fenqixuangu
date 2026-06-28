# 2026-06-15 数据质量报告

## 概览

- 总标的: **239**
- 数据成功: **233**
- 数据失败: **6**
- 最新日线 MA 覆盖正常: **238**
- 日线/分钟收盘价校验通过: **235**

## 数据源

| 类型 | 来源 | 数量 |
|---|---|---:|
| 日线 | cache | 236 |
| 日线 | tencent_daily | 2 |
| 日线 | unknown | 1 |
| 5分钟 | sina_5m | 237 |
| 5分钟 | cache | 1 |
| 5分钟 | unknown | 1 |

## 失败标的

| 代码 | 名称 | D0日期 | 错误 |
|---|---|---|---|
| 001206 | 依依股份 | 2026-06-12 | 001206 data quality check failed: daily/minute close cross-check failed: matched=34, max_diff=0.03999999999999915 |
| 001233 | 海安集团 | 2026-06-12 | 001233 data quality check failed: daily history rows 133 < required 180 |
| 001257 | 盛龙股份 | 2026-06-15 | 001257 data quality check failed: daily history rows 51 < required 180 |
| 002961 | 瑞达期货 | 2026-06-15 | 002961 data quality check failed: daily/minute close cross-check failed: matched=34, max_diff=0.030000000000001137 |
| 002976 | 瑞玛精密 | 2026-06-12 | 002976 data quality check failed: daily/minute close cross-check failed: matched=34, max_diff=0.21000000000000085 |
| 603407 | 长裕集团 | 2026-06-15 | 603407 missing required daily moving averages on latest row: ma30 |

## 质量提示

| 代码 | 名称 | 日线源 | 5分钟源 | 提示 |
|---|---|---|---|---|
| 001206 | 依依股份 | cache | sina_5m | daily/minute close cross-check failed: matched=34, max_diff=0.03999999999999915 |
| 001233 | 海安集团 | tencent_daily | sina_5m | daily history rows 133 < required 180 |
| 001257 | 盛龙股份 | tencent_daily | sina_5m | daily history rows 51 < required 180 |
| 002961 | 瑞达期货 | cache | sina_5m | daily/minute close cross-check failed: matched=34, max_diff=0.030000000000001137 |
| 002976 | 瑞玛精密 | cache | sina_5m | daily/minute close cross-check failed: matched=34, max_diff=0.21000000000000085 |

## 字段说明

| 字段 | 含义 |
|---|---|
| daily_history_rows | 参与日线 MA 计算的日线历史交易日数量 |
| daily_required_days | 当前策略要求的最少日线历史交易日数量 |
| daily_ma_coverage_ok | 最新行 MA5/MA10/MA20/MA30 是否都存在 |
| daily_minute_close_check_ok | 日线收盘与当日最后一根 5 分钟收盘是否基本一致 |
| missing_*_count | 对应字段缺失数量 |