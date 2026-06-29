# 2026-06-10 数据质量报告

## 概览

- 总标的: **192**
- 数据成功: **188**
- 数据失败: **4**
- 最新日线 MA 覆盖正常: **192**
- 日线/分钟收盘价校验通过: **189**

## 数据源

| 类型 | 来源 | 数量 |
|---|---|---:|
| 日线 | cache | 191 |
| 日线 | tencent_daily | 1 |
| 5分钟 | sina_5m | 192 |

## 失败标的

| 代码 | 名称 | D0日期 | 错误 |
|---|---|---|---|
| 001369 | 双欣材料 | 2026-06-09 | 001369 data quality check failed: daily history rows 105 < required 180 |
| 002225 | 濮耐股份 | 2026-06-09 | 002225 data quality check failed: daily/minute close cross-check failed: matched=30, max_diff=0.020000000000000018 |
| 002976 | 瑞玛精密 | 2026-06-09 | 002976 data quality check failed: daily/minute close cross-check failed: matched=30, max_diff=0.21000000000000085 |
| 003043 | 华亚智能 | 2026-06-10 | 003043 data quality check failed: daily/minute close cross-check failed: matched=30, max_diff=0.030000000000001137 |

## 质量提示

| 代码 | 名称 | 日线源 | 5分钟源 | 提示 |
|---|---|---|---|---|
| 001369 | 双欣材料 | tencent_daily | sina_5m | daily history rows 105 < required 180 |
| 002225 | 濮耐股份 | cache | sina_5m | daily/minute close cross-check failed: matched=30, max_diff=0.020000000000000018 |
| 002976 | 瑞玛精密 | cache | sina_5m | daily/minute close cross-check failed: matched=30, max_diff=0.21000000000000085 |
| 003043 | 华亚智能 | cache | sina_5m | daily/minute close cross-check failed: matched=30, max_diff=0.030000000000001137 |

## 字段说明

| 字段 | 含义 |
|---|---|
| daily_history_rows | 参与日线 MA 计算的日线历史交易日数量 |
| daily_required_days | 当前策略要求的最少日线历史交易日数量 |
| daily_ma_coverage_ok | 最新行 MA5/MA10/MA20/MA30 是否都存在 |
| daily_minute_close_check_ok | 日线收盘与当日最后一根 5 分钟收盘是否基本一致 |
| missing_*_count | 对应字段缺失数量 |