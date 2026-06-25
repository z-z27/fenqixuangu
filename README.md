# 短线强势股分歧承接预案系统

这是一个研究型 A 股盘后预案项目，按 `short_term_quant_divergence_strategy_v0_2.md` 落地第一版代码。

当前版本优先解决数据链路：

- 收集最近涨停池；
- 只保留主板股票；
- 对涨停股票补充日线与 5 分钟线；
- 计算基础指标、关键成本区和四类评分；
- 在 D1 收盘后生成 D2 低吸验证预案；
- 保留本地缓存，减少重复请求公开行情接口。

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

## 常用命令

收集某天涨停池：

```powershell
python -m src.cli collect-limitups --date 2026-06-25
```

收集最近若干自然日内的涨停池，并汇总去重：

```powershell
python -m src.cli collect-limitups --lookback-days 5
```

为涨停池中的主板股票补充日线和 5 分钟数据：

```powershell
python -m src.cli collect-bars --limitup-file data/processed/recent_limitups.csv --days 40
```

生成 D2 交易预案：

```powershell
python -m src.cli generate-signals --limitup-file data/processed/recent_limitups.csv --days 40
```

一键执行完整流程：

```powershell
python -m src.cli run-daily --lookback-days 5 --days 40
```

## 数据目录

```text
data/
  raw/limit_ups/          原始涨停池
  raw/daily/              个股日线缓存导出
  raw/minute_5m/          个股 5 分钟缓存导出
  cache/                  pickle 缓存
  processed/              汇总后的候选池
reports/
  daily_signals/          每日预案 CSV / Markdown
```

## 主板过滤口径

当前只保留：

- 上海主板：`600`、`601`、`603`、`605` 开头；
- 深圳主板：`000`、`001`、`002`、`003` 开头。

默认排除 ST、退市、北交所、科创板、创业板。

## 数据源

涨停池优先使用 AkShare 东方财富涨停池接口，失败时降级为东方财富实时行情近似筛选。

个股日线和 5 分钟线优先使用东方财富直连，保留 AkShare 包装器作为降级来源。公开接口不稳定，代码会把空表也视为失败，并在可用时回退到缓存。
