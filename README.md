# 短线强势股分歧承接预案系统

这是一个研究型 A 股盘后预案与因子迭代项目，目标不是直接给出“确定性买卖点”，而是建立一套可以持续运行的短线强势股研究闭环：

```text
获得数据 -> 分析数据 -> 因子调节 -> 回测验证 -> 正式使用
```

当前策略方向聚焦主板涨停股后的分歧承接机会：先获得涨停池与历史行情，再生成全候选样本，分析收益分布和因子表现，基于数据调节因子权重和过滤条件，经过回测验证后再用于每日盘后预案。

> 本项目用于量化研究、复盘和交易预案生成，不构成投资建议。实盘使用前必须经过足够样本量的历史验证、分阶段小仓验证和人工复核。

---

## 一、项目目标

系统的核心目标是搭建一个可迭代的短线研究框架：

1. **获得数据**
   - 收集 A 股涨停池。
   - 过滤主板股票，排除 ST、退市、北交所、科创板、创业板。
   - 为候选股票补充日线和 5 分钟线。
   - 保留本地缓存，减少重复请求公开行情接口。

2. **分析数据**
   - 对每个候选股生成信号、评分、关键区间和未来收益路径。
   - 不只看最终是否执行成功，而是保留全部候选样本。
   - 重点分析全样本收益分布，例如 D3 收盘收益、D3 最高收益、亏损尾部、机会尾部。

3. **因子调节**
   - 分析图形质量、承接质量、活跃资金、题材联动等因子。
   - 比较因子分位数、收益均值、目标收益率和亏损率。
   - 根据数据调整评分权重、过滤阈值和执行规则。

4. **回测验证**
   - 使用历史区间重新生成每日信号。
   - 对全候选、可交易候选、TopN 选择和实际执行结果分别评估。
   - 输出交易记录、因子快照、运行日志、收益分布和每日汇总。

5. **正式使用**
   - 每日盘后运行数据获取和预案生成。
   - 第二天只按预案验证关键区间，不临时追高。
   - 实盘使用时结合人工复核、仓位控制和风险限制。

---

## 二、当前策略框架

当前版本围绕“涨停后分歧承接”生成 D2 低吸验证预案。

主要处理流程：

```text
涨停池
  -> 主板过滤
  -> 日线 / 5分钟线补数
  -> 指标计算
  -> 图形、活跃、承接、题材评分
  -> 生成低吸区间 / 失效位
  -> 输出 D2 预案
  -> 历史回测与全样本收益分析
```

当前总分结构：

```text
total_score = graph_quality_score * 0.35
            + active_money_score * 0.25
            + support_score * 0.25
            + theme_score * 0.15
```

信号类型主要包括：

| 类型 | 含义 |
|---|---|
| `D2_LOW_ABSORB` | 推荐交易候选，正常仓位预案 |
| `D2_WATCH_OR_SMALL` | 小仓观察候选 |
| `WATCH_ONLY` | 只观察，不参与执行 |

---

## 三、数据规则

### 1. 主板过滤口径

当前只保留：

- 上海主板：`600`、`601`、`603`、`605` 开头；
- 深圳主板：`000`、`001`、`002`、`003` 开头。

默认排除：

- ST；
- 退市股；
- 北交所；
- 科创板；
- 创业板。

### 2. 5 分钟线窗口规则

当前 5 分钟线采用“目标窗口”和“最低可用窗口”分离的规则：

```text
目标抓取：40 个交易日
最低合格：20 个交易日
```

含义：

- 系统会尽量获取 40 个交易日的 5 分钟线；
- 实际可用交易日数大于等于 20，就视为正常数据；
- 少于 20 个交易日，才判定为数据质量失败；
- 20 到 39 个交易日不再视为异常，也不单独记 warning。

这个设计用于避免公开 5 分钟数据接口历史覆盖不足时，把大量可用样本误判为失败。

### 3. 日线规则

日线默认使用较长历史窗口，以支持均线、趋势和图形结构计算。

默认配置：

```text
FQ_DAILY_HISTORY_DAYS=180
FQ_5MIN_DAYS=40
FQ_MIN_5MIN_DAYS=20
FQ_INDICATOR_WARMUP_TRADING_DAYS=120
```

如需临时覆盖，可以使用环境变量调整。

---

## 四、安装依赖

```powershell
python -m pip install -r requirements.txt
```

建议使用虚拟环境，避免和本机其他量化、爬虫或数据分析包冲突。

---

## 五、常用命令

### 1. 收集某天涨停池

```powershell
python -m src.cli collect-limitups --date 2026-06-25
```

### 2. 收集最近若干自然日内的涨停池，并汇总去重

```powershell
python -m src.cli collect-limitups --lookback-days 5
```

### 3. 为涨停池中的主板股票补充日线和 5 分钟数据

```powershell
python -m src.cli collect-bars --limitup-file data/processed/recent_limitups.csv --days 40
```

`--days 40` 表示目标获取 40 个交易日的数据；最终 5 分钟线只要达到最低可用窗口 20 个交易日，就会通过质量检查。

### 4. 生成 D2 交易预案

```powershell
python -m src.cli generate-signals --limitup-file data/processed/recent_limitups.csv --days 40
```

输出文件：

```text
reports/daily_signals/signals_<date>.csv
reports/daily_signals/signals_<date>.md
reports/data_quality/data_quality_<date>.csv
reports/data_quality/data_quality_<date>.md
```

### 5. 一键执行日常盘后流程

```powershell
python -m src.cli run-daily --lookback-days 5 --days 40
```

日常使用时建议先看：

1. `reports/data_quality/data_quality_<date>.md`
2. `reports/daily_signals/signals_<date>.md`
3. `reports/daily_signals/signals_<date>.csv`

---

## 六、命令参数说明

### 1. `collect-limitups`

用途：收集涨停池，并按主板规则过滤。

```powershell
python -m src.cli collect-limitups --date 2026-06-25 --lookback-days 5 --force-refresh
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--date` | 今天 | 指定锚定日期，例如 `2026-06-25`。 | 复盘历史日期时填写；日常盘后可不填。 |
| `--lookback-days` | `1` | 从锚定日期向前扫描多少个自然日。 | 日常建议用 `5`，可以覆盖周末和短假期。 |
| `--force-refresh` | `False` | 忽略本地缓存，重新请求涨停池接口。 | 接口异常、缓存怀疑过期或数据明显不对时使用。 |

输出：

```text
data/processed/recent_limitups.csv
```

---

### 2. `collect-bars`

用途：读取涨停池，为候选股票补充日线和 5 分钟线。

```powershell
python -m src.cli collect-bars --limitup-file data/processed/recent_limitups.csv --days 40 --max-codes 200 --force-refresh
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--limitup-file` | `data/processed/recent_limitups.csv` | 候选涨停池文件。 | 通常使用默认值。 |
| `--days` | 配置默认值 | 目标获取多少个交易日的 5 分钟线。 | 建议 `40`；最终 5 分钟线满足 20 个交易日即可合格。 |
| `--max-codes` | 不限制 | 最多处理多少只股票。 | 调试时可设 `20` 或 `50`；正式研究建议不限制或设较大值。 |
| `--force-refresh` | `False` | 忽略本地缓存，重新拉日线和 5 分钟线。 | 缓存异常或数据质量异常时使用；会明显增加请求时间。 |

输出：

```text
data/raw/daily/
data/raw/minute_5m/
data/cache/daily/
data/cache/minute_5m/
```

---

### 3. `generate-signals`

用途：基于候选池和行情数据生成 D2 交易预案。

```powershell
python -m src.cli generate-signals --limitup-file data/processed/recent_limitups.csv --days 40 --max-codes 200 --force-refresh
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--limitup-file` | `data/processed/recent_limitups.csv` | 候选涨停池文件。 | 通常使用默认值。 |
| `--days` | 配置默认值 | 目标使用多少个交易日的 5 分钟线构建信号。 | 建议 `40`。 |
| `--max-codes` | 不限制 | 最多生成多少只股票的信号。 | 调试时使用；正式运行尽量不限制。 |
| `--force-refresh` | `False` | 重新拉取行情数据再生成信号。 | 数据异常时使用。 |

输出：

```text
reports/daily_signals/signals_<date>.csv
reports/daily_signals/signals_<date>.md
reports/data_quality/data_quality_<date>.csv
reports/data_quality/data_quality_<date>.md
```

---

### 4. `run-daily`

用途：日常盘后一键执行“涨停池 -> 补行情 -> 生成预案”。

```powershell
python -m src.cli run-daily --date 2026-06-25 --lookback-days 5 --days 40 --max-codes 300 --force-refresh
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--date` | 今天 | 指定盘后处理日期。 | 日常盘后通常不填；复盘时填写。 |
| `--lookback-days` | `5` | 向前扫描多少个自然日的涨停池。 | 保持 `5` 比较稳。 |
| `--days` | 配置默认值 | 目标获取多少个交易日的 5 分钟线。 | 建议 `40`。 |
| `--max-codes` | 不限制 | 最多处理多少只候选股票。 | 正式使用不建议限制；测试时可限制。 |
| `--force-refresh` | `False` | 忽略缓存重新拉数据。 | 数据源异常、缓存异常或回测复核时使用。 |

这是日常正式使用的主命令。

---

### 5. `backtest-run`

用途：从历史涨停池开始，完整执行“收集数据 -> 生成每日信号 -> 回测评估”。

```powershell
python -m src.cli backtest-run `
  --start-date 2026-06-01 `
  --end-date 2026-06-25 `
  --top-n 3 `
  --hold-days 10 `
  --target-return-pct 7 `
  --stop-loss-pct 3 `
  --lookback-days 5 `
  --signal-days 40 `
  --eval-days 40 `
  --include-all-allowed `
  --include-small `
  --entry-price-mode zone_max
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--start-date` | 必填 | 回测开始日期。 | 建议覆盖完整市场阶段，不要只跑单日。 |
| `--end-date` | 必填 | 回测结束日期。 | 结束日期后需要有足够未来行情用于评估。 |
| `--top-n` | `3` | 每日按排序选前 N 只作为执行候选。 | 执行回测可用 `3`；研究全候选可用 `999`。 |
| `--hold-days` | `10` | 单笔最多观察多少个交易日。 | 当前短线研究常用 `10`。 |
| `--target-return-pct` | `7.0` | 目标收益率，例如 `7` 表示 7%。 | 用于成功/机会判断。 |
| `--stop-loss-pct` | `3.0` | 止损阈值，例如 `3` 表示 -3%。 | 用于回测风险判断。 |
| `--lookback-days` | `5` | 每个信号日向前看多少自然日涨停池。 | 建议 `5`。 |
| `--signal-days` | 配置默认值 | 生成信号时目标使用多少个交易日的 5 分钟线。 | 建议 `40`。 |
| `--eval-days` | 配置默认值 | 评估未来收益时目标获取多少个交易日的 5 分钟线。 | 建议 `40`。 |
| `--max-codes` | 不限制 | 每天最多处理多少只股票。 | 调试时限制；正式研究尽量不限制。 |
| `--force-refresh` | `False` | 忽略缓存，重新请求数据。 | 数据异常时使用。 |
| `--include-all-allowed` | `False` | 回测时纳入所有 allowed 候选，而不仅 TopN。 | 做研究时建议开启。 |
| `--include-small` | `False` | 是否纳入小仓试探信号。 | 想研究边缘候选时开启。 |
| `--entry-price-mode` | `zone_max` | 入场价模式。`zone_max` 使用低吸区间上沿；`confirmation_close` 使用确认收盘价。 | 默认 `zone_max` 更贴近低吸预案。 |

输出目录：

```text
reports/backtest_runs/<start>_<end>/
```

---

### 6. `watch_backtest`

用途：包装 `backtest-run`，一边跑历史回测，一边输出文件级进度。

```powershell
python -m src.watch_backtest `
  --start-date 2026-06-01 `
  --end-date 2026-06-25 `
  --top-n 3 `
  --signal-days 40 `
  --eval-days 40 `
  --interval-seconds 5
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--start-date` | 必填 | 回测开始日期。 | 同 `backtest-run`。 |
| `--end-date` | 必填 | 回测结束日期。 | 同 `backtest-run`。 |
| `--top-n` | `3` | 每日执行候选数量。 | 执行回测用 `3`，研究用更大值。 |
| `--hold-days` | `10` | 最大观察天数。 | 同 `backtest-run`。 |
| `--target-return-pct` | `7.0` | 目标收益率。 | 同 `backtest-run`。 |
| `--stop-loss-pct` | `3.0` | 止损阈值。 | 同 `backtest-run`。 |
| `--lookback-days` | `5` | 涨停池回看自然日。 | 建议 `5`。 |
| `--signal-days` | `40` | 信号生成目标 5 分钟线窗口。 | 默认已经是 `40`。 |
| `--eval-days` | `40` | 未来评估目标 5 分钟线窗口。 | 默认已经是 `40`。 |
| `--max-codes` | 不限制 | 每天最多处理多少只股票。 | 调试时可限制。 |
| `--force-refresh` | `False` | 忽略缓存重新拉取。 | 数据异常时使用。 |
| `--include-all-allowed` | `False` | 纳入全部 allowed 候选。 | 研究时建议开启。 |
| `--include-small` | `False` | 纳入小仓试探候选。 | 研究边缘信号时开启。 |
| `--entry-price-mode` | `zone_max` | 入场价模式。 | 默认即可。 |
| `--interval-seconds` | `5` | 进度输出间隔秒数。 | 数据多时可调成 `10` 或 `30`。 |

---

### 7. `watch_research`

用途：完整运行历史回测后，自动复制结果并生成全候选研究报告。

```powershell
python -m src.watch_research `
  --start-date 2026-06-01 `
  --end-date 2026-06-25 `
  --top-n 999 `
  --signal-days 40 `
  --eval-days 40 `
  --include-all-allowed
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--start-date` | 必填 | 研究区间开始日期。 | 样本越完整越有价值。 |
| `--end-date` | 必填 | 研究区间结束日期。 | 确保之后有未来行情可评估。 |
| `--top-n` | `999` | 研究默认尽量保留大量候选。 | 保持 `999`，不要过早只看 Top3。 |
| `--hold-days` | `10` | 最大观察天数。 | 默认即可。 |
| `--target-return-pct` | `7.0` | 目标收益率。 | 用于成功/机会判断。 |
| `--stop-loss-pct` | `3.0` | 止损阈值。 | 用于回测风险判断。 |
| `--target-min-return-pct` | 代码默认 | 研究样本中“较好收益”的下限。 | 默认即可。 |
| `--target-max-return-pct` | 代码默认 | 研究样本中“机会收益”的上限或参考阈值。 | 默认即可。 |
| `--return-loss-cutoff-pct` | `-3.0` | 收益分布中亏损组阈值。 | 默认用 -3%。 |
| `--return-quantiles` | `5` | 因子分位数数量。 | 默认五分位。 |
| `--with-factor-discovery` | `False` | 是否额外生成旧版因子发现报告。 | 需要兼容旧报告时开启。 |
| `--discovery-min-group-size` | `3` | 旧版因子分组最小样本数。 | 默认即可。 |
| `--lookback-days` | `5` | 涨停池回看自然日。 | 建议 `5`。 |
| `--signal-days` | 配置默认值 | 生成信号目标 5 分钟线窗口。 | 建议 `40`。 |
| `--eval-days` | 配置默认值 | 未来评估目标 5 分钟线窗口。 | 建议 `40`。 |
| `--max-codes` | 不限制 | 每天最多处理多少只股票。 | 调试时限制；正式研究不限制。 |
| `--force-refresh` | `False` | 忽略缓存重新拉数据。 | 数据异常时使用。 |
| `--include-all-allowed` | `False` | 纳入所有 allowed 候选。 | 做全候选研究时建议开启。 |
| `--include-small` | `False` | 纳入小仓试探候选。 | 研究边缘信号时开启。 |
| `--entry-price-mode` | `zone_max` | 入场价模式。 | 默认即可。 |
| `--interval-seconds` | `5` | 子回测进度输出间隔。 | 数据多时可调大。 |
| `--output-root` | `reports/research_runs` | 研究输出根目录。 | 默认即可。 |
| `--run-name` | 自动生成 | 指定研究目录名称。 | 对重要实验建议手动命名。 |

---

### 8. `backtest-history`

用途：读取已经存在的每日 `signals_*.csv`，不重新生成信号，只做历史回测。

```powershell
python -m src.cli backtest-history `
  --signals-dir reports/daily_signals `
  --start-date 2026-06-01 `
  --end-date 2026-06-25 `
  --top-n 3 `
  --include-all-allowed
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--signals-dir` | `reports/daily_signals` | 已生成信号文件所在目录。 | 默认即可。 |
| `--start-date` | 必填 | 回测开始日期。 | 必填。 |
| `--end-date` | 必填 | 回测结束日期。 | 必填。 |
| `--top-n` | `3` | 每天选前 N 只。 | 执行验证用 `3`。 |
| `--hold-days` | `10` | 最大观察天数。 | 默认即可。 |
| `--target-return-pct` | `7.0` | 目标收益率。 | 默认即可。 |
| `--stop-loss-pct` | `3.0` | 止损阈值。 | 默认即可。 |
| `--include-all-allowed` | `False` | 是否纳入所有 allowed 候选。 | 研究时建议开启。 |
| `--include-small` | `False` | 是否纳入小仓试探候选。 | 研究边缘信号时开启。 |
| `--entry-price-mode` | `zone_max` | 入场价模式。 | 默认即可。 |

---

### 9. `backtest-top3`

用途：对单个信号文件做 TopN 回测。

```powershell
python -m src.cli backtest-top3 `
  --signals-file reports/daily_signals/signals_2026-06-25.csv `
  --top-n 3 `
  --target-return-pct 7
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--signals-file` | 示例文件 | 单日信号 CSV。 | 指定要评估的某一天信号。 |
| `--top-n` | `3` | 选前 N 只信号。 | 默认 `3`。 |
| `--target-return-pct` | `7.0` | 目标收益率。 | 默认 `7`。 |
| `--include-small` | `False` | 是否纳入小仓试探候选。 | 想看边缘候选时开启。 |
| `--fetch-through-date` | 不指定 | 未来行情拉取到哪一天。 | 复盘指定未来截止日时使用。 |
| `--days` | 配置默认值 | 目标 5 分钟线窗口。 | 建议 `40`。 |
| `--force-refresh` | `False` | 忽略缓存重新拉数据。 | 数据异常时使用。 |
| `--entry-price-mode` | `zone_max` | 入场价模式。 | 默认即可。 |

---

### 10. `validate-data`

用途：对缓存中的日线、5 分钟线和均线计算做数据验收。

```powershell
python -m src.cli validate-data --limitup-file data/processed/recent_limitups.csv --days 40 --max-codes 100
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--limitup-file` | `data/processed/recent_limitups.csv` | 要验证的候选池文件。 | 默认即可。 |
| `--days` | 配置默认值 | 验证时使用的目标数据窗口。 | 建议 `40`。 |
| `--max-codes` | 不限制 | 最多验证多少只股票。 | 调试时可限制。 |
| `--reference-root` | `F:\dataaccept` | 外部对照数据目录。 | 有对照数据时使用。 |

---

### 11. `review-failed-data`

用途：复查数据质量失败的样本，并输出排除或修复参考。

```powershell
python -m src.cli review-failed-data --quality-file reports/data_quality/data_quality_2026-06-25.csv --days 40 --force-refresh
```

| 参数 | 默认值 | 含义 | 使用建议 |
|---|---:|---|---|
| `--quality-file` | 不指定 | 要复查的数据质量 CSV。 | 指定某天失败较多的质量文件。 |
| `--days` | 配置默认值 | 复查时目标数据窗口。 | 建议 `40`。 |
| `--force-refresh` | `False` | 忽略缓存重新拉数据。 | 判断是否缓存导致失败时使用。 |

---

## 七、历史回测输出

一次性历史回测：

```powershell
python -m src.cli backtest-run `
  --start-date 2026-06-01 `
  --end-date 2026-06-25 `
  --top-n 3 `
  --hold-days 10 `
  --target-return-pct 7 `
  --stop-loss-pct 3 `
  --lookback-days 5 `
  --signal-days 40 `
  --eval-days 40
```

输出目录：

```text
reports/backtest_runs/<start>_<end>/
```

主要产物：

```text
daily_signals/                                每日信号
data_quality/                                 每日数据质量报告
backtest_results/history_trades_*.csv         全量交易/候选记录
backtest_results/history_summary_*.csv        回测汇总
backtest_results/history_factor_stats_*.csv   因子快照
backtest_results/history_run_log_*.csv        每日运行日志
backtest_results/history_future_fetch_*.csv   未来行情补数日志
```

---

## 八、研究模式：全候选收益分布分析

研究模式用于回答下面这些问题：

- 候选池整体收益分布是什么样？
- 哪些因子在盈利样本中更常见？
- 哪些因子在亏损样本中更常见？
- 当前 TopN 或执行规则是否选中了真正更好的候选？
- 机会收益和收盘留存收益之间是否存在明显衰减？

运行命令：

```powershell
python -m src.watch_research `
  --start-date 2026-06-01 `
  --end-date 2026-06-25 `
  --top-n 999 `
  --signal-days 40 `
  --eval-days 40 `
  --include-all-allowed
```

研究结果目录：

```text
reports/research_runs/<run_name>/research_results/
```

重点文件：

```text
research_samples_<start>_<end>.csv              研究样本
factor_compare_<start>_<end>.csv                旧版因子对比
return_samples_<start>_<end>.csv                全候选收益样本
return_distribution_report_<start>_<end>.md     收益分布报告
return_bucket_compare_<start>_<end>.csv         收益桶比较
factor_quantile_report_<start>_<end>.csv        因子分位数报告
profit_loss_compare_<start>_<end>.csv           盈亏组比较
daily_return_summary_<start>_<end>.csv          每日收益汇总
```

当前全样本研究重点不是简单的 `success/failed`，而是：

| 字段 | 含义 |
|---|---|
| `candidate_d3_close_return_pct` | 候选股 D3 收盘收益，衡量收盘留存能力 |
| `candidate_d3_max_return_pct` | 候选股 D3 区间最高收益，衡量机会空间 |
| `candidate_d3_max_drawdown_pct` | 候选股 D3 区间最大回撤，衡量风险暴露 |

推荐研究顺序：

```text
全样本收益分布
  -> 盈利尾部 / 亏损尾部分组
  -> 因子分位数
  -> 因子调节
  -> 重新回测
```

---

## 九、因子调节原则

因子调节不要直接根据单次结果改权重，建议按下面顺序进行：

1. 先确认样本覆盖是否正常：
   - 每个交易日是否有涨停池；
   - 每天是否有足够信号；
   - 数据质量失败是否异常集中。

2. 再看全样本收益分布：
   - 平均收益；
   - 中位数；
   - 目标收益率；
   - 亏损率；
   - 机会收益和收盘收益差异。

3. 再看因子分位数：
   - Q5 是否明显好于 Q1；
   - 高分组是否降低亏损率；
   - 因子是否只在少数样本中有效。

4. 最后才调整：
   - 因子权重；
   - 硬过滤阈值；
   - TopN 排序规则；
   - 执行条件；
   - 买入区间和失效位规则。

调节后必须重新跑完整历史区间，不能只看单日或少量样本。

---

## 十、正式使用流程

正式盘后使用建议采用下面流程：

```text
收盘后运行 run-daily
  -> 查看数据质量报告
  -> 查看 signals markdown
  -> 人工排除异常标的
  -> 次日只按预案验证低吸区间和失效位
  -> 收盘后记录执行与未执行样本
  -> 周期性合并进研究样本
```

日常命令：

```powershell
python -m src.cli run-daily --lookback-days 5 --days 40
```

实盘使用纪律：

- 不临时追高；
- 不因为分数高而忽略失效位；
- 不把 `WATCH_ONLY` 直接升级为交易标的；
- 不用单日表现调整策略；
- 每次策略调整后必须重新回测；
- 研究样本必须保留未执行、未选中、执行失败、执行成功等所有候选记录。

---

## 十一、数据目录

```text
data/
  raw/limit_ups/              原始涨停池
  raw/daily/                  个股日线导出
  raw/minute_5m/              个股 5 分钟线导出
  cache/daily/                日线缓存
  cache/minute_5m/            5 分钟线缓存
  cache/limit_ups/            涨停池缓存
  processed/                  汇总后的候选池

reports/
  daily_signals/              每日预案 CSV / Markdown
  data_quality/               每日数据质量报告
  backtest_runs/              历史回测结果
  research_runs/              全候选研究结果
```

---

## 十二、数据源说明

涨停池：

- 优先使用 AkShare 东方财富涨停池接口；
- 当查询当日且涨停池接口不可用时，可降级使用东方财富实时行情近似筛选。

个股行情：

- 日线优先使用腾讯日线接口；
- 腾讯日线失败时降级到新浪日线；
- 5 分钟线使用新浪 5 分钟接口；
- 所有公开接口都可能不稳定，因此项目会写入本地缓存和数据质量报告。

如果发现某段时间信号数量异常少，应优先检查：

```text
reports/backtest_runs/<start>_<end>/backtest_results/history_run_log_<start>_<end>.csv
reports/backtest_runs/<start>_<end>/data_quality/data_quality_<date>.csv
```
