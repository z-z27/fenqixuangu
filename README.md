# 短线强势股分歧承接研究系统

本项目当前定位是**研究系统**，不是自动策略生成器，也不是直接实盘执行系统。

当前核心目标是围绕 A 股主板涨停后的分歧承接机会，建立一条可复现的研究链路：

```text
历史候选样本 → 基础因子分析 → 人工讨论数学模型 → ranking_backtest 验证模型
```

本项目不构成投资建议。任何模型、因子或回测结果都只能作为研究材料，不能直接代表实盘收益。

---

## 1. 当前研究边界

### 1.1 已完成的工程层

当前代码按三层组织：

```text
第 1 层：history_samples
  生成干净的历史候选样本 history_candidates_*.csv

第 2 层：factor analysis
  只做基础因子分析，不自动生成 ranking_model

第 3 层：ranking_backtest
  验证人工构建的 ranking_model_*.json
```

### 1.2 明确不做的事情

第二层不会自动生成数学模型，也不会自动生成 `ranking_model_*.json`。

数学模型必须由研究者基于真实历史样本和因子分析结果共同讨论确定，然后再写成模型 JSON，用第三层验证。

---

## 2. 核心日期定义

```text
D0 = 涨停日
D1 = 首次分歧观察日 / 盘后生成候选样本的日期
D2 = 计划观察或买入日
D3 = 重点收益验证日
```

当前研究目标不是验证某个具体买点是否成交，而是先研究：

```text
D1 盘后候选 → D2/D3 是否存在足够好的收益机会
```

核心标签：

```text
target7  = candidate_d3_max_return_pct >= 7%
target10 = candidate_d3_max_return_pct >= 10%
```

---

## 3. 第 1 层：生成历史候选样本

### 3.1 命令

建议先用较小区间测试，确认本机数据接口和缓存正常：

```powershell
python -m src.cli generate-history-samples `
  --start-date 2026-06-01 `
  --end-date 2026-06-20 `
  --lookback-days 5 `
  --hold-days 10 `
  --max-codes 50
```

如果测试成功，再扩大区间，例如：

```powershell
python -m src.cli generate-history-samples `
  --start-date 2026-01-01 `
  --end-date 2026-06-20 `
  --lookback-days 5 `
  --hold-days 10
```

Linux / macOS 可以把反引号换成反斜杠：

```bash
python -m src.cli generate-history-samples \
  --start-date 2026-06-01 \
  --end-date 2026-06-20 \
  --lookback-days 5 \
  --hold-days 10 \
  --max-codes 50
```

### 3.2 输出目录

第一层输出在：

```text
reports/history_samples/<start-date>_<end-date>/
```

关键文件：

```text
history_candidates_<start-date>_<end-date>.csv
history_candidates_summary_<start-date>_<end-date>.csv
history_generation_log_<start-date>_<end-date>.csv
history_future_fetch_<start-date>_<end-date>.csv
history_candidates_review_<start-date>_<end-date>.md
```

### 3.3 第一层输出含义

`history_candidates_*.csv` 是后续所有研究的基础。

它应该包含：

- D1 已知因子；
- 候选基础信息；
- future label，例如 `candidate_d3_max_return_pct`；
- `target7` / `target10`；
- 诊断字段，例如 `candidate_evaluable`、`future_trade_days_available`。

它不应该包含执行回测字段，例如：

```text
executed
selected_for_execution
selected_by_topn
buy_price
zone_buy_price
confirmation_price
execution_date
buy_time
entry_price_mode
execution_reason
target_hit
stop_hit
first_outcome
failure_reason
d3_realized_return_pct
d3_sell_reason
```

---

## 4. 第 2 层：基础因子分析

### 4.1 命令

在第一层成功生成 `history_candidates_*.csv` 后，运行：

```powershell
python -m src.cli analyze-factors `
  --samples-file reports/history_samples/2026-06-01_2026-06-20/history_candidates_2026-06-01_2026-06-20.csv `
  --target-return-pct 7 `
  --min-bucket-size 10
```

如果要分析全部可评估候选，而不是默认只看 `eligible_for_trade == True` 的候选，可以加：

```powershell
--all-candidates
```

### 4.2 输出目录

第二层输出在：

```text
reports/factor_analysis/<start-date>_<end-date>/
```

关键文件：

```text
factor_summary_<start-date>_<end-date>.csv
factor_buckets_<start-date>_<end-date>.csv
factor_daily_stability_<start-date>_<end-date>.csv
factor_pair_review_<start-date>_<end-date>.csv
factor_analysis_report_<start-date>_<end-date>.md
```

### 4.3 第二层职责

第二层只负责提供研究证据：

```text
哪些因子与 target7 有关系？
哪些因子方向可能是 higher_better / lower_better？
哪些因子分桶区间表现更好？
哪些因子跨日期稳定？
哪些因子组合可能有增益？
```

第二层不会生成最终模型。

`ranking_model_*.json` 必须由研究者基于这些分析结果讨论后手工构建。

---

## 5. 交给 ChatGPT 分析时需要提供哪些文件

跑完第一层和第二层后，把下面文件发给 ChatGPT：

### 必须提供

```text
reports/history_samples/<区间>/history_candidates_<区间>.csv
reports/history_samples/<区间>/history_candidates_summary_<区间>.csv
reports/factor_analysis/<区间>/factor_summary_<区间>.csv
reports/factor_analysis/<区间>/factor_buckets_<区间>.csv
reports/factor_analysis/<区间>/factor_daily_stability_<区间>.csv
reports/factor_analysis/<区间>/factor_pair_review_<区间>.csv
reports/factor_analysis/<区间>/factor_analysis_report_<区间>.md
```

### 建议同时提供

```text
reports/history_samples/<区间>/history_generation_log_<区间>.csv
reports/history_samples/<区间>/history_future_fetch_<区间>.csv
```

这些文件用于判断是否存在数据缺口、未来行情抓取失败、样本不可评估等问题。

---

## 6. 与 ChatGPT 共同构建数学模型

当你提供上述文件后，下一步不是继续写代码，而是一起研究：

```text
1. 哪些因子可用；
2. 哪些因子应该排除；
3. 哪些因子只在特定区间有效；
4. 是否存在组合条件；
5. Top3 排名函数应该如何定义；
6. ranking_model_*.json 应该如何表达这个模型。
```

模型可能是：

```text
线性加权模型
分桶加分模型
阈值触发模型
组合规则模型
混合模型
```

但模型必须来自真实样本分析，不允许在没有数据的情况下凭空生成。

---

## 7. 第 3 层：验证人工构建的 ranking_model

当我们共同构建好 `ranking_model_*.json` 后，再运行：

```powershell
python -m src.cli ranking-backtest `
  --samples-file reports/history_samples/2026-06-01_2026-06-20/history_candidates_2026-06-01_2026-06-20.csv `
  --model-file reports/manual_models/ranking_model_v001.json `
  --top-n 3
```

第三层只验证模型，不生成模型。

它会输出：

```text
ranking_backtest_summary_*.csv
ranking_backtest_daily_*.csv
ranking_backtest_topn_*.csv
ranking_backtest_failures_*.csv
ranking_backtest_*.md
```

关注指标：

```text
daily_hit_rate
topn_target7_rate
avg_top1_candidate_d3_max_return_pct
avg_topn_candidate_d3_max_return_pct
failure dates
```

---

## 8. 推荐当前执行顺序

### Step 1：小样本跑通

```powershell
python -m src.cli generate-history-samples `
  --start-date 2026-06-01 `
  --end-date 2026-06-20 `
  --lookback-days 5 `
  --hold-days 10 `
  --max-codes 50
```

### Step 2：确认输出存在

检查是否生成：

```text
reports/history_samples/2026-06-01_2026-06-20/history_candidates_2026-06-01_2026-06-20.csv
```

### Step 3：跑因子分析

```powershell
python -m src.cli analyze-factors `
  --samples-file reports/history_samples/2026-06-01_2026-06-20/history_candidates_2026-06-01_2026-06-20.csv `
  --target-return-pct 7 `
  --min-bucket-size 10
```

### Step 4：把文件发给 ChatGPT

至少发送：

```text
history_candidates_2026-06-01_2026-06-20.csv
history_candidates_summary_2026-06-01_2026-06-20.csv
factor_summary_2026-06-01_2026-06-20.csv
factor_buckets_2026-06-01_2026-06-20.csv
factor_daily_stability_2026-06-01_2026-06-20.csv
factor_pair_review_2026-06-01_2026-06-20.csv
factor_analysis_report_2026-06-01_2026-06-20.md
```

我会基于这些文件和你一起讨论并构造数学模型。

---

## 9. 当前不要做的事

在还没有真实样本和因子分析结果之前，不要做：

```text
不要调正式策略权重
不要改 signal_engine.py
不要生成 ranking_model_*.json
不要跑 ranking-backtest
不要讨论 Top3 是否已经达到 80% 命中率
```

正确顺序永远是：

```text
先有样本 → 再有分析 → 再有人类模型 → 最后验证
```
