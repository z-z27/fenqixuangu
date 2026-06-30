# 短线强势股分歧承接研究系统

本项目用于研究 A 股主板涨停后的分歧承接机会。它不是自动交易系统，也不构成投资建议。

当前主线分成两条：

- 日常使用：生成当天 D2 交易预案，并用已验证的研究模型给候选排序。
- 研究验证：生成历史候选样本，分析因子，手工构建 ranking model，再用 ranking backtest 验证。

## 当前 Daily 主模型

日常入口已经默认使用 v002 排序模型：

```text
reports/manual_models/ranking_model_v002_core_momentum_support.json
```

默认参数：

```text
top_n = 3
score_column = research_score
model_id = ranking_model_v002_core_momentum_support
```

v002 会对 daily signals 增加这些字段：

```text
ranking_model_id
research_score
daily_rank
model_topn
```

排序规则：

```text
allowed=True 且 signal_type=D2_LOW_ABSORB 的候选
按 research_score 降序生成 daily_rank
默认展示 Top3 主选
```

原始策略分 `total_score` 会保留，不会删除。`daily_rank` 和 `model_topn` 只写入日常 daily signals，不写入 history sample 研究链路。

## 日常使用

一键生成涨停池、补数据、生成信号和 v002 Top3 主选：

```powershell
python -m src.cli run-daily
```

指定日期：

```powershell
python -m src.cli run-daily `
  --date 2026-06-29 `
  --lookback-days 5
```

只基于已有涨停池生成 daily signals：

```powershell
python -m src.cli generate-signals `
  --limitup-file data/processed/recent_limitups.csv
```

手动切换模型或 TopN：

```powershell
python -m src.cli run-daily `
  --ranking-model reports/manual_models/ranking_model_v001_core_momentum.json `
  --top-n 3
```

日常输出目录：

```text
reports/daily_signals/
reports/data_quality/
```

Markdown 报告会包含：

```text
v002 Top3 主选
推荐交易 - D2 低吸
观察列表
字段说明
```

## 研究链路

研究链路保持独立，不使用 daily ranking 字段污染样本。

```text
history_samples -> factor analysis -> manual ranking_model JSON -> ranking_backtest
```

### 1. 生成历史候选样本

```powershell
$env:FQ_MIN_5MIN_DAYS="4"

python -m src.cli generate-history-samples `
  --start-date 2026-05-06 `
  --end-date 2026-06-29 `
  --lookback-days 5 `
  --signal-days 10 `
  --eval-days 10 `
  --hold-days 10 `
  --workers 6
```

输出目录：

```text
reports/history_samples/<start-date>_<end-date>/
```

关键输出：

```text
history_candidates_<start-date>_<end-date>.csv
history_candidates_summary_<start-date>_<end-date>.csv
history_generation_log_<start-date>_<end-date>.csv
history_future_fetch_<start-date>_<end-date>.csv
history_candidates_review_<start-date>_<end-date>.md
```

### 2. 因子分析

```powershell
python -m src.cli analyze-factors `
  --samples-file reports/history_samples/2026-05-06_2026-06-29/history_candidates_2026-05-06_2026-06-29.csv `
  --target-return-pct 7 `
  --min-bucket-size 10
```

如果要分析所有可评估候选，而不是只看 `eligible_for_trade=True`：

```powershell
--all-candidates
```

### 3. 验证手工模型

```powershell
python -m src.cli ranking-backtest `
  --samples-file reports/history_samples/2026-05-06_2026-06-29/history_candidates_2026-05-06_2026-06-29.csv `
  --model-file reports/manual_models/ranking_model_v002_core_momentum_support.json `
  --top-n 3
```

关注指标：

```text
daily_hit_rate
topn_target7_rate
avg_top1_candidate_d3_max_return_pct
avg_topn_candidate_d3_max_return_pct
failure dates
```

## v002 模型定位

v002 是当前 daily 主排序模型，适合用于 Top3 组合筛选，不建议只看 Top1。

在 `2026-05-06~2026-06-29` 历史样本验证中，v002 Top3 相比 v001 Top3 提升了 TopN 内部候选质量和平均 D3 最大收益，但日命中率略低。因此当前执行口径是：

```text
以 v002 Top3 为主选池
不要把 daily_rank=1 单独当成唯一信号
继续用后续新样本做外推跟踪
```

## 数据与版本管理

以下内容属于运行产物，不应提交到 Git：

```text
data/cache/
data/raw/
data/processed/*.csv
reports/daily_signals/
reports/data_quality/
reports/backtest_results/
reports/backtest_runs/
reports/history_samples/
reports/factor_analysis/
reports/ranking_backtests/
reports/research_runs/
reports/warmup/
__pycache__/
*.pyc
```

需要保留在 Git 中的模型资产：

```text
reports/manual_models/*.json
```

## 风险边界

所有模型、因子、回测结果都只是研究材料。A 股短线样本容易受行情阶段、流动性、数据质量和涨停池构造影响。任何 daily 输出都需要人工复核，不应直接等同于实盘收益。
