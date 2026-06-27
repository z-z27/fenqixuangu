# 回测流程设计 v0.1

## 目标

回测要验证的是“D1 收盘后生成 D2 预案，D2 盘中只按预案执行”的交易链路，不验证盘中临时选股能力。

当前核心优化目标：

- 每个信号日只选择排序后的前 3 只。
- 默认只从 `D2_LOW_ABSORB` 的 normal 仓位标的中选前 3；`D2_WATCH_OR_SMALL` 只做单独扩展统计，不进入默认主回测。
- 买入必须发生在合规低吸位置，不能按开盘价、最低价或事后最优价假设成交。
- 默认买价使用低吸区上沿 `low_absorb_max`，这是合规买点里的保守成交价；同时保留站回 VWAP 的确认价用于复核。
- 目标不是“所有信号平均赚钱”，而是“每日 Top3 在合规买点成交后，平均能否接近或达到 7% 收益机会”。
- 7% 目标拆成两个指标：D2 盘中最高浮盈是否达到 7%，以及 D2 收盘收益是否仍有质量。

核心边界：

- 信号生成日只能使用当日收盘已经存在的数据。
- D2 执行只能使用触发时点之前已经出现的 5 分钟 K。
- 任何数据质量失败的标的直接排除。
- 不允许用 D2 收盘结果反推是否买入。
- 不允许用后续涨停、后续收益反推 D1 是否合格。

## 数据输入

每个回测样本日需要：

- 当日及以前的涨停池。
- 至少 180 个交易日的日线历史，用于 MA5/MA10/MA20/MA30 与趋势特征。
- 至少 40 个交易日的 5 分钟数据，用于 D1/D2 承接和执行判断。
- 数据质量报告，所有 failed 标的排除。

回测区间不能只拉起止日期本身，需要扩展：

- 向前扩展至少 180 个交易日，用于指标 warmup。
- 向后扩展至少 5 个交易日，用于持仓收益、止盈、止损和次日表现统计。

## 单日回测时序

1. 以交易日 T 为信号日。
2. 只截取 `<= T` 的日线和 5 分钟数据。
3. 用 T 日收盘后的状态生成 T+1 预案。
4. 在 T+1 的 5 分钟 K 中逐根模拟执行。
5. 如果触发买入，记录买入时间、买入价、失效价和仓位级别。
6. 从买入后开始统计 T+1 至 T+5 的收益路径。
7. 根据退出规则生成最终交易结果。

## 入场规则

默认主回测只回测 `allowed=True` 且 `signal_type == D2_LOW_ABSORB` 的信号。

排序规则：

1. normal 仓位优先。
2. `total_score` 高者优先。
3. 分数相同时依次比较 `graph_quality_score`、`support_score`、`active_money_score`。
4. 每个信号日只保留前 3 只。

信号分层：

- `D2_LOW_ABSORB`：默认进入 Top3 排序池。
- `D2_WATCH_OR_SMALL`：默认不进入主回测，可用参数单独纳入对照。
- `WATCH_ONLY`：不交易，只保留观察统计。

D2 触发条件：

- 回踩进入 `low_absorb_min ~ low_absorb_max`。
- 未有效跌破 `invalid_price`，或跌破后在配置时间内收回。
- 收盘价重新站回当日 VWAP。
- 不能追高；没有回踩则不成交。
- 默认成交价按 `low_absorb_max` 计；如果要按确认后追入价复核，可切换为 `confirmation_close`。

## 退出规则

第一版先做机械规则，避免人为解释：

- 买入后跌破 `invalid_price` 且 30 分钟内不能收回，止损。
- 买入当日收盘低于 VWAP，次日优先退出。
- 买入后出现涨停，次日不能继续强势则退出。
- 持有到 T+5 仍未触发退出，按 T+5 收盘退出。

## 输出字段

交易明细：

- signal_date
- execution_date
- daily_rank
- code
- name
- signal_type
- position_level
- total_score
- buy_time
- buy_price
- invalid_price
- exit_time
- exit_price
- exit_reason
- holding_days
- return_pct
- max_favorable_pct
- max_adverse_pct
- d2_max_return_pct
- d2_close_return_pct
- target_7_hit
- d1_return_pct
- d2_return_pct
- d3_return_pct
- d5_return_pct
- data_quality_status

汇总统计：

- 每日 Top3 数量。
- 可评估样本数。
- 合规买点成交数。
- 成交率。
- D2 盘中最高浮盈达到 7% 的比例。
- D2 盘中最高浮盈均值。
- D2 收盘收益均值。
- D2 盘中最高浮盈中位数。
- D2 收盘收益中位数。
- 最大回撤。
- normal/small 分组表现。
- 按 total_score、graph/support/active/theme 分箱表现。
- 按 days_since_d0 分组表现。

## 第一阶段实现顺序

1. 先实现单日报告的 Top3 选择和 D2 执行模拟。
2. 输出 Top3 明细 CSV 和 Markdown 汇总。
3. 再扩展到多日信号回放。
4. 接入数据质量门槛，只允许 accepted/ok 数据进入回测。
5. 加入 7% 目标达成率、D2 最高浮盈、D2 收盘收益。
6. 最后再加入参数扫描，避免一开始就过拟合。

当前单日报告验证命令：

```powershell
python -m src.cli backtest-top3 --signals-file reports/daily_signals/signals_2026-06-25.csv --top-n 3 --target-return-pct 7
```

如果本地还没有 D2 的 5 分钟缓存，可以只为了回测评估补到 D2 日期：

```powershell
python -m src.cli backtest-top3 --signals-file reports/daily_signals/signals_2026-06-25.csv --top-n 3 --target-return-pct 7 --fetch-through-date 2026-06-26
```

注意：`--fetch-through-date` 只用于事后评估 D2 执行，不允许回头改写 T 日信号。

完整历史回测命令：

```powershell
python -m src.cli backtest-run --start-date 2026-01-01 --end-date 2026-06-30 --top-n 3 --hold-days 10 --target-return-pct 7 --stop-loss-pct 3
```

`backtest-run` 是正式回测入口，不要求提前准备信号 CSV。它按日期逐日执行正常项目流程：

1. 拉取信号日 T 及 lookback 范围内的涨停池。
2. 对候选股拉取并缓存 `<= T` 的日线和 5 分钟线。
3. 使用严格数据质量门槛生成 T 日收盘后的 D2 预案，MA5/MA10/MA20/MA30 仍要求足够历史数据。
4. 保存 `signals_YYYY-MM-DD.csv/md` 和 `data_quality_YYYY-MM-DD.csv/md` 作为审计快照。
5. 再拉取 T 之后的 5 分钟数据，只用于事后验证。
6. 汇总候选收益、Top3 执行、目标收益、止损、失败原因和因子分箱。

完整回测产物写入隔离目录，避免覆盖日常运行报告：

- `reports/backtest_runs/<start>_<end>/daily_signals/`
- `reports/backtest_runs/<start>_<end>/data_quality/`
- `reports/backtest_runs/<start>_<end>/backtest_results/`

完整回测不会覆盖 `data/processed/recent_limitups.csv`，日常 `run-daily` 的工作文件保持独立。

回放已保存信号快照的命令：

```powershell
python -m src.cli backtest-history --signals-dir reports/daily_signals --start-date 2026-01-01 --end-date 2026-06-30 --top-n 3 --hold-days 10 --target-return-pct 7 --stop-loss-pct 3
```

`backtest-history` 只读取已经生成好的 `signals_YYYY-MM-DD.csv`，用于复盘、审计或重新评估旧信号。正式跑新样本应使用 `backtest-run`。两者共同遵守：

- 信号生成阶段：只能使用信号日及以前的数据。
- 事后评估阶段：读取信号日之后的 5 分钟数据，统计候选收益、合规买点触发、执行收益、止盈止损和失败原因。

输出文件：

- `history_trades_<start>_<end>.csv`：逐信号明细。所有信号都会保留，包含 `selected_by_topn`、`selected_for_execution`、候选 D2/D3/D5/D10 收益、执行 D2/D3/D5/D10 收益、`target_hit`、`stop_hit`、`first_outcome`、`failure_reason`。
- `history_summary_<start>_<end>.csv`：区间总览，包括样本数、入选数、成交数、成交率、7% 命中率、止损率、平均/中位收益。
- `history_factor_stats_<start>_<end>.csv`：按排名、涨停后天数、信号类型、仓位、承接类型、总分、图形分、承接分、活跃资金分、题材分、低吸区宽度、失效距离做分箱统计。
- `history_review_<start>_<end>.md`：可读版摘要。
- `history_run_log_<start>_<end>.csv/md`：逐信号日运行日志，记录涨停池、信号数、质量通过数、未来数据补取成功/失败数。
- `history_future_fetch_<start>_<end>.csv`：逐代码未来验证数据补取日志，记录失败原因，避免把数据源问题误判为策略失败。

如果要研究“不是只买 Top3，而是所有 allowed 样本的因子表现”，使用：

```powershell
python -m src.cli backtest-history --signals-dir reports/daily_signals --start-date 2026-01-01 --end-date 2026-06-30 --include-all-allowed
```

## 关键风险

- 当前涨停池来自近期数据，历史长区间回测会有幸存者偏差。
- 公共数据源的历史复权数据可能被后续公司行为改写。
- 日线源和 5 分钟源不同，必须持续做收盘价交叉校验。
- 新股或历史不足标的不能为了扩大样本而补空值。
