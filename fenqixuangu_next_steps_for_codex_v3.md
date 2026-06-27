# fenqixuangu 下一阶段改进策略：先验证思路，再用多日回测寻找调节因子

## 0. 先对当前想法做判断

当前想法是：

```text
通过很多天的历史回测，记录所有候选、触发、失败、收益比例、回撤比例等数据，
然后从这些数据中寻找更合适的调节因子和参数方案。
```

这个方向是正确的，而且应该作为项目下一阶段的核心工作。

原因是：当前系统已经能生成规则信号，但还不能充分回答“哪些规则真的有效、哪些规则只是看起来合理”。要提升准确度，必须先把大量历史信号跑成结构化复盘数据，再从数据中统计哪些因子与收益、回撤、触发率、失败率有关。

不过，这个想法要落地时必须注意几个边界：

```text
1. 不能只记录涨了多少，也要记录有没有真实触发买点。
2. 不能只记录成功样本，也要记录失败、未触发、数据异常样本。
3. 不能只看 D2，当至少记录 D2/D3/D5/D10 的表现。
4. 不能只看最大涨幅，也要看最大回撤和先止损还是先达标。
5. 参数不能直接在全部历史数据上调到最好，否则容易过拟合。
6. 所有收益比例必须基于明确价格，例如 entry_price、D1 close、D2 open。
7. 候选表现和执行表现要分开统计。
```

所以，改进策略应该写入项目，但写入方式不应该只是“多跑回测”，而应该是：

```text
多日批量回测
全量信号记录
候选收益记录
执行收益记录
失败原因分类
因子分桶统计
参数组合扫描
分时间段验证
```

---

## 1. 下一阶段核心目标

把项目从“单日预案生成系统”升级为：

```text
多日历史信号复盘系统
```

系统需要持续回答：

```text
哪些信号真的有效？
哪些信号只是没触发，不应算交易失败？
哪些信号触发后容易亏？
哪些分数、指标、形态和后续收益相关？
哪些阈值能提高命中率？
哪些阈值虽然提高胜率，但会显著减少机会？
哪些参数在不同市场环境下表现不同？
```

---

## 2. 必须先区分三类数据

### 2.1 候选数据

候选数据用于判断系统选股池质量。

即使 D2 没有触发低吸买点，也要记录这只票后续有没有涨。

示例字段：

```text
candidate_d2_max_return_pct
candidate_d3_max_return_pct
candidate_d5_max_return_pct
candidate_d10_max_return_pct
candidate_d2_close_return_pct
candidate_d5_close_return_pct
```

这些字段可以回答：

```text
这个票其实选对了吗？
是不是低吸区设得太保守导致没买到？
没触发但后面下跌，说明过滤有效；
没触发但后面大涨，说明买点规则可能需要优化。
```

### 2.2 执行数据

执行数据只统计真正触发 D2 买点的交易。

示例字段：

```text
triggered
trigger_time
entry_price
d2_max_return_pct
d2_close_return_pct
d3_max_return_pct
d5_max_return_pct
d10_max_return_pct
d2_max_drawdown_pct
d5_max_drawdown_pct
target_hit
stop_hit
first_outcome
```

这些字段可以回答：

```text
触发后是否真的有交易价值？
买点是否太早？
失效位是否太松或太紧？
触发后是先涨到目标，还是先打到止损？
```

### 2.3 排序数据

排序数据用于判断 Top1、Top2、Top3 是否真的优于后排。

示例字段：

```text
daily_rank
rank_method
selected_by_topn
top_n
```

这些字段可以回答：

```text
total_score 排序是否真的有效？
support_score 排序是否更好？
days_since_d0=1 是否应该优先？
Top1 是否长期优于 Top2/Top3？
```

---

## 3. 多日批量回测是第一优先级

### 3.1 新增命令

建议新增：

```powershell
python -m src.cli backtest-history ^
  --signals-dir reports/daily_signals ^
  --start-date 2026-01-01 ^
  --end-date 2026-06-30 ^
  --top-n 3 ^
  --hold-days 10 ^
  --target-return-pct 7 ^
  --stop-loss-pct 3
```

支持回测全量 allowed 信号：

```powershell
python -m src.cli backtest-history ^
  --signals-dir reports/daily_signals ^
  --start-date 2026-01-01 ^
  --end-date 2026-06-30 ^
  --include-all-allowed ^
  --hold-days 10
```

### 3.2 输出文件

```text
reports/backtest_results/history_trades_<start>_<end>.csv
reports/backtest_results/history_summary_<start>_<end>.csv
reports/backtest_results/history_factor_stats_<start>_<end>.csv
reports/backtest_results/history_review_<start>_<end>.md
```

### 3.3 每条记录必须包含

```text
signal_date
code
name
d0_date
days_since_d0
consecutive_boards
signal_type
position_level
support_type

total_score
graph_quality_score
active_money_score
support_score
theme_score

low_absorb_min
low_absorb_max
invalid_price

daily_rank
selected_by_topn

triggered
trigger_time
entry_price
execution_reason

candidate_d2_max_return_pct
candidate_d3_max_return_pct
candidate_d5_max_return_pct
candidate_d10_max_return_pct

d2_max_return_pct
d2_close_return_pct
d2_max_drawdown_pct
d3_max_return_pct
d3_close_return_pct
d3_max_drawdown_pct
d5_max_return_pct
d5_close_return_pct
d5_max_drawdown_pct
d10_max_return_pct
d10_close_return_pct
d10_max_drawdown_pct

target_hit
stop_hit
first_outcome
failure_reason
data_reason
```

---

## 4. 收益比例和增长比例必须统一定义

为了后续统计稳定，所有增长比例都必须有统一计算口径。

### 4.1 执行收益

只对 triggered=True 的样本计算。

```text
dN_max_return_pct = (D2 触发后至 DN 窗口最高价 / entry_price - 1) * 100
dN_close_return_pct = (DN 窗口最后收盘价 / entry_price - 1) * 100
dN_max_drawdown_pct = (D2 触发后至 DN 窗口最低价 / entry_price - 1) * 100
```

### 4.2 候选收益

对所有信号都可以计算，用于判断候选是否选对。

建议基准价使用：

```text
candidate_base_price = D1 close
```

计算：

```text
candidate_dN_max_return_pct = (DN 窗口最高价 / D1 close - 1) * 100
candidate_dN_close_return_pct = (DN 窗口最后收盘价 / D1 close - 1) * 100
candidate_dN_max_drawdown_pct = (DN 窗口最低价 / D1 close - 1) * 100
```

注意：候选收益不是实际交易收益，只用于判断选股池质量。

### 4.3 触发机会收益

对未触发样本，可以记录：

```text
missed_opportunity_pct
```

定义：

```text
如果未触发，但 D2/D3/D5 后续最大涨幅达到目标，则说明可能是低吸区或触发条件太保守。
```

---

## 5. 失败原因分类要进入回测结果

每条信号都要有：

```text
failure_reason
failure_detail
```

建议分类：

```text
not_triggered
D2 没有触发低吸区或没有站回 VWAP。

zone_too_low
未触发，但后续大涨，说明低吸区可能过低。

zone_too_high
触发后快速跌破失效位，说明低吸区可能过高。

break_invalid
跌破失效位后未能收回。

weak_repair
触发后修复弱，收盘表现差。

late_signal
days_since_d0 太大，分歧时效衰减。

bad_graph
图形分或后续走势显示趋势已破坏。

fake_support
看似承接，但实际没有主动修复。

high_volume_fail
高位巨量失败后弱修复。

theme_weak
题材联动弱或题材负反馈。

data_issue
数据质量问题。

unknown
暂无法判断。
```

失败原因的目的不是描述亏损，而是指导后续调参。

例如：

```text
zone_too_low 多：低吸区可能过于保守。
zone_too_high 多：低吸区可能太激进。
late_signal 多：days_since_d0 应该收紧。
high_volume_fail 多：高位巨量失败过滤要加强。
theme_weak 多：题材过滤或题材权重要提高。
```

---

## 6. 因子统计分析

多日回测数据沉淀后，要对因子做分桶统计。

### 6.1 优先分析的因子

```text
days_since_d0
consecutive_boards
signal_type
position_level
support_type

total_score
graph_quality_score
active_money_score
support_score
theme_score

low_absorb_width_pct
invalid_distance_pct

d1_amount_ratio
d1_close_position
d1_amplitude
d1_ma5_distance
d1_ma10_distance
d1_above_vwap_ratio
d1_low_to_close_repair_pct
d1_tail_repair_ratio
```

### 6.2 分桶方式

示例：

```text
graph_quality_score:
0-50
50-60
60-70
70-80
80-100

support_score:
0-45
45-55
55-65
65-75
75-100

days_since_d0:
1
2
3
>3
```

### 6.3 每个桶统计

```text
count
trigger_rate
target_hit_rate
stop_hit_rate
avg_candidate_d5_max_return_pct
avg_d5_max_return_pct
avg_d5_close_return_pct
avg_d5_max_drawdown_pct
median_d5_max_return_pct
profit_loss_ratio
```

输出：

```text
reports/backtest_results/factor_bins_<start>_<end>.csv
reports/backtest_results/factor_bins_<start>_<end>.md
```

---

## 7. 参数组合扫描

### 7.1 目标

用批量回测验证参数，而不是凭感觉调参。

### 7.2 新增命令

```powershell
python -m src.cli sweep-strategy-params ^
  --signals-dir reports/daily_signals ^
  --start-date 2026-01-01 ^
  --end-date 2026-06-30 ^
  --top-n 3 ^
  --graph-min-list 55,60,65,70 ^
  --support-min-list 55,60,65,70 ^
  --active-min-list 50,55,60,65 ^
  --days-since-d0-max-list 1,2,3 ^
  --target-return-pct 7 ^
  --stop-loss-pct 3
```

### 7.3 扫描参数

```text
min_graph_quality_trade
55, 60, 65, 70

min_active_money
50, 55, 60, 65

min_support_trade
55, 60, 65, 70

weak_support_min
40, 45, 50, 55

days_since_d0_max
1, 2, 3

theme_score_min
40, 50, 60

total_score_min
60, 65, 70, 75
```

### 7.4 每组参数输出

```text
param_set_id
selected_count
triggered_count
trigger_rate
target_hit_count
target_hit_rate
stop_hit_count
stop_hit_rate
avg_d5_max_return_pct
avg_d10_max_return_pct
avg_max_drawdown_pct
median_return_pct
profit_loss_ratio
score
```

### 7.5 综合评分

初版可以使用：

```text
score = target_hit_rate * 0.35
      + avg_d5_max_return_pct * 0.25
      + profit_loss_ratio * 0.20
      + trigger_rate * 0.10
      - abs(avg_max_drawdown_pct) * 0.10
```

后续根据实际结果调整。

---

## 8. 防止参数过拟合

参数扫描不能只找全历史最优参数。

必须至少做时间段验证：

```text
前 70% 日期：用于寻找参数
后 30% 日期：用于验证参数
```

更好的方式是滚动验证：

```text
用过去 3 个月调参，验证下 1 个月；
窗口向前滚动；
统计参数是否稳定。
```

输出字段：

```text
train_period
test_period
train_score
test_score
score_decay
```

如果某组参数在训练期很好、测试期明显变差，说明可能过拟合。

---

## 9. 排序方案对比

当前不应只依赖 total_score 排序。应批量对比多种排序方案。

### 9.1 新增命令

```powershell
python -m src.cli backtest-ranking-variants ^
  --signals-dir reports/daily_signals ^
  --start-date 2026-01-01 ^
  --end-date 2026-06-30 ^
  --top-n 3
```

### 9.2 排序方案

```text
baseline_total
按 total_score 排序。

graph_first
优先 graph_quality_score，再 total_score。

support_first
优先 support_score，再 graph_quality_score。

days_first
优先 days_since_d0 = 1，再 total_score。

theme_first
优先 theme_score，再 total_score。

strict_support
只选 support_type = A。

strict_graph_support
graph_quality_score >= 65 且 support_score >= 65。

no_small
排除 small 仓位。

anti_late_signal
排除 days_since_d0 > 2。

hybrid_conservative
support_type = A
days_since_d0 <= 2
graph_quality_score >= 65
support_score >= 65
theme_score >= 50
```

### 9.3 输出指标

```text
variant_name
selected_count
trigger_rate
target_hit_rate
stop_hit_rate
avg_d5_max_return_pct
avg_d10_max_return_pct
avg_max_drawdown_pct
profit_loss_ratio
top1_target_hit_rate
top2_target_hit_rate
top3_target_hit_rate
```

---

## 10. 低吸区间与失效位优化

多日回测后，应专门分析：

```text
low_absorb_min
low_absorb_max
invalid_price
```

新增字段：

```text
low_absorb_width_pct = (low_absorb_max / low_absorb_min - 1) * 100
invalid_distance_pct = (entry_price / invalid_price - 1) * 100
```

重点统计：

```text
low_absorb_width_pct 过窄是否导致 not_triggered 或 zone_too_low 增加？
low_absorb_width_pct 过宽是否导致 stop_first 增加？
invalid_distance_pct 过小是否导致过早止损？
invalid_distance_pct 过大是否导致回撤过大？
```

根据统计结果再调整低吸区和失效位生成规则。

---

## 11. 报告增强

新增历史复盘报告：

```text
reports/backtest_results/history_review_<start>_<end>.md
```

报告结构：

```text
## Summary
- 回测日期范围
- 交易日数量
- 信号数量
- 可评估数量
- 触发数量
- 触发率
- 目标命中率
- 止损率
- 平均 D5 最大收益
- 平均 D5 最大回撤
- 盈亏比

## Candidate Performance
候选层面表现。

## Execution Performance
执行层面表现。

## Ranking Performance
Top1/Top2/Top3 表现。

## By Signal Type
按信号类型统计。

## By Support Type
按承接类型统计。

## By Days Since D0
按时效统计。

## By Score Bucket
按分数分桶统计。

## Failure Reasons
失败原因分布。

## Actionable Findings
自动输出可执行结论。
```

示例可执行结论：

```text
- days_since_d0 = 1 的 target_hit_rate 显著高于 2/3，建议优先保留 D2 最佳窗口。
- support_type = B 的平均回撤高于 A，建议 B 类只保留 small。
- graph_quality_score < 60 的样本亏损集中，建议提高图形硬过滤阈值。
- zone_too_low 占比较高，说明低吸区设置可能过于保守。
- high_volume_fail 样本亏损集中，建议强化高位巨量失败过滤。
```

---

## 12. 开发顺序

### Step 1：实现 backtest-history

目标：

```text
支持多日批量回测。
```

验收：

```text
能读取多个 signals_YYYY-MM-DD.csv；
能按 start-date/end-date 过滤；
能输出 history_trades 和 history_summary。
```

### Step 2：记录完整收益比例

目标：

```text
每条信号记录候选收益和执行收益。
```

验收：

```text
包含 candidate_d2/d3/d5/d10 收益；
包含 triggered 后 d2/d3/d5/d10 收益和回撤。
```

### Step 3：加入失败原因

目标：

```text
每条信号都有 failure_reason。
```

验收：

```text
报告中能看到失败原因分布。
```

### Step 4：实现因子分桶统计

目标：

```text
从多日数据中找有效调节因子。
```

验收：

```text
能按 score、days_since_d0、support_type 等分桶输出表现。
```

### Step 5：实现排序方案对比

目标：

```text
验证当前排序是否最优。
```

验收：

```text
能输出多种排序方案的对比结果。
```

### Step 6：实现参数组合扫描

目标：

```text
用数据寻找更合适的阈值。
```

验收：

```text
能输出不同参数组合在训练期和验证期的表现。
```

---

## 13. 最小可交付版本

MVP 定义为：

```text
多日信号复盘回测系统
```

必须包含：

```text
1. backtest-history 命令；
2. 支持 start-date / end-date；
3. 支持 TopN 和全量 allowed 回测；
4. 记录候选后续收益比例；
5. 记录触发后收益比例；
6. 记录最大回撤比例；
7. 记录是否触发、触发时间、买入价；
8. 记录 target_hit、stop_hit、first_outcome；
9. 记录 failure_reason；
10. 输出 CSV 和 Markdown；
11. 输出按因子分组的统计结果。
```

---

## 14. 最终原则

后续所有改动都应该遵守：

```text
先记录，再判断；
先批量回测，再调参数；
先区分候选表现和执行表现，再谈准确度；
先看收益，也要看回撤；
先看训练期，也要看验证期；
所有参数调整都必须能被数据解释。
```

提升准确度的核心路径是：

```text
多日回测
全量记录
收益比例统计
失败归因
因子分桶
参数扫描
分时间段验证
规则优化
```
