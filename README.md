# 短线强势股分歧承接研究系统

本项目用于研究 A 股主板涨停后的分歧承接机会，核心目标是从涨停后进入分歧/承接阶段的候选中，筛出次日到后续短窗口具备较高弹性的 Top3 组合。

本项目不是自动交易系统，也不构成投资建议。所有输出都应作为研究和人工复核材料使用。

---

## 当前项目状态

截至当前版本，项目已经完成从交易构想到模型验证的第一阶段闭环：

```text
交易思路形成
→ 历史候选样本生成
→ 因子分析
→ v002 手工排序模型
→ v004a 加权 logistic walk-forward 模型
→ v005 Top3 组合选择器
→ fixed-grid holdout 验证
→ daily v005 shadow flow
```

当前主观察策略是：

```text
policy_v005_v002_regime_fallback
```

它的含义是：

```text
默认使用 v005 fixed-grid Top3 组合
在特定 regime 风险条件下 fallback 到 v002 Top3
```

当前日常输出已经支持同时生成：

```text
v5 policy final Top3     # 主买入观察清单
v5 baseline Top3         # v005 裸 fixed-grid 对照
v2 Top3                  # 旧主流程模型，对照 + fallback 来源
v004a Top3               # logistic 模型直接 Top3，对照
```

日常使用时，只应优先看：

```text
Primary buy list = policy_v005_v002_regime_fallback
```

---

## 交易研究口径

本项目研究的不是普通强势股排序，而是一个较窄的短线结构：

```text
D0 涨停
→ D1 分歧/承接观察
→ D2 低吸/开盘后承接
→ 观察 D2 open 到 D3 high 的短线弹性
```

当前核心目标列是：

```text
target7_d2open_d3high
```

含义是：

```text
从 D2 open 到 D3 high 的最大收益是否达到 7%
```

结算研究中常用的 realized 口径是：

```text
如果 target7_d2open_d3high=True，则按 +7% 结算
否则按 d2open_d3close_return_pct / realized_return_pct 记录
```

因此，本项目更关注 Top3 组合的稳定性，而不是单票 Top1 的绝对胜率。

重点观察指标：

```text
top3_target_rate              # Top3 单票命中率
top3_all_hit_rate             # Top3 全中率
hit_count_0_days              # 0-hit 天数
avg_top3_realized_return      # Top3 等权结算收益
fallback_days                 # fallback 触发天数
changed_from_baseline_days    # policy 相对 v005 baseline 是否改变
```

---

## 当前模型分工

### 1. v002：旧 daily 主排序模型

模型文件：

```text
reports/manual_models/ranking_model_v002_core_momentum_support.json
```

定位：

```text
逐票打分 ranking model
旧 run-daily 默认模型
v005 fallback 的来源
长期对照组
```

特点：

```text
输入 daily signals
→ 对每只候选计算 research_score
→ 按 research_score 生成 daily_rank
→ 输出 v2 Top3
```

v002 仍然保留，不应该删除。原因：

```text
1. v005 fallback policy 需要 v002 Top3
2. v002 是稳定对照组
3. v2/v5 分歧本身是后续复盘的重要信息
```

---

### 2. v004a：加权 logistic walk-forward 模型

核心脚本：

```text
src/v004a.py
```

当前固定 daily 使用的系数文件：

```text
reports/v004a/grid_v2_scored/v004a_coefficients.csv
```

当前固定参数：

```text
v004a_l2 = 0.30
v004a_positive_weight = 1.5
coefficient_predict_date = 2026-06-26
```

定位：

```text
不是最终买入策略
而是给 v005 提供候选池和 v004a rank/score
```

v004a 的训练方式是 walk-forward：对每个 predict_date，只使用更早日期训练，再对当前日期打分。daily v005 阶段固定使用 `coefficient_predict_date=2026-06-26` 对应的一组系数，避免后续 7 月 forward 样本反向污染模型。

---

### 3. v005：Top3 set-level 组合选择器

核心脚本：

```text
src/v005_set_selector.py
```

v005 不是普通单票 ranking model。它不是：

```text
输入股票
→ 每只股票打分
→ 直接排序 Top3
```

它实际是：

```text
读取 v004a / v002 scored candidates
→ 取 v004a Top15 作为候选池
→ 枚举 C(15,3) 个 Top3 组合
→ 根据 fixed grid 的组合特征打分
→ 选择组合分最高的 Top3
```

当前固定参数：

```text
top_n = 3
candidate_top_k = 15
grid_id = 4
v004a_l2 = 0.30
v004a_positive_weight = 1.5
```

当前使用的 fixed grid_id=4 参数为：

```text
min_total_rank_weight = 0.5
avg_total_rank_weight = 0.0
contains_v004a_top3_bonus = 0.0
contains_v002_top3_bonus = 0.0
extreme_vwap_penalty = 0.05
extreme_close_low_penalty = 0.05
extreme_price_penalty = 0.05
rank_dispersion_weight = 0.02
```

v005 的主要价值在于：

```text
它选择的是 Top3 组合，不是单票第一名
它更关注组合稳定性和 0-hit 风险
它能利用 v004a 与 v002 的分歧信息
```

---

### 4. fallback policy：v005 + v002 regime gate

核心脚本：

```text
src/v005_fallback_gate.py
```

当前主策略：

```text
policy_v005_v002_regime_fallback
```

当前 gate 逻辑：

```python
extreme_confirm = v002_extreme_vwap_count >= 2 and v002_extreme_close_low_count >= 2
close_low_dominant = v002_extreme_close_low_count >= 3
v005_weak = v005_avg_v002_rank >= 12

fallback = (extreme_confirm or close_low_dominant) and v005_weak
```

解释：

```text
如果 v002 Top3 同时出现较强的极端承接/结构信号，且 v005 选出的组合在 v002 体系里明显偏弱，才切换到 v002 Top3。
否则保持 v005 fixed-grid Top3。
```

重要原则：

```text
fallback 不应该频繁触发
fallback 是 regime 风险修正，不是每天重新择优
```

---

## 当前验证结果摘要

### 研究内验证

在早期研究阶段，v002、v004a、v005 都经过了历史样本和 walk-forward / set-level 检查。当前最终策略不是裸 v002，也不是裸 v004a，而是：

```text
policy_v005_v002_regime_fallback
```

此前主验证样本中，该策略曾达到：

```text
Top3 all-hit rate: 60%
单票 target hit rate: 83.33%
0-hit days: 0
avg realized: 5.0688%
```

### 新增 fixed-grid holdout：2026-06-26 ~ 2026-06-30

使用固定规则：

```text
coefficient_predict_date = 2026-06-26
grid_id = 4
不重新选 grid
不重新训练 fallback
不做新因子发现
```

新增 3 个成熟 signal_date：

```text
2026-06-26
2026-06-29
2026-06-30
```

主策略结果：

```text
policy_v005_v002_regime_fallback = baseline_v005_fixed_grid
```

原因：这 3 天 fallback 均未触发。

结果摘要：

```text
Top3 单票命中率: 66.67%
Top3 全中率: 33.33%
0-hit 天数: 0
avg_top3_realized_return: +5.0792%
```

每日结果：

```text
2026-06-26: 600888,603067,002654 | hit_count=1/3 | avg_realized=+3.7532%
2026-06-29: 002106,603903,000620 | hit_count=3/3 | avg_realized=+7.0000%
2026-06-30: 002653,600857,002180 | hit_count=2/3 | avg_realized=+4.4843%
```

对照组结果：

```text
v005/policy: top3_target_rate=66.67%, all_hit=33.33%, avg_realized=+5.0792%
v004a Top3:  top3_target_rate=22.22%, all_hit=0%,     avg_realized=-0.6810%
v002 Top3:   top3_target_rate=22.22%, all_hit=0%,     avg_realized=-2.4359%
```

当前结论：

```text
v005 fixed-grid + fallback policy 已通过小样本 forward 初测
但样本仍然偏少，不能过早认定长期稳定
现阶段应冻结模型，继续滚动 forward 记录
```

---

## 项目模块关系

### 数据和信号层

| 模块 | 作用 |
|---|---|
| `src/loaders.py` | 行情数据、涨停池、日线/5分钟线缓存读取与更新 |
| `src/signal_engine.py` | 从单票日线/分钟线生成 D2 分歧承接 signal |
| `src/report.py` | 写 daily signals、data quality、markdown 报告 |
| `src/cli.py` | 原始主 CLI，包含 collect/run-daily/history-samples 等命令 |

---

### 历史样本和评估层

| 模块 | 作用 |
|---|---|
| `src/history_samples.py` | 生成干净历史候选样本，不写执行字段 |
| `src/backtester.py` | 传统执行回测、历史 signals 回测、candidate path metrics |
| `src/research_models.py` | 因子分析 |
| `src/ranking_backtest.py` | 验证 manual ranking model JSON，例如 v002 |

历史样本核心输出：

```text
reports/history_samples/<start>_<end>/history_candidates_<start>_<end>.csv
```

该文件是 v002/v004a/v005 研究链路的重要输入。

---

### 模型研究层

| 模块 | 作用 |
|---|---|
| `src/daily_ranking.py` | daily signals 上加载 manual ranking JSON 并生成 research_score/daily_rank |
| `src/v004a.py` | 加权 logistic walk-forward 研究模型 |
| `src/v004b.py` | pairwise ranking 研究分支，目前不是主线 |
| `src/v005_set_selector.py` | v005 Top3 set-level 组合选择器 |
| `src/v005_objective_sweep.py` | v005 objective sweep 研究辅助 |
| `src/v005_failure_attribution.py` | v005 失败归因辅助 |
| `src/v005_fallback_gate.py` | v005/v002 fallback policy 研究与对照 |

---

### fixed-grid holdout 层

| 模块 | 作用 |
|---|---|
| `src/v005_fixed_grid_holdout.py` | 用固定 v004a 系数 + 固定 grid_id 对新增样本做 holdout 验证 |

它的原则是：

```text
不重新训练
不重新选 grid
不调 fallback
只验证固定策略在新增成熟样本上的表现
```

---

### daily v005 生产观察层

| 模块 | 作用 |
|---|---|
| `src/v005_daily_selector.py` | 基于当天 signals 生成 v005 primary buy list 和对照组 |
| `src/run_daily_v005.py` | CLI 包装模块，一键生成 v2 daily signals + v005 daily 输出 |

当前推荐日常入口是：

```text
python -m src.run_daily_v005
```

而不是直接把 `src.cli run-daily` 替换掉。

---

## 日常运行方式

### 1. 更新代码并做语法检查

```powershell
cd F:\fenqixuangu
git checkout research-sample-analysis
git pull

python -m py_compile src/v005_daily_selector.py
python -m py_compile src/run_daily_v005.py
```

---

### 2. 每天生成 v2 + v005 输出

```powershell
python -m src.run_daily_v005 `
  --date 2026-07-03 `
  --lookback-days 5 `
  --days 10 `
  --workers 6 `
  --coefficient-predict-date 2026-06-26 `
  --grid-id 4
```

日期换成当天交易日即可。

输出终端会直接打印：

```text
primary strategy
fallback_triggered
primary_buy_codes
v005_baseline_codes
v002_codes
v004a_codes
markdown
```

其中最重要的是：

```text
primary_buy_codes
```

它就是当天 `policy_v005_v002_regime_fallback` 的最终 Top3。

---

### 3. 输出目录

原 v2 daily signals：

```text
reports/daily_signals/signals_YYYY-MM-DD.csv
reports/daily_signals/signals_YYYY-MM-DD.md
```

v005 daily 输出：

```text
reports/daily_v005/YYYY-MM-DD/
```

关键文件：

```text
v005_daily_report_YYYY-MM-DD.md
v005_daily_decision_YYYY-MM-DD.csv
v005_daily_selection_YYYY-MM-DD.csv
v005_daily_selected_combos_YYYY-MM-DD.csv
v005_daily_baseline_top3_YYYY-MM-DD.csv
v005_daily_scored_candidates_YYYY-MM-DD.csv
v005_daily_run_meta_YYYY-MM-DD.csv
```

优先阅读：

```text
reports/daily_v005/YYYY-MM-DD/v005_daily_report_YYYY-MM-DD.md
```

报告第一块是：

```text
Primary buy list
```

这一块才是主买入观察清单。

---

### 4. 如果已经有 signals 文件

可以不重新抓取涨停池和数据，直接基于已有 daily signals 生成 v005：

```powershell
python -m src.v005_daily_selector `
  --signals-file reports/daily_signals/signals_2026-07-03.csv `
  --coefficient-predict-date 2026-06-26 `
  --grid-id 4
```

---

## 日常输出如何解读

### Primary buy list

只看：

```text
strategy_role = primary_buy
strategy = policy_v005_v002_regime_fallback
is_primary_buy = True
```

这些行才是主买入观察清单。

---

### Control rows

以下内容只用于观察和复盘：

```text
baseline_control            # v005 baseline
fallback_source_control     # v002 Top3
model_control               # v004a Top3
```

不要把 control rows 误当成买入清单。

---

### fallback 解释

如果：

```text
fallback_triggered = False
action = keep_v005_fixed_grid
```

则：

```text
primary_buy_codes = v005_baseline_codes
```

如果：

```text
fallback_triggered = True
action = fallback_to_v002_regime_policy
```

则：

```text
primary_buy_codes = v002_codes
```

---

## fixed-grid holdout 验证流程

日常输出只是当日推理，不知道未来结果。等标签成熟后，需要用 holdout 脚本做事后验证。

### 1. 生成新增成熟样本

例如验证 2026-07-01 到 2026-07-03：

```powershell
$env:FQ_MIN_5MIN_DAYS = "4"

python -m src.cli generate-history-samples `
  --start-date 2026-07-01 `
  --end-date 2026-07-03 `
  --lookback-days 5 `
  --signal-days 10 `
  --eval-days 10 `
  --hold-days 10 `
  --workers 6
```

### 2. 跑 fixed-grid holdout

```powershell
python -m src.v005_fixed_grid_holdout `
  --samples-file reports/history_samples/2026-07-01_2026-07-03/history_candidates_2026-07-01_2026-07-03.csv `
  --output-dir reports/v005_fixed_grid_holdout_2026-07-01_2026-07-03 `
  --coefficient-predict-date 2026-06-26 `
  --grid-id 4
```

### 3. 看结果

重点看 summary 中的：

```text
policy_v005_v002_regime_fallback
baseline_v005_fixed_grid
v002_top3_control
v004a_top3_control
```

以及：

```text
top3_target_rate
top3_all_hit_rate
hit_count_0_days
avg_top3_realized_return
fallback_days
changed_from_baseline_days
```

---

## forward 样本管理原则

当前项目最大的限制不是模型复杂度，而是有效 forward 样本少。

交易思路是在 2026 年 5 月底形成的，因此后续验证中要特别注意样本污染：

```text
已经参与规则发现/调参的日期 = research sample
规则冻结后新增成熟的日期 = forward holdout sample
```

当前建议：

```text
每 2~3 个交易日批量验证一次
每 5 个成熟 signal_date 做一次小复盘
每 15~20 个成熟 signal_date 做一次正式复盘
到 30+ 个成熟 signal_date 后再考虑 v006 或调参
```

在累计足够 forward 样本之前，不建议：

```text
重新选 grid
重新调 fallback gate
新增因子后马上替换主策略
因为单日亏损就改模型
用 7 月结果反向修 6 月模型
```

当前最重要的是：

```text
冻结模型
固定口径
滚动验证
记录分歧
```

---

## 推荐运行节奏

### 每个交易日盘后

```powershell
python -m src.run_daily_v005 `
  --date YYYY-MM-DD `
  --lookback-days 5 `
  --days 10 `
  --workers 6 `
  --coefficient-predict-date 2026-06-26 `
  --grid-id 4
```

查看：

```text
reports/daily_v005/YYYY-MM-DD/v005_daily_report_YYYY-MM-DD.md
```

只看第一块：

```text
Primary buy list
```

---

### 每 2~3 个交易日

等 D2 open → D3 high 标签成熟后，跑：

```text
generate-history-samples
v005_fixed_grid_holdout
```

把结果追加到 forward tracking 记录中。

---

## 当前冻结配置

在进入下一阶段前，当前不建议改动以下配置：

```text
model policy: policy_v005_v002_regime_fallback
v005 grid_id: 4
v005 candidate_top_k: 15
v005 top_n: 3
v004a l2: 0.30
v004a positive_weight: 1.5
v004a coefficient_predict_date: 2026-06-26
v2 model: reports/manual_models/ranking_model_v002_core_momentum_support.json
```

---

## 常用命令速查

### 原 v2 daily flow

```powershell
python -m src.cli run-daily `
  --date YYYY-MM-DD `
  --lookback-days 5
```

### 当前推荐 v005 daily flow

```powershell
python -m src.run_daily_v005 `
  --date YYYY-MM-DD `
  --lookback-days 5 `
  --days 10 `
  --workers 6 `
  --coefficient-predict-date 2026-06-26 `
  --grid-id 4
```

### 基于已有 signals 文件生成 v005

```powershell
python -m src.v005_daily_selector `
  --signals-file reports/daily_signals/signals_YYYY-MM-DD.csv `
  --coefficient-predict-date 2026-06-26 `
  --grid-id 4
```

### 生成历史样本

```powershell
python -m src.cli generate-history-samples `
  --start-date YYYY-MM-DD `
  --end-date YYYY-MM-DD `
  --lookback-days 5 `
  --signal-days 10 `
  --eval-days 10 `
  --hold-days 10 `
  --workers 6
```

### fixed-grid holdout

```powershell
python -m src.v005_fixed_grid_holdout `
  --samples-file reports/history_samples/YYYY-MM-DD_YYYY-MM-DD/history_candidates_YYYY-MM-DD_YYYY-MM-DD.csv `
  --output-dir reports/v005_fixed_grid_holdout_YYYY-MM-DD_YYYY-MM-DD `
  --coefficient-predict-date 2026-06-26 `
  --grid-id 4
```

### v004a 研究重跑

```powershell
python -m src.cli train-v004a `
  --samples-file reports/history_samples/2026-05-06_2026-06-25/history_candidates_2026-05-06_2026-06-25.csv `
  --output-dir reports/v004a/grid_v2_scored
```

### v005 set selector 研究重跑

```powershell
python -m src.v005_set_selector `
  --scored-file reports/v004a/grid_v2_scored/v004a_scored_candidates.csv `
  --output-dir reports/v005_set_selector
```

---

## 目录说明

```text
src/
  cli.py                         # 原始 CLI 主入口
  loaders.py                     # 数据加载和缓存
  signal_engine.py               # D2 分歧承接 signal 生成
  daily_ranking.py               # v2 manual ranking daily 应用
  history_samples.py             # 历史候选样本生成
  ranking_backtest.py            # manual ranking model 验证
  v004a.py                       # logistic walk-forward 研究
  v005_set_selector.py           # v005 Top3 组合选择器
  v005_fallback_gate.py          # fallback policy 研究
  v005_fixed_grid_holdout.py     # fixed-grid forward holdout 验证
  v005_daily_selector.py         # v005 daily 推理输出
  run_daily_v005.py              # v005 daily CLI 包装

reports/
  manual_models/                 # v001/v002 manual model JSON
  daily_signals/                 # 原 v2 daily signals 输出
  daily_v005/                    # v005 daily primary buy list + controls
  history_samples/               # 历史候选样本
  v004a/                         # v004a 研究输出
  v005_set_selector/             # v005 set selector 研究输出
  v005_fixed_grid_holdout*/      # fixed-grid holdout 输出
```

---

## Git 管理建议

通常不提交运行产物：

```text
data/cache/
data/raw/
data/processed/*.csv
reports/daily_signals/
reports/data_quality/
reports/daily_v005/
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

应保留的模型/配置资产：

```text
reports/manual_models/*.json
reports/v004a/grid_v2_scored/v004a_coefficients.csv
```

是否保留 scored candidates / research reports，视仓库大小和复现实验需要决定。

---

## 风险边界

1. 本项目是研究系统，不是自动交易系统。
2. 当前 v005 结论基于小样本 forward 初测，不能视为长期稳定收益证明。
3. A 股短线策略受行情阶段、题材强度、流动性、停牌/复牌、数据质量、涨停池构造影响很大。
4. daily 输出必须人工复核，尤其要检查一字板、流动性、公告、监管风险、题材退潮和极端高位风险。
5. 后续最重要的工作不是频繁调参，而是严格记录 fixed policy 的 forward 表现。

---

## 当前一句话总结

```text
本项目当前已经完成 v005 fixed-grid + v002 fallback 的研究闭环，进入固定版本 daily shadow flow 与 forward rolling validation 阶段。
```
